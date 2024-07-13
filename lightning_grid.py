import os
import csv

import torch
import torchaudio
from pytorch_lightning import LightningModule

from cosine import WarmupCosineScheduler
from datamodule.transforms import TextTransform

from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import get_model_conf
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.length_bonus import LengthBonus
import wandb

def print_stats(stats_dict):
    print(f"\n{20*'-'}STATS{20*'-'}")
    for name, value in stats_dict.items():
        print(f"{name}: {value}")
    print(f"{50*'-'}\n")

def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())

class ModelModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        if self.cfg.data.modality == "audio":
            self.backbone_args = self.cfg.model.audio_backbone
        elif self.cfg.data.modality == "video":
            self.backbone_args = self.cfg.model.visual_backbone

        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list
        self.model = E2E(len(self.token_list), self.backbone_args)

        # language model configuration
        rnnlm = self.cfg.rnnlm
        rnnlm_conf = self.cfg.rnnlm_conf
        penalty = self.cfg.decode.penalty
        ctc_weight = self.cfg.decode.ctc_weight
        lm_weight = self.cfg.decode.lm_weight
        beam_size = self.cfg.decode.beam_size
        
        # beam search decoder
        self.beam_search = get_beam_search_decoder(self.model, self.token_list, rnnlm, rnnlm_conf, penalty, ctc_weight, lm_weight, beam_size)
        print(f"init: self.device: {self.device}")
        self.beam_search.to(device=self.device).eval()

        # -- initialise
        if self.cfg.pretrained_model_path:
            ckpt = torch.load(self.cfg.pretrained_model_path, map_location=lambda storage, loc: storage)
            if self.cfg.transfer_frontend:
                tmp_ckpt = {k: v for k, v in ckpt["model_state_dict"].items() if k.startswith("trunk.") or k.startswith("frontend3D.")}
                self.model.encoder.frontend.load_state_dict(tmp_ckpt)
            elif self.cfg.transfer_encoder:
                tmp_ckpt = {k.replace("encoder.", ""): v for k, v in ckpt.items() if k.startswith("encoder.")}
                self.model.encoder.load_state_dict(tmp_ckpt, strict=True)
            else:
                self.model.load_state_dict(ckpt)
        
        self.epoch_loss = 0.0
        self.epoch_acc = 0.0
        self.epoch_loss_ctc = 0.0
        self.epoch_loss_att = 0.0
        self.epoch_size = 0.0
        self.print_every = 100
        self.result_data = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([{"name": "model", "params": self.model.parameters(), "lr": self.cfg.optimizer.lr}], weight_decay=self.cfg.optimizer.weight_decay, betas=(0.9, 0.98))
        scheduler = WarmupCosineScheduler(optimizer, self.cfg.optimizer.warmup_epochs, self.cfg.trainer.max_epochs, len(self.trainer.datamodule.train_dataloader()))
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def forward(self, sample):
        """
        sample : (B, 1, T, 88, 88)
        """
        print(f"self.device = {self.device}")
        print(f"sample.shape = {sample.shape}")
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)
        enc_feat, _ = self.model.encoder(sample.unsqueeze(0).to(self.device), None)
        print(f"enc_feat.shape = {enc_feat.shape}")
        enc_feat = enc_feat.squeeze(0) # (B, T, C)
        print(f"After squeeze: enc_feat.shape = {enc_feat.shape}")

        nbest_hyps = self.beam_search(enc_feat)
        print(f"\ntype of nbest_hyps: {type(nbest_hyps)}, {len(nbest_hyps)}, {type(nbest_hyps[0])}")
        temp = nbest_hyps[0].asdict()
        print(temp.keys())
        print(f"\nscore : {temp['score']}")
        print(f"scores: {temp['scores']}")
        print(f"yseq: {temp['yseq']}")
        print(f"states: {len(temp['states']['decoder'])}")
        # print(f"{nbest_hyps[0].asdict()['score']}")
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        return predicted

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="train")

    def validation_step(self, batch, batch_idx):
        """
            - batch: tensor of shape (T, 1, H, W)
        """
        inputs = batch['input'].transpose(0, 1).unsqueeze(0) # (B=1, C=1, T, H, W)
        enc_feat, _ = self.model.encoder(inputs.to(self.device), None)
        enc_feat = enc_feat.squeeze(0) # (T, D)

        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")

        token_id = batch["target"]
        actual = self.text_transform.post_process(token_id)
        word_distance = compute_word_level_distance(actual, predicted)

        self.total_edit_distance += word_distance
        self.total_length += len(actual.split())
        wer = self.total_edit_distance/self.total_length
        self.log("wer_iter", wer, on_step=True, logger=True, batch_size=1)

        if self.cfg.verbose:
            print(f"\n{'*' * 70}")
            print(f"{batch_idx} GT: {actual}")
            print(f"{batch_idx} Pred: {predicted}")

            print(f"{batch_idx} dist = {word_distance}, len: {len(actual.split())}")
            print(f"{batch_idx} Sentence WER: {word_distance/len(actual.split())}")
            print(f"{batch_idx} Cur WER: {wer}")
            print(f"{'*' * 70}")

        # Results.csv data for this epoch
        if self.loggers and self.result_data:
            gt_len = len(actual)
            wd = word_distance
            data = [
                actual,
                predicted,
                gt_len,
                wd,
                wd/gt_len,
                self.total_length,
                self.total_edit_distance,
                wer
            ]
            self.result_data.append(data)

        return

    def test_step(self, sample, sample_idx):
        enc_feat, _ = self.model.encoder(sample["input"].unsqueeze(0).to(self.device), None)
        enc_feat = enc_feat.squeeze(0)

        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")

        token_id = sample["target"]
        actual = self.text_transform.post_process(token_id)
        word_distance = compute_word_level_distance(actual, predicted)

        if self.cfg.verbose:
            print(f"\n{'*' * 70}")
            print(f"{sample_idx} GT: {actual}")
            print(f"{sample_idx} Pred: {predicted}")

            print(f"{sample_idx} dist = {word_distance}, len: {len(actual.split())}")
            print(f"{sample_idx} WER: {word_distance/len(actual.split())}")
            print(f"{'*' * 70}")
        self.total_edit_distance += word_distance
        self.total_length += len(actual.split())
        return

    def _step(self, batch, batch_idx, step_type):
        B, T, C, H, W = batch['inputs'].shape
        inputs = batch['inputs'].transpose(1, 2) # (B, C, T, H, W)

        loss, loss_ctc, loss_att, acc = self.model(inputs, batch["input_lengths"], batch["targets"])
        batch_size = len(batch["inputs"])

        if step_type == "train":
            self.epoch_loss += loss.item() * batch_size
            self.epoch_loss_ctc += loss_ctc.item() * batch_size
            self.epoch_loss_att += loss_att.item() * batch_size
            self.epoch_acc += acc * batch_size
            self.epoch_size += batch_size

            # self.log("loss_step", self.epoch_loss/self.epoch_size, on_step=True, logger=True, prog_bar=True)
            # self.log("loss_ctc_step", self.epoch_loss_ctc/self.epoch_size, on_step=True, logger=True)
            # self.log("loss_att_step", self.epoch_loss_att/self.epoch_size, on_step=True, logger=True)
            # self.log("decoder_acc_step", self.epoch_acc/self.epoch_size, on_step=True, logger=True, prog_bar=True)
            # self.log("iteration", self.global_step, on_step=True, logger=True)
        else:
            self.log("loss_val", loss, batch_size=batch_size)
            self.log("loss_ctc_val", loss_ctc, batch_size=batch_size)
            self.log("loss_att_val", loss_att, batch_size=batch_size)
            self.log("decoder_acc_val", acc, batch_size=batch_size)

        if step_type == "train":
            self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        return loss

    def on_train_epoch_start(self):
        # sampler = self.trainer.train_dataloader.loaders.batch_sampler
        # if hasattr(sampler, "set_epoch"):
        #     sampler.set_epoch(self.current_epoch)
        self.epoch_loss = 0.0
        self.epoch_acc = 0.0
        self.epoch_loss_ctc = 0.0
        self.epoch_loss_att = 0.0
        self.epoch_size = 0.0
        return super().on_train_epoch_start()
    
    def on_train_epoch_end(self) -> None:
        log_dict = {
            "loss_epoch": self.epoch_loss/self.epoch_size,
            "loss_ctc_epoch": self.epoch_loss_ctc/self.epoch_size,
            "loss_att_epoch": self.epoch_loss_att/self.epoch_size,
            "decoder_acc_epoch": self.epoch_acc/self.epoch_size,
            "epoch": self.current_epoch
        }
        self.log_dict(log_dict, logger=True)
        print_stats(log_dict)

        return super().on_train_epoch_end()

    def on_validation_epoch_start(self):
        self.total_length = 0
        self.total_edit_distance = 0

        if self.loggers:
            results_dir = os.path.join(self.loggers[0].log_dir, f"results")
            self.results_dir = results_dir

            os.makedirs(results_dir, exist_ok=True)
            results_fp = os.path.join(results_dir, f"test_results_epoch{self.current_epoch}.csv")
            self.results_fp = results_fp
            row_names = [
                "Ground Truth Text",
                "Predicted Text",
                "Length",
                "Word Distance",
                "WER",
                "Total Length",
                "Total Word Distance",
                "Final WER"
            ]
            self.result_data = [row_names]

    def on_validation_epoch_end(self) -> None:
        wer = self.total_edit_distance/self.total_length
        log_dict = {
            "wer_test_epoch": wer,
            "epoch": self.current_epoch
        }
        if self.cfg.wandb:
            wandb.log(log_dict)
        else:
            self.log_dict(log_dict, logger=True)
        print_stats(log_dict)

        if self.loggers and self.result_data:
            with open(self.results_fp, mode='w') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerows(self.result_data)
                print(f"{self.current_epoch = } Successfully written the test results data at {self.results_fp}")

        return super().on_validation_epoch_end()

    def on_test_epoch_start(self):
        self.total_length = 0
        self.total_edit_distance = 0

    def on_test_epoch_end(self):
        wer = self.total_edit_distance/self.total_length
        log_dict = {
            "wer_test_epoch": wer,
            "epoch": self.current_epoch
        }
        if self.cfg.wandb:
            wandb.log(log_dict)
        else:
            self.log_dict(log_dict, logger=True)
        print_stats(log_dict)
        return super().on_test_epoch_end()

def get_beam_search_decoder(model, token_list, rnnlm=None, rnnlm_conf=None, penalty=0, ctc_weight=0.1, lm_weight=0., beam_size=40):
    if not rnnlm:
        lm = None
    else:
        lm_args = get_model_conf(rnnlm, rnnlm_conf)
        lm_model_module = getattr(lm_args, "model_module", "default")
        lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
        lm = lm_class(len(token_list), lm_args)
        torch_load(rnnlm, lm)
        lm.eval()

    scorers = {
        "decoder": model.decoder,
        "ctc": CTCPrefixScorer(model.ctc, model.eos),
        "length_bonus": LengthBonus(len(token_list)),
        "lm": lm
    }
    scorers["lm"] = lm
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": lm_weight,
        "length_bonus": penalty,
    }

    print(weights)
    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )
