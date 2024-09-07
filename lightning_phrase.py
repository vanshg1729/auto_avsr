import os
import csv
import numpy as np

import torch
from torch.optim.lr_scheduler import StepLR
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

from utils.norm_utils import get_weight_norms, get_grad_norms
import wandb

def get_lr(opt):
    return opt.param_groups[0]['lr']

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
        
        self.log_dir = None
        self.epoch_loss = 0.0
        self.epoch_acc = 0.0
        self.epoch_loss_ctc = 0.0
        self.epoch_loss_att = 0.0
        self.epoch_size = 0.0
        self.print_every = 100
        self.cur_train_step = 0
        self.result_data = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([{"name": "model", "params": self.model.parameters(), "lr": self.cfg.optimizer.lr}], weight_decay=self.cfg.optimizer.weight_decay, betas=(0.9, 0.98))
        # scheduler = WarmupCosineScheduler(optimizer, self.cfg.optimizer.warmup_epochs, self.cfg.trainer.max_epochs, len(self.trainer.datamodule.train_dataloader()))
        # scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        # scheduler = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
        # return [optimizer], [scheduler]
        return [optimizer]

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

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
            - batch: tensor of shape (T, C=1, H, W)
            NOTE: This is only to be used with Batch Size = 1 and no collate function
        """
        # Preparing the input for the model
        idx = dataloader_idx # idx is same as dataloader index
        T = batch['input'].shape[0]
        inputs = batch['input'].transpose(0, 1).unsqueeze(0) # (B=1, C=1, T, H, W)
        print(f"")
        # inputs = inputs.to(self.device)
        input_lengths = torch.tensor([T]).to(self.device)
        targets = batch['target'].unsqueeze(0).unsqueeze(0)

        # Getting Encoder features for CTC Beam search decoder
        enc_feat, _ = self.model.encoder(inputs, None)
        enc_feat = enc_feat.squeeze(0) # (T, D)

        # Model losses
        # print(f"{inputs.shape = } | {input_lengths = } | {targets = }")
        loss, loss_ctc, loss_att, acc = self.model(inputs, input_lengths, targets)
        batch_size = 1
        self.val_loss[idx] += loss.item() * batch_size
        self.val_loss_ctc[idx] += loss_ctc.item() * batch_size
        self.val_loss_att[idx] += loss_att.item() * batch_size
        self.val_acc[idx] += acc * batch_size
        self.val_epoch_size[idx] += batch_size
        # print(f"{batch_idx = } | {enc_feat.shape = }")

        # Getting the predicted output of the model using beam search
        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")

        # Calculating the word distance
        video_path = batch['video_path']
        token_id = batch["target"] # (Lmax,)
        actual = batch['transcript']
        # actual = self.text_transform.post_process(token_id)
        word_distance = compute_word_level_distance(actual, predicted) # edit distance
        word_distance = torch.tensor(word_distance).to(self.device) # (1, )
        gt_len = torch.tensor(len(actual.split())).to(self.device) # length of GT sentence (1, )
        data_id = batch['idx'].to(self.device) # (1, )

        # Gathering the quantities across devices
        wd_gahter = self.all_gather(word_distance, sync_grads=False).reshape(-1) # (W, )
        gt_len_gather = self.all_gather(gt_len, sync_grads=False).reshape(-1) # (W, )
        data_ids = self.all_gather(data_id, sync_grads=False).reshape(-1) # (W, )

        # Putting the gathered quantities on cpu
        wd_gather = wd_gahter.to('cpu').numpy() # (W, )
        gt_len_gather = gt_len_gather.to('cpu').numpy() # (W, )
        data_ids = data_ids.to('cpu').numpy() # (W, )

        # Combining the gathered quantities with previous data
        self.ids[idx] = np.concatenate([self.ids[idx], data_ids], axis=0) # (D, )
        self.word_dists[idx] = np.concatenate([self.word_dists[idx], wd_gather], axis=0) # (D, )
        self.sent_lengths[idx] = np.concatenate([self.sent_lengths[idx], gt_len_gather], axis=0) # (D, )
        
        # Calculating the unique ids of the datapoints
        _, unique_ids = np.unique(self.ids[idx], return_index=True, axis=0)

        # Printing the ids of data
        # if self.global_rank == 0:
        #     print(f"\n{'*' * 70}")
        #     print(f"{self.ids[idx] = }")
        #     print(f"{unique_ids = }")
        #     print(f"{'*' * 70}")

        # Calculating WER based on dataloader index
        # We are only considering unique data ids here
        self.total_edit_distance[idx] = self.word_dists[idx][unique_ids].sum()
        self.total_length[idx] = self.sent_lengths[idx][unique_ids].sum()
        wer = self.total_edit_distance[idx]/self.total_length[idx]
        
        # Logging only from Global Rank 0 process
        if self.global_rank == 0:
            if dataloader_idx == 0:
                self.log("test_wer_iter", wer, on_step=True, on_epoch=False, logger=True, batch_size=1)
            else:
                self.log(f"val{idx}_wer_iter", wer, on_step=True, on_epoch=False, batch_size=1)

        # Printing Stats for this datapoint
        if self.cfg.verbose:
            print(f"\n{'*' * 70}"
                  + f"\n{dataloader_idx = }"
                  + f"\n{data_id} video_path: {video_path}"
                  + f"\n{data_id} GT: {actual}"
                  + f"\n{data_id} Pred: {predicted}"
                  + f"\n{data_id} dist = {word_distance}, len: {len(actual.split())}"
                  + f"\n{data_id} Sentence WER: {word_distance/len(actual.split())}"
                  + f"\n{data_id} Cur WER: {wer}"
                  + f"\n{'*' * 70}")

        # Results.csv data for this epoch
        if self.loggers and self.result_data:
            gt_len = gt_len.item()
            wd = word_distance.item()
            data = [
                data_id.item(),
                video_path,
                actual,
                predicted,
                gt_len,
                wd,
                wd/gt_len,
                self.total_length[idx],
                self.total_edit_distance[idx],
                wer
            ]
            self.result_data[idx].append(data)

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

        torch.cuda.empty_cache()
        loss, loss_ctc, loss_att, acc = self.model(inputs, batch["input_lengths"], batch["targets"])
        batch_size = len(batch["inputs"])

        if step_type == "train":
            self.epoch_loss += loss.item() * batch_size
            self.epoch_loss_ctc += loss_ctc.item() * batch_size
            self.epoch_loss_att += loss_att.item() * batch_size
            self.epoch_acc += acc * batch_size
            self.epoch_size += batch_size
            opt = self.optimizers()
            self.cur_lr = get_lr(opt)
            if self.global_rank == 0:
                # print(f"{self.cur_train_step = } | {self.cur_lr = }")
                lr_dict = {'train_step': self.cur_train_step, 'lr': self.cur_lr}
                if self.cfg.wandb:
                    wandb.log(lr_dict)

            self.cur_train_step += 1

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

        if step_type == "train" and self.global_rank == 0:
            self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        return loss

    def on_train_batch_start(self, batch, batch_idx):
        weight_norms_dict = get_weight_norms(self.model)
        if self.global_rank == 0:
            if self.cfg.wandb:
                wandb.log(weight_norms_dict)
            # self.log_dict(weight_norms_dict, on_step=True, on_epoch=False,
            #               logger=True)
        
        return super().on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(self, out, batch, batch_idx):
        grad_norms_dict = get_grad_norms(self.model)
        if self.global_rank == 0:
            if self.cfg.wandb:
                wandb.log(grad_norms_dict)
            
            # TODO : This only logs at intervals of 50 for some reason (figure it out)
            # self.log_dict(grad_norms_dict, on_step=True, on_epoch=False,
            #               logger=True)
        
        return super().on_train_batch_end(out, batch, batch_idx)

    def on_train_epoch_start(self):
        if self.log_dir is None:
            if self.loggers:
                self.log_dir = self.loggers[0].log_dir
                print(f"{self.log_dir = }")
                if self.cfg.wandb and self.global_rank == 0:
                    wandb.log({'log_dir': self.log_dir})
            
        self.epoch_loss = torch.tensor(0.0).to(self.device)
        self.epoch_acc = torch.tensor(0.0).to(self.device)
        self.epoch_loss_ctc = torch.tensor(0.0).to(self.device)
        self.epoch_loss_att = torch.tensor(0.0).to(self.device)
        self.epoch_size = torch.tensor(0.0).to(self.device)
        return super().on_train_epoch_start()
    
    def on_train_epoch_end(self) -> None:
        # Gathering the values across GPUs
        self.epoch_loss = self.all_gather(self.epoch_loss, sync_grads=False).sum()
        self.epoch_loss_ctc = self.all_gather(self.epoch_loss_ctc, sync_grads=False).sum()
        self.epoch_loss_att = self.all_gather(self.epoch_loss_att, sync_grads=False).sum()
        self.epoch_acc = self.all_gather(self.epoch_acc, sync_grads=False).sum()
        self.epoch_size = self.all_gather(self.epoch_size, sync_grads=False).sum()

        # Logging from process with global rank = 0
        if self.global_rank == 0:
            log_dict = {
                "train_loss_epoch": self.epoch_loss/self.epoch_size,
                "train_loss_ctc_epoch": self.epoch_loss_ctc/self.epoch_size,
                "train_loss_att_epoch": self.epoch_loss_att/self.epoch_size,
                "train_decoder_acc_epoch": self.epoch_acc/self.epoch_size,
                "epoch": self.current_epoch
            }
            self.log_dict(log_dict, logger=True)
            print_stats(log_dict)

        return super().on_train_epoch_end()

    def on_validation_epoch_start(self):
        if self.log_dir is None:
            if self.loggers:
                self.log_dir = self.loggers[0].log_dir
                print(f"{self.log_dir = }")
                if self.cfg.wandb and self.global_rank == 0:
                    wandb.log({'log_dir': self.log_dir})

        self.num_val_loaders = len(self.trainer.val_dataloaders) # V
        self.total_length = [0 for i in range(self.num_val_loaders)]
        self.total_edit_distance = [0 for i in range(self.num_val_loaders)]
        self.result_data = [[] for i in range(self.num_val_loaders)]

        # Validation loss and accuracies
        self.val_loss = torch.zeros(self.num_val_loaders).to(self.device) # (V, )
        self.val_loss_ctc = torch.zeros(self.num_val_loaders).to(self.device) # (V, )
        self.val_loss_att = torch.zeros(self.num_val_loaders).to(self.device) # (V, )
        self.val_acc = torch.zeros(self.num_val_loaders).to(self.device) # (V, )
        self.val_epoch_size = torch.zeros(self.num_val_loaders).to(self.device) # (V, )

        # For storing the metrics across GPUs and dataloaders
        # each individual [] will store the metrics for that val loader
        self.ids = [np.array([]) for i in range(self.num_val_loaders)]
        self.word_dists = [np.array([]) for i in range(self.num_val_loaders)]
        self.sent_lengths = [np.array([]) for i in range(self.num_val_loaders)]

        if self.loggers:
            results_dir = os.path.join(self.loggers[0].log_dir, f"results")
            self.results_dir = results_dir
            self.results_filepaths = ['' for idx in range(self.num_val_loaders)]

            # Making the results.csv files to analyse the generated output text
            os.makedirs(results_dir, exist_ok=True)
            for i in range(self.num_val_loaders):
                if i == 0:
                    results_filename = f"test_results_epoch{self.current_epoch}.csv"
                else:
                    results_filename = f"val{i}_results_epoch{self.current_epoch}.csv"

                results_fp = os.path.join(results_dir, results_filename)
                self.results_filepaths[i] = results_fp

                # Wrote the column names to the results csv file (only from global rank = 0)
                if self.global_rank == 0:
                    row_names = [
                        "Index",
                        "Video Path",
                        "Ground Truth Text",
                        "Predicted Text",
                        "Length",
                        "Word Distance",
                        "WER",
                        "Total Length",
                        "Total Word Distance",
                        "Final WER"
                    ]
                    with open(self.results_filepaths[i], mode='w') as file:
                        writer = csv.writer(file, delimiter=',')
                        writer.writerow(row_names)

    def on_validation_epoch_end(self) -> None:
        # Gathering the values across GPUs
        # sizes = (W, V)
        self.val_loss = self.all_gather(self.val_loss, sync_grads=False).sum(dim=0) # (V, )
        self.val_loss_ctc = self.all_gather(self.val_loss_ctc, sync_grads=False).sum(dim=0) # (V, )
        self.val_loss_att = self.all_gather(self.val_loss_att, sync_grads=False).sum(dim=0) # (V, )
        self.val_acc = self.all_gather(self.val_acc, sync_grads=False).sum(dim=0) # (V, )
        self.val_epoch_size = self.all_gather(self.val_epoch_size, sync_grads=False).sum(dim=0) # (V, )

        # Logging from process with global rank = 0
        if self.global_rank == 0:
            for idx in range(self.num_val_loaders):
                if idx == 0:
                    val_type = "test"
                else:
                    val_type = f"val{idx}"
                log_dict = {
                    f"{val_type}_loss_epoch": self.val_loss[idx]/self.val_epoch_size[idx],
                    f"{val_type}_loss_ctc_epoch": self.val_loss_ctc[idx]/self.val_epoch_size[idx],
                    f"{val_type}_loss_att_epoch": self.val_loss_att[idx]/self.val_epoch_size[idx],
                    f"{val_type}_decoder_acc_epoch": self.val_acc[idx]/self.val_epoch_size[idx],
                    f"epoch": self.current_epoch
                }
                self.log_dict(log_dict, logger=True)
                print_stats(log_dict)

        # Loop over the results of all the different validation loaders
        for idx in range(self.num_val_loaders):
            # Only do logging from Global Rank = 0 process
            if self.global_rank == 0:
                wer = self.total_edit_distance[idx]/self.total_length[idx]
                log_dict = {'epoch': self.current_epoch}
                if idx == 0:
                    log_dict['wer_test_epoch'] = wer
                else:
                    log_dict[f"wer_val{idx}_epoch"] = wer

                if self.cfg.wandb:
                    wandb.log(log_dict)
                else:
                    self.log_dict(log_dict, logger=True)
                print_stats(log_dict)

            # Write the csv results file for the corresponding val dataloader
            if self.loggers and self.result_data[idx]:
                with open(self.results_filepaths[idx], mode='a') as file:
                    writer = csv.writer(file, delimiter=',')
                    writer.writerows(self.result_data[idx])
                    print(f"{self.current_epoch = } Successfully written the results data at {self.results_filepaths[idx]}")

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
