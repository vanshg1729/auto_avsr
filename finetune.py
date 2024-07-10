import os
import hydra
import logging
import numpy as np
import random
from tqdm import tqdm

import torch
import torchaudio
from avg_ckpts import ensemble
import wandb

from cosine import WarmupCosineScheduler
from datamodule.transforms import TextTransform
from datamodule.data_module_phrase import DataModulePhrase

from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import get_model_conf
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.length_bonus import LengthBonus

def print_stats(stats_dict):
    print(f"\n{20*'-'}STATS{20*'-'}")
    for name, value in stats_dict.items():
        print(f"{name}: {value}")
    print(f"{50*'-'}\n")

def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())

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


class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.backbone_args = self.cfg.model.visual_backbone
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list
        self.model = E2E(len(self.token_list), self.backbone_args)
        # self.model.encoder.requires_grad_(False)
        # self.model.ctc.requires_grad_(False)
        self.model.encoder.frontend.frontend3D.requires_grad_(False)

        # language model configuration
        rnnlm = self.cfg.rnnlm
        rnnlm_conf = self.cfg.rnnlm_conf
        penalty = self.cfg.decode.penalty
        ctc_weight = self.cfg.decode.ctc_weight
        lm_weight = self.cfg.decode.lm_weight
        beam_size = self.cfg.decode.beam_size
        
        # beam search decoder
        self.beam_search = get_beam_search_decoder(self.model, self.token_list, rnnlm, rnnlm_conf, penalty, ctc_weight, lm_weight, beam_size)
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

        train_params = [param for param in self.model.parameters() if param.requires_grad == True]
        self.optimizer = torch.optim.AdamW(params=train_params, lr=self.cfg.optimizer.lr)

    def train(self, train_dataloader, epochs):

        self.model = self.model.to(self.device)
        for i in range(epochs):
            self.current_epoch = i
            self.epoch_loss = 0.0
            self.epoch_acc = 0.0
            self.epoch_loss_ctc = 0.0
            self.epoch_loss_att = 0.0
            self.epoch_size = 0.0
            self.print_every = 100

            for batch in tqdm(train_dataloader):
                self.optimizer.zero_grad()

                B, T, C, H, W = batch['inputs'].shape
                inputs = batch['inputs'].transpose(1, 2) # (B, C, T, H, W)

                inputs = inputs.to(self.device)
                input_lengths = batch['input_lengths'].to(self.device)
                targets = batch['targets'].to(self.device)

                loss, loss_ctc, loss_att, acc = self.model(inputs, input_lengths, targets)
                batch_size = len(batch["inputs"])

                self.epoch_loss += loss.item() * batch_size
                self.epoch_loss_ctc += loss_ctc.item() * batch_size
                self.epoch_loss_att += loss_att.item() * batch_size
                self.epoch_acc += acc * batch_size
                self.epoch_size += batch_size

                loss.backward()
                self.optimizer.step()
            
            log_dict = {
                "loss_epoch": self.epoch_loss/self.epoch_size,
                "loss_ctc_epoch": self.epoch_loss_ctc/self.epoch_size,
                "loss_att_epoch": self.epoch_loss_att/self.epoch_size,
                "decoder_acc_epoch": self.epoch_acc/self.epoch_size,
                "epoch": self.current_epoch
            }
            print_stats(log_dict)

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    trainer = Trainer(cfg)
    data_module = DataModulePhrase(cfg)
    train_dataloader = data_module.train_dataloader()

    model = trainer.model
    print(model)
    
    # trainer.train(train_dataloader, epochs=5)

if __name__ == '__main__':
    main()