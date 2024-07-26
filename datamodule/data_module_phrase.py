import os
import numpy as np
import random

import torch
from pytorch_lightning import LightningDataModule

from .phrase_dataset import PhraseDataset
from .transforms import VideoTransform

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# https://github.com/facebookresearch/av_hubert/blob/593d0ae8462be128faab6d866a3a926e2955bde1/avhubert/hubert_dataset.py#L517
def pad(samples, pad_val=0.0):
    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    if len(samples[0].shape) == 1:
        collated_batch = collated_batch.unsqueeze(1)  # targets
    elif len(samples[0].shape) == 2:
        pass  # collated_batch: [B, T, 1]
    elif len(samples[0].shape) == 4:
        pass  # collated_batch: [B, T, C, H, W]
    return collated_batch, lengths


def collate_pad(batch):
    batch_out = {}
    for data_type in batch[0].keys():
        if data_type == 'idx':
            batch_out['ids'] = torch.tensor(
                [s[data_type] for s in batch]
            )
            continue
        pad_val = -1 if data_type == "target" else 0.0
        c_batch, sample_lengths = pad(
            [s[data_type] for s in batch if s[data_type] is not None], pad_val
        )
        batch_out[data_type + "s"] = c_batch
        batch_out[data_type + "_lengths"] = torch.tensor(sample_lengths)
    return batch_out


class DataModulePhrase(LightningDataModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        # self.total_gpus = self.cfg.trainer.devices * self.cfg.trainer.num_nodes

    def _dataloader(self, ds, collate_fn):
        g = torch.Generator()
        g.manual_seed(0)

        return torch.utils.data.DataLoader(
            ds,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            generator=g
        )

    def train_dataloader(self):
        ds_args = self.cfg.data.dataset
        train_ds = PhraseDataset(
            root_dir=ds_args.root_dir,
            label_path=ds_args.train_file,
            video_transform=VideoTransform("train"),
            subset="train",
        )
        return self._dataloader(train_ds, collate_pad)

    # def val_dataloader(self):
    #     ds_args = self.cfg.data.dataset
    #     val_ds = AVDataset(
    #         root_dir=ds_args.root_dir,
    #         label_path=os.path.join(ds_args.root_dir, ds_args.label_dir, ds_args.val_file),
    #         subset="val",
    #         modality=self.cfg.data.modality,
    #         audio_transform=AudioTransform("val"),
    #         video_transform=VideoTransform("val"),
    #     )
    #     # sampler = ByFrameCountSampler(
    #     #     val_ds, self.cfg.data.max_frames_val, shuffle=False
    #     # )
    #     if self.total_gpus > 1:
    #         sampler = DistributedSamplerWrapper(sampler, shuffle=False, drop_last=True)
    #     return self._dataloader(val_ds, sampler, collate_pad)

    def _val_dataloader(self, ds):
        g = torch.Generator()
        g.manual_seed(0)

        dataloader = torch.utils.data.DataLoader(
            ds,
            batch_size=None,
            num_workers=1,
            worker_init_fn=seed_worker,
            generator=g
        )
        return dataloader

    def val_dataloader(self):
        ds_args = self.cfg.data.dataset

        val_dataloaders = []
        if ds_args.test_file is not None:
            dataset = PhraseDataset(
                root_dir=ds_args.root_dir,
                label_path=ds_args.test_file,
                video_transform=VideoTransform("test"),
                subset="test",
            )
            dataloader = self._val_dataloader(dataset)
            val_dataloaders.append(dataloader)

        if ds_args.val_file is not None:
            dataset = PhraseDataset(
                root_dir=ds_args.root_dir,
                label_path=ds_args.val_file,
                video_transform=VideoTransform("test"),
                subset="test",
            )
            dataloader = self._val_dataloader(dataset)
            val_dataloaders.append(dataloader)
        return val_dataloaders