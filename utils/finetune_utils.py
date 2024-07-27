import torch

def freeze_frontend3D(model):
    model.encoder.frontend.frontend3D.requires_grad_(False)

def unfreeze_frontend3D(model):
    model.encoder.frontend.frontend3D.requires_grad_(True)

def freeze_frontend(model):
    model.encoder.frontend.requires_grad_(False)

def unfreeze_frontend(model):
    model.encoder.frontend.requires_grad_(True)

def freeze_encoders(model):
    """
    This Freezes the 12 EncoderLayer of AutoAVSR
    """
    model.encoder.encoders.requires_grad_(False)

def unfreeze_encoders(model):
    model.encoder.encoders.requires_grad_(True)

def freeze_encoder(model):
    model.encoder.requires_grad_(False)

def unfreeze_encoder(model):
    model.encoder.requires_grad_(True)

def freeze_decoder(model):
    model.decoder.requires_grad_(False)

def unfreeze_decoder(model):
    model.decoder.requires_grad_(True)

def freeze_ctc(model):
    model.ctc.requires_grad_(False) 

def unfreeze_ctc(model):
    model.ctc.requires_grad_(True)

def finetune_full(model):
    model.requires_grad_(True)

def finetune_encoder(model):
    model.requires_grad_(False)
    unfreeze_encoder(model)

def finetune_frontend(model):
    model.requires_grad_(False)
    unfreeze_frontend(model)

def finetune_decoder(model):
    model.requires_grad_(False)
    unfreeze_decoder(model)

def finetune_ctc(model):
    model.requires_grad_(False)
    unfreeze_ctc(model)

def finetune_encoders(model):
    model.requires_grad_(False)
    unfreeze_encoders(model)

finetune_funcs = {
    "full": finetune_full,
    "encoder": finetune_encoder,
    "frontend": finetune_frontend,
    "encoders": finetune_encoders,
    "decoder": finetune_decoder,
    "ctc": finetune_ctc,
}