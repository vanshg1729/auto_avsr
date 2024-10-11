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

def unfreeze_encoders_top_half(model):
    num_blocks = len(model.encoder.encoders)
    for i in range(num_blocks // 2):
        for param in model.encoder.encoders[i].parameters():
            param.requires_grad_(True)

def unfreeze_encoders_middle_six(model):
    for i in range(3, 9):
        for param in model.encoder.encoders[i].parameters():
            param.requires_grad_(True)

def unfreeze_encoders_bottom_half(model):
    num_blocks = len(model.encoder.encoders)
    for i in range(num_blocks//2, num_blocks):
        for param in model.encoder.encoders[i].parameters():
            param.requires_grad_(True)

def unfreeze_encoders_feed_forward(model):
    num_blocks = len(model.encoder.encoders)
    for i in range(0, num_blocks):
        for name, param in model.encoder.encoders[i].named_parameters():
            if 'feed_forward' in name:
                param.requires_grad_(True)

def unfreeze_encoders_conv_module(model):
    num_blocks = len(model.encoder.encoders)
    for i in range(0, num_blocks):
        for name, param in model.encoder.encoders[i].named_parameters():
            if 'conv_module' in name:
                param.requires_grad_(True)

def unfreeze_encoders_self_attn(model):
    num_blocks = len(model.encoder.encoders)
    for i in range(0, num_blocks):
        for name, param in model.encoder.encoders[i].named_parameters():
            if 'self_attn' in name:
                param.requires_grad_(True)

def unfreeze_encoders_middle_six_conv_module(model):
    for i in range(3, 9):
        for name, param in model.encoder.encoders[i].named_parameters():
            if 'conv_module' in name:
                param.requires_grad_(True)

def unfreeze_encoders_layer3_4(model):
    for i in range(3, 5):
        for param in model.encoder.encoders[i].parameters():
            param.requires_grad_(True)

def unfreeze_encoders_layer5_6(model):
    for i in range(5, 7):
        for param in model.encoder.encoders[i].parameters():
            param.requires_grad_(True)

def unfreeze_encoders_layer7_8(model):
    for i in range(7, 9):
        for param in model.encoder.encoders[i].parameters():
            param.requires_grad_(True)

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

def finetune_encoders_top_half(model):
    model.requires_grad_(False)
    unfreeze_encoders_top_half(model)

def finetune_encoders_bottom_half(model):
    model.requires_grad_(False)
    unfreeze_encoders_bottom_half(model)

def finetune_encoders_middle_six(model):
    model.requires_grad_(False)
    unfreeze_encoders_middle_six(model)

def finetune_encoders_feed_forward(model):
    model.requires_grad_(False)
    unfreeze_encoders_feed_forward(model)

def finetune_encoders_conv_module(model):
    model.requires_grad_(False)
    unfreeze_encoders_conv_module(model)

def finetune_encoders_middle_six_conv_module(model):
    model.requires_grad_(False)
    unfreeze_encoders_middle_six_conv_module(model)

def finetune_encoders_self_attn(model):
    model.requires_grad_(False)
    unfreeze_encoders_self_attn(model)

def finetune_encoders_layer3_4(model):
    model.requires_grad_(False)
    unfreeze_encoders_layer3_4(model)

def finetune_encoders_layer5_6(model):
    model.requires_grad_(False)
    unfreeze_encoders_layer5_6(model)

def finetune_encoders_layer7_8(model):
    model.requires_grad_(False)
    unfreeze_encoders_layer7_8(model)

finetune_funcs = {
    "full": finetune_full,
    "encoder": finetune_encoder,
    "frontend": finetune_frontend,
    "encoders": finetune_encoders,
    "decoder": finetune_decoder,
    'encoders_top_half': finetune_encoders_top_half,
    'encoders_bottom_half': finetune_encoders_bottom_half,
    'encoders_middle_six': finetune_encoders_middle_six,
    'encoders_layer3_4': finetune_encoders_layer3_4,
    'encoders_layer5_6': finetune_encoders_layer5_6,
    'encoders_layer7_8': finetune_encoders_layer7_8,
    'encoders_feed_forward': finetune_encoders_feed_forward,
    'encoders_conv_module': finetune_encoders_conv_module,
    'encoders_self_attn': finetune_encoders_self_attn,
    'encoders_middle_six_conv_module': finetune_encoders_middle_six_conv_module,
    "ctc": finetune_ctc,
}