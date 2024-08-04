import os
import sys
import hydra
import torch
import math

def get_weight_norms(model):
    module_dict = {
        'frontend3D': model.encoder.frontend.frontend3D,
        'frontend': model.encoder.frontend,
        'encoders': model.encoder.encoders,
        'encoder': model.encoder,
        'decoder': model.decoder,
        'ctc': model.ctc,
        'model': model
    }

    weight_norm_dict = {}
    for module_name in module_dict:
        num_params = 0
        params_sum = 0
        weight_norm = 0
        model_module = module_dict[module_name]

        for name, param in model_module.named_parameters():
            params_sum += torch.sum(param * param)
            num_params += param.numel()
        
        if num_params:
            weight_norm = torch.sqrt(params_sum)/num_params 
            weight_norm = weight_norm.item()
        
        # print(f"{module_name} | {num_params = } | {math.sqrt(params_sum) = }")
        weight_norm_dict[f"{module_name}_weight_norm"] = weight_norm * 1e6
    
    return weight_norm_dict

def get_grad_norms(model):
    module_dict = {
        'frontend3D': model.encoder.frontend.frontend3D,
        'frontend': model.encoder.frontend,
        'encoders': model.encoder.encoders,
        'encoder': model.encoder,
        'decoder': model.decoder,
        'ctc': model.ctc,
        'model': model
    }

    grad_norms_dict = {}
    for module_name in module_dict:
        num_params = 0 # only considering params with non None grads
        grads_sum = 0
        grad_norm = 0
        model_module = module_dict[module_name]

        for name, param in model_module.named_parameters():
            grad = param.grad
            if grad is not None:
                grads_sum += torch.sum(grad * grad)
                num_params += grad.numel()

        if num_params:
            grad_norm = torch.sqrt(grads_sum)/num_params 
            grad_norm = grad_norm.item()
        
        # print(f"{module_name} | {num_params = } | {math.sqrt(grads_sum) = }")
        grad_norms_dict[f"{module_name}_grad_norm"] = grad_norm * 1e6
    
    return grad_norms_dict

# @hydra.main(version_base="1.3", config_path="configs", config_name="config")
# def main(cfg):
#     from lightning_lip2wav import ModelModule
#     modelmodule = ModelModule(cfg)
#     model = modelmodule.model
#     weight_norm_dict = get_weight_norms(model)
#     print(weight_norm_dict)
#     grad_norms_dict = get_grad_norms(model)
#     print(grad_norms_dict)
#     # for name, param in model.encoder.frontend.named_parameters():
#     #     print(f"{name} | {param.shape} | {param.numel()}")

# if __name__ == '__main__':
#     main()