import numpy as np
import torch

def generate_fixed_mask(new_weight,mask_line):  ##wys
    p_grad = new_weight.reshape(-1)
    mask = torch.zeros_like(p_grad, dtype=torch.float32)
    mask[:mask_line * 4096] = 1.0
    p_mask = mask
    p_mask = p_mask.to(p_grad.device)
    return p_mask

def generate_activation_mask(new_weight,mask_ratio):
    p_grad = new_weight.reshape(-1)
    p_mask = np.random.choice([1, 0], size=p_grad.size()[0], p=[mask_ratio, 1 - mask_ratio])
    p_mask = torch.from_numpy(p_mask).to(p_grad.device)
    return p_mask

if __name__ == '__main__':
    array = np.random.rand(14336, 4096).astype(np.float32)

    # 转换为PyTorch张量并移至GPU
    tensor = torch.from_numpy(array).to("cuda")
    #array_normal_f32 = np.random.randn(14336, 4096).astype(np.float32)
    a=generate_activation_mask(tensor,0.2)