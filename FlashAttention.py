import torch as th
import torch.nn as nn
import torch.nn.functional as F

# Implementation of Flash Attention

# Check out my implementation of attention in my Diffusion repo: https://github.com/aLadNamedPat/Diffusion-Models/blob/main/Classic%20Diffusion/Attention.py

class MultiHeadedFlashAttention(nn.Module):
    def __init__(
        self,
        n_channels : int,
        heads : int = 8,
    ) -> None:
        super(MultiHeadedFlashAttention, self).__init__()

        self.channels = n_channels
        self.num_heads = heads
        self.gamma = nn.Parameter(th.tensor([0.]))
        
        # Compute query, key and values together instead of separately
        self.qkv = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels * 3,
            kernel_size=1
        )
        

        self.l = nn.Conv2d(
            in_channels= n_channels,
            out_channels= n_channels,
            kernel_size= 1
        )
        self.softmax = nn.Softmax(dim = -1)

    
    def forward(
        self,
        input : th.Tensor
    ) -> th.Tensor:
    
        batch_size, channels, w, h = input.size()

        qkv = self.qkv(input)

        q = qkv[:, 0, :, :].view(batch_size, self.num_heads, -1,  w * h).permute(0, 1, 3, 2)
        k = qkv[:, 1, :, :].view(batch_size, self.num_heads, -1 , w * h)
        v = qkv[:, 2, :, :].view(batch_size, self.num_heads, -1, w * h)
        
        o = F.scaled_dot_product_attention(q, k, v)
        # F.scaled_dot_product_attention utilizes flash attention which offers significant speed ups in memory

        o = o.view(batch_size, channels, w, h)
        o = self.l(o)
        o = self.gamma * o + input

        return o