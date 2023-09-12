import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """Implementation of layer normalization for the data format: 
    channels_first. Thus the dimensions of the input and the output will be
    (batch_size, channels, height, width)
    
    Keyword arguments:
    
    arguments -- 
    normalized_dim - dimension of the tensors to be normalized
    eps - epsilon value for Layer Normalization equation

    Return: Normalized value of input tensor x.
    """
    
    def __init__(self, normalized_dim, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_dim))
        self.bias = nn.Parameter(torch.zeros(normalized_dim))
        self.eps = eps
        self.normalized_dim = (normalized_dim, )

    def forward(self, x):
        u = x.mean(1, keepdims = True)
        s = (x-u).pow(2).mean(1, keepdimes = True)
        x = (x-u)/ torch.sqrt( s + self.eps)
        x = self.weight[:,None, None]*x + self.bias[:,None, None]
        return x

class LayerNormChannelLast(nn.Module):

    """Implementation of layer normalization for the data format: 
    channels_last. Thus the dimensions of the input and the output will be
    (batch_size, height, width, channels)
    
    Keyword arguments:
    
    arguments -- 
    normalized_dim - dimension of the tensors to be normalized
    eps - epsilon value for Layer Normalization equation

    Return: Normalized value of input tensor x."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super(LayerNormChannelLast, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x: Tensor) -> Tensor:
        #Just call layer norm from nn.Functional
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    


class DualPathResponseFusionAttention(nn.Module):
  """
    Official implementation of our proposed Dual Path Response Fusion Attention module.
  For better clarity, refer to the symbols given in the paper (Fig 3.)

  
  Keyword arguments:
  argument -- F_g, F_l and F_int are tunable dimensions for Conv 1x1 layers
  Return: returns a torch Tensor n_u2 + theta_fuse
  """
  

  def __init__(self, F_g: int , F_l: int , F_int: int) -> None:
        super(DualPathResponseFusionAttention,self).__init__()
        self.W_f  = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_u = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.gelu = nn.GELU(approximate="none")
        
  def forward(self, u: Tensor , f: Tensor) -> Tensor:
        n_u = self.W_u(u)
        n_u2 = self.gelu(n_u)

        n_f = self.W_f(f)
        psi = self.gelu(n_u + n_f)
        psi = self.W(psi)
        theta_fuse = n_f * psi

        return n_u2 + theta_fuse
  

def conv_relu(in_channels: int , out_channels: int , kernel: int, padding:int) -> function:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )