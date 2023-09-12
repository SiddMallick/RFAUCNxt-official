import torch
import torch.nn as nn
from torch import Tensor
from model_utils import LayerNorm, LayerNormChannelLast
from timm.models.layers import trunc_normal_, DropPath

class ConvNextEncoderBlock(nn.Module):
    """sumary_line
    
    Keyword arguments:
    argument -- description
    Return: return_description
    """
    def __init__(self, dim: int , drop_path:float = 0., layer_scale_init_value: float=1e-6 ) -> None:
        super(ConvNextEncoderBlock, self).__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size =7, padding = 3, groups = dim) #Depth-wise Convolution
        self.layer_norm = LayerNormChannelLast(dim, eps = 1e-6)
        self.pw_conv1 = nn.Linear(dim, 4 * dim) #Pointwise or 1x1 Conv layer
        self.gelu = nn.GELU()
        self.pw_conv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
                                    layer_scale_init_value * torch.ones((dim)),
                                    requires_grad = True) if layer_scale_init_value> 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x: Tensor) -> Tensor:
        input_x = x
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1) # Permuting Dimensions --- (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.gelu(x)
        x = self.pw_conv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # Permuting Dimensions --- (N, H, W, C) -> (N, C, H, W)

        x = input_x + self.drop_path(x)
        return x

class ConvNextDecoderBlock(nn.Module):
    """sumary_line
    
    Keyword arguments:
    argument -- description
    Return: return_description
    """
    
    def __init__(self, input_dim:int, output_dim:int, stride:int = 1 , padding:int = 1) -> None:
        super(ConvNextDecoderBlock, self).__init__()

        self.mod_convNext_block = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size = 7, stride = 1, padding = 1, groups = output_dim),
            nn.BatchNorm2d(output_dim),
            nn.Conv2d(output_dim, output_dim, kernel_size = 1, stride = 1, padding = 1),
            nn.GELU(approximate = 'none'),
            nn.Conv2d(output_dim, output_dim, kernel_size = 1, stride = 1, padding = 1)
        )

        self.conv_skip_connection = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(output_dim),
        )

        self.gelu = nn.GELU(approximate = 'none')

    def forward(self, x : Tensor) -> Tensor:
        return self.gelu(self.mod_convNext_block(x) + self.conv_skip_connection(x))

