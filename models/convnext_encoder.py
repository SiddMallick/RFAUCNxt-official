import torch
import torch.nn as nn
from .model_utils import LayerNorm
from .blocks import ConvNextEncoderBlock
from timm.models.layers import trunc_normal_, DropPath

class ConvNext(nn.Module):
    """
    Standard ConvNeXt model taken from 
    "Liu, Zhuang, et al. "A convnet for the 2020s." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022."
    
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1000,
                 depths = [3,3,9,3], dims = [96, 192, 384, 768], drop_path_rate: int = 0.,
                 layer_scale_init_value:int  = 1e-6, head_init_scale: int = 1.,
                 ) -> None:
        

        super(ConvNext, self).__init__()
        
        #Stem operation and 3 downsampling layers
        self.downsample_layers = nn.ModuleList() 

        stem_op = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6)
        )

        self.downsample_layers.append(stem_op)

        #Staging downsampling layers
        for i in range(3):
            downsample = nn.Sequential(
                LayerNorm(dims[i], eps = 1e-6),
                nn.Conv2d(dims[i], dims[i+1], kernel_size = 2, stride = 2)
            )
            self.downsample_layers.append(downsample)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        current = 0

        for i in range(4):
            #Construct each block one by one
            stage = nn.Sequential(
                *[ConvNextEncoderBlock(dim=dims[i], drop_path=dp_rates[current+j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
        
            self.stages.append(stage)
            current += depths[i]

        #This is the standard layer normalization for the last layer
        self.norm = nn.LayerNorm(dims[-1], eps = 1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        
        x = self.head(x)
        return x
        
weight_urls = {
    "convnext_tiny" : "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
}


def convnext_build(model_size: str = 'convnext_tiny', pretrained: bool = False, **kwargs) -> ConvNext:
    """sumary_line
    
    Keyword arguments:
    argument -- description
    Return: return_description
    """

    if model_size == 'convnext_tiny':
        model = ConvNext(depths = [3, 3, 9, 3], dims = [96,192,384,768], **kwargs)

        if pretrained:
            pretrained_weight_url = weight_urls['convnext_tiny']
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_weight_url, map_location = "cpu")
            model.load_state_dict(checkpoint["model"])
        return model
    elif model_size == 'convnext_small':
        model = ConvNext(depths = [3, 3, 27, 3], dims = [96, 192, 384, 768], **kwargs)

        if pretrained:
            pretrained_weight_url = weight_urls['convnext_small']
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_weight_url, map_location = "cpu")
            model.load_state_dict(checkpoint["model"])
        return model
    
    elif model_size == 'convnext_base':
        model = ConvNext(depths = [3, 3, 27, 3], dims = [128, 256, 512, 1024], **kwargs)

        if pretrained:
            pretrained_weight_url = weight_urls['convnext_base']
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_weight_url, map_location = "cpu")
            model.load_state_dict(checkpoint["model"])
        return model
    elif model_size == 'convnext_large':
        model = ConvNext(depths = [3, 3, 27, 3], dims = [192, 384, 768, 1536], **kwargs)

        if pretrained:
            pretrained_weight_url = weight_urls['convnext_large']
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_weight_url, map_location = "cpu")
            model.load_state_dict(checkpoint["model"])
        return model
    
    

