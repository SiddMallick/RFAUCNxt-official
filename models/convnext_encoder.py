import torch
import torch.nn as nn
from model_utils import LayerNorm
from blocks import ConvNextEncoderBlock

class ConvNext(nn.Module):
    """
    Standard ConvNeXt model taken from 
    "Liu, Zhuang, et al. "A convnet for the 2020s." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022."
    
    """
    def __init__(self, in_channels = 3, num_classes = 100,
                 depths = [3,3,9,3], dims = [96,192,384,768], drop_path_rate = 0.,
                 layer_scale_init_value = 1e-6, head_init_scale = 1.,
                 ):
        

        super(ConvNext, self).__init__()
        
        #Stem operation and 3 downsampling layers
        self.encoder_downsampling_collection = nn.ModuleList() 

        stem_op = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6)
        )

        self.encoder_downsampling_collection.append(stem_op)

        #Staging downsampling layers
        for i in range(3):
            downsample = nn.Sequential(
                LayerNorm(dims[i], eps = 1e-6),
                nn.Conv2d(dims[i], dims[i+1], kernel_size = 2, stride = 2)
            )
            self.encoder_downsampling_collection.append(downsample)

        self.convnext_stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linespace(0, drop_path_rate, sum(depths))]

        current = 0

        for i in range(4):
            #Construct each block one by one
            stage = nn.Sequential(
                *[ConvNextEncoderBlock(dim=dims[i], drop_path=dp_rates[current+j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
        
            self.convnext_stages.append(stage)
            current += depths[i]

        #This is the standard layer normalization for the last layer
        self.layer_norm = nn.LayerNorm(dims[-1], eps = 1e-6)
        self.head = nn.Linear(dims[-1], num_classes)


    def forward(self, x):

        for i in range(4):
            x = self.encoder_downsampling_collection[i](x)
            x = self.convnext_stages[i](x)
        
        x = self.head(x)
        return x
        


