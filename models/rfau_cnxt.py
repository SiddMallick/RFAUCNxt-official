import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from convnext_encoder import convnext_build
from model_utils import conv_relu, DualPathResponseFusionAttention
from blocks import ConvNextDecoderBlock

class ResponseFusionAttentionUConvNextTiny(nn.Module):
  def __init__(self, pretrained_encoder_backbone: bool = True, n_class: int = 1, **kwargs) -> None:
    super(ResponseFusionAttentionUConvNextTiny,self).__init__()
    #Down part of convunext
    self.encoder = convnext_build(
      model_size = 'convnext_tiny',
      pretrained = pretrained_encoder_backbone,
      **kwargs
    )
    self.downsampling_layers = list(self.encoder.downsample_layers.children())
    self.encoder_layers = list(self.encoder.stages.children())
    self.layer0 = nn.Sequential(self.downsampling_layers[0], self.encoder_layers[0])
    self.layer0_1x1 = conv_relu(96,96,1,0) 
    self.layer1 = nn.Sequential(self.downsampling_layers[1], self.encoder_layers[1])
    self.layer1_1x1 = conv_relu(192,192,1,0) 
    self.layer2 = nn.Sequential(self.downsampling_layers[2], self.encoder_layers[2])
    self.layer2_1x1 = conv_relu(384,384,1,0)

    #bottleneck
    self.bottleneck = nn.Sequential(self.downsampling_layers[3])
    self.bott_1x1 = conv_relu(768,768,1,0)
    #Up part of convunext
    self.ups = nn.ModuleList()

    for feature in [384,192,96]:
      self.ups.append(
          nn.ConvTranspose2d(
              feature*2, feature, kernel_size = 2, stride = 2,
          )
      )
      self.ups.append(DualPathResponseFusionAttention(F_g = feature, F_l = feature, F_int = feature))
      self.ups.append(ConvNextDecoderBlock(feature*2, feature,1,1))

    #Last conv layer
    self.conv_last = nn.Conv2d(96, n_class, kernel_size = 1)

  def forward(self, input: torch.Tensor) -> torch.Tensor:

    layer0 = self.layer0(input)
    layer1 = self.layer1(layer0)
    layer2 = self.layer2(layer1)

    bottleneck = self.bottleneck(layer2)
    bottleneck = self.bott_1x1(bottleneck)

    x = self.ups[0](bottleneck) #upsample 2
    layer2 = self.layer2_1x1(layer2)
    layer2 = self.ups[1](u = x, f = layer2)
    x = torch.cat([x,layer2], dim = 1)
    x = self.ups[2](x) #Double Convolutions

    x = self.ups[3](x) #upsample1
    layer1 = self.layer1_1x1(layer1)
    layer1 = self.ups[4](u = x, f = layer1)
    x = torch.cat([x,layer1] , dim = 1)
    x = self.ups[5](x) #Double Convolutions

    x = self.ups[6](x)
    layer0 = self.layer0_1x1(layer0)
    layer0 = self.ups[7](u=x, f=layer0)
    x = torch.cat([x,layer0],dim = 1)
    x = self.ups[8](x)

    mask = self.conv_last(x)

    return nn.functional.interpolate(mask, size=(224,224), mode="bilinear", align_corners=False)


class ResponseFusionAttentionUConvNextSmall(nn.Module):
  def __init__(
      self, pretrained_encoder_backbone: bool = True, n_class: int = 1, **kwargs
  ) -> None:
    super(ResponseFusionAttentionUConvNextSmall,self).__init__()
    #Down part of convunext
    self.encoder = convnext_build(
      model_size = 'convnext_small', pretrained = pretrained_encoder_backbone, **kwargs
    )

    self.downsampling_layers = list(self.encoder.downsample_layers.children())
    self.encoder_layers = list(self.encoder.stages.children())
    self.layer0 = nn.Sequential(self.downsampling_layers[0], self.encoder_layers[0])
    self.layer0_1x1 = conv_relu(96,96,1,0)    
    self.layer1 = nn.Sequential(self.downsampling_layers[1], self.encoder_layers[1])
    self.layer1_1x1 = conv_relu(192,192,1,0)    
    self.layer2 = nn.Sequential(self.downsampling_layers[2], self.encoder_layers[2])
    self.layer2_1x1 = conv_relu(384,384,1,0)
    #bottleneck
    self.bottleneck = nn.Sequential(self.downsampling_layers[3]) 
    self.bott_1x1 = conv_relu(768,768,1,0)
    #Up part of convunext
    self.ups = nn.ModuleList()

    for feature in [384,192,96]:
      self.ups.append(
          nn.ConvTranspose2d(
              feature*2, feature, kernel_size = 2, stride = 2,
          )
      )
      self.ups.append(DualPathResponseFusionAttention(F_g = feature, F_l = feature, F_int = feature))
      self.ups.append(ConvNextDecoderBlock(feature*2, feature,1,1))

    #Last conv layer
    self.conv_last = nn.Conv2d(96, n_class, kernel_size = 1)

  def forward(self, input: torch.Tensor) -> torch.Tensor:

    layer0 = self.layer0(input)
    layer1 = self.layer1(layer0)
    layer2 = self.layer2(layer1)

    bottleneck = self.bottleneck(layer2)
    bottleneck = self.bott_1x1(bottleneck)

    x = self.ups[0](bottleneck) #upsample 2
    layer2 = self.layer2_1x1(layer2)
    layer2 = self.ups[1](u = x, f = layer2)
    x = torch.cat([x,layer2], dim = 1)
    x = self.ups[2](x) #Double Convolutions

    x = self.ups[3](x) #upsample1
    layer1 = self.layer1_1x1(layer1)
    layer1 = self.ups[4](u = x, f = layer1)
    x = torch.cat([x,layer1] , dim = 1)
    x = self.ups[5](x) #Double Convolutions

    x = self.ups[6](x)
    layer0 = self.layer0_1x1(layer0)
    layer0 = self.ups[7](u = x, f = layer0)
    x = torch.cat([x,layer0],dim = 1)
    x = self.ups[8](x)

    mask = self.conv_last(x)
    

    return nn.functional.interpolate(mask, size=(224,224), mode="bilinear", align_corners=False)
  

class ResponseFusionAttentionUConvNextBase(nn.Module):
  def __init__(
      self, pretrained_encoder_backbone: bool = True , n_class: int = 1, **kwargs
  ) -> None:
    super(ResponseFusionAttentionUConvNextBase ,self).__init__()
    #Down part of convunext
    self.encoder = convnext_build(
      model_size = 'convnext_base', pretrained = pretrained_encoder_backbone, **kwargs
    )
    self.downsampling_layers = list(self.encoder.downsample_layers.children())
    self.encoder_layers = list(self.encoder.stages.children())
    self.layer0 = nn.Sequential(self.downsampling_layers[0], self.encoder_layers[0])
    self.layer0_1x1 = conv_relu(128,128,1,0) 
    self.layer1 = nn.Sequential(self.downsampling_layers[1], self.encoder_layers[1])
    self.layer1_1x1 = conv_relu(256,256,1,0) 
    self.layer2 = nn.Sequential(self.downsampling_layers[2], self.encoder_layers[2])
    self.layer2_1x1 = conv_relu(512,512,1,0) 

    #bottleneck
    self.bottleneck = nn.Sequential(self.downsampling_layers[3])
    self.bott_1x1 = conv_relu(1024,1024,1,0)
    #Up part of convunext
    self.ups = nn.ModuleList()

    for feature in [512,256,128]:
      self.ups.append(
          nn.ConvTranspose2d(
              feature*2, feature, kernel_size = 2, stride = 2,
          )
      )
      self.ups.append(DualPathResponseFusionAttention(F_g = feature, F_l = feature, F_int = feature))
      self.ups.append(ConvNextDecoderBlock(feature*2, feature,1,1))

    #Last conv layer
    self.conv_last = nn.Conv2d(128, n_class, kernel_size = 1)

  def forward(self, input: torch.Tensor) -> torch.Tensor:

    layer0 = self.layer0(input)
    layer1 = self.layer1(layer0)
    layer2 = self.layer2(layer1)

    bottleneck = self.bottleneck(layer2)
    bottleneck = self.bott_1x1(bottleneck)

    x = self.ups[0](bottleneck) #upsample 2
    layer2 = self.layer2_1x1(layer2)
    layer2 = self.ups[1](u = x, f = layer2)
    x = torch.cat([x,layer2], dim = 1)
    x = self.ups[2](x) #Decoder Convolutions

    x = self.ups[3](x) #upsample1
    layer1 = self.layer1_1x1(layer1)
    layer1 = self.ups[4](u = x, f = layer1)
    x = torch.cat([x,layer1] , dim = 1)
    x = self.ups[5](x) #Decoder Convolutions

    x = self.ups[6](x)
    layer0 = self.layer0_1x1(layer0)
    layer0 = self.ups[7](u=x, f=layer0)
    x = torch.cat([x,layer0],dim = 1)
    x = self.ups[8](x)

    mask = self.conv_last(x)
    
    return nn.functional.interpolate(mask, size=(224,224), mode="bilinear", align_corners=False)



class ResponseFusionAttentionUConvNextLarge(nn.Module):
  def __init__(
      self,  pretrained_encoder_backbone: bool = True, n_class: int = 1, **kwargs
  ) -> None:
    super(ResponseFusionAttentionUConvNextLarge,self).__init__()
    #Down part of convunext
    self.encoder = convnext_build(
      model_size = 'convnext_large', pretrained = pretrained_encoder_backbone, **kwargs
    )
    self.downsampling_layers = list(self.encoder.downsample_layers.children())
    self.encoder_layers = list(self.encoder.stages.children())
    self.layer0 = nn.Sequential(self.downsampling_layers[0], self.encoder_layers[0])
    self.layer0_1x1 = conv_relu(192,192,1,0) 
    self.layer1 = nn.Sequential(self.downsampling_layers[1], self.encoder_layers[1])
    self.layer1_1x1 = conv_relu(384,384,1,0) 
    self.layer2 = nn.Sequential(self.downsampling_layers[2], self.encoder_layers[2])
    self.layer2_1x1 = conv_relu(768,768,1,0) 

    #bottleneck
    self.bottleneck = nn.Sequential(self.downsampling_layers[3])
    self.bott_1x1 = conv_relu(1536,1536,1,0)
    #Up part of convunext
    self.ups = nn.ModuleList()

    for feature in [768,384,192]:
      self.ups.append(
          nn.ConvTranspose2d(
              feature*2, feature, kernel_size = 2, stride = 2,
          )
      )
      self.ups.append(DualPathResponseFusionAttention(F_g = feature, F_l = feature, F_int = feature))
      self.ups.append(ConvNextDecoderBlock(feature*2, feature,1,1))

    #Last conv layer
    self.conv_last = nn.Conv2d(192, n_class, kernel_size = 1)

  def forward(self, input : torch.Tensor) -> torch.Tensor:

    layer0 = self.layer0(input)
    layer1 = self.layer1(layer0)
    layer2 = self.layer2(layer1)

    bottleneck = self.bottleneck(layer2)
    bottleneck = self.bott_1x1(bottleneck)

    x = self.ups[0](bottleneck) #upsample 2
    layer2 = self.layer2_1x1(layer2)
    layer2 = self.ups[1](u = x, f = layer2)
    x = torch.cat([x,layer2], dim = 1)
    x = self.ups[2](x) #Double Convolutions

    x = self.ups[3](x) #upsample1
    layer1 = self.layer1_1x1(layer1)
    layer1 = self.ups[4](u = x, f = layer1)
    x = torch.cat([x,layer1] , dim = 1)
    x = self.ups[5](x) #Double Convolutions

    x = self.ups[6](x)
    layer0 = self.layer0_1x1(layer0)
    layer0 = self.ups[7](u=x, f=layer0)
    x = torch.cat([x,layer0],dim = 1)
    x = self.ups[8](x)

    mask = self.conv_last(x)
    

    return nn.functional.interpolate(mask, size=(224,224), mode="bilinear", align_corners=False)
  

  

