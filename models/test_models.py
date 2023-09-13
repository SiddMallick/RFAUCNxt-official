import torch
from rfau_cnxt import ResponseFusionAttentionUConvNextTiny, ResponseFusionAttentionUConvNextBase, ResponseFusionAttentionUConvNextSmall, ResponseFusionAttentionUConvNextLarge
from convnext_encoder import ConvNext, convnext_build

def test():
  x = torch.randn((3,3,224,224))
  model = ResponseFusionAttentionUConvNextLarge(pretrained_encoder_backbone=True)
  preds = model(x)
  print(preds.size())
  assert preds.size() == (3,1,224,224)

test()