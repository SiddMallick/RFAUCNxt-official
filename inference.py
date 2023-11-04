from models.rfau_cnxt import *
import torch
from PIL import Image
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from utils.train_test_functions import load_checkpoint
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('dir', type = str, help = "Path and name of the image file")
parser.add_argument('model_pth_dir', type = str, help = "Path to the .pth file")
parser.add_argument('--model_size', type = str, default='tiny', help = "Size of the RFAUCNxt model")
parser.add_argument('--output_dir','-o', type = str, default = '/', help = "Path and name of the image file")
parser.add_argument('--pretrained', '-p', type = bool, default = True, help = "enable/disable pretrained convnext encoder backbone")

args = parser.parse_args()


if __name__ == '__main__':

    image = Image.open(args.dir)

    model = ResponseFusionAttentionUConvNextTiny(pretrained_encoder_backbone=args.pretrained)
    if args.model_size == 'small':
        model = ResponseFusionAttentionUConvNextSmall(pretrained_encoder_backbone=args.pretrained)
    if args.model_size == 'base':
        model = ResponseFusionAttentionUConvNextBase(pretrained_encoder_backbone=args.pretrained)
    if args.model_size == 'large':
        model = ResponseFusionAttentionUConvNextLarge(pretrained_encoder_backbone=args.pretrained)
    
    transform = A.Compose([
                    A.Resize(height=224, width=224),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                    ),
                    ToTensorV2(),
                ])

    image_tensor= transform(image = np.array(image))
    model = ResponseFusionAttentionUConvNextTiny(pretrained_encoder_backbone=True)
    load_checkpoint(checkpoint=torch.load(args.model_pth_dir, map_location="cuda:0"), model=model)

    output_tensor = torch.sigmoid(model(torch.unsqueeze(image_tensor["image"], 0)))
    output_tensor = (output_tensor>0.5).float()

    print(torch.squeeze(output_tensor, 0).size())

    output_img = transforms.ToPILImage()(torch.squeeze(output_tensor,0))

    if args.output_dir == '/':
        path = args.dir.split('/')
        
        output_img.save(path[-1][:-4] + '_prediction.png')
    else:
        output_img.save(args.output_dir)