import argparse
from utils.data_loader import get_loaders
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from models.rfau_cnxt import ResponseFusionAttentionUConvNextTiny, ResponseFusionAttentionUConvNextBase, ResponseFusionAttentionUConvNextSmall, ResponseFusionAttentionUConvNextLarge
import torch.optim as optim
import torch
from utils.train_test_functions import train_fn, check_accuracy, save_predictions_as_imgs, get_metrics_data
import pandas as pd
from torchvision.models import convnext_tiny
from utils.loss_functions import *

from models.convnext_encoder import convnext_build

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 224
IMAGE_WIDTH  = 224
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "/mnt/d/refuge_cropped_final/refuge_cropped_disc/train/"
TRAIN_MASK_DIR = "/mnt/d/refuge_cropped_final/refuge_cropped_disc/train_masks/"
VAL_IMG_DIR = "/mnt/d/refuge_cropped_final/refuge_cropped_disc/test/"
VAL_MASK_DIR = "/mnt/d/refuge_cropped_final/refuge_cropped_disc/test_masks/"
VERTHETA = 0.25

def main(loss_fn):
  train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
  val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
  
  
  model = ResponseFusionAttentionUConvNextTiny().to(DEVICE)
  
  optimizer = optim.AdamW(model.parameters(), lr= LEARNING_RATE)

  train_loader, val_loader = get_loaders(
      TRAIN_IMG_DIR,
      TRAIN_MASK_DIR,
      VAL_IMG_DIR,
      VAL_MASK_DIR,
      BATCH_SIZE,
      train_transform,
      val_transforms
  )

  scaler = torch.cuda.amp.GradScaler()
  for epoch in range(NUM_EPOCHS):
    print("Epoch:",epoch)
    train_fn(train_loader, model, optimizer, loss_fn, scaler)

    #save model
    # checkpoint = {
    #     "state_dict":model.state_dict(),
    #     "optimizer":optimizer.state_dict(),
    # }
    #save_checkpoint(checkpoint)
    #check accuracy
    check_accuracy(val_loader, model, device =DEVICE)

    #print some examples
    save_predictions_as_imgs(
        val_loader, model, folder = "/mnt/d/results/", device =DEVICE
    )


proposed_loss_fn = JBDCLoss(alpha = VERTHETA)
main(proposed_loss_fn)
metric = get_metrics_data()
df = pd.DataFrame(metric)
df.to_csv('/mnt/d/results/disc_results_proposed_loss.csv')