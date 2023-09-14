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
import argparse
from models.convnext_encoder import convnext_build


#Define the arguments to be parsed from input when train.py is run
parser = argparse.ArgumentParser()

parser.add_argument('train_img', type = str, help = "directory for train images")
parser.add_argument('train_mask', type = str, help = "directory for train masks")
parser.add_argument('test_img', type = str, help = "directory for test images")
parser.add_argument('test_masks', type = str, help = "directory for test masks")
parser.add_argument('result_dir', type = str, help = "directory for storing results")

parser.add_argument('--epochs', '-e', type = int, 
                    default= 50, help = "number of training epochs")

parser.add_argument('--lr', type = float, 
                    default= 1e-4, help = "learning rate for optimizer")

parser.add_argument('--batch_size', '-B', type = int, 
                    default= 16, help = "batch size per training epoch")

parser.add_argument('--loss_fn', type = str, 
                    default= 'jdbc', help = "loss function choice")

parser.add_argument('--model_size', '-m', type = str, 
                    default= 'tiny', help = "model size of RFAUCNxt")

parser.add_argument('--vertheta', '-v', type = float, 
                    default= 0.25, help = "joint parameter for jdbc")

parser.add_argument('--num_workers', '-w', type = int, default=2, help="number of cpu workers for training")

parser.add_argument('--pin_mem', type = bool, default=True, help ="pin memory for dataset loaders" )

parser.add_argument('--optimizer', type = str, default = 'adamw', help = "gradient optimizer.")



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

def train_setup_and_run(loss_fn):
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

    
    #check accuracy
    check_accuracy(val_loader, model, device =DEVICE)

    #print some examples
    save_predictions_as_imgs(
        val_loader, model, folder = "/mnt/d/results/", device =DEVICE
    )




proposed_loss_fn = JBDCLoss(alpha = VERTHETA)
train_setup_and_run(proposed_loss_fn)
metric = get_metrics_data()
df = pd.DataFrame(metric)
df.to_csv('/mnt/d/results/disc_results_proposed_loss.csv')