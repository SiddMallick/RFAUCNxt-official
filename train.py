import argparse
from utils.data_loader import get_loaders
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from models.rfau_cnxt import ResponseFusionAttentionUConvNextTiny, ResponseFusionAttentionUConvNextBase, ResponseFusionAttentionUConvNextSmall, ResponseFusionAttentionUConvNextLarge
import torch.optim as optim
import torch
from utils.train_test_functions import train_fn, compute_metrics, save_predictions_as_imgs, get_metrics_data
import pandas as pd
from torchvision.models import convnext_tiny
from utils.loss_functions import *
import argparse
from models.convnext_encoder import convnext_build
import os


#Define the arguments to be parsed from input when train.py is run
parser = argparse.ArgumentParser()

parser.add_argument('dir', type = str, help = "directory of dataset")
parser.add_argument('--result_dir', '-r', type = str, default= '/results' , help = "directory for storing results")
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

args = parser.parse_args()

#Define macros which are constants for this application
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 224
IMAGE_WIDTH  = 224



def train_setup_and_run(loss_fn):
  
  #Define image augmentation transformations for both train and validation purposes
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
  
  
  model = ResponseFusionAttentionUConvNextTiny() #Default model to be loaded
  if args.model_size == 'small':
     model = ResponseFusionAttentionUConvNextSmall()
  if args.model_size == 'base':
     model = ResponseFusionAttentionUConvNextBase()
  if args.model_size == 'large':
     model = ResponseFusionAttentionUConvNextLarge()
  
  model = model.to(DEVICE)

  optimizer = optim.AdamW(model.parameters(), lr = args.lr) #Default choice of optimizer
  if args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr = args.lr)

  train_loader, val_loader = get_loaders(
      args.dir + '/train',
      args.dir + '/train_masks',
      args.dir + '/test',
      args.dir + '/test_masks',
      args.batch_size,
      train_transform,
      val_transforms,
      num_workers= args.num_workers,
      pin_memory= args.pin_mem,
  )

  num_epochs = args.epochs
  scaler = torch.cuda.amp.GradScaler()

  for epoch in range(num_epochs):
    print("Epoch:",epoch)
    train_fn(train_loader, model, optimizer, loss_fn, scaler)

    
    #compute metrics
    compute_metrics(val_loader, model, device = DEVICE)

    #print batchwise results in result folder
    save_predictions_as_imgs(
        val_loader, model, folder = args.result_dir, device =DEVICE
    )



if __name__ == '__main__':

    loss_fn = JBDCLoss(alpha = args.vertheta)
    train_setup_and_run(loss_fn)
    metric = get_metrics_data()
    df = pd.DataFrame(metric)
    df.to_csv(os.path.join(args.result_dir,'metric_results.csv'))