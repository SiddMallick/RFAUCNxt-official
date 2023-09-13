import torch
from torchmetrics import MatthewsCorrCoef, JaccardIndex, CohenKappa, Recall, Precision, ROC, Specificity, PrecisionRecallCurve
from collections import defaultdict
import torchvision
from tqdm import tqdm


metric = defaultdict(list)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def save_checkpoint(state, filename = "/content/gdrive/MyDrive/results_for_comparison/refuge_unetplus.pth.tar"):
  print("=> Saving Checkpoint")
  torch.save(state,filename)

def load_checkpoint(checkpoint, model):
  print("=> Loading Checkpoint")
  model.load_state_dict(checkpoint["state_dict"])

def save_predictions_as_imgs(
    loader, model, folder = "/content/gdrive/MyDrive/refuge_unet/", device = "cuda"
):

  model.eval()
  for idx, (x,y) in enumerate(loader):
    x = x.to(device = device)
    with torch.no_grad():
      preds = torch.sigmoid(model(x))
      preds = (preds>0.5).float()

    torchvision.utils.save_image(
        preds, f"{folder}/pred_{idx}.png"
    )
    torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}.png")
  
  model.train()

def check_accuracy(loader, model, train_loss, loss_fn, device = "cuda"):
  num_correct = 0
  num_pixels = 0
  dice_score = 0
  jaccard_score = 0
  val_loss = 0
 
  model.eval()

  with torch.no_grad():
    for x,y in loader:
      x = x.to(device)
      y_in = y.to(device).type(torch.cuda.IntTensor)
      y = y.to(device).unsqueeze(1)
      predictions = model(x)
      preds1 = torch.sigmoid(predictions)
      preds = (preds1 > 0.5).float()
      
      val_loss += loss_fn(predictions, y) 

      num_correct += (preds == y).sum()
      num_pixels += torch.numel(preds)
      dice_score += (2 * (preds * y).sum())/(
          (preds + y).sum() + 1e-8
      )


      jaccard = JaccardIndex(task='binary').to(device)
      jaccard_score += jaccard(preds, y_in.unsqueeze(1))
  metric["Train_loss"].append(train_loss)
  metric["Val_loss"].append((val_loss/len(loader)).detach().cpu().numpy().item())
  metric["Pixel_Wise_Acc"].append((num_correct/num_pixels*100).detach().cpu().numpy().item())
  metric["Dice_Score"].append((dice_score/len(loader)).detach().cpu().numpy().item())
  metric["Jaccard_Score"].append((jaccard_score/len(loader)).detach().cpu().numpy().item())
  
  print(
      f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
  )

  print(f"Train Loss: {train_loss}")
  print(f"Val Loss: {val_loss/len(loader)}")
  print(f"Dice Score: {dice_score/len(loader)}")
  print(f"Jaccard Score: {jaccard_score/len(loader)}")
  

  model.train()

def train_fn(loader, model, optimizer, loss_fn, scaler):
  loop = tqdm(loader)
  train_loss = []
  for _, (data, targets) in enumerate(loop):
    data = data.to(device=DEVICE)
    targets = targets.float().unsqueeze(1).to(device = DEVICE)

    #forward
    with torch.cuda.amp.autocast():
      predictions = model(data)
      loss = loss_fn(predictions, targets)
    
    #backward
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    train_loss.append(loss.item())
    loop.set_postfix(loss = loss.item())

  return sum(train_loss)/len(train_loss)