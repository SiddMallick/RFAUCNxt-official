import torch
from torchmetrics import MatthewsCorrCoef, JaccardIndex, CohenKappa, Recall, Precision, ROC, Specificity, PrecisionRecallCurve
from collections import defaultdict
import torchvision
from tqdm import tqdm


metric = defaultdict(list)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_metrics_data():
  return metric

def save_checkpoint(state, filename = "refuge_rfaucnxt_tiny.pth.tar"):
  print("=> Saving Checkpoint")
  torch.save(state,filename)

def load_checkpoint(checkpoint, model):
  print("=> Loading Checkpoint")
  model.load_state_dict(checkpoint["state_dict"])

def save_predictions_as_imgs(
    loader, model, folder, device = "cuda"
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

def compute_metrics(loader, model, device = "cuda"):
  num_correct = 0
  num_pixels = 0
  dice_score = 0
  jaccard_score = 0
  mcc_score = 0
  model.eval()

  with torch.no_grad():
    for x,y in loader:
      x = x.to(device)
      y_in = y.to(device).type(torch.cuda.IntTensor)
      y = y.to(device).unsqueeze(1)
      preds = torch.sigmoid(model(x))
      preds = (preds > 0.5).float()
      #preds_int = (preds>0.5).int()
      num_correct += (preds == y).sum()
      num_pixels += torch.numel(preds)
      dice_score += (2 * (preds * y).sum())/(
          (preds + y).sum() + 1e-8
      )
      jaccard = JaccardIndex(task='binary').to(device)
      jaccard_score += jaccard(preds, y_in.unsqueeze(1))
      matthews_corrcoef = MatthewsCorrCoef(task = 'binary',num_classes=2).to(device)
      #mcc_score += matthews_corrcoef(preds.squeeze(1), y_in)
  #metric["Pixel_Wise_Acc"].append((num_correct/num_pixels*100).detach().cpu().numpy().item())
  metric["Dice_Score"].append((dice_score/len(loader)).detach().cpu().numpy().item())
  metric["Jaccard_Score"].append((jaccard_score/len(loader)).detach().cpu().numpy().item())
  #metric["MCC_Score"].append((mcc_score/len(loader)).detach().cpu().numpy().item())
  print(
      f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
  )

  print(f"Dice Score: {dice_score/len(loader)}")
  print(f"Jaccard Score: {jaccard_score/len(loader)}")
  #print(f"MCC Score: {mcc_score/len(loader)}")
  model.train()

def train_fn(loader, model, optimizer, loss_fn, scaler):
  loop = tqdm(loader)
  for batch_idx, (data, targets) in enumerate(loop):
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

    loop.set_postfix(loss = loss.item())
