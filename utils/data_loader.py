import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class DatasetLoad(Dataset):
  def __init__(self,image_dir: str, mask_dir: str, transform: object = None)->None:
    self.image_dir = image_dir
    self.mask_dir = mask_dir
    self.transform = transform
    self.images = os.listdir(image_dir)

  def __len__(self) -> int:
    return len(self.images)

  def __getitem__(self, index: int):
    img_path = os.path.join(self.image_dir, self.images[index])
    mask_path = os.path.join(self.mask_dir, self.images[index])
    image = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"), dtype = np.float32)
    mask[mask == 0] = 2.0
    mask[mask==128] = 1.0
    mask[mask==255] = 0.0

    if self.transform is not None:
      augmentations = self.transform(image = image, mask = mask)
      image = augmentations["image"]
      mask = augmentations["mask"]


    return image, mask
  


def get_loaders(
    train_dir, train_maskdir, val_dir, val_maskdir, batch_size, train_transform, val_transform, num_workers = 4, pin_memory = True):

  train_ds = DatasetLoad(
      image_dir = train_dir,
      mask_dir = train_maskdir,
      transform = train_transform,
  )

  train_loader = DataLoader(
      train_ds,
      batch_size = batch_size,
      num_workers = num_workers,
      pin_memory = pin_memory,
      shuffle = True,
  )

  val_ds = DatasetLoad(
      image_dir = val_dir,
      mask_dir = val_maskdir,
      transform = val_transform,
  )

  val_loader = DataLoader(
      val_ds,
      batch_size = batch_size,
      num_workers = num_workers,
      pin_memory = pin_memory,
      shuffle = False,
  )


  return train_loader, val_loader