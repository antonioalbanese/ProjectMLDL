from PIL import Image
from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple

class Exemplar(Dataset):
  
  def __init__(self, exemplar_set, transform=None):
    self.data = []
    self.targets = []
    self.transform = transform
    
    for index, exemplar_i in enumerate(exemplar_set):
      self.data += exemplar_i
      self.targets += [index]*len(exemplar_i)

  def __getitem__(self, index: int) -> Tuple[Any, Any]:
    """
    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data[index], self.targets[index]
    
    img = Image.fromarray(img) # Return a PIL image

    if self.transform is not None:
        img = self.transform(img)

    return index, img, target
  
  def __len__(self) -> int:
    return len(self.targets)
