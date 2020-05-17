from torchvision.datasets import CIFAR100
from torch.utils.data import Subset
import pandas as pd
import random

# splitting the 100 classes in [num_groups] groups
# and the indexes of the images belonging to those classes as well
def split_classes(dataset, num_groups):
  assert 100 % num_groups == 0

  classes = list(range(100))
  classes_per_group = 100 / num_groups
  classes_split = []
  indexes_split = []

  for _ in range(num_groups):
    random.seed(42)
    subset = random.sample(classes, classes_per_group)
    classes_split.append(subset)
    indexes = [idx for idx in range(len(dataset)) if dataset.__getitem__(idx)[1] in subset]
    indexes_split.append(indexes)
    classes = list(set(classes) - set(subset))
  return classes_split, indexes_split



# IncrementalCIFAR class store the known CIFAR100 dataset and some info helpful for the 
# incremental learning process: the  groups and their splitting
class IncrementalCIFAR():

  def __init__(self, root, num_groups = 10, train=True, transform=None, target_transform=None, download=False):
        
        self.dataset = CIFAR100(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.num_groups = num_groups
        self.classes_split, self.indexes_split = split_classes(num_groups)

  def __getitem__(self, index):
    return self.dataset.__getitem__(index)

  def __len__(self):
    return self.dataset.__len__()

# get the subset of the dataset relative to the [group_index] group
  def get_group(self, group_index):
    indexes = self.indexes_split[group_index]
    return Subset(self.dataset, indexes)