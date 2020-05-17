from torchvision.datasets import CIFAR100
from torch.utils.data import Subset
import random 

# splitting the 100 classes in [num_groups] groups
# and the indexes of the images belonging to those classes as well
def split_labels(dataset, num_groups):

  labels = list(range(100))
  labels_per_group = 100 // num_groups
  labels_split = []
  indexes_split = [[] for _ in range(num_groups)]

  for _ in range(num_groups):
    random.seed(42)
    subset = random.sample(labels, labels_per_group)
    labels_split.append(subset)
    labels = list(set(labels) - set(subset))

  for idx in range(len(dataset)):
    label = dataset.__getitem__(idx)[1]
    for i in range(num_groups):
      if label in labels_split[i]:
        indexes_split[i].append(idx)
        break
      
  return labels_split, indexes_split



# IncrementalCIFAR class stores the CIFAR100 dataset and some info helpful for the 
# incremental learning process: the splitting of the groups and of the indexes
class IncrementalCIFAR():

  def __init__(self, root, num_groups = 10, train=True, transform=None, target_transform=None, download=False):
        self.dataset = CIFAR100(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.num_groups = num_groups
        self.labels_split, self.indexes_split = split_labels(self.dataset, num_groups)

  # get the subset of the dataset relative to the [group_index] group
  def get_group(self, group_index):
    indexes = self.indexes_split[group_index]
    return Subset(self.dataset, indexes)
