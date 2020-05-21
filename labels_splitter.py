import random 
import torch
# LabelsSplitter class
# define the labels splitting, given the total number of labels and the number of groups
# WARNING: it is needed that num_labels % num_groups == 0


def transforms_labels_onehot(labels, num_labels):
  transformed = []
  for i in range(len(labels)):
    onehot = []
    for label in range(num_labels):
      onehot.append(1 if labels[i] == label else 0)
    transformed.append(onehot)
  return torch.Tensor(transformed)

class LabelsSplitter():
  def __init__(self, num_labels, num_groups, seed = 42):
    assert num_labels % num_groups == 0

    self.labels_split = []
    labels = list(range(num_labels))
    labels_per_group = num_labels // num_groups

    for i in range(num_groups):
      random.seed(seed)
      subset = random.sample(labels, labels_per_group)
      self.labels_split.append(subset)
      labels = list(set(labels) - set(subset))
