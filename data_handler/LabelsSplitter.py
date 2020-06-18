import random 

# [LabelsSplitter] class
# define the labels splitting, given the total number of labels and the number of groups
# [num_labels]: the total number of classes to be split
# [num_groups]: the number of groups to create, each of which will contain num_label//num_groups labels
# WARNING: it is needed that num_labels % num_groups == 0

class LabelsSplitter():
  def __init__(self, num_labels, num_groups, seed = 42):

    self.labels_split = []
    labels = list(range(num_labels))
    labels_per_group = num_labels // num_groups

    for i in range(num_groups):
      random.seed(seed)
      subset = random.sample(labels, labels_per_group)
      self.labels_split.append(subset)
      labels = list(set(labels) - set(subset))
