from torchvision.datasets import CIFAR100
from torchvision.datasets import VisionDataset
import pandas as pd
from PIL import Image

DEFAULT_LABELS = list(range(10))
DEFAULT_SPLIT = [list(range(i*10, (i+1)*10)) for i in range(10)]
DEFAULT_DATA_DIR = "./data"

cifar_train = CIFAR100(DEFAULT_DATA_DIR, train = True, download = True)
cifar_test = CIFAR100(DEFAULT_DATA_DIR, train = False, download = True)
data = {"train": cifar_train.data, "test": cifar_test.data}
targets = {"train": cifar_train.targets, "test": cifar_test.targets}

# SubCIFAR extracts from the CIFAR100 dataset a subset of classes
# [root]: where to find the data
# [labels_split] : the splitting of the labels
# [labels] : the labels stored in this subset of cifar100
# [train] : whether to load the training or test data
# [transform] : the transform performed on the training data
class SubCIFAR(VisionDataset):

  def __init__(self, root = DEFAULT_DATA_DIR, labels_split = DEFAULT_SPLIT, labels = DEFAULT_LABELS, train=True, transform=None, target_transform=None):
    super(SubCIFAR, self).__init__(root, transform=transform, target_transform=target_transform)

    self.all_labels = [label for split in labels_split for label in split]
    self.stored_labels = labels
    mode = "train" if train else "test"
    
    images = []
    labels = []
    for i in range(len(data[mode])):
      if targets[mode][i] in self.stored_labels:
        images.append(data[mode][i])
        labels.append(targets[mode][i])
    self.dataFrame = pd.DataFrame(zip(images, labels), columns=["image", "label"]) 

  def add_samples(self, new_samples):
    new_dataframe = pd.DataFrame(new_samples, columns=["image", "label"])
    self.dataFrame = pd.concat([self.dataFrame, new_dataframe], ignore_index = True)


# return the image and the mapped index according to its position in the [all_labels] list 
# necessary for the training phase where only labels in the [0, num_of_labels-1] labels are accepted
  def __getitem__(self, index):
    image = self.dataFrame["image"].iloc[index]
    label = self.dataFrame["label"].iloc[index]

    image = Image.fromarray(image).convert("RGB")
    if self.transform is not None:
      image = self.transform(image)
    if self.target_transform is not None:
      label = self.target_transform(label)

    return image, self.all_labels.index(label)

  def __len__(self):
    return len(self.dataFrame)
