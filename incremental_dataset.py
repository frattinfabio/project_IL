from torchvision.datasets import CIFAR100
from torchvision.datasets import VisionDataset
import pandas as pd
from PIL import Image

# WARNING: it is required to have already downloaded the data under the [root] folder
# execute the following lines of code before instantiting a SubCIFAR object
# ----------------------------------------------------------------
# from torchvsion.dataset.utils import  download_and_extract_archive
# download_and_extract_archive("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", root, filename = "cifar-100-python.tar.gz", md5 = 'eb9058c3a382ffc7106e4002c42a8d85')
# ----------------------------------------------------------------

DEFAULT_LABELS = list(range(10))
DEFAULT_SPLIT = [list(range(i*10, (i+1)*10)) for i in range(10)]

# SubCIFAR extracts from the CIFAR100 dataset a subset of classes
# [root]: where to find the data
# [labels_split] : the splitting of the labels
# [labels] : the labels stored in this subset of cifar100
# [train] : whether to load the training or test data
# [transform] : the transform performed on the training data
class SubCIFAR(VisionDataset):

  def __init__(self, root, labels_split = DEFAULT_SPLIT, labels = DEFAULT_LABELS, train=True, transform=None, target_transform=None):
    super(SubCIFAR, self).__init__(root, transform=transform, target_transform=target_transform)

    self.all_labels = [label for split in labels_split for label in split]
    self.stored_labels = labels
    cifar_full = CIFAR100(root, train = train, download = False)
    data = cifar_full.data
    targets = cifar_full.targets
    images = []
    labels = []
    for i in range(len(data)):
      if targets[i] in self.stored_labels:
        images.append(data[i])
        labels.append(targets[i])
    self.dataFrame = pd.DataFrame(zip(images, labels), columns=["image", "label"]) 

  def add_samples(new_samples):
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
