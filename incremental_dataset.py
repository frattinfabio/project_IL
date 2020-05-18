from torchvision.datasets import CIFAR100
from torchvision.datasets import VisionDataset
import random 
import pandas as pd
from PIL import Image



# SubCIFAR for incremental learning
# stores only the last 10 classes of the [classes] parameter if training data
# stores all the classes of the [classes] parameter if test data
class SubCIFAR(VisionDataset):

  def __init__(self, root, classes = list(range(10)), train=True, transform=None, target_transform=None):
    super(SubCIFAR, self).__init__(root, transform=transform, target_transform=target_transform)

    self.train = train
    self.classes = classes
    if train:
      classes_to_store = classes[-10:]
    else:
      classes_to_store = classes 

    cifar_full = CIFAR100(root, train = train, download = False)
    data = cifar_full.data
    targets = cifar_full.targets

    images = []
    labels = []
    for i in range(len(data)):
      if targets[i] in classes_to_store:
        images.append(data[i])
        labels.append(targets[i])
    self.dataFrame = pd.DataFrame(zip(images, labels), columns=["image", "label"])    

# return the image and the mapped index according to its position in the [classes] list 
# necessary for the training phase where only labels in the [0, num_of_labels-1] labels are accepted
  def __getitem__(self, index):
    image = self.dataFrame["image"].iloc[index]
    label = self.dataFrame["label"].iloc[index]

    image = Image.fromarray(image)
    if self.transform is not None:
      image = self.transform(image)
    if self.target_transform is not None:
      label = self.target_transform(label)

    return image, self.classes.index(label)

  def __len__(self):
    return len(self.dataFrame)
