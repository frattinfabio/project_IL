import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# Nearest mean of examplars classifier
class NMEClassifier():
  def __init__(self):
      self.net = None
      self.means = None
      
  def update(self, step, net, train_dataloader):
      self.net = net
      n_known_classes = self.net.fc.out_features
      self.means = [torch.zeros((self.net.fc.in_features,)).cuda() for _ in range(n_known_classes)]
      num_images = torch.zeros(n_known_classes,)

      with torch.no_grad():
        self.net = self.net.cuda()
        self.net = self.net.train(False)

        for images, labels in train_dataloader:
          images = images.cuda()
          labels = labels.cuda()
          features = self.net(images, output = 'features')
          features = F.normalize(features, p = 2)
          for feature, label in zip(features, labels):
            self.means[label] += feature
            num_images[label] += 1
        for label in range(n_known_classes):
          self.means[label] /= num_images[label]

      self.means = torch.stack(self.means).cuda()
      self.means = F.normalize(self.means, p = 2)

  # predict the labels for the batch [input_images]
  # according to the nearest mean criterion 
  def classify(self, input_images):
    with torch.no_grad():
      self.net.train(False)
      features = self.net(input_images, output = 'features')
      features = F.normalize(features, p = 2)
      preds = []
      for feature in features:
        distances = torch.pow(self.means.cuda() - feature, 2).sum(-1)
        preds.append(distances.argmin().item())
      return torch.Tensor(preds).cuda()
