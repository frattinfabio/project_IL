import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# [FCClassifier] simply classifies according to the output of the network by choosing the most likable class
class FCClassifier():
  def __init__(self):
      self.net = None

  def update(self, step, net, train_dataloader):
      self.net = net

  def classify(self, input_images):
    self.net = self.net.cuda()
    self.net.train(False)
    with torch.no_grad():
      output = self.net(input_images, output = 'fc')
      preds = torch.argmax(output, dim = 1).cuda()
      return preds
