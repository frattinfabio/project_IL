import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

class FCClassifier():
  def __init__(self):
      self.net = None

  def update(self, step, net, train_dataloader):
      self.net = net

  def classify(self, input_images):
    self.net.train(False)
    self.net.cuda()
    with torch.no_grad():
      output = self.net(input_images, output = 'fc')
      preds = torch.argmax(output, dim = 1).cuda()
      return preds
