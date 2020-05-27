import numpy as np
import torch
from torchvision import transforms

class NearestMeanOfExamplarsClassifier():
  def __init__(self, net, examplars, transform):
    with torch.no_grad():
      self.net = net
      self.net.train(False)

      self.means = []
      for i in range(len(examplars)):
        examplar_set = examplars[i][:,0]
        features_mean = torch.zeros((net.linear.in_features,))

        for j in range(len(examplar_set)):
          tensor = transform(examplar_set[j]).unsqueeze(0).cuda()
          features = self.net.features_extraction(tensor).squeeze(0).cpu()
          features = features / torch.norm(features, p = 2)
          features_mean += features
        features_mean /= len(examplar_set)
        features_mean = features_mean / torch.norm(features_mean, p = 2)
        self.means.append(features_mean)

      self.means = torch.stack(self.means)
      
        
  def classify(self, input_images):
    with torch.no_grad():
      self.net.train(False)
      features = self.net.features_extraction(input_images)
      features = features / torch.norm(features, p = 2)
      preds = []
      for feature in features:
        distances = torch.pow(self.means.cuda() - feature, 2).sum(-1)
        preds.append(distances.argmin().item())
      return torch.Tensor(preds).cuda()
