import torch

class NearestMeanOfExamplarsClassifier():
  def __init__(self, feature_extractor, feat_dim, examplars):
    self.feature_extractor = feature_extractor
    self.means = []

    for i in range(len(examplars)):
      examplar_set = examplars[i][:,0]
      features_mean = torch.zeros((feat_dim,))
      
      for j in range(len(examplar_set)):
        tensor = transforms.ToTensor()(examplar_set[j]).unsqueeze(0).cuda()
        features = sefl.feature_extractor(tensor).squeeze(0).cpu()
        features = features / torch.norm(features, p = 2)
        features_mean += features
      features_mean /= len(examplar_set)
      features_mean = features_mean / torch.norm(features_mean, p = 2)
      self.means.append(features_mean)
      
    self.means = torch.stack(self.means)
      
        
  def classify(self, input_images):
    features = self.feature_extractor(input_images.cuda())
    features = features / torch.norm(features, p = 2)
    preds = []
    for feature in features:
      distances = torch.pow(self.means.cuda() - feature, 2).sum(-1)
      preds.append(distances.argmin().item())
    return np.array(preds)

