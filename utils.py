import torch 

def transforms_labels_onehot(labels, num_labels):
  transformed = []
  for i in range(len(labels)):
    onehot = []
    for label in range(num_labels):
      onehot.append(1 if labels[i] == label else 0)
    transformed.append(onehot)
  return torch.Tensor(transformed)
