import torch 

# transform the [labels] array into a one_hot encoding array
def transform_labels_onehot(labels, num_labels):
  onehot = torch.zeros((len(labels), num_labels))
  for i in range(len(labels)):
    onehot[i][labels[i]] = 1
  return onehot
