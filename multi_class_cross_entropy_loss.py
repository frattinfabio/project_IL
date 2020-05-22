import torch.nn as nn

# function MultiClassCrossEntropy taken from the GitHub repo https://github.com/ngailapdi/LWF

class MultiClassCrossEntropyLoss(nn.Module):

  def __init__(self, reduction='mean', T = 1):
    super(MultiClassCrossEntropyLoss, self).__init__()
    self.reduction = reduction
    self.T = T
        
  def forward(self, input, target):
    output = output.narrow(1, 0, target.shape[1])
    output = torch.log_softmax(output/self.T, dim=1)
    target = torch.softmax(target/self.T, dim=1)
    loss = (output * target).mean(dim=1)
    if self.reduction == 'mean':
        loss = -torch.mean(loss)
    elif self.reduction == 'sum':
        loss = -torch.sum(loss)
    else:
        loss = -loss

    return loss
  
    
