from torch.nn.modules.loss import _WeightedLoss

# function MultiClassCrossEntropy taken from the GitHub repo https://github.com/ngailapdi/LWF

class MultiClassCrossEntropyLoss(_WeightedLoss):

  def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', T = 1):
        super(MultiClassCrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.T = T
        
  def MultiClassCrossEntropy(logits, labels, T):
    labels = Variable(labels.data, requires_grad=False).cuda()
    outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
    labels = torch.softmax(labels/T, dim=1)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return Variable(outputs.data, requires_grad=True).cuda()
    
  def forward(self, input, target):
    return MultiClassCrossEntropy(input, targets, self.T)
  
    
