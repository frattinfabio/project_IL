import torch

def evaluate(dataloader, classifier):
  running_corrects = 0
  for images, labels in dataloader:
    images = images.cuda()
    labels = labels.cuda()
    preds = classifier.classify(images)
    running_corrects += torch.sum(preds == labels.data).data.item()
  accuracy = running_corrects / float(len(dataloader.dataset))
  return accuracy
  
# evaluate the classifier on both the new task test set and on the old ones
#  returns the accuracies for new, old and combined sets
def evaluate_incremental(new_dataloader, old_dataloader, classifier):
  n_new_classes = len(new_dataloader.dataset.stored_labels)
  n_old_classes = len(old_dataloader.dataset.stored_labels)
  tot_classes = n_new_classes + n_old_classes

  new_test_accuracy = evaluate(new_dataloader, classifier)
  if n_old_classes > 0:
    old_test_accuracy = evaluate(old_dataloader, classifier)
    overall_test_accuracy = (new_test_accuracy*n_new_classes + old_test_accuracy*n_old_classes)/(tot_classes)
  else:
    old_test_accuracy = None
    overall_test_accuracy = new_test_accuracy

  return {"new" : new_test_accuracy, "old" : old_test_accuracy, "overall": overall_test_accuracy}
