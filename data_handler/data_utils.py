from project_IL.data_handler.SubCIFAR import SubCIFAR

def load_data(step, labels_split, exemplars = None):

  new_labels = labels_split[step]
  old_labels = [label for split in labels_split[:step] for label in split]

  train_dataset = SubCIFAR(labels_split = labels_split, labels = new_labels, train = True, transform = train_params["train_transform"])
  if exemplars is not None and step > 0:
    print("Adding exemplars to the training dataset...")
    for i in range(len(exemplars)):
      train_dataset.add_samples(exemplars[i])

  new_test_dataset = SubCIFAR(labels_split = labels_split, labels = new_labels, train = False, transform = train_params["test_transform"])
  old_test_dataset = SubCIFAR(labels_split = labels_split, labels = old_labels, train = False, transform = train_params["test_transform"])

  train_dataloader = DataLoader(train_dataset, batch_size = train_params["BATCH_SIZE"], shuffle = True, num_workers = 4, drop_last = True)
  new_test_dataloader = DataLoader(new_test_dataset, batch_size = train_params["BATCH_SIZE"], num_workers = 4)
  old_test_dataloader = DataLoader(old_test_dataset, batch_size = train_params["BATCH_SIZE"], num_workers = 4)

  return train_dataloader, new_test_dataloader, old_test_dataloader
