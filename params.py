from torchvision import transforms
from project_IL.classifiers.NMEClassifier import NMEClassifier
from project_IL.classifiers.FCClassifier import FCClassifier

# train parameters from the iCaRL paper
train_params_base = {
"LR": 2 ,
"MOMENTUM": 0.9 ,
"WEIGHT_DECAY": 1e-5,
"STEP_MILESTONES": [49,63],
"GAMMA": 0.2,
"NUM_EPOCHS": 70,
"BATCH_SIZE": 128,
"train_transform": transforms.Compose([
                                      transforms.RandomCrop(32, padding = 4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                      ]),
"test_transform": transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                     ]),
}

train_params_cosine = {
  "LR": 0.1,
  "MOMENTUM": 0.9 ,
  "WEIGHT_DECAY": 5e-4,
  "STEP_MILESTONES": [80,120],
  "GAMMA": 0.1,
  "NUM_EPOCHS": 160,
  "BATCH_SIZE": 128,
  "train_transform": transforms.Compose([
                                        transforms.RandomCrop(32, padding = 4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                        ]),
  "test_transform": transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                      ]),
}


# [approach_params_*] stores all the option according to the different approach (*) we want to pursue
# "classification_loss" : the type of classification loss (see CustomizedLoss.py for the list of possible loss behaviour)
# "distillation_loss" : the type of distillation loss
# "classifier": the classifier used (see the project_IL/classifier repo)
# "use_*": whether to use the specified technique (*) in the learning phase (boolean)
# "n_exemplars": the total number of exemplars stored
# "exemplars_selection": the policy to select the exemplars

approach_params_finetuning = {
"classification_loss": "bce",
"classifier": FCClassifier(),
"distillation_loss": None,
"use_distillation" : False,
"use_variation" : False,
"use_exemplars": False,
}

approach_params_lwf = {
"classification_loss": "bce",
"distillation_loss": "icarl",
"classifier": FCClassifier(),
"use_distillation" : False,
"use_variation" : False,
"use_exemplars": False,
}

approach_params_icarl = {
"classification_loss": "bce",
"distillation_loss": "icarl",
"classifier": NMEClassifier(),
"use_distillation" : True,
"use_variation" : False,
"use_exemplars": True,
"n_exemplars": 2000,
"exemplars_selection" : "random"
}

approach_params_variation = {
"classification_loss": "icarl",
"distillation_loss": "icarl",
"classifier": NMEClassifier(),
"use_distillation" : True,
"use_variation" : True,
"use_exemplars": True,
"n_exemplars": 2000,
"exemplars_selection" : "random"
}

approach_params_cosine = {
"classification_loss": "ce",
"distillation_loss": "lfc",
"classifier": FCClassifier(),
"use_distillation" : True,
"use_variation" : False,
"use_exemplars": True,
"n_exemplars": 2000,
"exemplars_selection" : "random"
}

def get_params(method):
    if method == "FINETUNING":
        return train_params_base, approach_params_finetuning
    elif method == "LWF":
        return train_params_base, approach_params_lwf
    elif method == "ICARL":
        return train_params_base, approach_params_icarl
    elif method == "COSINE":
        return train_params_cosine, approach_params_cosine
    else:
        return train_params_base, approach_params_variation
