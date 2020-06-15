import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import numpy as np

from project_IL.utils import transform_labels_onehot
from project_IL.data_handler.SubCIFAR import SubCIFAR
from project_IL.data_handler.LabelsSplitter import LabelsSplitter
from project_IL.model.CustomizedLoss import CustomizedLoss
from project_IL.nets.resnet import resnet32

class IncrementalLearner():

    def __init__(self, num_classes, num_groups, splitter_seed, approach_params, train_params):
        self.num_classes = num_classes
        self.num_groups = num_groups
        self.classes_per_group = num_classes//num_groups

        self.train_params = train_params
        self.approach_params = approach_params

        self.splitter = LabelsSplitter(num_classes, num_groups, seed = splitter_seed)

        # initializing the nets
        # [net]: the main net incrementally trained
        # [prev_net]: the version of the net at the previous step, useful for the distillation term
        # [ft_net]: the fine-tuned network for the variation, useful for the classification term
        self.net = resnet32()
        self.net.fc = nn.Linear(self.net.fc.in_features, self.classes_per_group)
        self.init_weights = torch.nn.init.kaiming_normal_(self.net.fc.weight)
        parameters_to_optimize = self.net.parameters()
        self.optimizer = optim.SGD(parameters_to_optimize , lr = train_params["LR"], momentum = train_params["MOMENTUM"], weight_decay = train_params["WEIGHT_DECAY"])
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = train_params["STEP_MILESTONES"], gamma = train_params["GAMMA"])

        self.loss = CustomizedLoss(approach_params["classification_loss"], approach_params["distillation_loss"])

        # setting the enviroment according to the [approach_params]
        self.use_distillation = approach_params["use_distillation"]
        self.use_variation = self.use_distillation and approach_params["use_variation"]
        self.use_exemplars = approach_params["use_exemplars"]
        if self.use_distillation:
            self.prev_net = None
            if self.use_variation:
                self.ft_net = None
                self.ft_optimizer = None
                self.ft_scheduler = None
                self.ft_loss = CustomizedLoss(approach_params["classification_loss"], None)
        if self.use_exemplars:
            self.exemplars = []
            self.K = approach_params["n_exemplars"]
        else:
            self.exemplars = None

        self.current_step = -1
        self.n_known_classes = 0

    def step(self):
        self.current_step = self.current_step + 1

    def update_nets(self):
        if self.current_step > 0:
            print("Updating networks...")
            # save the state of the network at the previous step in [prev_net]
            if self.use_distillation:
                self.prev_net = copy.deepcopy(self.net)

            # update the main [net]
            old_weights = self.net.fc.weight.data
            self.net.fc = nn.Linear(self.net.fc.in_features, (self.current_step + 1) * self.classes_per_group)
            self.net.fc.weight.data = torch.cat((old_weights, self.init_weights))
            parameters_to_optimize = self.net.parameters()
            self.optimizer = optim.SGD(parameters_to_optimize , lr = self.train_params["LR"], momentum = self.train_params["MOMENTUM"], weight_decay = self.train_params["WEIGHT_DECAY"])
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = self.train_params["STEP_MILESTONES"], gamma = self.train_params["GAMMA"])

            # prepare the [ft_net] to be fine-tuned
            if self.use_variation:
                self.ft_net = copy.deepcopy(self.net)
                parameters_to_optimize = self.ft_net.parameters()
                self.ft_optimizer = optim.SGD(parameters_to_optimize , lr = self.train_params["LR"], momentum = self.train_params["MOMENTUM"], weight_decay = self.train_params["WEIGHT_DECAY"])
                self.ft_scheduler = optim.lr_scheduler.MultiStepLR(self.ft_optimizer, milestones = self.train_params["STEP_MILESTONES"], gamma = self.train_params["GAMMA"])

    def train(self, dataloader):
        print("Training the main net...")
        n_new_classes = self.classes_per_group
        self.n_known_classes = (self.current_step + 1) * self.classes_per_group

        # bringing all the nets to cuda
        self.net = self.net.cuda()
        self.net.train(True)
        if self.current_step > 0:
            if self.use_distillation:
                self.prev_net = self.prev_net.cuda()
                self.prev_net.train(False)
                if self.use_variation:
                    self.ft_net = self.ft_net.cuda()
                    self.ft_net.train(False)

        cudnn.benchmarks
        log_step = 0
        for epoch in range(self.train_parms["NUM_EPOCHS"]):
            print(f"\rEpoch {epoch + 1}/{self.train_parms['NUM_EPOCHS']}...", end = "")
            for images, labels in dataloader:
                images = images.cuda()
                labels = transform_labels_onehot(labels, self.n_known_classes).cuda()
                output, features = self.net(images, output = 'all')

                # defining input and targets for classification and distillation loss
                # depending on the type of loss
                class_input = output
                class_target = labels

                if self.use_distillation and self.current_step > 0:
                    # if distillation is used, change input and target of classfication to new classes only
                    class_input = class_input[:, -n_new_classes:]
                    # the variation requires the classification targets to be the output of the ft_net
                    if self.use_variation:
                        ft_output = self.ft_net(images)
                        class_target = ft_output[:, -n_new_classes:]
                    else:
                        class_target = class_target[:, -n_new_classes:]
                    prev_output, prev_features = self.prev_net(images, output = 'all')
                    # "lfc" loss requires the previous and current features to compute the loss
                    if self.approach_params["distillation_loss"] == "lfc":
                        dist_input = features[:, :-n_new_classes]
                        dist_target = prev_features
                    else:
                        dist_input = output[:, :-n_new_classes]
                        dist_target = prev_output
                else:
                    dist_input, dist_target = None

                self.optimizer.zero_grad()

                loss = self.loss(class_input, class_target, dist_input, dist_target)
                loss.backward()
                self.optimizer.step()
                log_step = log_step + 1
            self.scheduler.step()
        print("")

    def update_exemplars(self):
        def get_features_representation(dataloader):
            mean = torch.zeros((self.net.fc.in_features,)).cuda()
            batch_features = []
            tot_images = 0

            self.net.train(False)
            with torch.no_grad():
                for images, _ in dataloader:
                    images = images.cuda()
                    features = self.net(images, output = "features")
                    batch_features.append(features)
                    mean += torch.sum(features)
                    tot_images += features.shape[0]
                mean /= tot_images
                batch_features = torch.cat(batch_features).cuda()

            return F.normalize(batch_features, p = 2), mean/torch.norm(mean, p = 2)

        def get_closest_exemplar_idx(label_mean, features, idx_taken):
            features = F.normalize(features, p = 2)
            distances = torch.pow(label_mean - features, 2).sum(-1)
            for idx in idx_taken:
                distances[idx] = 100000
            return distances.argmin().item()

        if self.use_exemplars:
            print("Updating exemplars...")

            n_old_classes = self.n_known_classes - self.classes_per_group
            m = self.K // self.n_known_classes
            new_labels = self.splitter.labels_split[self.current_step]

            if self.current_step > 0:
                for i in range(n_old_classes):
                    self.exemplars[i] = self.exemplars[i][:m]

            self.net = self.net.cuda()
            self.net.train(False)
            for label in new_labels:
                exemplar_set = []
                dataset = SubCIFAR(labels_split = self.splitter.labels_split, labels = [label], train = True, transform = self.train_params["test_transform"])
                dataloader = DataLoader(dataset, batch_size = self.train_params["BATCH_SIZE"], num_workers = 4 )
                features, label_mean = get_features_representation(dataloader)
                exemplars_mean = torch.zeros((self.net.fc.in_features,)).cuda()

                num_exemplars = min(m, len(dataset))
                idx_taken = []
                if self.approach_params["exemplars_selection"] == 'herding':
                    for _ in range(num_exemplars):
                        idx = get_closest_exemplar_idx(label_mean, features + exemplars_mean, idx_taken)
                        exemplars_mean += features[idx]
                        idx_taken.append(idx)
                else:
                    idx_taken = random.sample(range(len(dataset)), num_exemplars)

                for idx in idx_taken:
                    exemplar = dataloader.dataset.dataFrame.iloc[idx]
                    exemplar_set.append((exemplar["image"], exemplar["label"]))

                self.exemplars.append(np.array(exemplar_set))

    def train_ft(self, dataloader):
        if self.use_variation:
            print("Training the ft-net...")
            self.n_known_classes = (self.current_step + 1) * self.classes_per_group
            self.ft_net = self.ft_net.cuda()
            self.ft_net.train(True)
            cudnn.benchmark
            log_step = 0
            for epoch in range(self.train_parms["NUM_EPOCHS"]):
                print(f"\rEpoch {epoch + 1}/{self.train_params['NUM_EPOCHS']}...", end = "")
                for images, labels in dataloader:
                    images = images.cuda()
                    labels = transform_labels_onehot(labels, self.n_known_classes).cuda()
                    output = self.ft_net(images)
                    self.ft_optimizer.zero_grad()
                    loss = self.ft_loss(output, labels)
                    loss.backward()
                    self.ft_optimizer.step()
                    log_step = log_step + 1
                self.ft_scheduler.step()
            print("")
