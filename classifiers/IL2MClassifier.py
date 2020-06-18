import torch

class IL2MClassifier():
    def __init__(self):
        self.net = None
        self.current_step = None
        self.mean_train_scores = {}
        self.mean_examplars_scores = {}
        self.confidences = {}

    def update(self, step, net, train_dataloader):
        self.net = net
        self.current_step = step

        num_new_classes = len(train_dataloader.dataset.stored_labels)
        num_old_classes = len(self.mean_train_scores)
        # initialize the confidence for the new state of the net
        self.confidences[step] = 0
        # initialize the scores for the new classes in the current state, to be used in future states
        # it contains both the score and the step the relative class was trained
        for i in range(num_new_classes):
            self.mean_train_scores[num_old_classes + i] = [step, 0]  
        # re-initialize the scores for the old classes in the current state  
        self.mean_examplars_scores = {k: 0 for k in range(num_old_classes)}
        # num_images contains the number of images per class in train_dataloader 
        num_images = [0 for _ in range(num_new_classes + num_old_classes)]

        self.net = self.net.cuda()
        self.net.train(False)
        with torch.no_grad():
            for images, labels in train_dataloader:
                images = images.cuda()
                labels = labels.cuda()
                scores = net(images)
                for score, label in zip(scores, labels):
                    score = score.cpu()
                    label = label.cpu().item()
                    self.confidences[step] += torch.max(score).item()
                    # the train dataloader already contains both the new data and the examplars
                    # if label < num_old_classes it means it is an examplar
                    if label >= num_old_classes:
                        self.mean_train_scores[label][1] += score[label]
                    else:
                        self.mean_examplars_scores[label] += score[label]
                    num_images[label] += 1

        self.confidences[step] /= len(train_dataloader.dataset)
        # dividing for the number of images per label
        for i in range(num_old_classes):
            self.mean_examplars_scores[i] /= num_images[i]
        for i in range(num_new_classes):
            self.mean_train_scores[num_old_classes + i][1] /= num_images[num_old_classes + i]

    def rectify(self, score, num_old_classes):
        lambda_rect = 1.3
        for i in range(num_old_classes):
            old_step = self.mean_train_scores[i][0]
            mu_p_class = self.mean_train_scores[i][1]
            mu_n_class = self.mean_examplars_scores[i]
            conf_p = self.confidences[old_step]
            conf_n = self.confidences[self.current_step]
            score[i] = lambda_rect * score[i] * (mu_p_class/mu_n_class) * (conf_n/conf_p)
        return score

    def classify(self, images):
        num_old_classes = len(self.mean_examplars_scores)
        preds = []
        with torch.no_grad():
            self.net = self.net.cuda()
            self.net.train(False)
            scores = self.net(images)

            for score in scores:
                score = score.cpu()
                pred = torch.argmax(score).item()
                if pred >= num_old_classes:
                    # rectify scores for old classes
                    score = self.rectify(score, num_old_classes)
                    pred = torch.argmax(score).item()
                preds.append(pred)

        return torch.Tensor(preds).cuda()
