import torch
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import RandomUnderSampler

# classifier using the NN policy
# [k_values]: the n_neighbors value to be tried 
class KNNClassifier():
    def __init__(self, k_values = [9, 11, 13, 15]):
        self.net = None
        self.k_param_grid = {"n_neighbors": k_values}
        self.classifier = KNeighborsClassifier(weights = "distance")

    def update(self, step, net, train_dataloader):
        self.net = net
        images_tot = None
        labels_tot = None
        self.net = self.net.cuda()
        self.net.train(False)
        with torch.no_grad():
            for images, labels in train_dataloader:
                if images_tot is None:
                    images_tot = images
                    labels_tot = labels
                else:
                    # Take all the images and labels to be fitted by the classifier
                    images_tot = torch.cat((images_tot, images), 0)
                    labels_tot = torch.cat((labels_tot, labels), 0)

            images_tot = images_tot.cuda()
            features = self.net(images_tot, output = 'features')
            features = features.cpu()

            # undersampling to tackle the unbalance between the new data and old examplars
            rus = RandomUnderSampler()
            features, labels_tot = rus.fit_resample(features, labels_tot)
            # selecting the best K through grid search with cross-validation
            gs = GridSearchCV(estimator = self.classifier, param_grid = self.k_param_grid, cv = 4)
            gs.fit(features, labels_tot)
            self.classifier = gs.best_estimator_

    def classify(self, images):
        preds = []
        self.net = self.net.cuda()
        self.net.train(False)
        with torch.no_grad():
            features = self.net(images, output = 'features')
            features = features.cpu()
            # Classifier's predictions
            preds = self.classifier.predict(features)
        return torch.Tensor(preds).cuda()
