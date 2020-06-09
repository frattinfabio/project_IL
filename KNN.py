import torch
from sklearn.neighbors import KNeighborsClassifier

class KNNClassifier():
    def __init__(self, K = 3):
        self.net = None
        self.K = K
        # Inizialization of the classifier with K neighbors (3 as default value)
        self.classifier = KNeighborsClassifier(n_neighbors = self.K)
        
    def update(self, net, train_dataloader):
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
            # Fit the classifier on the input features and the labels
            self.classifier.fit(features, labels_tot)
           
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