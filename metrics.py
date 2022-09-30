#from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F



def thresholded_output_transform(y_pred):
    y_pred = torch.round(y_pred) # if p <= 0.5 --> returns 0
    return y_pred

def classification_accuracy(predictions, targets):

        pred = F.softmax(predictions, dim=-1) #softmax per row (over the columns/classes)
        #pred = thresholded_output_transform(pred) #th of 0.5
        corrects = (pred.argmax(1) == targets.argmax(1)).cpu().numpy().mean() 
        return corrects

def comp_confmat(actual, predicted, classn=10):

    # extract the different classes
    classes = range(classn)#np.unique(actual)

    # initialize the confusion matrix
    confmat = torch.zeros((len(classes), len(classes)), device="cuda")

    # loop across the different combinations of actual / predicted classes
    for i in range(len(classes)):
        for j in range(len(classes)):

           # count the number of instances in each combination of actual / predicted classes
           confmat[j, i] = torch.sum((actual == classes[i]) & (predicted == classes[j]))

    return confmat

def conf_matrix(predictions, targets, classn=3):

    nsp_m = comp_confmat(targets,predictions,classn=classn)

    return nsp_m
