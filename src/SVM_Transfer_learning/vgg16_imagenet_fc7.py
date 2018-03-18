# load necessary modules
import os
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.svm import SVC
import pickle
from utils.create_dataset import *

# Train a linear SVM classifier on the features
# of the fc7 layer of VGG16 and save the model to a file
def train_svm(X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
	tuned_parameters = {'kernel': ['linear'], 'C':[1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}
	
	X = np.concatenate((X_train, X_valid), axis=0)
	Y = np.concatenate((Y_train, Y_valid), axis=0)

	my_test_fold = [-1] * 27947 + [0]*3508

	ps = PredefinedSplit(test_fold=my_test_fold)
	sv = GridSearchCV(SVC(), tuned_parameters, cv=ps, scoring='accuracy', verbose=1)
	sv.fit(X, Y)

	return sv.predict(X_test)


X_train, Y_train = create_dataset('../../data/transfer_learning/VGG16_ImageNet_fc7/Train/', '../../data/ferplus/Train48/')
X_valid, Y_valid = create_dataset('../../data/transfer_learning/VGG16_ImageNet_fc7/Valid/', '../../data/ferplus/Valid48/')
X_test, Y_test = create_dataset('../../data/transfer_learning/VGG16_ImageNet_fc7/Test/', '../../data/ferplus/Test48/')


train_svm(X_train, Y_train, X_valid, Y_valid)