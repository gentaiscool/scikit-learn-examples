from sklearn import metrics
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report, log_loss, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

import datetime as dt

from util import *

# MODEL
class LogisticRegression():
# Logistic Regression Class
# using SGDClassifier built-in function
    def __init__(self, name, lr, l1_ratio, penalty):
        self.model = linear_model.SGDClassifier(loss="log", eta0=lr, l1_ratio=0.15, verbose=0, n_iter=1, penalty=penalty)
        self.name = name
    
    def fit(self, x, y):
        """ Training data """
        # Input
        # x: input
        # y: target
        
        # Output 
        # train_time: train time 
        
        start = dt.datetime.now()
        self.model.fit(x, y)
        train_time = (dt.datetime.now()-start).microseconds
        return train_time
    
    def predict(self, x):
        """ Predict label """
        # Input
        # x: input
        
        # Output 
        # predict_proba: probability of the prediction 
        # predict: prediction (0,1) 
        # eval_time: evaluation time 
        
        start = dt.datetime.now()
        predict = self.model.predict(x)
        predict_proba = self.model.predict_proba(x)
        eval_time = (dt.datetime.now()-start).microseconds
        return predict_proba, predict, eval_time
    
    def get_eval_per_epoch(self, it, true, pred_prob, pred):
        loss = log_loss(true, pred_prob)
        accuracy = accuracy_score(true, pred)
        return loss, accuracy
    
    def evaluation(self, true, pred_prob, pred):
        """ Evaluation to get metrics """
        print(confusion_matrix(true, pred))
        get_measures(true, pred)
        fpr, tpr, thresholds = metrics.roc_curve(true, pred_prob[:,1], pos_label=1)
        print("accuracy:", accuracy_score(true, pred))
        print("log loss:",log_loss(true, pred_prob))
        print("auc:",metrics.auc(fpr, tpr))
        print(classification_report(true, pred))
        print()
       
        
class NN():
# NN Class
# A neural network model class using MLPClassifier built-in function
    def __init__(self, name, activation, solver, lr, hidden_layer_sizes, parameters=None, is_grid_search=False):
        self.model = MLPClassifier(solver=solver, activation=activation, alpha=0.1, 
                                   hidden_layer_sizes=hidden_layer_sizes, random_state=0,
                                   learning_rate="adaptive", learning_rate_init=lr, early_stopping=False)
        self.name = name        
        self.parameters = parameters
        self.is_grid_search = is_grid_search
        
        if self.is_grid_search:
            self.grid_model = GridSearchCV(self.model, parameters, cv=5)
    
    def fit(self, x, y):
        """ Training data """
        # Input
        # x: input
        # y: target
        
        # Output 
        # train_time: train time 
        start = dt.datetime.now()
        if not self.is_grid_search:
            self.model.fit(x, y)
        else:
            self.grid_model.fit(x, y)
            self.best_parameter = self.grid_model.best_params_
            print("Best parameter:", self.best_parameter)
        train_time = (dt.datetime.now()-start).microseconds
        return train_time
    
    def predict(self, x):
        """ Predict label """
        start = dt.datetime.now()
        predict = self.model.predict(x)
        predict_proba = self.model.predict_proba(x)
        eval_time = (dt.datetime.now()-start).microseconds
        return predict_proba, predict, eval_time
    
    def get_eval_per_epoch(self, it, true, pred_prob, pred):
        loss = log_loss(true, pred_prob)
        accuracy = accuracy_score(true, pred)
        return loss, accuracy
    
    def evaluation(self, true, pred_proba, pred):
        """ Evaluation to get metrics """
        print(confusion_matrix(true, pred))
        get_measures(true, pred)
        fpr, tpr, thresholds = metrics.roc_curve(true, pred_proba[:,1], pos_label=1)
        print("accuracy:", accuracy_score(true, pred))
        print("log loss:",log_loss(true, pred_proba))
        print("auc:",metrics.auc(fpr, tpr))
        print(classification_report(true, pred))
        print()
        
class SVM():
# SVM Class
# Support Vector Machine class using svm built-in function   
    def __init__(self, name, kernel, gamma, parameters=None, is_grid_search=False):
        self.model = svm.SVC(kernel=kernel, gamma=gamma, probability=True)
        self.name = name        
        self.parameters = parameters
        self.gamma = gamma
        self.is_grid_search = is_grid_search
        
        if self.is_grid_search:
            self.grid_model = GridSearchCV(self.model, parameters, cv=5)
    
    def fit(self, x, y):
        """ Training data """
        # Input
        # x: input
        # y: target
        
        # Output 
        # train_time: train time 
        
        start = dt.datetime.now()
        if not self.is_grid_search:
            self.model.fit(x, y)
        else:
            self.grid_model.fit(x, y)
            self.best_parameter = self.grid_model.best_params_
            print("Best parameter:", self.best_parameter)
        train_time = (dt.datetime.now()-start).microseconds
        return train_time
    
    def predict(self, x):
        """ Predict label """
        start = dt.datetime.now()
        predict = self.model.predict(x)
        predict_proba = self.model.predict_proba(x)
        eval_time = (dt.datetime.now()-start).microseconds
        return predict_proba, predict, eval_time
    
    def get_eval_per_epoch(self, it, true, pred_proba, pred):
        loss = log_loss(true, pred_proba)
        accuracy = accuracy_score(true, pred)
        return loss, accuracy
    
    def evaluation(self, true, pred_proba, pred):
        """ Evaluation to get metrics """
        print(confusion_matrix(true, pred))
        get_measures(true, pred)
        fpr, tpr, thresholds = metrics.roc_curve(true, pred_proba[:,1], pos_label=1)
        print("accuracy:", accuracy_score(true, pred))
        print("log loss:",log_loss(true, pred_proba))
        print("auc:",metrics.auc(fpr, tpr))
        print(classification_report(true, pred))
        print()