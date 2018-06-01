import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, log_loss, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

from util import *
from model import *
from config import *

files = ["breast-cancer.npz", "diabetes.npz", "digit.npz", "iris.npz", "wine.npz"]

IMG_DIR = "./img/"

if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

def run_logistic_regression():
    print("###########################################")
    print("########### Logistic Regression ###########")
    print("###########################################")
    for file in files:
        name = file.replace(".npz", "")
        print("----------------------------------")
        print("# Dataset: " + name)
        print("----------------------------------")
        
        lr = 0.001 # step size
        l1_ratio = 0.5# regularizer
        penalty = "l1"
        num_epoch = 9
        
        train_x, train_y, test_x, test_y = load_data(file)
        
        lr_model = LogisticRegression(name, lr, l1_ratio, penalty)
        
        train_losses = []
        train_accuracies = []
        train_times = []
        
        test_losses = []
        test_accuracies = []
        test_times = []
        
        for it in tqdm(range(1, num_epoch)):
            lr_model.fit(train_x, train_y) # train 1 iter every epoch

            train_pred_prob, train_pred, train_time = lr_model.predict(train_x)
            train_loss, train_accuracy = lr_model.get_eval_per_epoch((it + 1), train_y, train_pred_prob, train_pred)
            train_times.append(train_time)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            
            test_pred_prob, test_pred, test_time = lr_model.predict(test_x)
            test_loss, test_accuracy = lr_model.get_eval_per_epoch((it + 1), test_y, test_pred_prob, test_pred)
            test_times.append(test_time)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
        
        print("TRAIN EVALUATION")
        lr_model.evaluation(train_y, train_pred_prob, train_pred)
        print()
        print("TEST EVALUATION")
        lr_model.evaluation(test_y, test_pred_prob, test_pred)

        # DRAW DIAGRAM
        
        x = []
        for i in range(1, num_epoch):
            x.append(i)
            
        plt.plot(x, train_losses)
        plt.plot(x, test_losses)

        plt.title("Dataset " + name)
        plt.legend(['train loss', 'test loss'], loc='upper left')
        plt.savefig(IMG_DIR + name + "_lr")
        if args["show"]:
            plt.show()
        
def run_nn():
    print("######################################")
    print("########### Neural Network ###########")
    print("######################################")
    for file in files:
        name = file.replace(".npz", "")
        print("----------------------------------")
        print("# Dataset: " + name)
        print("----------------------------------")
        
        # hyper-parameters
        solver = "sgd"
        lr = 0.001
        hidden_layer_sizes = (1)
        activation = "tanh"
        val_split = 0.2
        scores = ["accuracy", "recall", "precision", ""]
        num_epochs = 9
        
        # grid-search
        parameters = {'hidden_layer_sizes':[(1),(2),(3),(4),(5),
                                            (6),(7),(8),(9),(10)]
                    ,'learning_rate_init':[(0.1), (0.01), (0.001)
                    ]}
            
        train_x, train_y, test_x, test_y = load_data(file)
        
        nn_model = NN(name, activation, solver, lr, hidden_layer_sizes, parameters, True)
        nn_model.fit(train_x, train_y)
        
        hidden_layer_sizes = nn_model.best_parameter["hidden_layer_sizes"]
        lr = nn_model.best_parameter["learning_rate_init"]
        
        # use best parameter after cross-validation
        print("Use best parameter after cross-validation")
        nn_model = NN(name, activation, solver, lr, hidden_layer_sizes, None, False)
        
        train_time = nn_model.fit(train_x, train_y)
        print("train time:", train_time)
        
        it = 0
        
        train_pred_prob, train_pred, _ = nn_model.predict(train_x)
        train_loss, train_accuracy = nn_model.get_eval_per_epoch((it + 1), train_y, train_pred_prob, train_pred)
        
        test_pred_prob, test_pred, test_time = nn_model.predict(test_x)
        test_loss, test_accuracy = nn_model.get_eval_per_epoch((it + 1), test_y, test_pred_prob, test_pred)
        print("test time:", test_time)
        
        print("TRAIN EVALUATION")
        nn_model.evaluation(train_y, train_pred_prob, train_pred)
        
        print("TEST EVALUATION")
        nn_model.evaluation(test_y, test_pred_prob, test_pred)

def run_svm_linear():
    print("##############################################")
    print("########### SVM with linear kernel ###########")
    print("##############################################")
    for file in files:
        name = file.replace(".npz", "")
        print("----------------------------------")
        print("# Dataset: " + name)
        print("----------------------------------")
        
        # hyper-parameters
        scores = ["accuracy", "recall", "precision", ""]
            
        train_x, train_y, test_x, test_y = load_data(file)
        gamma = 'auto'
        
        kernel="linear"
        svm_model = SVM(name, kernel, gamma, None, False)
        train_time = svm_model.fit(train_x, train_y)
        
        it = 0
        train_pred_prob, train_pred, _ = svm_model.predict(train_x)
        train_loss, train_accuracy = svm_model.get_eval_per_epoch((it + 1), train_y, train_pred_prob, train_pred)
        print("train time:", train_time)
        
        test_pred_prob, test_pred, test_time = svm_model.predict(test_x)
        test_loss, test_accuracy = svm_model.get_eval_per_epoch((it + 1), test_y, test_pred_prob, test_pred)
        print("test time:", test_time)
        
        print("TRAIN EVALUATION")
        svm_model.evaluation(train_y, train_pred_prob, train_pred)
        
        print("TEST EVALUATION")
        svm_model.evaluation(test_y, test_pred_prob, test_pred)

def run_svm_rbf():
    # SVM with RBF kernel
    print("###########################################")
    print("########### SVM with RBF kernel ###########")
    print("###########################################")
    for file in files:
        name = file.replace(".npz", "")
        print("----------------------------------")
        print("# Dataset: " + name)
        print("----------------------------------")
        
        # hyper-parameters
        scores = ["accuracy", "recall", "precision", ""]
        
        # grid-search
        parameters = {'gamma':[(1),(0.1),(0.01),(0.001)]}
            
        train_x, train_y, test_x, test_y = load_data(file)
        gamma = 1
        
        kernel="rbf"
        svm_model = SVM(name, kernel, gamma, parameters, True)
        svm_model.fit(train_x, train_y)
        
        gamma = svm_model.best_parameter["gamma"]
        
        # use best parameter after cross-validation
        print("Use best parameter after cross-validation")
        svm_model = SVM(name, kernel, gamma, None, False)
        train_time = svm_model.fit(train_x, train_y)
        
        it = 0
        train_pred_prob, train_pred, _ = svm_model.predict(train_x)
        train_loss, train_accuracy = svm_model.get_eval_per_epoch((it + 1), train_y, train_pred_prob, train_pred)
        print("train time:", train_time)
        
        test_pred_prob, test_pred, test_time = svm_model.predict(test_x)
        test_loss, test_accuracy = svm_model.get_eval_per_epoch((it + 1), test_y, test_pred_prob, test_pred)
        print("test time:", test_time)
        
        print("TRAIN EVALUATION")
        svm_model.evaluation(train_y, train_pred_prob, train_pred)
        
        print("TEST EVALUATION")
        svm_model.evaluation(test_y, test_pred_prob, test_pred)

if __name__ == "__main__":
    if args["model"] == "lr":
        run_logistic_regression()
    elif args["model"] == "nn":
        run_nn()
    elif args["model"] == "svm_linear":
        run_svm_linear()
    elif args["model"] == "svm_rbf":
        run_svm_rbf()