import os
import argparse

parser = argparse.ArgumentParser(description='Project 1 COMP5212 Machine Learning')
parser.add_argument('-model','--model', help='model (lr, nn, svm_linear, svm_rbf)', required=False)
parser.add_argument('-show','--show', help='show losses over time for linear regression', required=False)

args = vars(parser.parse_args())
print(args)

if args["model"] == "lr":
    print("Linear Regression")
elif args["model"] == "nn":
    print("Neural Network")
elif args["model"] == "svm_linear":
    print("SVM linear kernel")
elif args["model"] == "svm_rbf":
    print("SVM RBF kernel")
else:
    print("wrong argument, only supports (lr, nn, svm_linear, svm_rbf)")