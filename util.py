import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, log_loss, accuracy_score

DATA_DIR = '../datasets/'

def load_data(filename):
    """
        Read data from npz file (numpy format)
        Input: 
            filename: string
        Output:
            train_x, train_y, test_x, test_y: array
    """
    data = np.load(DATA_DIR + filename)
    train_x = data['train_X']
    train_y = data['train_Y']
    test_x = data['test_X']
    test_y = data['test_Y']
    
    return train_x, train_y, test_x, test_y

# EVALUATION FUNCTIONS
def get_tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def get_tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
def get_fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def get_fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]

def get_measures(y_true, y_pred):
    tp = get_tp(y_true, y_pred)
    tn = get_tn(y_true, y_pred)
    fp = get_fp(y_true, y_pred)
    fn = get_fn(y_true, y_pred)
    print("tp:", tp, "tn", tn, "fp", fp, "fn", fn)
    return tp, tn, fp, fn