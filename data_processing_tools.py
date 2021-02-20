
import numpy as np
import datetime
import pickle
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve, auc
from matplotlib import pyplot as plt
from collections import Generator


def date_transfer(s: str,
                  possible_pattern: list = None) -> datetime.datetime:
    """
    Transferring string-like date object into datetime.datetime format
    :param s: string-like date object
    :param possible_pattern: possible patterns for datetime string
    :return: None if fail to recognize, datetime
    """
    if not possible_pattern:
        possible_pattern = ['%%Y/%m/%d', '%Y-%m-%d']
    res = 0
    for pattern in possible_pattern:
        try:
            res = datetime.datetime.strptime(s, pattern)
        except ValueError:
            continue
    if res:
        return res
    else:
        print('Unrecognizable pattern {}.'.format(s))
        return None


def data_gen(x: np.array,
             n: int) -> Generator:
    """
    Data generator: splitting x into x//n parts and generate each part using next() function
    :param x: the data to be generated
    :param n: size of each part
    :return: generated data
    """
    while True:
        for i in range(x.shape[0] // n):
            yield x.iloc[i * n:(i + 1) * n]


def save_obj(obj, name):
    """
    Save an object to local, using pickle
    :param obj: ...
    :param name: ...
    :return: None
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    Loading an local object using pickle
    :param name: ...
    :return: Loaded object
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def draw_roc_curve(yt, yp):
    fpr, tpr, thresholds = roc_curve(yt, yp)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()