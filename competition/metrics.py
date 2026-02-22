
from sklearn.metrics import roc_auc_score

def binary_auc(y_true, y_pred):
    return float(roc_auc_score(y_true, y_pred))
