import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve


def plot_precision_recall(y_true, y_score, title='Precision-recall curve'):
    plt.title(title)
    plt.ylabel('Precision: tp/(tp+fp)')
    plt.xlabel('Recall (true positive rate): tp/(tp+fn)')
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    plt.plot(recall, precision)


def plot_roc(y_true, y_score, title='ROC curve'):
    plt.title(title)
    plt.xlabel('False positive rate: fp/(fp+tn)')
    plt.ylabel('True positive rate (recall): tp/(tp+fn)')
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    plt.plot(fpr, tpr)


def plot_precision_vs_score(y_true, y_score, title='Precision vs score'):
    plt.title(title)
    plt.xlabel('Score threshold')
    plt.ylabel('Precision: tp/(tp+fp)')
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    plt.plot(thresholds, precision[:-1])


def precision_at_score_percentile(y_true, y_score, pctile=90, max_pctile=100):
    y_true = np.ravel(y_true)
    y_score = np.ravel(y_score)
    ys = y_true[(y_score >= np.percentile(y_score, pctile)) & (y_score <= np.percentile(y_score, max_pctile))]
    count = len(ys)
    if count == 0:
        return 0
    return np.sum(ys) / count


def precision_by_score_buckets(y_true, y_score, n_buckets=100):
    y_true = np.ravel(y_true)
    y_score = np.ravel(y_score)
    y_true_sorted = y_true[np.argsort(y_score)]
    chunks = np.array_split(y_true_sorted, n_buckets)
    sums = np.array([sum(x) for x in chunks])
    counts = np.array([len(x) for x in chunks])
    return sums / counts


def plot_precision_by_score_buckets(y_true, y_score, n_buckets=100, title='Precision by score bucket'):
    precs = precision_by_score_buckets(y_true, y_score, n_buckets)
    plt.plot(precs, marker='o', linestyle='')
    plt.title(title)


def plot_bucket_counts(y_true, y_score, bucketsize=100):
    nbuckets = int(len(y_true) / bucketsize)
    counts = bucketsize * precision_by_score_buckets(y_true, y_score, nbuckets)
    bucket_avg = bucketsize * (sum(y_true) / len(y_true))
    plt.plot(counts, marker='o', linestyle='')
    plt.plot([0, nbuckets], [bucket_avg, bucket_avg], '--')
    plt.xlabel('Buckets of %d ordered by score' % bucketsize)
    plt.ylabel('True positive count per bucket')
    plt.title('Predictions ranked by score, buckets of %d' % bucketsize)

