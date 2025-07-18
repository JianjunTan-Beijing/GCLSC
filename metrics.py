from sklearn.metrics.cluster import pair_confusion_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np
import os
import pandas as pd
import csv
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Cluster Accuracy (CA) function
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy using linear sum assignment.
    """
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values
    
    y_pred = y_pred.astype(np.int64)
    label_to_number = {label: number for number, label in enumerate(set(y_true))}
    label_numerical = np.array([label_to_number[i] for i in y_true])

    y_true = label_numerical.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size

# Compute metrics function
def compute_metrics(y_true, y_pred, X=None):
    """
    计算评估指标：ARI, NMI, CA, Silhouette, Calinski-Harabasz
    """
    metrics = {}
    metrics["ARI"] = ARI(y_true, y_pred)
    metrics["NMI"] = NMI(y_true, y_pred)
    metrics["CA"] = cluster_acc(y_true, y_pred)
    
    if X is not None and len(np.unique(y_pred)) > 1:  
        metrics["Silhouette"] = silhouette_score(X, y_pred)
        metrics["Log_Calinski-Harabasz"] = np.log10(calinski_harabasz_score(X, y_pred))
    #else:
        #metrics["Silhouette"] = None  
        #metrics["Calinski-Harabasz"] = None  
    
    return metrics

# Traverse folder and compute metrics
def traverse_folder_compute_metrics(root_path, save_path):
    for filename in os.listdir(root_path):
        if "gt_label" not in filename:
            continue
        pre_filename, ext = os.path.splitext(filename)
        _, _, method_name, dataset_name = pre_filename.split("_", 3)

        gt_label_path = os.path.join(root_path, filename)
        pd_label_path = os.path.join(root_path, filename.replace("gt_label", "pd_label"))
        gt_label = pd.read_csv(gt_label_path, index_col=0).iloc[:, 0]
        pd_label = pd.read_csv(pd_label_path, index_col=0).iloc[:, 0]

        metrics = compute_metrics(gt_label, pd_label)

        print("{} - {}: \n {}".format(dataset_name, method_name, metrics))

        # Saving to CSV
        csv_path = os.path.join(save_path, "new_metrics.csv")
        need_head = False
        if os.path.exists(csv_path) != True:
            need_head = True
        f = open(csv_path, "a", encoding="utf-8", newline="")
        csv_writer = csv.writer(f)
        record = [dataset_name, method_name]
        head = ["dataset", "method"]
        for key in metrics.keys():
            record.append(metrics[key])
            head.append(key)
        if need_head == True:
            csv_writer.writerow(head)
        csv_writer.writerow(record)
        f.close()