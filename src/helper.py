import os
import math
import pickle

import numpy as np
from collections import Counter


class colors:
    """Class for colors"""
    RED = '\033[31;1m'
    GREEN = '\033[32;1m'
    YELLOW = '\033[33;1m'
    BLUE = '\033[34;1m'
    MAGENTA = '\033[35;1m'
    CYAN = '\033[36;1m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'


def getColorEntry(val):
    """Colored value output if colorized flag is activated."""
    if not isinstance(val, float) or math.isnan(val):
        return colors.ENDC
    if val < .20:
        return colors.RED
    elif val < .40:
        return colors.YELLOW
    elif val < .60:
        return colors.BLUE
    elif val < .80:
        return colors.CYAN
    else:
        return colors.GREEN


def counts_array_to_data_list(counts_array, max_size=None):
    if max_size is None:
        max_size = np.sum(counts_array)  # max of counted array entry
    counts_array = (counts_array / np.sum(counts_array) * max_size).astype("uint32")
    counts_dict = {}
    for i in range(1, len(counts_array) + 1):
        counts_dict[i] = counts_array[i - 1]
    return list(Counter(counts_dict).elements())


def get_save_path_metrics_i(i, metaseg_root, subdir):
    return os.path.join(metaseg_root, "metrics", subdir, "%04d.p" % i)


def get_save_path_components_i(i, metaseg_root, subdir):
    return os.path.join(metaseg_root, "components", subdir, "%04d.p" % i)


def metrics_dump(metrics, i, metaseg_root, subdir):
    dump_path = get_save_path_metrics_i(i, metaseg_root, subdir)
    dump_dir = os.path.dirname(dump_path)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    pickle.dump(metrics, open(dump_path, "wb"))


def components_dump(components, i, metaseg_root, subdir):
    dump_path = get_save_path_components_i(i, metaseg_root, subdir)
    dump_dir = os.path.dirname(dump_path)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    pickle.dump(components.astype('int16'), open(dump_path, "wb"))


def metrics_load(i, metaseg_root, subdir):
    return pickle.load(open(get_save_path_metrics_i(i, metaseg_root, subdir), "rb"))


def components_load(i, metaseg_root, subdir):
    return pickle.load(open(get_save_path_components_i(i, metaseg_root, subdir), "rb"))


def concatenate_metrics(metaseg_root, subdir, num_imgs):
    metrics = metrics_load(0, metaseg_root, subdir)
    start = list([0, len(metrics["S"])])
    for i in range(1, num_imgs):
        m = metrics_load(i, metaseg_root, subdir)
        start += [start[-1] + len(m["S"])]
        for j in metrics:
            metrics[j] += m[j]
    return metrics, start


def metrics_to_nparray(metrics, names, normalize=False, non_empty=False, all_metrics=None):
    if all_metrics is None:
        all_metrics = []
    I = range(len(metrics['S_in']))
    if non_empty:
        I = np.asarray(metrics['S_in']) > 0
    M = np.asarray([np.asarray(metrics[m])[I] for m in names])
    if all_metrics == []:
        MM = M.copy()
    else:
        MM = np.asarray([np.asarray(all_metrics[m])[I] for m in names])
    if normalize:
        for i in range(M.shape[0]):
            if names[i] != "class":
                M[i] = (np.asarray(M[i]) - np.mean(MM[i], axis=-1)) / (np.std(MM[i], axis=-1) + 1e-10)
    M = np.squeeze(M.T)
    return M


def metrics_to_dataset(metrics, nclasses, non_empty=False, all_metrics=None):
    if all_metrics is None:
        all_metrics = []
    exclude = ["iou", "iou0", "prc"]
    X_names = sorted([m for m in metrics if m not in exclude])
    class_names = ["cprob" + str(i) for i in range(nclasses) if "cprob" + str(i) in metrics]
    Xa = metrics_to_nparray(metrics, X_names, normalize=True, non_empty=non_empty, all_metrics=all_metrics)
    classes = metrics_to_nparray(metrics, class_names, normalize=True, non_empty=non_empty, all_metrics=all_metrics)
    ya = metrics_to_nparray(metrics, ["iou"], non_empty=non_empty)
    y0a = metrics_to_nparray(metrics, ["iou0"], non_empty=non_empty)   ### 1, if iou=0
    return Xa, classes, ya, y0a, X_names, class_names
