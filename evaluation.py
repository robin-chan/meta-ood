import argparse
import os
import time
import sys

import numpy as np
import torch
import pickle

from config import config_evaluation_setup
from src.imageaugmentations import Compose, Normalize, ToTensor
from src.model_utils import inference
from scipy.stats import entropy
from src.calc import calc_precision_recall, calc_sensitivity_specificity
from src.helper import concatenate_metrics
from meta_classification import meta_classification


class eval_pixels(object):
    """
    Evaluate in vs. out separability on pixel-level
    """

    def __init__(self, params, roots, dataset):
        self.params = params
        self.epoch = params.val_epoch
        self.alpha = params.pareto_alpha
        self.batch_size = params.batch_size
        self.roots = roots
        self.dataset = dataset
        self.save_dir_data = os.path.join(self.roots.io_root, "results/entropy_counts_per_pixel")
        self.save_dir_plot = os.path.join(self.roots.io_root, "plots")
        if self.epoch == 0:
            self.pattern = "baseline"
            self.save_path_data = os.path.join(self.save_dir_data, "baseline.p")
        else:
            self.pattern = "epoch_" + str(self.epoch) + "_alpha_" + str(self.alpha)
            self.save_path_data = os.path.join(self.save_dir_data, self.pattern + ".p")

    def counts(self, loader, num_bins=100, save_path=None, rewrite=False):
        """
        Count the number in-distribution and out-distribution pixels
        and get the networks corresponding confidence scores
        :param loader: dataset loader for evaluation data
        :param num_bins: (int) number of bins for histogram construction
        :param save_path: (str) path where to save the counts data
        :param rewrite: (bool) whether to rewrite the data file if already exists
        """
        print("\nCounting in-distribution and out-distribution pixels")
        if save_path is None:
            save_path = self.save_path_data
        if not os.path.exists(save_path) or rewrite:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                print("Create directory", save_dir)
                os.makedirs(save_dir)
            bins = np.linspace(start=0, stop=1, num=num_bins + 1)
            counts = {"in": np.zeros(num_bins, dtype="int64"), "out": np.zeros(num_bins, dtype="int64")}
            inf = inference(self.params, self.roots, loader, self.dataset.num_eval_classes)
            for i in range(len(loader)):
                probs, gt_train, _, _ = inf.probs_gt_load(i)
                ent = entropy(probs, axis=0) / np.log(self.dataset.num_eval_classes)
                counts["in"] += np.histogram(ent[gt_train == self.dataset.train_id_in], bins=bins, density=False)[0]
                counts["out"] += np.histogram(ent[gt_train == self.dataset.train_id_out], bins=bins, density=False)[0]
                print("\rImages Processed: {}/{}".format(i + 1, len(loader)), end=' ')
                sys.stdout.flush()
            torch.cuda.empty_cache()
            pickle.dump(counts, open(save_path, "wb"))
        print("Counts data saved:", save_path)

    def oodd_metrics_pixel(self, datloader=None, load_path=None):
        """
        Calculate 3 OoD detection metrics, namely AUROC, FPR95, AUPRC
        :param datloader: dataset loader
        :param load_path: (str) path to counts data (run 'counts' first)
        :return: OoD detection metrics
        """
        if load_path is None:
            load_path = self.save_path_data
        if not os.path.exists(load_path):
            if datloader is None:
                print("Please, specify dataset loader")
                exit()
            self.counts(loader=datloader, save_path=load_path)
        data = pickle.load(open(load_path, "rb"))
        fpr, tpr, _, auroc = calc_sensitivity_specificity(data, balance=True)
        fpr95 = fpr[(np.abs(tpr - 0.95)).argmin()]
        _, _, _, auprc = calc_precision_recall(data)
        if self.epoch == 0:
            print("\nOoDD Metrics - Epoch %d - Baseline" % self.epoch)
        else:
            print("\nOoDD Metrics - Epoch %d - Lambda %.2f" % (self.epoch, self.alpha))
        print("AUROC:", auroc)
        print("FPR95:", fpr95)
        print("AUPRC:", auprc)
        return auroc, fpr95, auprc


def oodd_metrics_segment(params, roots, dataset, metaseg_dir=None):
    """
    Compute number of errors before / after meta classification and compare to baseline
    """
    epoch = params.val_epoch
    alpha = params.pareto_alpha
    thresh = params.entropy_threshold
    num_imgs = len(dataset)
    if epoch == 0:
        load_subdir = "baseline" + "_t" + str(thresh)
    else:
        load_subdir = "epoch_" + str(epoch) + "_alpha_" + str(alpha) + "_t" + str(thresh)
    if metaseg_dir is None:
        metaseg_dir = os.path.join(roots.io_root, "metaseg_io")
    try:
        m, _ = concatenate_metrics(metaseg_root=metaseg_dir, num_imgs=num_imgs,
                                   subdir="baseline" + "_t" + str(thresh))
        fp_baseline = len([i for i in range(len(m["iou0"])) if m["iou0"][i] == 1])
        m, _ = concatenate_metrics(metaseg_root=metaseg_dir, num_imgs=num_imgs,
                                   subdir="baseline" + "_t" + str(thresh) + "_gt")
        fn_baseline = len([i for i in range(len(m["iou"])) if m["iou0"][i] == 1])
    except FileNotFoundError:
        fp_baseline, fn_baseline = None, None
    m, _ = concatenate_metrics(metaseg_root=metaseg_dir, num_imgs=num_imgs,
                               subdir=load_subdir + "_gt")
    fn_training = len([i for i in range(len(m["iou"])) if m["iou0"][i] == 1])
    fn_meta, fp_training, fp_meta = meta_classification(params=params, roots=roots, dataset=dataset).remove()

    if epoch == 0:
        print("\nOoDD Metrics - Epoch %d - Baseline - Entropy Threshold %.2f" % (epoch, thresh))
    else:
        print("\nOoDD Metrics - Epoch %d - Lambda %.2f - Entropy Threshold %.2f" % (epoch, alpha, thresh))
    if fp_baseline is not None and fn_baseline is not None:
        print("Num FPs baseline                       :", fp_baseline)
        print("Num FNs baseline                       :", fn_baseline)
    if epoch > 0:
        print("Num FPs OoD training                   :", fp_training)
        print("Num FNs OoD training                   :", fn_training)
    print("Num FPs OoD training + meta classifier :", fp_meta)
    print("Num FNs OoD training + meta classifier :", fn_meta)
    return fp_baseline, fn_baseline, fp_training, fn_training, fp_meta, fn_meta


def main(args):
    config = config_evaluation_setup(args)
    if not args["pixel_eval"] and not args["segment_eval"]:
        args["pixel_eval"] = args["segment_eval"] = True

    transform = Compose([ToTensor(), Normalize(config.dataset.mean, config.dataset.std)])
    datloader = config.dataset(root=config.roots.eval_dataset_root, transform=transform)
    start = time.time()

    """Perform evaluation"""
    print("\nEVALUATE MODEL: ", config.roots.model_name)
    if args["pixel_eval"]:
        print("\nPIXEL-LEVEL EVALUATION")
        eval_pixels(config.params, config.roots, config.dataset).oodd_metrics_pixel(datloader=datloader)

    if args["segment_eval"]:
        print("\nSEGMENT-LEVEL EVALUATION")
        oodd_metrics_segment(config.params, config.roots, datloader)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nFINISHED {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':
    """Get Arguments and setup config class"""
    parser = argparse.ArgumentParser(description='OPTIONAL argument setting, see also config.py')
    parser.add_argument("-train", "--TRAINSET", nargs="?", type=str)
    parser.add_argument("-val", "--VALSET", nargs="?", type=str)
    parser.add_argument("-model", "--MODEL", nargs="?", type=str)
    parser.add_argument("-epoch", "--val_epoch", nargs="?", type=int)
    parser.add_argument("-alpha", "--pareto_alpha", nargs="?", type=float)
    parser.add_argument("-pixel", "--pixel_eval", action='store_true')
    parser.add_argument("-segment", "--segment_eval", action='store_true')
    main(vars(parser.parse_args()))
