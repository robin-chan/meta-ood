import os
from src.dataset.cityscapes_coco_mixed import CityscapesCocoMix
from src.dataset.lost_and_found import LostAndFound
from src.dataset.fishyscapes import Fishyscapes

TRAINSETS   = ["Cityscapes+COCO"]
VALSETS     = ["LostAndFound", "Fishyscapes"]
MODELS      = ["DeepLabV3+_WideResNet38", "DualGCNNet_res50"]

TRAINSET    = TRAINSETS[0]
VALSET      = VALSETS[0]
MODEL       = MODELS[0]
IO          = "/home/chan/io/ood_detection/"

class cs_coco_roots:
    """
    OoD training roots for Cityscapes + COCO mix
    """
    model_name  = MODEL
    init_ckpt   = os.path.join("/home/chan/io/cityscapes/weights/", model_name + ".pth")
    cs_root     = "/home/datasets/cityscapes/"
    coco_root   = "/home/datasets/COCO/2017"
    io_root     = IO + "meta_ood_" + model_name
    weights_dir = os.path.join(io_root, "weights/")


class laf_roots:
    """
    LostAndFound config class
    """
    model_name = MODEL
    init_ckpt = os.path.join("/home/chan/io/cityscapes/weights/", model_name + ".pth")
    eval_dataset_root = "/home/datasets/lost_and_found/"
    eval_sub_dir = "laf_eval"
    io_root = os.path.join(IO + "meta_ood_" + model_name, eval_sub_dir)
    weights_dir = os.path.join(io_root, "..", "weights/")


class fs_roots:
    """
    Fishyscapes config class
    """
    model_name = MODEL
    init_ckpt = os.path.join("/home/chan/io/cityscapes/weights/", model_name + ".pth")
    eval_dataset_root = "/home/datasets/fishyscapes/"
    eval_sub_dir = "fs_eval"
    io_root = os.path.join(IO + "meta_ood_" + model_name, eval_sub_dir)
    weights_dir = os.path.join(io_root, "..", "weights/")


class params:
    """
    Set pipeline parameters
    """
    training_starting_epoch = 0
    num_training_epochs     = 1
    pareto_alpha            = 0.9
    ood_subsampling_factor  = 0.1
    learning_rate           = 1e-5
    crop_size               = 480
    val_epoch               = num_training_epochs
    batch_size              = 8
    entropy_threshold       = 0.7


#########################################################################

class config_training_setup(object):
    """
    Setup config class for training
    If 'None' arguments are passed, the settings from above are applied
    """
    def __init__(self, args):
        if args["TRAINSET"] is not None:
            self.TRAINSET = args["TRAINSET"]
        else:
            self.TRAINSET = TRAINSET
        if self.TRAINSET == "Cityscapes+COCO":
            self.roots = cs_coco_roots
            self.dataset = CityscapesCocoMix
        else:
            print("TRAINSET not correctly specified... bye...")
            exit()
        if args["MODEL"] is not None:
            tmp = getattr(self.roots, "model_name")
            roots_attr = [attr for attr in dir(self.roots) if not attr.startswith('__')]
            for attr in roots_attr:
                if tmp in getattr(self.roots, attr):
                    rep = getattr(self.roots, attr).replace(tmp, args["MODEL"])
                    setattr(self.roots, attr, rep)
        self.params = params
        params_attr = [attr for attr in dir(self.params) if not attr.startswith('__')]
        for attr in params_attr:
            if attr in args:
                if args[attr] is not None:
                    setattr(self.params, attr, args[attr])
        roots_attr = [self.roots.weights_dir]
        for attr in roots_attr:
            if not os.path.exists(attr):
                print("Create directory:", attr)
                os.makedirs(attr)


class config_evaluation_setup(object):
    """
    Setup config class for evaluation
    If 'None' arguments are passed, the settings from above are applied
    """
    def __init__(self, args):
        if args["VALSET"] is not None:
            self.VALSET = args["VALSET"]
        else:
            self.VALSET = VALSET
        if self.VALSET == "LostAndFound":
            self.roots = laf_roots
            self.dataset = LostAndFound
        if self.VALSET == "Fishyscapes":
            self.roots = fs_roots
            self.dataset = Fishyscapes
        self.params = params
        params_attr = [attr for attr in dir(self.params) if not attr.startswith('__')]
        for attr in params_attr:
            if attr in args:
                if args[attr] is not None:
                    setattr(self.params, attr, args[attr])
        roots_attr = [self.roots.io_root]
        for attr in roots_attr:
            if not os.path.exists(attr):
                print("Create directory:", attr)
                os.makedirs(attr)
