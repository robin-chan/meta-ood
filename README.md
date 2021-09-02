
# Entropy Maximization and Meta Classification for Out-of-Distribution Detection in Semantic Segmentation  
  
**Abstract** Deep neural networks (DNNs) for the semantic segmentation of images are usually trained to operate on a predefined closed set of object classes. This is in contrast to the "open world" setting where DNNs are envisioned to be deployed to. From a functional safety point of view, the ability to detect so-called "out-of-distribution" (OoD) samples, i.e., objects outside of a DNN's semantic space, is crucial for many applications such as automated driving.
We present a two-step procedure for OoD detection. Firstly, we utilize samples from the COCO dataset as OoD proxy and introduce a second training objective to maximize the softmax entropy on these samples. Starting from pretrained semantic segmentation networks we re-train a number of DNNs on different in-distribution datasets and evaluate on completely disjoint OoD datasets. Secondly, we perform a transparent post-processing step to discard false positive OoD samples by so-called "meta classification". To this end, we apply linear models to a set of hand-crafted metrics derived from the DNN's softmax probabilities.
Our method contributes to safer DNNs with more reliable overall system performance.

* More details can be found in the preprint https://arxiv.org/abs/2012.06575
* Training with [Cityscapes](https://www.cityscapes-dataset.com/) and [COCO](https://cocodataset.org), evaluation with [LostAndFound](http://www.6d-vision.com/lostandfounddataset) and [Fishyscapes](https://fishyscapes.com/)
  
## Requirements  
  
This code was tested with **Python 3.6.10** and **CUDA 10.2**. The following Python packages were installed via **pip 20.2.4**, see also ```requirements.txt```: 
```  
Cython == 0.29.21  
h5py == 3.1.0  
scikit-learn == 0.23.2  
scipy == 1.5.4  
torch == 1.7.0  
torchvision == 0.8.1
pycocotools == 2.0.2
```
**Dataset preparation**: In ```preparation/prepare_coco_segmentation.py``` a preprocessing script can be found in order prepare the COCO images serving as OoD proxy for OoD training. This script basically generates binary segmentation masks for COCO images not containing any instances that could also be assigned to one of the Cityscapes (train-)classes. Execute via:
```  
python preparation/prepare_coco_segmentation.py
```
Regarding the Cityscapes dataset, the dataloader used in this repo assumes that the *labelTrainId* images are already generated according to the [official Cityscapes script](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py).

**Cython preparation**: Make sure that the Cython script ```src/metaseg/metrics.pyx``` (on the machine where the script is deployed to) is compiled. If it has not been compiled yet:  
```  
cd src/metaseg/  
python metrics_setup.py build_ext --inplace  
cd ../../  
```  
For pretrained weights, see [https://github.com/NVIDIA/semantic-segmentation/tree/sdcnet](https://github.com/NVIDIA/semantic-segmentation/tree/sdcnet) (for DeepLabv3+) and [https://github.com/lxtGH/GALD-DGCNet](https://github.com/lxtGH/GALD-DGCNet) (for DualGCNNet).
The weights after OoD training can be downloaded [here for DeepLabv3+](https://uni-wuppertal.sciebo.de/s/kCgnr0LQuTbrArA/download) and [here for DualGCNNet](https://uni-wuppertal.sciebo.de/s/VAXiKxZ21eAF68q/download).
  
## Quick start  
  
Modify settings in ```config.py```. All files will be saved in the directory defined via ```io_root``` (Different roots for each datasets that is used). Then run:  
```  
python ood_training.py  
python meta_classification.py  
python evaluation.py  
```  
  
## More options
  
For better automation of experiments,  **command-line options** for ```ood_training.py```, ```meta_classification.py``` and ```evaluation.py``` are available.  
  
Use the ```[-h]``` argument for details about which parameters in ```config.py``` can be modified via command line. Example:  
```  
python ood_training.py -h  
```  
  
If no command-line options are provided, the settings in ```config.py``` are applied.

