# Gaze-on-Objects (GOO) Project
A repository featuring evaluation of state-of-the-art research papers on the task of **Gaze Estimation** (locating the specific point a human in an image is looking at) and the novel task of **Gaze Object Detection** (identifying the object in an image a human in the same image is looking at).

![GOO_GIF](./tools/GOO_GIF.gif)

## Paper

[Arxiv](https://arxiv.org/abs/2105.10793)

To appear at [CVPR2021 3rd International Workshop on Gaze Estimation and Prediction in the Wild (GAZE 2021)](https://gazeworkshop.github.io/2021/)

## Datasets

1. **GazeFollow**: A dataset for evaluation on the Gaze Estimation task. Composed of images of humans in different scenarios with their heads and gaze points annotated.
2. **GOO-Synth**: A *synthetic* dataset for evaluation on the Gaze Object Detection task. Composed of images of scenes in a virtual marketplace environment, where the human's head, gaze point, and gazed object is annotated. 
3. **GOO-Real**: A smaller, accompanying dataset for GOOSynth, composed of real-world images of humans in a marketplace environment, where the human's head, gaze point, and gazed object is annotated. Designed for domain adaptation of models trained on GooSynth from simulation to real-world applications.

## Baseline Evaluation

The following baselines are stable and found in the *master* branch:

1. A. Recasens, A. Khosla, C. Vondrick and A. Torralba. **"Where are they looking?"** 
2. Dongze Lian, Zehao Yu, Shenghua Gao. **"Believe It or Not, We Know What You Are Looking at!"**
3. Chong, Eunji and Wang, Yongxin and Ruiz, Nataniel and Rehg, James M. **"Detecting Attended Visual Targets in Video"**.

## dataset
The [dataset](./dataset/) directory contains instructions on how to download GOO-Synth and GOO-Real, keys to access the annotations, as well as lookup tables for the object detection and segmentation classes.
    
## gazefollowing

The gazefollowing directory contains the code used in implementing selected gazefollowing methods for evaluation on the GazeFollow and GOO dataset.

Documentation on this directory's installation and usage can be found in the [readme](https://github.com/upeee/GazeOnObjects/blob/master/gazefollowing/README.md).

## Citation
If you find this work useful, please cite:

```
@inproceedings{tomas2021goo,
  title={GOO: A Dataset for Gaze Object Prediction in Retail Environments},
  author={Tomas, Henri and Reyes, Marcus and Dionido, Raimarc and Ty, Mark and Casimiro, Joel and Atienza, Rowel and Guinto, Richard},
  booktitle = {CVPR Workshops (CVPRW)},
  year={2021},
  pubstate={published},
  tppubtype={inproceedings}
}
```
