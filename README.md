# Gaze-on-Objects (GOO) Project
A repository featuring evaluation of state-of-the-art research papers on the task of **Gaze Estimation** (locating the specific point a human in an image is looking at) and the novel task of **Gaze Object Detection** (identifying the object in an image a human in the same image is looking at).

## Datasets

1. **GazeFollow**: A dataset for evaluation on the Gaze Estimation task. Composed of images of humans in different scenarios with their heads and gaze points annotated.
2. **GOO-Synth**: A *synthetic* dataset for evaluation on the Gaze Object Detection task. Composed of images of scenes in a virtual marketplace environment, where the human's head, gaze point, and gazed object is annotated. 
3. **GOO-Real**: A smaller, accompanying dataset for GOOSynth, composed of real-world images of humans in a marketplace environment, where the human's head, gaze point, and gazed object is annotated. Designed for domain adaptation of models trained on GooSynth from simulation to real-world applications.

## Baseline Evaluation

The following baselines are stable and found in the *master* branch:

1. A. Recasens, A. Khosla, C. Vondrick and A. Torralba. **"Where are they looking?"** 
2. Dongze Lian, Zehao Yu, Shenghua Gao. **"Believe It or Not, We Know What You Are Looking at!"**
3. Chong, Eunji and Wang, Yongxin and Ruiz, Nataniel and Rehg, James M. **"Detecting Attended Visual Targets in Video"**.
    
## Documentation

Documentation on this repository's installation and usage can be found in the [readme](https://github.com/upeee/GazeOnObjects/blob/master/gazefollowing/README.md) under the gazefollowing directory. 
