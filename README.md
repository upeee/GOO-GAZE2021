# Gaze-on-Objects (GOO) Project
A repository featuring evaluation of state-of-the-art research papers on the task of **Gaze Estimation** (locating the specific point a human in an image is looking at) and the novel task of **Gaze Object Detection** (identifying the object in an image a human in the same image is looking at).

## Datasets

1. **GazeFollow**: A dataset for evaluation on the Gaze Estimation task. Composed of images of humans in different scenarios with their heads and gaze points annotated.
2. **GOOSynth**: A *synthetic* dataset for evaluation on the Gaze Object Detection task. Composed of images of scenes in a virtual marketplace environment, where the human's head, gaze point, and gazed object is annotated. 
3. **GOOReal**: A smaller, accompanying dataset for GOOSynth, composed of real-world images of humans in a marketplace environment, where the human's head, gaze point, and gazed object is annotated. Designed for domain adaptation of models trained on GooSynth from simulation to real-world applications.

## Baseline Evaluation

The following baselines are stable and found in the *master* branch:

1. A. Recasens, A. Khosla, C. Vondrick and A. Torralba. **"Where are they looking?"** 
2. Dongze Lian, Zehao Yu, Shenghua Gao. **"Believe It or Not, We Know What You Are Looking at!"**

The following baselines are works in progress

3. Chong, Eunji, et al. **"Detecting Attended Visual Targets in Video."** Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
    * implemented in the *chong* branch
4. GOOv1: Gazemask with Item Penalizing Loss (IPL) and Automatic Domain Randomization (ADR)
    * to be implemented in the *master* branch
5. Gaze Object Transformer
    * implemented in the *gazeobject_transformer* branch
