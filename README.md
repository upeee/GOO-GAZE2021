# Gaze-on-Objects (GOO) Project
A repository featuring evaluation of state-of-the-art research papers on the task of **Gaze Estimation** (locating the specific point a human in an image is looking at) and the novel task of **Gaze Object Detection** (identifying the object in an image a human in the same image is looking at).

## Datasets

1. **GazeFollow**: A dataset for evaluation on the Gaze Estimation task. Composed of images of humans in different scenarios with their heads and gaze points annotated.
2. **GOOSynth**: A *synthetic* dataset for evaluation on the Gaze Object Detection task. Composed of images of scenes in a virtual marketplace environment, where the human's head, gaze point, and gazed object is annotated. 
3. **GOOReal**: A smaller, accompanying dataset for GOOSynth, composed of real-world images of humans in a marketplace environment, where the human's head, gaze point, and gazed object is annotated. Designed for domain adaptation of models trained on GooSynth from simulation to real-world applications.

## Baseline Evaluation