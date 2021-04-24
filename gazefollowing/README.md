# Gazefollowing

This folder contains the code for implementing, training, and testing of models on the **gaze estimation** task. The following metrics are used to evaluate how accurate the model is on this task:

- **L2** : L2 norm'ed distance between predicted gaze point and the ground truth.
- **Angular Error** : Angle difference between the predicted gaze line from eyes to gaze point and the ground truth.
- **AUC** : Computed by getting the AUC of the 5x5 predicted heatmap of the gaze point and the ground truth.

# Gaze Object Detection

For future work, we hope to come up with novel metrics that are more suited to the task of Gaze Object Detection than the current metrics used in Gazefollowing.

## Installation 
For a minimal list of requirements:

* python >= 3.7.*
* pytorch >= 1.1
* torchvision >= 0.3.0
* cudatoolkit >= 9.0
* opencv, scipy, scikit-image, scikit-learn, tqdm

Exact versions of these packages used to run the evaluation is provided in [requirements.txt](./requirements.txt).

## Baselines

1. A. Recasens, A. Khosla, C. Vondrick and A. Torralba. **"Where are they looking?"** 
    * Select by setting --baseline='recasens'
2. Dongze Lian, Zehao Yu, Shenghua Gao. **"Believe It or Not, We Know What You Are Looking at!"**
    * Select by setting --baseline='gazenet'
3. Gazemask (ours) method can be enabled, keep in mind this does not include IPL or ADR as of this version.
    * Select by setting the flag --gazemask 
    
## Usage
Training on GOOSynth:
```
python main.py --baseline='gazenet' \
--train_dir='../goosynth/1person/GazeDatasets/' \
--train_annotation='../goosynth/picklefiles/trainpickle2to19human.pickle' \
--test_dir='../goosynth/test/' \
--test_annotation='../goosynth/picklefiles/testpickle120.pickle' \
--log_file='training.log' \
--save_model_dir='./saved_models/temp/' \
```

To continue training from previous run:
```
python main.py --baseline='gazenet' \
--train_dir='../goosynth/1person/GazeDatasets/' \
--train_annotation='../goosynth/picklefiles/trainpickle2to19human.pickle' \
--test_dir='../goosynth/test/' \
--test_annotation='../goosynth/picklefiles/testpickle120.pickle' \
--log_file='training.log' \
--save_model_dir='./saved_models/temp/' \
--resume_training
--resume_path='./saved_models/gazenet_goo/model_epoch25.pth.tar'\
```

To evaluate model on Gaze estimation and GOO metrics:
```
!python evaluate.py \
--test_dir='../goosynth/test/'\
--test_annotation='../goosynth/picklefiles/testpickle120.pickle'\
--resume_path='./saved_models/gazenet_goo/model_epoch25.pth.tar'\
```

For an example of the expected output of the scripts, you can check out the ipynb notebooks that were used for testing.

## Sample Predictions
Please refer to [modeltester.ipynb](https://github.com/upeee/GazeOnObjects/blob/master/gazefollowing/modeltester.ipynb), which shows how to predict on a set of images with a pretrained gazenet model. 

To see example outputs using a pretrained model, see images in [sample_out](https://github.com/upeee/GazeOnObjects/tree/master/gazefollowing/sample_out). Images are from the goosynth dataset, however sample outputs can also be produced from the GooReal and GazeFollow dataset by tweaking the modeltester.ipynb notebook. 

![sample_out/out_0.png](https://github.com/upeee/GazeOnObjects/blob/master/gazefollowing/sample_out/out_0.png)

## Contributing
See [CONTRIBUTING.md](https://github.com/upeee/GazeOnObjects/blob/master/gazefollowing/CONTRIBUTING.md) found in this directory.
