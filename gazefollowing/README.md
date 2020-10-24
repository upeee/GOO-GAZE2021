# Gazefollowing

This folder contains the code for implementing, training, and testing of models on the **gaze estimation** task. 

## Baselines

1. **Recasens et al**
2. **Gazenet**

## Usage
'''
!python main.py --baseline='gazenet' \
--train_dir='/hdd/HENRI/goosynth/1person/GazeDatasets/' \
--train_annotation='/hdd/HENRI/goosynth/picklefiles/trainpickle2to19human.pickle' \
--test_dir='/hdd/HENRI/goosynth/test/' \
--test_annotation='/hdd/HENRI/goosynth/picklefiles/testpickle120.pickle' \
--log_file='main_test.log' \
--save_model_dir='./saved_models/temp2/' \
'''