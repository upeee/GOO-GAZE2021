# Adding your own baselines

## Summary
The following functions/classes/modules must be implemented and put in the corresponding directories. In the future, we are looking to make the test() function model independent.
'''
from models.yourModel import GazeNet   
from dataloader.yourModel import GooDataset, GazeDataset
from training.train_yourModel import train, test, GazeOptimizer
'''

## Step-by-step
### 1. Define your Model in ./models/
- There are currently no requirements for defining the model as long as it is a pytorch module, the input matches your custom dataloader and the output is easily connected to your train() and test() functions. 

### 2. Define your Dataloader in ./dataloader/
- Should inherit the pytorch Dataset class.
- Additional details on how to parse the GOO dataset as well as the dataset proper will be released soon.

### 3. Define train() and test() in training/train_<>.py
'''
train(net, train_data_loader, optimizer, epoch, logger)
'''
The function should accept the above arguments and train a single epoch of your model. Use logger to save train and test errors in a .log file. Custom functions involved in training such as the model loss function should also be implemented in the same file. 

'''
test(net, test_data_loader,logger)
'''
The function should accept the above arguments and evaluate your model output on the following metrics minimum: **L2** and **Angular Error**. 

### 4. Define the class GazeOptimizer in training/train_<>.py
Optimizer initialization, as well as learning rate scheduling should be implemented under this class. Given the epoch, a class method should be able to return the optimizer value. This class is implemented outside of train() so optimizer values can be easily accessed and saved by only altering the main file. 
