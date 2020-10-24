# Adding your own baselines

## 1. Define your Model in ./models/
- There are currently no requirements for defining the model as long as it is a pytorch module, as long as the input matches your custom dataloader and the output is easily connected to your train() and test() functions. 

## 2. Define your Dataloader in ./dataloader/
- Should inherit the pytorch Dataset class.
- Additional details on how to parse this dataset as well as the dataset proper will be released soon.

## 3. Define your 
