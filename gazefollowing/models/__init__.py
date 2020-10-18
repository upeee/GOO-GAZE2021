import torch
import torch.nn as nn
import torch.optim as optim
import utils
import os

def save_checkpoint(model, optimizer, epoch, save_path, modelIdx=-1):

    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
    }

    filename = "model_{}_adrIdx{}.pth.tar".format(epoch, modelIdx)

    if modelIdx < 0:
        filename = "model_epoch{}.pth.tar".format(epoch)
    else:
        filename = "model_epoch{}_adrIdx{}.pth.tar".format(epoch, modelIdx)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(state, save_path+filename)

def resume_checkpoint(model, optimizer, resume_path):

    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))

        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            print('=> Optimizer has different parameter groups. Usually this will occur for staged optimizers (GazeNet, GazeMask)')

        print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))

        return model, optimizer, checkpoint['epoch']
