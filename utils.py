import random
import numpy as np
import torch
import os
import torch.nn as nn
import copy
import config
from models.generator import Generator

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    # print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    # print("=> Loading checkpoint")
    model.load_state_dict(checkpoint)
    return model