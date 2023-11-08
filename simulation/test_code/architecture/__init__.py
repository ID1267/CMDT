import torch
from .DAUHST import DAUHST

def model_generator(method):
    if 'dstuf' in method:
        num_iter = int(method.split('_')[1][0])
        model = DAUHST()
    else:
        print(f'Method {method} is not defined !')
    return model
