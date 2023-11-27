import torch
from .CMDT import DAUHST

def model_generator(method):
    if 'CMDT' in method:
        num_iter = int(method.split('_')[1][0])
        model = DAUHST(num_iterations=num_iter)
    else:
        print(f'Method {method} is not defined !')
    return model
