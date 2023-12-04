import torch
from .CMDT import DFUF

def model_generator(method):
    if 'CMDT' in method:
        num_iter = int(method.split('_')[1][0])
        model = DFUF(num_iterations=num_iter)
    else:
        print(f'Method {method} is not defined !')
    return model
