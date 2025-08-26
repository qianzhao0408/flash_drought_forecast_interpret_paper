
import torch.nn as nn
import torch
import numpy as np

from typing import Callable, List, Tuple
from torch import nn, Tensor


# 固定随机种子 避免随机数干扰
def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# 文件名年转为索引
def parse_years_from_index(index):
    # Directly extracting years based on known structure
    parts = index.split('_')
    split_years = []
    for part in parts:
        # Checking if the part has a length of 9 (e.g., 20032011) which indicates year range
        if len(part) == 8 and part.isdigit():
            return int(part[:4]), int(part[4:])
        
        elif len(part) == 4 and part.isdigit():
            split_years.append(int(part[:4]))
             
    if(len(split_years)==2):
        return split_years[0], split_years[1]
    return "Error", "Error"  # Returning an error indication if no valid year range is found

# loss 统计用
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# IG 拆分input
def gauss_legendre_builders() -> Tuple[
    Callable[[int], List[float]], Callable[[int], List[float]]
]:
    def step_sizes(n: int) -> List[float]:
        assert n > 0, "The number of steps has to be larger than zero"
        # Scaling from 2 to 1
        return list(0.5 * np.polynomial.legendre.leggauss(n)[1])

    def alphas(n: int) -> List[float]:
        assert n > 0, "The number of steps has to be larger than zero"
        # Scaling from [-1, 1] to [0, 1]
        return list(0.5 * (1 + np.polynomial.legendre.leggauss(n)[0]))

    return step_sizes, alphas

def _reshape_and_sum(
    tensor_input: Tensor, num_steps: int, num_examples: int, layer_size: Tuple[int, ...]
) -> Tensor:
    return torch.sum(
        tensor_input.reshape((num_steps, num_examples) + layer_size), dim=0
    )
    

import sys
class DualWriter:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        # This flush method is needed for compatibility with the standard stdout interface.
        self.terminal.flush()
        self.log.flush()
       
       
# 创建窗口 
def create_windows(input_x, target, input_window=100, forecast_window=30, step=30):
    """
    Creates input and target windows from time series data.
    
    :param data: A tensor of shape (time_steps, batch_size, feature_dim)
    :param input_window: Number of time steps in the input window
    :param forecast_window: Number of time steps in the forecast window
    :param step: Step size for the sliding window
    :return: A tuple of input windows and target windows
    """
    num_steps = input_x.shape[0]
    inputs = []
    targets = []
    start_list = []
    end_list = []
    for start in range(0, num_steps - input_window - forecast_window + 1, step):
        end = start + input_window
        input_window_data = input_x[start:end, :, :]
        target_window_data = target[end:end + forecast_window, :, :]
        inputs.append(input_window_data)
        targets.append(target_window_data)
        start_list.append(start)
        end_list.append(end + forecast_window)

    # Stack all windows to create a new dimension for windows
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    # start_list = torch.stack(start_list)
    # end_list = torch.stack(end_list)

    return inputs, targets, start_list, end_list