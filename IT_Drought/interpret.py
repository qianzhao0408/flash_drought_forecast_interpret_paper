
import torch
import sys
# sys.path.append('..')
import torch.nn as nn
import argparse
import numpy as np
from util import AverageMeter
import matplotlib.pyplot as plt
import math
import mat73
from tqdm import tqdm
import os, time
import gc
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import nn, Tensor
import numpy as np
from typing import Callable, List, Tuple
from tqdm import tqdm

import dataset_3tasks0927maskfuturex as dataset_3tasks
from model import build_model
from util import set_seed, parse_years_from_index, gauss_legendre_builders, _reshape_and_sum

# nan_mask_local = mat73.loadmat("/fs/scratch/PAS2599/data_v0822_NLDAS_0125d/CONUS_daily/Drought_idx_NLDAS.mat")['Drought_idx']
# nan_mask_local = torch.tensor(nan_mask_local)
nan_mask_local = torch.load("/fs/ess/PAS2599/zhao4243/DL_input/Drought_idx_NLDASpt_v0115.pth").permute(0,1,3,2).flatten(start_dim=2) # 200，400，44，365

def create_windows(input_x, target, input_window=52, forecast_window=26, step1=0, step2=3, training = False, method = 'qian'):
    if(method == 'sequential'):
        
        num_steps = input_x.shape[0]
        inputs = []
        targets = []
        start_list = []
        end_list = []
        for start in range(0, num_steps - input_window - forecast_window + 1, step1):
            end = start + input_window
            input_window_data = input_x[start:end + forecast_window, :, :]
            target_window_data = target[end:end + forecast_window, :, :]
            inputs.append(input_window_data)
            targets.append(target_window_data)
            start_list.append(start)
            end_list.append(end + forecast_window)
        
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        start_list = torch.tensor(start_list).to(target.device)
        end_list = torch.tensor(end_list).to(target.device)
        return inputs, targets, start_list, end_list
    
    num_steps = input_x.shape[0]
    # 切分为两段数据
    mid_point = num_steps // 2
    
    # 第一段数据
    input_x_part1 = input_x[:mid_point, :, :]
    target_part1 = target[:mid_point, :, :]
    
    # 第二段数据
    input_x_part2 = input_x[mid_point:, :, :]
    target_part2 = target[mid_point:, :, :]
    
    def process_data(input_data, target_data):
        
        if(training):
            input_data = input_data[:-40]
        num_steps = input_data.shape[0]
        inputs = []
        targets = []
        start_list = []
        end_list = []
        
        current_step = step1  # 初始步长为 step1（即 0）
        start = 0  # 初始化起始位置为 0
        
        while start <= num_steps - input_window - forecast_window:
            end = start + input_window
            input_window_data = input_data[start:end+ forecast_window, :, :]
            target_window_data = target_data[end:end + forecast_window, :, :]
            
            inputs.append(input_window_data)
            targets.append(target_window_data)
            start_list.append(start)
            end_list.append(end + forecast_window)
            
            # 更新起始位置
            start += current_step + forecast_window + input_window
            
            # 交替步长
            if current_step == step1:
                current_step = step2  # 下次用 step2（3）
            else:
                current_step = step1  # 下次用 step1（0）

        # 堆叠所有窗口数据
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        start_list = torch.tensor(start_list).to(target.device)
        end_list = torch.tensor(end_list).to(target.device)
        
        return inputs, targets, start_list, end_list
    

    def process_random_data(input_data, target_data):
        num_steps = input_data.shape[0]
        window_size = input_window + forecast_window
        
        # 计算可以完全不重叠的窗口数量
        num_windows = num_steps // window_size
        
        # 创建可能的窗口起点
        possible_starts = np.arange(0, num_steps - window_size + 1, window_size)
        
        # 使用 numpy 随机选择 num_windows 个窗口起点
        random_starts = np.random.choice(possible_starts, size=num_windows, replace=False)
        
        inputs = []
        targets = []
        start_list = []
        end_list = []
        
        for start in random_starts:
            end = start + input_window
            input_window_data = input_data[start:end + forecast_window, :, :]
            target_window_data = target_data[end:end + forecast_window, :, :]
            
            inputs.append(input_window_data)
            targets.append(target_window_data)
            start_list.append(start)
            end_list.append(end + forecast_window)

        # 堆叠所有窗口数据
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        start_list = torch.tensor(start_list).to(target.device)
        end_list = torch.tensor(end_list).to(target.device)
        
        return inputs, targets, start_list, end_list
    if(method == 'random'):
        inputs_part1, targets_part1, start_list_part1, end_list_part1 = process_random_data(input_x_part1, target_part1)
        inputs_part2, targets_part2, start_list_part2, end_list_part2 = process_random_data(input_x_part2, target_part2)
    elif(method == 'qian'):
        inputs_part1, targets_part1, start_list_part1, end_list_part1 = process_data(input_x_part1, target_part1)
        inputs_part2, targets_part2, start_list_part2, end_list_part2 = process_data(input_x_part2, target_part2)
    else:
        raise NotImplementedError
    
    # 将两部分数据分别堆叠到一起
    inputs = torch.cat((inputs_part1, inputs_part2), dim=0)
    targets = torch.cat((targets_part1, targets_part2), dim=0)
    start_list_part2 = start_list_part2 + mid_point
    end_list_part2 = end_list_part2 + mid_point
    # print(start_list_part1, start_list_part2)
    start_list = torch.cat((start_list_part1, start_list_part2), dim=0)   
    end_list = torch.cat((end_list_part1, end_list_part2), dim=0)    

    return inputs, targets, start_list, end_list



def preprocess_agg(args,x_num_static_batch, x_num_time_batch, y_batch, mean_values, agg_distance, model, device):
    x_num_static_batch = x_num_static_batch.to(device)
    x_num_time_batch = x_num_time_batch.to(device)
    y_batch = y_batch.to(device)
    nan_mask_static = torch.isnan(x_num_static_batch)
    expanded_replacement_values = mean_values['static'].expand_as(x_num_static_batch)
    x_num_static_batch[nan_mask_static] = expanded_replacement_values[nan_mask_static]
    
    if(not args.agg):
        x_num_time_batch = x_num_time_batch.contiguous().view(x_num_time_batch.shape[0], x_num_time_batch.shape[1], -1)
        y_batch = y_batch.contiguous().view(y_batch.shape[0], y_batch.shape[1], -1).permute(2,0,1)
        x_num_time_batch = x_num_time_batch.permute(2,0,1)
        if(args.ablate_target_idx is None):
            x_num_time_batch = torch.cat((x_num_time_batch, y_batch), dim=2)
        else:
            ablate_indices = sorted(args.ablate_target_idx, reverse=True)
            y_batch_append = y_batch.clone()  # 克隆 y_batch 以避免修改原始张量
            for idx in ablate_indices:
                y_batch_append = torch.cat((y_batch_append[:, :, :idx], y_batch_append[:, :, idx+1:]), dim=2)
            
            # y_batch_append = torch.cat((y_batch[:, :, :args.ablate_target_idx], y_batch[:, :, args.ablate_target_idx+1:]), dim=2)
            x_num_time_batch = torch.cat((x_num_time_batch, y_batch_append), dim=2)
        current_x_num_static_batch =  x_num_static_batch   
        # x_num_time_batch = x_num_time_batch[:,:,:-args.unused_time]
        current_x_num_time_batch = x_num_time_batch
    else:
        x_num_time_batch = x_num_time_batch.contiguous().view(x_num_time_batch.shape[0], x_num_time_batch.shape[1],x_num_time_batch.shape[2], -1)
        y_batch = y_batch.contiguous().view(y_batch.shape[0], y_batch.shape[1], y_batch.shape[2], -1)
        if(args.ablate_target_idx is None):
            x_num_time_batch = torch.cat((x_num_time_batch, y_batch), dim=2)
        else:
            ablate_indices = sorted(args.ablate_target_idx, reverse=True)
            y_batch_append = y_batch.clone()  # 克隆 y_batch 以避免修改原始张量
            for idx in ablate_indices:
                y_batch_append = torch.cat((y_batch_append[:, :, :idx], y_batch_append[:, :, idx+1:]), dim=2)
            # y_batch_append = torch.cat((y_batch[:, :, :args.ablate_target_idx], y_batch[:, :, args.ablate_target_idx+1:]), dim=2)
            x_num_time_batch = torch.cat((x_num_time_batch, y_batch_append), dim=2)
        x_num_time_batch = x_num_time_batch
        current_x_num_static_batch = x_num_static_batch[:, 12, :].clone()
        current_x_num_time_batch = x_num_time_batch[:,12,:,:].clone().permute(2,0,1)

        y_batch = y_batch[:,12,:,:].permute(2,0,1)
    return current_x_num_static_batch,current_x_num_time_batch, y_batch, x_num_static_batch, x_num_time_batch
    
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # all_dataset,  all_loader, mean_values, nan_mask_list, year_index = dataset_3tasks.load0822_all(args)
    if('-' in args.split_set or 'all' in args.split_set ):
        args.split = args.split_set
        all_dataset,  all_loader, mean_values, nan_mask_list, year_index = dataset_3tasks.load0822_all(args)
    else:
    # train_dataset, test_dataset, train_loader, test_loader, mean_values, nan_mask_list, year_index  = dataset_3tasks.load0822(args)
        train_dataset, test_dataset, val_dataset, train_loader, test_loader, val_loader, mean_values, nan_mask_list, year_index = dataset_3tasks.load0822(args)
        nan_mask_list = nan_mask_list.to(device)
        if(args.split_set=='train'):
            all_dataset = train_dataset
            all_loader = train_loader
        elif(args.split_set == 'test'):
            all_dataset = test_dataset
            all_loader = test_loader
        elif(args.split_set == 'val'):
            all_dataset = val_dataset
            all_loader = val_loader
        
    
    
    labels = [name.split('_')[0] for name in all_dataset.input_time_names]
   
    y_labels = [all_dataset.drought_indices_names[i].split('_')[0] for i in range(len(all_dataset.drought_indices_names))]

    labels+=y_labels
    colors = [
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0',
        '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#800000'
    ]
    batch_size = args.bs
    for k,v in mean_values.items():
        mean_values[k] = v.to(device)
    args.forecast_window=args.single_time+args.unused_time-1 #zq edit
    model = build_model(args, device, num_tasks = len(all_dataset.drought_indices_names))

    if(not args.debug):
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint, strict=True) 
    
    n_steps = 20
    step_sizes_func, alphas_func = gauss_legendre_builders()
    step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)

    step_sizes_tensor = torch.tensor(step_sizes).float().view(n_steps, 1).to(device)
    if(not args.agg):
        alphas_tensor = torch.tensor(alphas).view(-1, 1, 1, 1).to(device).float()
        alphas_tensor_current = torch.tensor(alphas).view(-1, 1, 1, 1).to(device).float()
    else:
        alphas_tensor_current = torch.tensor(alphas).view(-1, 1, 1, 1).to(device).float()
        alphas_tensor = torch.tensor(alphas).view(-1, 1, 1, 1, 1).to(device).float()
        
        
    full_length = len(all_loader)

    def integrated_gradient(current_x_num_static_batch, current_x_num_time_batch, x_num_time, x_num_static, x_cata, model, agg_distance, start_list, end_list, window_idx, selected_skip_mask, index, futurex):
        # scaled_x_num_time =  torch.cat([alpha * x_num_time for alpha in alphas], dim=0 ).requires_grad_()  #拓展20次
        scaled_x_cata = torch.tile(x_cata, (n_steps,))
        if(not args.agg): # x_num_time torch.Size([572, 9, 14])      
            x_num_time = x_num_time.permute(1,2,0)[:,:,  start_list[window_idx]:args.input_window+start_list[window_idx]] #9,14,52
            
            if args.inter_var_idx is None:
                scaled_x_num_time = (alphas_tensor * x_num_time).reshape(-1, x_num_time.shape[1], x_num_time.shape[2]).requires_grad_()  #优化了五倍的时间
            else:
                scaled_x_num_time = (alphas_tensor * x_num_time).reshape(-1, x_num_time.shape[1], x_num_time.shape[2]) # 180，14，52
                left_part = scaled_x_num_time[:, :args.inter_var_idx, :]        
                target_part = scaled_x_num_time[:, args.inter_var_idx, :]      
                right_part = scaled_x_num_time[:, args.inter_var_idx+1:, :]    
                x_i = target_part.detach().clone().requires_grad_(True)
                x_i_3d = x_i.unsqueeze(1)
                scaled_x_num_time = torch.cat([left_part, x_i_3d, right_part], dim=1)

            scaled_x_static = (alphas_tensor * x_num_static).view(-1, x_num_static.shape[1]) #优化了五倍的时间
        else: #  x_num_time torch.Size([9, 25, 14, 572])   
            x_num_static_batch = x_num_static
            x_num_time_batch = x_num_time[:,:,:, start_list[window_idx]:args.input_window+start_list[window_idx]]
            current_x_num_time_batch = current_x_num_time_batch.permute(1,2,0)[:,:, start_list[window_idx]:args.input_window+start_list[window_idx]]
            
            scaled_distance =torch.tile(agg_distance, (n_steps,1,1))
            if args.inter_var_idx is None:
                scaled_current_x_num_time = (alphas_tensor_current * current_x_num_time_batch).reshape(-1, current_x_num_time_batch.shape[1], current_x_num_time_batch.shape[2]).requires_grad_()
            else:
                scaled_current_x_num_time = (alphas_tensor_current * current_x_num_time_batch).reshape(-1, current_x_num_time_batch.shape[1], current_x_num_time_batch.shape[2]) # 180，14，52
                left_part = scaled_current_x_num_time[:, :args.inter_var_idx, :]        
                target_part = scaled_current_x_num_time[:, args.inter_var_idx, :]      
                right_part = scaled_current_x_num_time[:, args.inter_var_idx+1:, :]    
                x_i = target_part.detach().clone().requires_grad_(True)
                x_i_3d = x_i.unsqueeze(1)
                scaled_current_x_num_time = torch.cat([left_part, x_i_3d, right_part], dim=1)
                
            # x_i = x_i_full[:, start_list[window_idx]:args.input_window+start_list[window_idx]]
            scaled_current_x_static = (alphas_tensor_current * current_x_num_static_batch).view(-1, current_x_num_static_batch.shape[1])
            
            scaled_x_static = (alphas_tensor * x_num_static_batch).view(-1, x_num_static_batch.shape[1], x_num_static_batch.shape[2])
            scaled_x_num_time = (alphas_tensor * x_num_time_batch).view(-1, x_num_time_batch.shape[1], x_num_time_batch.shape[2], x_num_time_batch.shape[3])
            
            weighted_time_features = model.agg(scaled_current_x_static, scaled_x_static.to(device), scaled_current_x_num_time, scaled_x_num_time.to(device) , distances = scaled_distance.flatten(start_dim=1).to(device))
            scaled_x_static = scaled_current_x_static
            # scaled_x_num_time = weighted_time_features[:,:, start_list[window_idx]:args.input_window+start_list[window_idx]]
            scaled_x_num_time = weighted_time_features
            # current_x_num_time_batch = current_x_num_time_batch[:,:, start_list[window_idx]:args.input_window+start_list[window_idx]]
            # print(current_x_num_time_batch.shape)
            
        # futurex_batch_test = torch.flatten(all_dataset.futurex[index], start_dim=2).to(device)[:,:,start_list[window_idx]+args.input_window:start_list[window_idx]+args.input_window+args.futurex_length ]
        futurex_batch_test = torch.flatten(futurex, start_dim=2,end_dim=3).to(device)[:,:,start_list[window_idx]+args.input_window]
        if(args.futurex_length!=0):
            
            if args.inter_var_idx_future is None or args.inter_var_idx_future==-1:
                scaled_futurex_batch_test  = (alphas_tensor_current * futurex_batch_test).reshape(-1, futurex_batch_test.shape[1], futurex_batch_test.shape[2]).requires_grad_()
            else:
                scaled_futurex_batch_test  = (alphas_tensor_current * futurex_batch_test).reshape(-1, futurex_batch_test.shape[1], futurex_batch_test.shape[2]) # 
                left_part_future = scaled_futurex_batch_test[:, :args.inter_var_idx_future, :]        
                target_part_future = scaled_futurex_batch_test[:, args.inter_var_idx_future, :]      
                right_part_future = scaled_futurex_batch_test[:, args.inter_var_idx_future+1:, :]    
                x_i_future = target_part_future.detach().clone().requires_grad_(True)
                x_i_future_3d = x_i_future.unsqueeze(1)
                scaled_futurex_batch_test = torch.cat([left_part_future, x_i_future_3d, right_part_future], dim=1)

            
        with torch.autograd.set_grad_enabled(True):
            dec_inp = torch.zeros_like(scaled_x_num_time[ :, : , -args.unused_time:]).float()
            if(args.futurex_length!=0):
                futurex_batch_test_time_window = scaled_futurex_batch_test
                
                for src_dim, tgt_dim in zip(args.futurex_dim_idx, args.decinp_dim_idx):
                    dec_inp[:, tgt_dim, :args.futurex_length] = futurex_batch_test_time_window[:, src_dim, :args.futurex_length]
                # dec_inp[:,  0:2,:args.futurex_length] = futurex_batch_test_time_window[:,0:2,:] # (unused_time, 32, 9) 
                # dec_inp[:,  3:8,:args.futurex_length] = futurex_batch_test_time_window[:,2:,:] 
            dec_inp = torch.cat([scaled_x_num_time, dec_inp], dim=2).float().to(device)
            if(args.use_decoder):
                # print(scaled_x_num_time.shape)
                inter_prediction = model(scaled_x_num_time.to(device), scaled_x_static.to(device), scaled_x_cata.to(device),tgt=dec_inp, local_batch_first=True)
                if(torch.isnan(inter_prediction[0]).sum()>0):
                    return None, None,  None
                    # print(torch.isnan(inter_prediction[0]).sum(),torch.isnan(futurex_batch_test[:,0]).sum(),torch.isnan(futurex_batch_test[:,1]).sum(),torch.isnan(futurex_batch_test[:,2]).sum())
                # inter_prediction = model(scaled_x_num_time, scaled_x_static, scaled_x_cata, local_batch_first=True)# 100, 14, 542
            else:
                inter_prediction = model(scaled_x_num_time, scaled_x_static, scaled_x_cata, local_batch_first=True)# 100, 14, 542

        # torch.unbind(forward_out) is a list of scalar tensor tuples and
        # contains batch_size * #steps elements
        results = [[] for _ in range(len(all_dataset.drought_indices_names))]
        futureresults = [[] for _ in range(len(all_dataset.drought_indices_names))]
        valid_times = [[] for _ in range(len(all_dataset.drought_indices_names))]
        for task_idx in range(len(all_dataset.drought_indices_names)):
            
            if(args.single_task_name is not None and all_dataset.drought_indices_names[task_idx] != args.single_task_name):
                continue
            results[task_idx] = []
            for target_idx in tqdm(range(args.inter_unused_time), disable=True):
            # for target_idx in (range(year_index[task_idx][0], year_index[task_idx][1])):
                current_time = start_list[window_idx] + args.input_window+target_idx
                ifskip = selected_skip_mask[:, current_time]
                if(sum(ifskip)==0):
                    continue
                if(args.futurex_length!=0 ):
                    if args.inter_var_idx is None:
                        grads = torch.autograd.grad(torch.unbind(inter_prediction[task_idx][:,target_idx]), (scaled_x_num_time, scaled_futurex_batch_test),retain_graph=True)
                    else:
                        if(args.inter_var_idx_future == -1):
                            grads = torch.autograd.grad(torch.unbind(inter_prediction[task_idx][:,target_idx]), (x_i),retain_graph=True,allow_unused=True)
                        else:
                            grads = torch.autograd.grad(torch.unbind(inter_prediction[task_idx][:,target_idx]), (x_i, x_i_future),retain_graph=True,allow_unused=True)
                        
                else:
                    if args.inter_var_idx is None:
                        grads = torch.autograd.grad(torch.unbind(inter_prediction[task_idx][:,target_idx]), (scaled_x_num_time),retain_graph=True)
                    else:
                        grads = torch.autograd.grad(torch.unbind(inter_prediction[task_idx][:,target_idx]), (x_i),retain_graph=True)
                        
                scaled_grads = [
                    grad.contiguous().view(n_steps, -1)
                    * step_sizes_tensor
                    for grad in grads
                ]
                total_grads = tuple(
                    _reshape_and_sum(
                        scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:]
                    )
                    for (scaled_grad, grad) in zip(scaled_grads, grads)
                )
                # grad * input
                if(not args.agg):
                    time_attributions = total_grads[0] * x_num_time if args.inter_var_idx is None else total_grads[0] * x_num_time[:, args.inter_var_idx, :]  
                else:
                    # print(total_grads[0][2])
                    time_attributions = total_grads[0] * current_x_num_time_batch if args.inter_var_idx is None else total_grads[0] * current_x_num_time_batch[:, args.inter_var_idx, :]  

                if(args.futurex_length!=0  ):
                    if args.inter_var_idx_future is None:
                        future_attributions = total_grads[1]*futurex_batch_test
                        futureresults[task_idx].append(future_attributions.detach().cpu())
                    elif args.inter_var_idx_future == -1:
                        future_attributions = None
                    else:
                        future_attributions = total_grads[1] * futurex_batch_test[:, args.inter_var_idx, :]  
                        futureresults[task_idx].append( future_attributions.unsqueeze(1).detach().cpu())
                    
                if args.inter_var_idx is not None:
                    time_attributions = time_attributions.unsqueeze(1)
                # results[task_idx].append( time_attributions.detach().cpu().numpy())
                results[task_idx].append( time_attributions.detach().cpu())
                
                # print(time_attributions.shape)
                valid_times[task_idx].append(current_time.item())
            if(len(results[task_idx]) != 0):
                results[task_idx] = torch.stack(results[task_idx])
                if(args.futurex_length!=0 and args.inter_var_idx_future!=-1 ):
                    futureresults[task_idx] = torch.stack(futureresults[task_idx])
                else:
                    futureresults[task_idx] = None
                # print(results[task_idx].shape) #torch.Size([4, 9, 9, 52])
                return results, valid_times[task_idx],futureresults
            else:
                return None, None,  None
        # if(args.single_task_name is not None):
        #    results = [ results[all_dataset.drought_indices_names.index(args.single_task_name)] ]
        
        # if(len(results[task_idx]) != 0):
        # results = torch.stack(results)
        # # results = np.array(results)
        # return results, valid_times

    batch_idx = -1
    model.eval()
    # 遍历数据集中的每一个像元
    for x_num_static_batch, x_cata_batch, x_num_time_batch, y_batch, location, index, agg_distance, original_index, nldas, futurex in tqdm(all_loader, disable=False):    
        
        rows, cols = location[:, 0].numpy(), location[:, 1].numpy()

        # 索引出指定位置的元素
        selected_skip_mask = nan_mask_local[rows, cols] # 16,060
        # print(nan_mask_local[location].shape)
        batch_idx+=1
        if(batch_idx<args.start_idx):
            continue
        if(batch_idx>args.end_idx):
            print('finish')
            break
        x_cata_batch = x_cata_batch.to(device)
        #预处理数据 拉平 插值
        current_x_num_static_batch,current_x_num_time_batch, y_batch, x_num_static_batch, x_num_time_batch = preprocess_agg(args,x_num_static_batch, x_num_time_batch, y_batch, mean_values, agg_distance, model, device)
        
        # print(nan_mask_local[location].shape)
        if(args.agg):
            f_distance = agg_distance.flatten(start_dim=1).to(device)
            f_distance[:,12] = args.dis
            weighted_time_features = model.agg(current_x_num_static_batch, x_num_static_batch.to(device), current_x_num_time_batch.permute(1,2,0), x_num_time_batch.to(device) , distances = f_distance)
            current_x_num_time_batch = weighted_time_features.permute(2,0,1)
        #     print("current_x_num_time_batch",current_x_num_time_batch.shape)
        # else:
        #     print("current_x_num_time_batch",current_x_num_time_batch.shape) # [572, 9, 14])
        
        # x_num_time_window_split, y_batch_window_split, start_list, end_list  = create_windows(current_x_num_time_batch, y_batch, input_window=args.input_window, forecast_window=args.unused_time,step=args.window_step) #([15, 100, 32, 14])
        # print('current_x_num_time_batch',current_x_num_time_batch.shape) torch.Size([9, 9, 52])
        
        x_num_time_window_split, y_batch_window_split, start_list, end_list  = create_windows(current_x_num_time_batch, y_batch, input_window=args.input_window, forecast_window=args.single_time+args.unused_time-1,step1=args.window_step0, step2=args.window_step1,  training=False, method = args.sample_method)  #([15, 100, 32, 14])
        
        batch_inter = {}
        batch_inter['location'] = location
        batch_inter['index'] = index
        inter_results = []
        futurex_inter_results = []
        valid_times_windows = []
        count = 0
        for window_idx in tqdm(range(len(x_num_time_window_split)), disable=args.no_progress):
            # 梯度解释
            # print(window_idx)
            inter_r,valid_times,future_attributions = integrated_gradient(current_x_num_static_batch, current_x_num_time_batch, x_num_time_batch, x_num_static_batch, x_cata_batch.to(device) , model, agg_distance, start_list, end_list, window_idx, selected_skip_mask, index, futurex)
            if(inter_r):
                inter_results.append(inter_r[0]) # torch.Size([4, 9, 9, 52])  限制了只能单任务
                futurex_inter_results.append(future_attributions[0]) # torch.Size([4, 9, 9, 52])  限制了只能单任务
                valid_times_windows.extend(valid_times)
                # print(inter_r[0].shape)
                # count+=inter_r[0].shape[0]
                # print(len(valid_times_windows))
                # print(count)
            # else:
            
            #     print("skip a window")
            #     # None
        if(len(inter_results)==0):
            print('All nan in a batch, skipped')
            continue
        batch_inter['valid_times_windows'] = valid_times_windows
        #调整解释结果的shape    
        # batch_inter['interresults'] = torch.stack(inter_results).permute(3,1,0,2,4,5) # 1000,1,windowssize,10,9,52
        # print(torch.cat(inter_results).shape)
        batch_inter['interresults'] = torch.cat(inter_results).permute(1,0,2,3)  # 1000*windowssize, 10, 9, 52 -> 10, 1000*
        if(args.inter_var_idx_future!=-1):
            batch_inter['futureresults'] = torch.cat(futurex_inter_results).permute(1,0,2,3)  # 1000*windowssize, 10, 9, 52 -> 10, 1000*
        # print(batch_inter['interresults'].shape)
        # print(batch_inter['futureresults'].shape)
        torch.save(batch_inter,os.path.join(args.save_path, f'{args.bs*batch_idx}_{len(all_dataset)}_{batch_idx}_{full_length}.pth'))
        gc.collect()

import shlex, argparse

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=10, help='batch_size')
    parser.add_argument('--num_layers', type=int, default=2, help='Transformer的层数， 1层则可以确保使用指定范围')
    parser.add_argument('--decoder_layers', type=int, default=1, help='Transformer的层数')
    parser.add_argument('--embedding_dim', type=int, default=2, help='类型数据的embeeding维度，默认4，调参得到')
    parser.add_argument('--dim_feedforward', type=int, default=128, help='Transformer前向传播使用的隐层维度，256显存要求低一些，越高fit越好')
    parser.add_argument('--mlp_input', type=int, default=8, help='静态数据MLP的输入的维度，多少个静态变量')
    parser.add_argument('--mlp_hidden', type=int, default=8, help='静态数据MLP的中间层')
    parser.add_argument('--mlp_dim', type=int, default=10, help='静态数据MLP的输出维度')
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--save_path', type=str, default="../results/inter_results/", help='')
    parser.add_argument('--data_path', type=str, default="/fs/ess/PAS2599/zhao4243/DroughtPrediction/data/data_v0822_NLDAS_0125d/CONUS_daily", help='处理后的文件位置')
    parser.add_argument('--model', type=str, default="window", help='使用的模型，精简到只提供一个')
    parser.add_argument('--d_model', type=int, default=9, help='the number of time-varing attributes, e.g. 11 (predictors) or 14 (11 predictors and 3 drought indices)')
    parser.add_argument('--d_model_expanded', type=int, default=32, help='expanded dimension of  time-varing attributes, expand d_model to d_model_expanded')
    parser.add_argument('--inner_att_heads', type=int, default=2, help='the number if attention heads in transformer')
    parser.add_argument('--no_progress', action='store_true', default=False)
    parser.add_argument('--agg', action='store_true', default=False, help='if use local aggeragation')
    parser.add_argument('--dropout', type=float, default=0.1, help='')
    
    parser.add_argument('--full_length', type=int, default=2288, help='time variables full length')
    
    parser.add_argument('--no_ncld', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=64*60)
    parser.add_argument('--ablate', type=str, nargs='+', default=None, help='需要消融的变量')
    parser.add_argument('--dis', type=float, default=0.8, help='')
    parser.add_argument('--no_encoder_mask', action='store_true', default=False)
    parser.add_argument('--use_decoder_mask', action='store_true', default=False)
    parser.add_argument('--input_window', type=int, default=52)
    parser.add_argument('--window_step0', type=int, default=0, help='')
    parser.add_argument('--window_step1', type=int, default=3, help='') 
    parser.add_argument('--unused_time', type=int, default=26, help='leave the last 30 time not to be trained')
    parser.add_argument('--inter_unused_time', type=int, default=1, help='解释预测窗口中的多少个时间点')
    parser.add_argument('--single_task_name', type=str, default=None, help='单任务的名字')
    
    parser.add_argument('--model_path', type=str, default='/users/PAS2353/tanxuwei99/code/project_climate/DroughtPrediction/simple/results/run_results/forecast_927_52_26/lr_5e-5_num_layers_2_decoder_layers1_dim_feedforward_128_sample_method_random_agg_seed_13/model_window_lr_5e-05_num_layers_2_tfdim_128_embedding_dim_2_mlp_dim_10_loss_mae_d_model_expanded_32_agg_True_single_task_name_None_seed_13/20241006-035033/model_epoch_20.pth')
    parser.add_argument('--split_set', type=str, default='train', help='train是训练集的时间划分，val是验证集，test是测试集，all是全体，不划分。')
    parser.add_argument('--single_time', type=int, default=1, help='')
    parser.add_argument('--sample_method', type=str, default='sequential', help='窗口方法:random or qian， 为了解释额外加入了一个sequential，sequential是旧的窗口方法')
    parser.add_argument('--mask_path', type=str, default="/fs/scratch/PAS2599/TrainMask_US_v0822_NLDAS.mat", help='') 
    parser.add_argument('--futurex_length', type=int, default=10, help='')
    parser.add_argument('--futurex_location', type=str, default="/fs/ess/PAS2599/zhao4243/DL_input/data_v0822_NLDAS_0125d/CONUS_daily_forecast_ERA5/data_pixel/")
    parser.add_argument('--decinp_dim_idx', type=str, default="013456789")
    parser.add_argument('--futurex_dim_idx', type=str, default="012345678")
    
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--inter_var_idx', type=int, default=None, help='解释单一变量，不指定是解释所有变量，指定是解释d-model（消融后的）中的第几个变量')
    parser.add_argument('--inter_var_idx_future', type=int, default=None, help='解释单一未来变量，不指定是解释所有未来变量，-1是跳过解释未来变量，0或1或2等是指定解释某个变量')

    args = parser.parse_args()
    args.ablate_target_idx = None
    args.use_decoder = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    args.decinp_dim_idx = list([int(char) for char in args.decinp_dim_idx])
    args.futurex_dim_idx = list([int(char) for char in args.futurex_dim_idx])
    print(args)
    
    set_seed(args)
    main(args)
    # sys.stdout.log.close()
    # sys.stdout = sys.stdout.terminal