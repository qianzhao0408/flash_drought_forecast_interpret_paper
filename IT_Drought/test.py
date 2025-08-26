import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import torch.nn as nn
import argparse
import numpy as np
from util import AverageMeter
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import gc
from util import set_seed, gauss_legendre_builders, _reshape_and_sum, DualWriter 
import os, time, sys

import dataset_3tasks0927maskfuturex as dataset_3tasks
from model import build_model

def create_sequantial_windows(input_x, target, input_window=52, forecast_window=26, step=30, training = False):
    """
    Creates input and target windows from time series data.
    
    :param data: A tensor of shape (time_steps, batch_size, feature_dim)
    :param input_window: Number of time steps in the input window
    :param forecast_window: Number of time steps in the forecast window
    :param step: Step size for the sliding window
    :return: A tuple of input windows and target windows
    """
    
    if(training):
        input_data = input_data[:-40]
    num_steps = input_x.shape[0] # 104
    inputs = []
    targets = []
    start_list = []
    end_list = []
    # print(num_steps - input_window - forecast_window + 1)
    # print(step)
    for start in range(0, num_steps - input_window - forecast_window + 1, step):
        end = start + input_window
        input_window_data = input_x[start:end + forecast_window, :, :]
        target_window_data = target[end:end + forecast_window, :, :]
        inputs.append(input_window_data)
        targets.append(target_window_data)
        start_list.append(start)
        end_list.append(end + forecast_window)

    # Stack all windows to create a new dimension for windows
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    start_list = torch.tensor(start_list)
    end_list = torch.tensor(end_list)

    return inputs, targets, start_list, end_list

def create_windows(input_x, target, input_window=52, forecast_window=26, step1=0, step2=3, training = False, method = 'qian'):
    
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
            input_window_data = input_data[start:end + forecast_window, :, :]
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

    
    # inputs = []
    # targets = []
    # start_list = []
    # end_list = []
    # for start in range(0, num_steps - input_window - forecast_window + 1, step):
    #     end = start + input_window
    #     input_window_data = input_x[start:end, :, :]
    #     target_window_data = target[end:end + forecast_window, :, :]
    #     inputs.append(input_window_data)
    #     targets.append(target_window_data)
    #     start_list.append(start)
    #     end_list.append(end + forecast_window)

    # # Stack all windows to create a new dimension for windows
    # inputs = torch.stack(inputs)
    # targets = torch.stack(targets)

    return inputs, targets, start_list, end_list


def preprocess_agg(args,x_num_static_batch, x_num_time_batch, y_batch, mean_values, device):
    x_num_static_batch = x_num_static_batch.to(device)
    x_num_time_batch = x_num_time_batch.to(device)
    y_batch = y_batch.to(device)
    nan_mask_static = torch.isnan(x_num_static_batch)
    expanded_replacement_values = mean_values['static'].expand_as(x_num_static_batch)
    x_num_static_batch[nan_mask_static] = expanded_replacement_values[nan_mask_static]
    
    if(not args.agg):
        x_num_time_batch = x_num_time_batch.contiguous().view(x_num_time_batch.shape[0], x_num_time_batch.shape[1], -1)
        y_batch = y_batch.contiguous().view(y_batch.shape[0], y_batch.shape[1], -1).permute(2,0,1)
        # print(y_batch.shape) # torch.Size([572, 32, 3])
        x_num_time_batch = x_num_time_batch.permute(2,0,1)
        if(args.ablate_target_idx is None):
            # print(x_num_time_batch.shape)
            # print(y_batch.shape)
            x_num_time_batch = torch.cat((x_num_time_batch, y_batch), dim=2)
        else:
            ablate_indices = sorted(args.ablate_target_idx, reverse=True)
            y_batch_append = y_batch.clone()  # 克隆 y_batch 以避免修改原始张量
            for idx in ablate_indices:
                y_batch_append = torch.cat((y_batch_append[:, :, :idx], y_batch_append[:, :, idx+1:]), dim=2)
            
            x_num_time_batch = torch.cat((x_num_time_batch, y_batch_append), dim=2)
        current_x_num_static_batch =  x_num_static_batch   
        current_x_num_time_batch = x_num_time_batch
    else:
        x_num_time_batch = x_num_time_batch.contiguous().view(x_num_time_batch.shape[0], x_num_time_batch.shape[1],x_num_time_batch.shape[2], -1)
        y_batch = y_batch.contiguous().view(y_batch.shape[0], y_batch.shape[1], y_batch.shape[2], -1)
        # print(y_batch.shape)
        if(args.ablate_target_idx is None):
            x_num_time_batch = torch.cat((x_num_time_batch, y_batch), dim=2)
        else:
            ablate_indices = sorted(args.ablate_target_idx, reverse=True)
            y_batch_append = y_batch.clone()  # 克隆 y_batch 以避免修改原始张量
            for idx in ablate_indices:
                y_batch_append = torch.cat((y_batch_append[:, :, :idx], y_batch_append[:, :, idx+1:]), dim=2)
            x_num_time_batch = torch.cat((x_num_time_batch, y_batch_append), dim=2)
        x_num_time_batch = x_num_time_batch
        current_x_num_static_batch = x_num_static_batch[:, 12, :].clone()
        current_x_num_time_batch = x_num_time_batch[:,12,:,:].clone().permute(2,0,1)
        y_batch = y_batch[:,12,:,:].permute(2,0,1)
    return current_x_num_static_batch,current_x_num_time_batch, y_batch, x_num_static_batch, x_num_time_batch
    
def main(args):
    
    # Determine the device to be used for computation (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## Load the dataset
    all_dataset,  all_loader, mean_values, nan_mask_list, year_index = dataset_3tasks.load0822_all(args)
    # train_dataset, all_dataset, train_loader, all_loader, mean_values, nan_mask_list, year_index = dataset_3tasks.load0822(args)

    args.forecast_window=args.single_time+args.unused_time-1
    # Extract label names for plotting   
    labels = [name.split('_')[0] for name in all_dataset.input_time_names]
    # y_labels = ['ESI', 'SIF', 'SMsurface']
    y_labels = [all_dataset.drought_indices_names[i].split('_')[0] for i in range(len(all_dataset.drought_indices_names))]
    labels+=y_labels
    colors = [
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0',
        '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#800000'
    ]
    
    # 预载平均值
    for k,v in mean_values.items():
        mean_values[k] = v.to(device)
    # args.forecast_window=args.single_time+args.unused_time-1      
        
    # 加载模型
    model = build_model(args, device , num_tasks = len(all_dataset.drought_indices_names))
    if(not args.debug):
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint, strict=True) 
    
    if(args.loss=='mse'):
        loss_fn = nn.MSELoss(reduction='none')
    elif(args.loss=='mae'):
        loss_fn = nn.L1Loss(reduction='none')
        
    task_list = [_ for _ in range(len(all_dataset.drought_indices_names))]
    
    for epoch in range(1):

        
        ###########################################################################测试############################################################
        model.eval() # 模型选择evaluation状态，关闭dropout
        total_loss = 0
        test_losses = [AverageMeter() for _ in range(len(all_dataset.drought_indices_names))]
        test_all_losses = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(len(all_loader)))
        full_length = len(all_loader)
        
        output_target_pairs = [[] for i in range(len(all_dataset.drought_indices_names))]
        outputs_batch_list = []
        location_list = []
        index_list  = []
        for batch_test_idx, (x_num_static_batch, x_cata_batch, x_num_time_batch, y_batch, location, index, agg_distance, original_index, nldas, futurex) in enumerate(all_loader):
            
            if(batch_test_idx>3  and args.fasttest):
                break
            
            x_cata_batch = x_cata_batch.to(device)
            current_x_num_static_batch,current_x_num_time_batch_init, y_batch, x_num_static_batch, x_num_time_batch = preprocess_agg(args,x_num_static_batch, x_num_time_batch, y_batch, mean_values,  device)

            if(args.agg):
                f_distance = agg_distance.flatten(start_dim=1).to(device)
                f_distance[:,12] = args.dis
                weighted_time_features, att_weight = model.agg(current_x_num_static_batch, x_num_static_batch.to(device), current_x_num_time_batch_init.permute(1,2,0), x_num_time_batch.to(device) , distances = f_distance, show=True)
                current_x_num_time_batch = weighted_time_features.permute(2,0,1)
            else:
                current_x_num_time_batch = current_x_num_time_batch_init

                    
            # futurex_batch_test = torch.flatten(all_dataset.futurex[index], start_dim=2).to(device)
            futurex_batch_test = torch.flatten(futurex, start_dim=2,end_dim=3).to(device) # 32, 7, 44*365
            
            # x_num_time_window_split, y_batch_window_split, start_list, end_list  = create_windows(current_x_num_time_batch, y_batch, input_window=args.input_window, forecast_window=args.single_time+args.unused_time-1,step1=args.window_step0, step2=args.window_step1,  training=False, method = args.sample_method)  #([15, 100, 32, 14])
            # train_split = args.split=="train"
            if(args.sequantialwindows):
                
                x_num_time_window_split, y_batch_window_split, start_list, end_list  = create_sequantial_windows(current_x_num_time_batch, y_batch, input_window=args.input_window, forecast_window=args.single_time+args.unused_time-1,step=args.window_step0,  training=False)  #([15, 100, 32, 14])
                # print(x_num_time_window_split.shape) # torch.Size([287, 52, 512, 9]) torch.Size([1413, 413, 512, 11]) 365+48
            else:
                
                x_num_time_window_split, y_batch_window_split, start_list, end_list  = create_windows(current_x_num_time_batch, y_batch, input_window=args.input_window, forecast_window=args.single_time+args.unused_time-1,step1=args.window_step0, step2=args.window_step1,  training=False, method = args.sample_method)  #([15, 100, 32, 14])
            output_figs = [torch.full((x_cata_batch.shape[0],0), float('nan'))  for _ in task_list]
            target_figs = [torch.full((x_cata_batch.shape[0],0), float('nan'))  for _ in task_list]
            window_output_list = []
            all_batch_flag = True
            for window_idx in range(len(x_num_time_window_split)):
                x_num_time_window = x_num_time_window_split[window_idx][:-args.forecast_window,:,:]
                y_batch_window = y_batch_window_split[window_idx]
                
                with torch.no_grad():
                    
                    dec_inp = torch.zeros_like(current_x_num_time_batch[-args.unused_time:, :, :]).float()
                    # TODO
                    # dec_inp[-args.unused_time:, :, :-1] = torch.rand(dec_inp[-args.unused_time:, :, :-1].shape)
                    # dec_inp[:args.futurex_length, :, :-1] = x_num_time_window_split[window_idx][-args.forecast_window:-args.forecast_window+args.futurex_length,:,:-1]     
                    # futurex_batch_test_time_window = futurex_batch_test[:,:,start_list[window_idx]+args.input_window:start_list[window_idx]+args.input_window+10 ].permute(2,0,1) 
                    futurex_batch_test_time_window = futurex_batch_test[:,:,start_list[window_idx]+args.input_window,:].permute(2,0,1)
                    dec_inp[:args.futurex_length, :, 0] = futurex_batch_test_time_window[:,:,1] # (unused_time, 32, 9) 
                    dec_inp[:args.futurex_length, :, 1:2] = futurex_batch_test_time_window[:,:,4:5] 

                    # dec_inp[:args.futurex_length, :, 0] = futurex_batch_time_window[:,:,1] # (unused_time, 32, 9) 
                    # dec_inp[:args.futurex_length, :, 1:2] = futurex_batch_time_window[:,:,4:5] # (unused_time, 32, 9) 
                
                    #  --------- qian edit 0528 start
                    # dec_inp[:args.futurex_length, :, 0:2] = futurex_batch_test_time_window[:,:,0:2] # (unused_time, 32, 9) 
                    # dec_inp[:args.futurex_length, :, 3:10] = futurex_batch_test_time_window[:,:,2:] 
                    # for src_dim, tgt_dim in zip(args.futurex_dim_idx, args.decinp_dim_idx):
                    #     dec_inp[:args.futurex_length, :, tgt_dim] = futurex_batch_test_time_window[:, :, src_dim]
                    #  --------- qian edit 0528 end

                    dec_inp = torch.cat([x_num_time_window, dec_inp], dim=0).float().to(device)
                    x_cata_batch_need = x_cata_batch.to(device)
                    current_x_num_static_batch_need = current_x_num_static_batch.to(device)
                    x_num_time_window_need = x_num_time_window.to(device)
                    
                    # 跳过某些nan
                    if(torch.isnan(futurex_batch_test_time_window).any()):
                        nan_sample =  torch.isnan(futurex_batch_test_time_window.permute(1,0,2)).any(dim=(1, 2)).to(device)
                        valid_indices = (~nan_sample).nonzero(as_tuple=True)[0]
                        
                        x_num_time_window_need = x_num_time_window_need[:,valid_indices]
                        current_x_num_static_batch_need = current_x_num_static_batch_need[valid_indices]
                        x_cata_batch_need = x_cata_batch_need[valid_indices]
                        dec_inp = dec_inp[:,valid_indices]
                        y_batch_window = y_batch_window[:,valid_indices]
                        
                    
                    outputs = model(x_num_time_window_need.to(device), current_x_num_static_batch_need.to(device), x_cata_batch_need.to(device),tgt=dec_inp)
            
                    # outputs = model(x_num_time_window.to(device), current_x_num_static_batch.to(device), x_cata_batch.to(device),tgt=dec_inp)
                window_output_list.append([item.cpu() for item in outputs])
                all_task_loss = 0
                for task_id in task_list:
                    nan_mask_list_batch = nan_mask_list[task_id][original_index].to(device)
                    # 跳过某些nan
                    if(torch.isnan(futurex_batch_test_time_window).any()):
                        nan_mask_list_batch = nan_mask_list_batch[valid_indices]
                    
                    
                    if(args.single_task_name is not None and all_dataset.drought_indices_names[task_id] != args.single_task_name):
                        test_losses[task_id].update(0)
                        continue
                    
                    output_task = outputs[task_id]
                    if(len(output_task.shape)==3):
                        output_task = output_task.squeeze()
                    if(len(output_task.shape)==1):
                        output_task = output_task.unsqueeze(dim=-1)
                    
                    # task_nan_mask = nan_mask_list[task_id][original_index, start_list[window_idx]+args.input_window:end_list[window_idx]].to(device)
                    task_nan_mask = nan_mask_list_batch[:, start_list[window_idx]+args.input_window:end_list[window_idx]].to(device)
                    target_task =  y_batch_window[:,:, task_id].float().to(device).permute(1,0)
                    
                    non_nan_mask  = (~(task_nan_mask))[:,-args.unused_time:]
                    if(non_nan_mask.sum()==0):
                        continue
                    task_loss = loss_fn(output_task[:,-args.unused_time:][non_nan_mask], target_task[:,-args.unused_time:][non_nan_mask]).mean()
                    all_task_loss += task_loss
                    test_losses[task_id].update(task_loss.item())       
                if(all_task_loss==0):
                    continue
                test_all_losses.update(all_task_loss.item())    
                total_loss += all_task_loss
                if epoch % args.log_interval == 0 and len(output_target_pairs[task_id]) == 0:
                    for task_id in task_list:
                        if(len(outputs[task_id].shape)==3):
                            outputs[task_id] = outputs[task_id].squeeze()
                        if(len(outputs[task_id].shape)==1):
                            outputs[task_id] = outputs[task_id].unsqueeze(dim=-1)
                        # print(outputs[task_id].shape)
                        # print(y_batch_window.shape)
                        # print(y_batch_window[:,:, task_id].float().to(device).permute(1,0).shape)
                        if(len(output_figs[task_id])!=len(outputs[task_id])):
                            all_batch_flag = False
                            
                        output_figs[task_id] = torch.cat((output_figs[task_id], outputs[task_id][:,-args.unused_time:].cpu()), dim=1)
                        target_figs[task_id] = torch.cat((target_figs[task_id], y_batch_window[:,:, task_id].float().to(device).permute(1,0)[:,-args.unused_time:].cpu()), dim=1)
                
            if epoch % args.log_interval == 0 and len(output_target_pairs[task_id]) == 0 and all_batch_flag:
                for task_id in task_list:
                    item  = (output_figs[task_id], target_figs[task_id])
                    output_target_pairs[task_id].append(item)
            if not args.no_progress:

                description = "Test Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Loss: {loss:.4f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_test_idx + 1,
                    iter=len(all_loader),
                    loss=test_all_losses.avg
                )

                for i, test_loss_avg in enumerate(test_losses):
                    description += "{}: {:.4f}. ".format(all_dataset.drought_indices_names[i].split('_')[0], test_loss_avg.avg)

                p_bar.set_description(description)
                p_bar.update()
            location_list.append(location)
            index_list.append(index)
            outputs_batch_list.append(window_output_list)
            
                
        torch.save({'outputs_batch_list':outputs_batch_list,'location':location_list,'index':index_list } ,os.path.join(args.save_path, f'test_output.pth'))
            
        if not args.no_progress:
            p_bar.close()
                    
        # print(f"Test: Total Loss:{test_all_losses.avg}, ESI Loss:{test_losses[0].avg}, SIF Loss: {test_losses[1].avg}, SM Loss: {test_losses[2].avg} ")
        
        print(f"Test: Total Loss::{test_all_losses.avg}", end="")
        for i, test_loss_avg in enumerate(test_losses):
            print("{} Loss: {}. ".format(all_dataset.drought_indices_names[i].split('_')[0], test_loss_avg.avg), end="")   
        print()

        if epoch % args.log_interval == 0:
            model_filename = os.path.join(args.save_path, f'model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), model_filename )
                        
            fig, axes = plt.subplots(len(all_dataset.drought_indices_names), 1, figsize=(14, 25))  # 7 rows, 1 column

            # Loop over the output_target_pairs and plot each one in a subplot
            for task_id in task_list:
                random_pair = output_target_pairs[task_id][0]
                Prediction, True_Value = random_pair
                Prediction = Prediction[2]
                True_Value = True_Value[2]
                non_nan_mask  = (~np.isnan(True_Value))
                task_max_value = all_dataset.max_rec[all_dataset.drought_indices_names[task_id]]
                Prediction = np.squeeze(Prediction) * task_max_value.cpu().numpy()
                True_Value = True_Value* task_max_value.cpu().numpy()
                axes.plot(Prediction, label='Predictions', color='blue', alpha=0.7)
                axes.plot(True_Value, label='True Values', color='orange', alpha=1)  # Slightly transparent
                axes.set_xlabel('Time')
                axes.set_ylabel('Values')
                axes.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(args.save_path, f'vis_{epoch}.png'))
            plt.close()
    



        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, help='train rounds')
    parser.add_argument('--bs', type=int, default=32, help='batch_size')
    parser.add_argument('--num_layers', type=int, default=2, help='Transformer的层数')
    parser.add_argument('--decoder_layers', type=int, default=1, help='Transformer的层数')
    parser.add_argument('--embedding_dim', type=int, default=2, help='类型数据的embeeding维度，默认4，调参得到')
    parser.add_argument('--dim_feedforward', type=int, default=128, help='Transformer前向传播使用的隐层维度，256显存要求低一些，越高fit越好')
    parser.add_argument('--mlp_input', type=int, default=8, help='静态数据MLP的输入的维度，多少个静态变量')
    parser.add_argument('--mlp_hidden', type=int, default=8, help='静态数据MLP的中间层')
    parser.add_argument('--mlp_dim', type=int, default=10, help='静态数据MLP的输出维度')
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--save_path', type=str, default="debug/", help='')
    parser.add_argument('--data_path', type=str, default="/fs/ess/PAS2599/zhao4243/DroughtPrediction/data/data_v0822_NLDAS_0125d/CONUS_daily", help='处理后的文件位置')
    parser.add_argument('--loss', type=str, default="mae", help='使用的loss function： mae或者mse')
    parser.add_argument('--model', type=str, default="window", help='使用的模型，精简到只提供一个')
    parser.add_argument('--d_model', type=int, default=9, help='the number of time-varing attributes, e.g. 11 (predictors) or 14 (11 predictors and 3 drought indices)')
    parser.add_argument('--d_model_expanded', type=int, default=32, help='时间变量使用一层网络拓展一次，拓展的维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='')
    
    parser.add_argument('--inner_att_heads', type=int, default=2, help='the number if attention heads in transformer')
    parser.add_argument('--no_progress', action='store_true', default=False)
    parser.add_argument('--agg', action='store_true', default=False, help='if use local aggeragation')
    
    parser.add_argument('--full_length', type=int, default=2288, help='time variables full length')

    parser.add_argument('--no_ncld', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--ablate', type=str, nargs='+', default=None, help='需要消融的变量')
    parser.add_argument('--dis', type=float, default=0.8, help='自己对自己的距离')
    parser.add_argument('--no_encoder_mask', action='store_true', default=False)
    parser.add_argument('--use_decoder_mask', action='store_true', default=False,help='防止在预测未来t时刻时，使用t+1时刻的x，参数要一直开着')
    parser.add_argument('--input_window', type=int, default=52, help='')
    parser.add_argument('--window_step0', type=int, default=0, help='')
    parser.add_argument('--window_step1', type=int, default=3, help='')
    parser.add_argument('--unused_time', type=int, default=26, help='')
    parser.add_argument('--sample_ratio', type=float, default=1, help='')
    parser.add_argument('--inter', type=str, default='ig', help='')
    parser.add_argument('--fasttest', action='store_true', default=False, help='')
    parser.add_argument('--single_task_name', type=str, default=None, help='单任务的名字')
    parser.add_argument('--model_path', type=str, default='/users/PAS2353/tanxuwei99/code/project_climate/DroughtPrediction/simple/qianresults/forecast0927_SMroot_NLDAS_1979_2022_test/tf_layer1_maeloss_inputwindow_52_predictlen_26_windowstep_0_3_sampleratio_1_dim_48_model_expanded_16_lr_1e-4_nouse_decoder_mask_agg_0125d_ablate_LCSIF/model_window_lr_0.0001_num_layers_1_tfdim_48_embedding_dim_4_mlp_dim_16_loss_mae_d_model_expanded_16_agg_True_single_task_name_SMroot_1979_2022.mat_SMroot_seed_13/20241002-105207/model_epoch_6.pth')
    parser.add_argument('--single_time', type=int, default=1, help='')
    
    parser.add_argument('--lambda1', type=float, default=1, help='')
    parser.add_argument('--newloss', action='store_true', default=False)
    parser.add_argument('--early_stop', type=int, default=5, help='')
    parser.add_argument('--sample_method', type=str, default='qian', help='窗口方法:random or qian')
    parser.add_argument('--split', type=str, default='test', help='train, val, test, all')
    parser.add_argument('--sequantialwindows', action='store_true', default=False)
    parser.add_argument('--mask_path', type=str, default="/fs/scratch/PAS2599/TrainMask_US_v0822_NLDAS.mat", help='') 
    parser.add_argument('--futurex_length', type=int, default=10, help='')
    # parser.add_argument('--decinp_dim_idx', type=str, default="013456789", help='未来窗口decoder填充的变量维度索引，索引位置是ablate之后剩余所有变量在d-model中的位置，,limited by d_model parameter')
    # parser.add_argument('--futurex_dim_idx', type=str, default="012345678", help='未来的变量维度索引，将ablate之后的剩余变量一一索引到specific idx of futurex full 9 variables, limited by the range of 0-8')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    print(args.ablate)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    run_detail = f'testing_model_{args.model}_num_layers_{args.num_layers}_tfdim_{args.dim_feedforward}_embedding_dim_{args.embedding_dim}_mlp_dim_{args.mlp_dim}_loss_{args.loss}_d_model_expanded_{args.d_model_expanded}_agg_{args.agg}_single_task_name_{args.single_task_name}_seed_{args.seed}'
    current_time = time.strftime("%Y%m%d-%H%M%S")
    args.save_path = os.path.join(args.save_path, run_detail, current_time)
    print(args.save_path, flush=True)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        os.makedirs(os.path.join(args.save_path, 'interpretation'))
    
    log_file_path = os.path.join(args.save_path, "log.txt")
    sys.stdout = DualWriter(log_file_path)
   
    print(args)
    set_seed(args)
    main(args)
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal
