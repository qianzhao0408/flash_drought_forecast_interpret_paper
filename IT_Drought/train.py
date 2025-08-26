import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
import numpy as np
from util import AverageMeter
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from util import set_seed, gauss_legendre_builders, _reshape_and_sum, DualWriter 
import os, time, sys, gc
import dataset
from model import build_model

def create_windows(input_x, target, input_window=52, forecast_window=26, step1=0, step2=3, training = False, method = 'random'):
    
    num_steps = input_x.shape[0]
    mid_point = num_steps // 2
    
    input_x_part1 = input_x[:mid_point, :, :]
    target_part1 = target[:mid_point, :, :]
    
    input_x_part2 = input_x[mid_point:, :, :]
    target_part2 = target[mid_point:, :, :]
    
    def process_data(input_data, target_data): # delete?
        
        if(training):
            input_data = input_data[:-40]
        num_steps = input_data.shape[0]
        inputs = []
        targets = []
        start_list = []
        end_list = []
        
        current_step = step1 
        start = 0 
        
        while start <= num_steps - input_window - forecast_window:
            end = start + input_window
            input_window_data = input_data[start:end + forecast_window, :, :]
            target_window_data = target_data[end:end + forecast_window, :, :]
            
            inputs.append(input_window_data)
            targets.append(target_window_data)
            start_list.append(start)
            end_list.append(end + forecast_window)
            
            start += current_step + forecast_window + input_window
            
            if current_step == step1:
                current_step = step2 
            else:
                current_step = step1 

        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        start_list = torch.tensor(start_list).to(target.device)
        end_list = torch.tensor(end_list).to(target.device)
        
        return inputs, targets, start_list, end_list

    def process_random_data(input_data, target_data):
        num_steps = input_data.shape[0]
        window_size = input_window + forecast_window
        num_windows = num_steps // window_size
        possible_starts = np.arange(0, num_steps - window_size + 1, window_size)
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

        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        start_list = torch.tensor(start_list).to(target.device)
        end_list = torch.tensor(end_list).to(target.device)
        
        return inputs, targets, start_list, end_list
    
    if(method == 'random'):
        inputs_part1, targets_part1, start_list_part1, end_list_part1 = process_random_data(input_x_part1, target_part1)
        inputs_part2, targets_part2, start_list_part2, end_list_part2 = process_random_data(input_x_part2, target_part2)
    else:
        raise NotImplementedError
    
    inputs = torch.cat((inputs_part1, inputs_part2), dim=0)
    targets = torch.cat((targets_part1, targets_part2), dim=0)
    start_list_part2 = start_list_part2 + mid_point
    end_list_part2 = end_list_part2 + mid_point
    start_list = torch.cat((start_list_part1, start_list_part2), dim=0)   
    end_list = torch.cat((end_list_part1, end_list_part2), dim=0)    

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
        x_num_time_batch = x_num_time_batch.permute(2,0,1)
        if(args.ablate_target_idx is None):
            x_num_time_batch = torch.cat((x_num_time_batch, y_batch), dim=2)
        else:
            ablate_indices = sorted(args.ablate_target_idx, reverse=True)
            y_batch_append = y_batch.clone() 
            for idx in ablate_indices:
                y_batch_append = torch.cat((y_batch_append[:, :, :idx], y_batch_append[:, :, idx+1:]), dim=2)
            
            x_num_time_batch = torch.cat((x_num_time_batch, y_batch_append), dim=2)
        current_x_num_static_batch =  x_num_static_batch   
        current_x_num_time_batch = x_num_time_batch
    else:
        x_num_time_batch = x_num_time_batch.contiguous().view(x_num_time_batch.shape[0], x_num_time_batch.shape[1],x_num_time_batch.shape[2], -1)
        y_batch = y_batch.contiguous().view(y_batch.shape[0], y_batch.shape[1], y_batch.shape[2], -1)
        if(args.ablate_target_idx is None):
            x_num_time_batch = torch.cat((x_num_time_batch, y_batch), dim=2)
        else:
            ablate_indices = sorted(args.ablate_target_idx, reverse=True)
            y_batch_append = y_batch.clone() 
            for idx in ablate_indices:
                y_batch_append = torch.cat((y_batch_append[:, :, :idx], y_batch_append[:, :, idx+1:]), dim=2)
            x_num_time_batch = torch.cat((x_num_time_batch, y_batch_append), dim=2)
        x_num_time_batch = x_num_time_batch
        current_x_num_static_batch = x_num_static_batch[:, 12, :].clone()
        current_x_num_time_batch = x_num_time_batch[:,12,:,:].clone().permute(2,0,1)
        y_batch = y_batch[:,12,:,:].permute(2,0,1)
    return current_x_num_static_batch,current_x_num_time_batch, y_batch, x_num_static_batch, x_num_time_batch
    
    
def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset, val_dataset, train_loader, test_loader, val_loader, mean_values, nan_mask_list, year_index = dataset.load0822(args)
    
    args.forecast_window=args.single_time+args.unused_time-1
    nan_mask_list = nan_mask_list.to(device)
    labels = [name.split('_')[0] for name in train_dataset.input_time_names]
    futurexlabels = ['a',"b","b","b","b","b","b","b","b","b","b","b","b","b","b","b","b","b","b","b","b","b"]
    # y_labels = ['ESI', 'SIF', 'SMsurface']
    y_labels = [train_dataset.drought_indices_names[i].split('_')[0] for i in range(len(train_dataset.drought_indices_names))]
    labels+=y_labels
    colors = [
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0',
        '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#800000'
    ]
    
    # 预载平均值
    for k,v in mean_values.items():
        mean_values[k] = v.to(device)      
        
    # 加载模型
    model = build_model(args, device , num_tasks = len(train_dataset.drought_indices_names))
    # 加载优化器，设置loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if(args.loss=='mse'):
        loss_fn = nn.MSELoss(reduction='none')
    elif(args.loss=='mae'):
        loss_fn = nn.L1Loss(reduction='none')
        
    # IG的初始化，20个step
    n_steps = 20
    step_sizes_func, alphas_func = gauss_legendre_builders()
    step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)

    step_sizes_tensor = torch.tensor(step_sizes).float().view(n_steps, 1).to(device)
    if(not args.agg):
        alphas_tensor = torch.tensor(alphas).view(-1, 1, 1, 1).to(device).float()
    else:
        alphas_tensor_current = torch.tensor(alphas).view(-1, 1, 1, 1).to(device).float()
        alphas_tensor = torch.tensor(alphas).view(-1, 1, 1, 1, 1).to(device).float()

    best_val_loss= 10000
    patient = 0
    
    task_list = [_ for _ in range(len(train_dataset.drought_indices_names))]
    args.num_tasks = len(train_dataset.drought_indices_names)
    
    
    for epoch in range(args.epochs):
        # 加载数据集到loader， 每个epoch重新加一次，打乱顺序
        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)
        losses = [AverageMeter() for _ in range(len(train_dataset.drought_indices_names))]
        all_losses = AverageMeter()
        model.train()
        if not args.no_progress:
            p_bar = tqdm(range(len(train_loader)))
            
        ###########################################################################训练############################################################
        first_epoch_loss = [[] for _ in range(len(train_dataset.drought_indices_names)+1)]
        for batch_idx,(x_num_static_batch, x_cata_batch, x_num_time_batch, y_batch, location, index, agg_distance, original_index, nldas, futurex) in enumerate(train_loader):
            
            model.train()
            # 加载数据
            # print(x_num_time_batch.shape)
            batch_size = x_num_time_batch.shape[0]
            current_x_num_static_batch, current_x_num_time_batch, y_batch, x_num_static_batch, x_num_time_batch = preprocess_agg(args,x_num_static_batch, x_num_time_batch, y_batch, mean_values, device)
            nldas = nldas.to(device)
            nldas = torch.flatten(nldas,start_dim=1)
            # 划分时间窗口为x_num_time_window_split，y_batch_window_split， 划分后的维度为  ([num_time_wiondows, input_window_length, batch_size, feature_dimension])
            # start_list是窗口的起始位置，end_list是窗口的结束位置+预测时间
            # print(current_x_num_time_batch.shape)
            # print(y_batch.shape)
            futurex_batch = torch.flatten(futurex, start_dim=2,end_dim=3).to(device) # 32, 7, 44*365, 10  torch.Size([32, 7, 11680, 10])
            x_num_time_window_split, y_batch_window_split, start_list, end_list  = create_windows(current_x_num_time_batch, y_batch, input_window=args.input_window, forecast_window=args.single_time+args.unused_time-1,step1=args.window_step0,step2=args.window_step1,  training=True, method = args.sample_method) 
            # print(x_num_time_window_split.shape) #torch.Size([14, 100, 32, 8]) Now 14,126,32,8
            
            nan_mask_list_batch = nan_mask_list[:, original_index,:] # 1 32 2288
            nan_mask_list_batch_window_list = []
            nldas_batch_window_list = []
            if(args.agg):
                # 切割 x_num_time_batch
                # print(x_num_time_batch.shape, 'x_num_time_batch') #torch.Size([32, 25, 8, 1664]) 
                
                # 初始化存储切割后Tensor的列表
                x_num_time_batch_window_list = []
                # 循环遍历每个窗口，并根据start_list和end_list切割
                for start, end in zip(start_list, end_list):
                    sliced_tensor = x_num_time_batch[:, :, :, start:end-(args.single_time+args.unused_time-1)]  # 在最后一维度进行切片操作
                    x_num_time_batch_window_list.append(sliced_tensor)
                    nan_mask_list_batch_window_list.append(nan_mask_list_batch[:,: , start:end])
                    nldas_batch_window_list.append(nldas[:, start:end])

                # 将多个切割后的Tensor在第0维度堆叠起来，形成新的Tensor
                x_num_time_batch_window_list = torch.stack(x_num_time_batch_window_list, dim=0)
                
            else:
                for start, end in zip(start_list, end_list):
                    nan_mask_list_batch_window_list.append(nan_mask_list_batch[:,: , start:end])
                    nldas_batch_window_list.append(nldas[:, start:end])

            nan_mask_list_batch_window_list = torch.stack(nan_mask_list_batch_window_list, dim=0) # window 1 32 57
            nldas_batch_window_list = torch.stack(nldas_batch_window_list, dim=0) # window 32 57
                
            
            # 保证每个像元最少一个窗口，随机采样sample_ratio的窗口
            num_windows = max(int(len(x_num_time_window_split) * args.sample_ratio),1)
            # selected_indices = np.random.choice(len(x_num_time_window_split), num_windows, replace=False)
            
            selected_indices_tensor = torch.randint(0, len(x_num_time_window_split), (num_windows, batch_size), device=device)
            batch_indices = torch.arange(batch_size)
            
            #遍历随机窗口
            for window_idx in range(num_windows):
                all_task_loss = 0    
                
                # 取出当前 window_idx 对应的 batch 内的索引，形状为 (batch_size,)
                selected_indices = selected_indices_tensor[window_idx]
                
                if(args.agg): # 特征聚合, 
                    f_distance = agg_distance.flatten(start_dim=1).to(device)
                    f_distance[:,12] = args.dis
                    # print(x_num_time_window_split.shape) #torch.Size([20, 78, 32, 10])
                    current_x_num_time_batch_windows = x_num_time_window_split[selected_indices, :-args.forecast_window, batch_indices].permute(0,2,1)
                    # print(current_x_num_time_batch_windows.shape) # torch.Size([32, 10, 52])
                    x_num_time_batch_window = x_num_time_batch_window_list[selected_indices, batch_indices, :, :, :] #(32, 25, 8, 57)
                    weighted_time_features = model.agg(current_x_num_static_batch, x_num_static_batch.to(device), current_x_num_time_batch_windows, x_num_time_batch_window.to(device) , distances = f_distance)
                    # x_num_time_window = weighted_time_features.permute(2,0,1)[start_list[window_idx]:start_list[window_idx]+args.input_window ] #取出当前窗口
                    x_num_time_window = weighted_time_features.permute(2,0,1)
                else:
                    # x_num_time_window = x_num_time_window_split[window_idx] #取出当前窗口
                    x_num_time_window = x_num_time_window_split[selected_indices, :-args.forecast_window, batch_indices].permute(1,0,2) #取出当前窗口 (26, 52, 32, 8) -> (32, 52, 8) -> (52, 32, 8) 
                
                # y_batch_window = y_batch_window_split[window_idx] #取出当前窗口 torch.Size([26, 5, 32, 1])
                y_batch_window = y_batch_window_split[selected_indices, :, batch_indices].permute(1,0,2) #取出当前窗口 
                nan_mask_list_batch_window = nan_mask_list_batch_window_list[selected_indices, :, batch_indices, :]  #  26 1 32 57
                nldas_batch_window = nldas_batch_window_list[selected_indices, batch_indices, :]
                # 用于decoder的输入，预留好需要预测的时间的位置，用0填充
                dec_inp = torch.zeros_like(current_x_num_time_batch[-args.unused_time:, :, :]).float()  # (unused_time, 32, 9) 

                futurex_batch_time_window = futurex_batch[:,:, start_list[window_idx]+args.input_window,:].permute(2,0,1) # 32, 7, 10
                # futurex_list = ['tas_1979_2022_forecastx.mat', 'vpd_1979_2022_forecastx.mat',  'sp_1979_2022_forecastx.mat', 'vs_1979_2022_forecastx.mat', 'pr_1979_2022_forecastx.mat', 'pet_1979_2022_forecastx.mat', 'netsolar_1979_2022_forecastx.mat','netthermal_1979_2022_forecastx.mat','snow_1979_2022_forecastx.mat']
                
                # ['tas_1979_2022_forecastx.mat', 'vpd_1979_2022_forecastx.mat',  'sp_1979_2022_forecastx.mat', 'vs_1979_2022_forecastx.mat', 'pr_1979_2022_forecastx.mat', 'pet_1979_2022_forecastx.mat', 'srad_1979_2022_forecastx.mat']
                
                for src_dim, tgt_dim in zip(args.futurex_dim_idx, args.decinp_dim_idx):
                    dec_inp[:args.futurex_length, :, tgt_dim] = futurex_batch_time_window[:, :, src_dim]
                
                # dec_inp[:args.futurex_length, :, 0:2] = futurex_batch_time_window[:,:,0:2] # (unused_time, 32, 9) 
                # dec_inp[:args.futurex_length, :, 3:10] = futurex_batch_time_window[:,:,2:] 
                
                #  futurex_batch 32, 7, 44*365
                dec_inp = torch.cat([x_num_time_window, dec_inp], dim=0).float().to(device)
                
                x_cata_batch_need = x_cata_batch.to(device)
                current_x_num_static_batch_need = current_x_num_static_batch.to(device)
                x_num_time_window_need = x_num_time_window.to(device)
                
                # 跳过某些nan
                if(torch.isnan(futurex_batch_time_window).any()):
                    nan_sample =  torch.isnan(futurex_batch_time_window.permute(1,0,2)).any(dim=(1, 2)).to(device)
                    valid_indices = (~nan_sample).nonzero(as_tuple=True)[0]
                    x_num_time_window_need = x_num_time_window_need[:,valid_indices]
                    current_x_num_static_batch_need = current_x_num_static_batch_need[valid_indices]
                    x_cata_batch_need = x_cata_batch_need[valid_indices]
                    dec_inp = dec_inp[:,valid_indices]
                    y_batch_window = y_batch_window[:,valid_indices]
                    nan_mask_list_batch_window = nan_mask_list_batch_window[valid_indices]
                    nldas_batch_window = nldas_batch_window[valid_indices]
                    
                    # print("x_num_time_window",x_num_time_window.shape)
                    # print(current_x_num_static_batch.shape)
                    # print(x_cata_batch.shape)
                    # print(dec_inp.shape)
                    # print(y_batch_window.shape)
                    # print(nan_mask_list_batch_window.shape)
                    # print(nldas_batch_window.shape)
                    # print(output_task[:,-args.unused_time:])
                    
                #模型预测
                outputs = model(x_num_time_window_need, current_x_num_static_batch_need.to(device), x_cata_batch_need,tgt=dec_inp)
                    
                # 遍历每个任务
                for task_id in task_list:
                    #如果给出了单任务的索引，只训练单任务
                    if(args.single_task_name is not None and train_dataset.drought_indices_names[task_id] != args.single_task_name):
                        losses[task_id].update(0)
                        continue
                    # 取出单任务的预测
                    output_task = outputs[task_id]
                    # 调整维度
                    if(len(output_task.shape)==3):
                        output_task = output_task.squeeze()
                    if(len(output_task.shape)==1):
                        output_task = output_task.unsqueeze(dim=-1)
                    # print(output_task.shape)
                    # output_task = output_task[]
                    # 取出target中的nan， 不训练nan的target
                    # task_nan_mask = nan_mask_list[task_id][original_index, start_list[window_idx]+args.input_window:end_list[window_idx]].to(device)
                    task_nan_mask = nan_mask_list_batch_window[:, task_id, args.input_window:]
                    non_nan_mask  = (~task_nan_mask)[:,-args.unused_time:]
                    
                    # if((task_nan_mask).any()):
                    #     print(task_nan_mask)
                    
                    target_task =  y_batch_window[:,:, task_id].float().to(device).permute(1,0)
                    if(non_nan_mask.sum()==0): #如果全部是nan 跳过本次
                        continue
                    # nldas_current_window = torch.flatten(nldas,start_dim=1)[:, start_list[window_idx]+args.input_window:end_list[window_idx]]
                    nldas_current_window = nldas_batch_window[:, args.input_window:] # torch.Size([32, 57]) -> torch.Size([32, 5])
                    output_task_non_nan = output_task[:,-args.unused_time:][non_nan_mask]
                    target_task_non_nan = target_task[:,-args.unused_time:][non_nan_mask]
                    nldas_current_window_non_nan = nldas_current_window[:,-args.unused_time:][non_nan_mask]
                    mae_loss = loss_fn(output_task_non_nan, target_task_non_nan) # 计算loss，元素
                    # print(nldas_current_window_non_nan)
                    if(args.newloss):
                        local_loss = torch.where(nldas_current_window_non_nan == 0, 
                                mae_loss, 
                                mae_loss * args.lambda1 * (1 - target_task_non_nan))
                    else:
                        local_loss=mae_loss
                    local_loss = local_loss.mean()
                    # print(local_loss)


                    all_task_loss += local_loss
                    losses[task_id].update(local_loss.item())
                    
                if(all_task_loss==0):
                    continue
                # 梯度更新
                optimizer.zero_grad()
                all_task_loss.backward()
                all_losses.update(all_task_loss.item())
                optimizer.step()
                

            # 实时log   
            if not args.no_progress:
                description = "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Loss: {loss:.4f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=len(train_loader),
                    loss=all_losses.avg
                )

                for i, loss_avg in enumerate(losses):
                    description += "{}: {:.4f}. ".format(train_dataset.drought_indices_names[i].split('_')[0], loss_avg.avg)

                p_bar.set_description(description)
                p_bar.update()


            # 记录 loss    
            if(epoch==0):
                
                first_epoch_loss[0].append(all_losses.avg)
                for first_epoch_idx in range(len(train_dataset.drought_indices_names)):
                    first_epoch_loss[first_epoch_idx+1].append(losses[first_epoch_idx].avg)
            if(batch_idx>5 and args.fasttest):
                break

        if not args.no_progress:
            p_bar.close()
               
                
        print(f"Epoch [{epoch+1}/{args.epochs}],Iter: {batch_idx:4}/{len(train_loader)}.  Total Loss:{all_losses.avg}", end="")
        for i, loss_avg in enumerate(losses):
            print("{} Loss: {}. ".format(train_dataset.drought_indices_names[i].split('_')[0], loss_avg.avg), end="")   
        print()
        
        if(epoch==0):
            torch.save(first_epoch_loss,os.path.join(args.save_path, f'first_epoch_loss.pth'))
        
        
        ###########################################################################测试############################################################
        model.eval() # 模型选择evaluation状态，关闭dropout
        total_loss = 0
        test_losses = [AverageMeter() for _ in range(len(train_dataset.drought_indices_names))]
        test_all_losses = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(len(test_loader)))
        full_length = len(test_loader)
        
        output_target_pairs = [[] for i in range(len(train_dataset.drought_indices_names))]
        for batch_test_idx, (x_num_static_batch, x_cata_batch, x_num_time_batch, y_batch, location, index, agg_distance, original_index, nldas, futurex) in enumerate(test_loader):
            
            # if(batch_test_idx >= args.sample_ratio * len(test_loader) and batch_test_idx>100): # 如果只使用部分
            #     break
            if(batch_test_idx>3  and args.fasttest):
                break
            
            #同训练
            x_cata_batch = x_cata_batch.to(device)
            current_x_num_static_batch,current_x_num_time_batch_init, y_batch, x_num_static_batch, x_num_time_batch = preprocess_agg(args,x_num_static_batch, x_num_time_batch, y_batch, mean_values,  device)
            #同训练
            if(args.agg):
                f_distance = agg_distance.flatten(start_dim=1).to(device)
                f_distance[:,12] = args.dis
                weighted_time_features = model.agg(current_x_num_static_batch, x_num_static_batch.to(device), current_x_num_time_batch_init.permute(1,2,0), x_num_time_batch.to(device) , distances = f_distance)
                current_x_num_time_batch = weighted_time_features.permute(2,0,1)
            else:
                current_x_num_time_batch = current_x_num_time_batch_init

            
            futurex_batch_test = torch.flatten(futurex, start_dim=2,end_dim=3).to(device) # 32, 7, 44*365
                    
            x_num_time_window_split, y_batch_window_split, start_list, end_list  = create_windows(current_x_num_time_batch, y_batch, input_window=args.input_window, forecast_window=args.single_time+args.unused_time-1,step1=args.window_step0, step2=args.window_step1,  training=False, method = args.sample_method)  #([15, 100, 32, 14])
            output_figs = [torch.full((x_cata_batch.shape[0],0), float('nan'))  for _ in task_list]
            target_figs = [torch.full((x_cata_batch.shape[0],0), float('nan'))  for _ in task_list]
            all_batch_flag = True
            for window_idx in range(len(x_num_time_window_split)):
                x_num_time_window = x_num_time_window_split[window_idx][:-args.forecast_window,:,:]
                y_batch_window = y_batch_window_split[window_idx]
                with torch.no_grad():
                    
                    dec_inp = torch.zeros_like(current_x_num_time_batch[-args.unused_time:, :, :]).float()
                    # TODO
                    # dec_inp[-args.unused_time:, :, :-1] = torch.rand(dec_inp[-args.unused_time:, :, :-1].shape)
                    # dec_inp[:args.futurex_length, :, :-1] = x_num_time_window_split[window_idx][-args.forecast_window:-args.forecast_window+args.futurex_length,:,:-1]   
                    
                    # futurex_batch_test_time_window = futurex_batch_test[:,:,start_list[window_idx]+args.input_window:start_list[window_idx]+args.input_window+10 ].permute(2,0,1)  # []
                    futurex_batch_test_time_window = futurex_batch_test[:,:,start_list[window_idx]+args.input_window,:].permute(2,0,1)
                    # ['tas_1979_2022.mat_tas', 'vpd_1979_2022.mat_vpd', 'SMsurf_1979_2022.mat_SMsurf', 'sp_1979_2022.mat_surfacepressure', 'vs_1979_2022.mat_vs', 'pr_1979_2022.mat_pr', 'pet_1979_2022.mat_pet', 'srad_1979_2022.mat_srad']
                    # ['tas_1979_2022_forecastx.mat', 'vpd_1979_2022_forecastx.mat',  'sp_1979_2022_forecastx.mat', 'vs_1979_2022_forecastx.mat', 'pr_1979_2022_forecastx.mat', 'pet_1979_2022_forecastx.mat', 'srad_1979_2022_forecastx.mat']
                    
                    for src_dim, tgt_dim in zip(args.futurex_dim_idx, args.decinp_dim_idx):
                        dec_inp[:args.futurex_length, :, tgt_dim] = futurex_batch_time_window[:, :, src_dim]
                    # dec_inp[:args.futurex_length, :, 0:2] = futurex_batch_test_time_window[:,:,0:2] # (unused_time, 32, 9) 
                    # dec_inp[:args.futurex_length, :, 3:10] = futurex_batch_test_time_window[:,:,2:] 
                    
                    
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
            
                all_task_loss = 0
                for task_id in task_list:
                    nan_mask_list_batch = nan_mask_list[task_id][original_index].to(device)
                    # 跳过某些nan
                    if(torch.isnan(futurex_batch_test_time_window).any()):
                        nan_mask_list_batch = nan_mask_list_batch[valid_indices]
                    
                    
                    if(args.single_task_name is not None and train_dataset.drought_indices_names[task_id] != args.single_task_name):
                        test_losses[task_id].update(0)
                        continue
                    
                    output_task = outputs[task_id]
                    if(len(output_task.shape)==3):
                        output_task = output_task.squeeze()
                    if(len(output_task.shape)==1):
                        output_task = output_task.unsqueeze(dim=-1)
                    
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
                        else:
                            
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
                    iter=len(train_loader),
                    loss=test_all_losses.avg
                )

                for i, test_loss_avg in enumerate(test_losses):
                    description += "{}: {:.4f}. ".format(train_dataset.drought_indices_names[i].split('_')[0], test_loss_avg.avg)

                p_bar.set_description(description)
                p_bar.update()

                

            # 解释变量 用于临时画图    x_num_static_batch, x_cata_batch, x_num_time_batch, y_batch
            if(batch_test_idx==283 or (( len(test_loader) <=283) and batch_test_idx==0)  or (args.fasttest and  batch_test_idx==0)):
                model.eval()
                
                agg_distance = agg_distance[:5]
                
                futurex_batch_test = torch.flatten(futurex, start_dim=2, end_dim=3).to(device)[:5,:,start_list[-1]+args.input_window]
                
                if(args.futurex_length!=0):
                    if(not args.agg):
                        scaled_futurex_batch_test  = (alphas_tensor * futurex_batch_test).reshape(-1, futurex_batch_test.shape[1], futurex_batch_test.shape[2]).requires_grad_()
                    else:
                        scaled_futurex_batch_test  = (alphas_tensor_current * futurex_batch_test).reshape(-1, futurex_batch_test.shape[1], futurex_batch_test.shape[2]).requires_grad_()
                
                if(not args.agg):
                    x_num_time_window = x_num_time_window_split[-1] 
                    y_batch_window = y_batch_window_split[-1]
                    x_num_time_batch = x_num_time_window[:,:5].permute(1,2,0)[:,:,:-args.forecast_window]
                    x_cata_batch = x_cata_batch[:5]
                    scaled_x_cata = torch.tile(x_cata_batch, (n_steps,))
                    x_num_static_batch = x_num_static_batch[:5]
                    scaled_x_num_time = (alphas_tensor * x_num_time_batch).reshape(-1, x_num_time_batch.shape[1], x_num_time_batch.shape[2]).requires_grad_()  #
                    scaled_x_static = (alphas_tensor * x_num_static_batch).view(-1, x_num_static_batch.shape[1]).requires_grad_() 
                else:

                    x_num_time_window = x_num_time_window_split[-1] 
                    y_batch_window = y_batch_window_split[-1]
                    x_cata_batch = x_cata_batch[:5]
                    scaled_x_cata = torch.tile(x_cata_batch, (n_steps,))
                    x_num_static_batch = x_num_static_batch[:5]
                    current_x_num_time_batch = current_x_num_time_batch_init[:,:5].permute(1,2,0)
                    current_x_num_static_batch = current_x_num_static_batch[:5]

                    x_num_time_batch = x_num_time_batch[:5]
                    scaled_distance =torch.tile(agg_distance, (n_steps,1,1))
                    scaled_current_x_num_time = (alphas_tensor_current * current_x_num_time_batch).reshape(-1, current_x_num_time_batch.shape[1], current_x_num_time_batch.shape[2]).requires_grad_()
                    scaled_current_x_static = (alphas_tensor_current * current_x_num_static_batch).view(-1, current_x_num_static_batch.shape[1])
                    
                    scaled_x_static = (alphas_tensor * x_num_static_batch).view(-1, x_num_static_batch.shape[1], x_num_static_batch.shape[2])
                    scaled_x_num_time = (alphas_tensor * x_num_time_batch).view(-1, x_num_time_batch.shape[1], x_num_time_batch.shape[2], x_num_time_batch.shape[3])
                    weighted_time_features = model.agg(scaled_current_x_static, scaled_x_static.to(device), scaled_current_x_num_time, scaled_x_num_time.to(device) , distances = scaled_distance.flatten(start_dim=1).to(device))
                    scaled_x_static = scaled_current_x_static
                    scaled_x_num_time = weighted_time_features[:,:, start_list[window_idx]:args.input_window+start_list[window_idx]]
                    current_x_num_time_batch = current_x_num_time_batch[:,:, start_list[window_idx]:args.input_window+start_list[window_idx]]
                
                with torch.autograd.set_grad_enabled(True):
                    dec_inp = torch.zeros_like(scaled_x_num_time[ :, : , -args.unused_time:]).float()
                    if(args.futurex_length!=0):
                        futurex_batch_test_time_window = scaled_futurex_batch_test
                        
                        for src_dim, tgt_dim in zip(args.futurex_dim_idx, args.decinp_dim_idx):
                            dec_inp[:, tgt_dim, :args.futurex_length ] = futurex_batch_test_time_window[:, src_dim,: ]

                    dec_inp = torch.cat([scaled_x_num_time, dec_inp], dim=2).float().to(device)
                    
                    inter_prediction = model(scaled_x_num_time.to(device), scaled_x_static.to(device), scaled_x_cata.to(device),tgt=dec_inp, local_batch_first=True)

                inter_results = {}
                for task_idx in range(len(train_dataset.drought_indices_names)):
                    inter_results[task_idx] = {'time':[],'static':[],'future':[]}
                    for target_idx in tqdm(range(args.unused_time), disable=args.no_progress):
                        if(args.futurex_length!=0):
                            grads = torch.autograd.grad(torch.unbind(inter_prediction[task_idx][:,target_idx]), (scaled_x_num_time, scaled_futurex_batch_test),retain_graph=True)
                        else:
                            grads = torch.autograd.grad(torch.unbind(inter_prediction[task_idx][:,target_idx]), (scaled_x_num_time),retain_graph=True)
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
                        if(not args.agg):
                            time_attributions = total_grads[0] * x_num_time_batch
                        else:
                            time_attributions = total_grads[0] * current_x_num_time_batch
                        if(args.futurex_length!=0):
                            future_attributions = total_grads[1]*futurex_batch_test
                            inter_results[task_idx]['future'].append( future_attributions.detach().cpu().numpy())
                        
                        inter_results[task_idx]['time'].append( time_attributions.detach().cpu().numpy())
                inter_results['location'] = location
                inter_results['original_index'] = original_index
                inter_results['index'] = index
                os.makedirs(os.path.join(args.save_path, 'interpretation',str(epoch)))
                torch.save(inter_results,os.path.join(args.save_path, f'interpretation/{epoch}/{batch_test_idx}_{full_length}.pth'))
                model.eval()

                rightlist = [0]
                for right in rightlist:
                    fig, axes = plt.subplots(len(train_dataset.drought_indices_names)+1, 1, figsize=(14, 25)) 
                    for task_id in range(len(train_dataset.drought_indices_names)):
                        all_attri = np.array(inter_results[task_id]['time'])
                        local_attri = all_attri[right][0]
                        for idx in range(local_attri.shape[0]) :
                            axes[task_id].plot( local_attri[idx][0:],label = labels[idx].split('_')[0], color=colors[idx])
                        axes[task_id].set_xlabel(y_labels[task_id])
                        axes[task_id].set_ylabel('Attribution Value')
                        axes[task_id].legend(fontsize='large')
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.save_path, f'interpretation/{epoch}/{right}_right_{batch_test_idx}_{full_length}.png'))
                    plt.close()
                
                    
                    fig, axes = plt.subplots(len(train_dataset.drought_indices_names)+1, 1, figsize=(14, 25)) 
                    for task_id in range(len(train_dataset.drought_indices_names)):
                        all_attri = np.array(inter_results[task_id]['time'])
                        local_attri = all_attri[right][0]
                        for idx in range(local_attri.shape[0]) :
                            axes[task_id].plot( local_attri[idx][-30:],label = labels[idx].split('_')[0], color=colors[idx])
                        axes[task_id].set_xlabel(y_labels[task_id])
                        axes[task_id].set_ylabel('Attribution Value')
                        axes[task_id].legend(fontsize='large')
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.save_path, f'interpretation/{epoch}/short_{right}_right_{batch_test_idx}_{full_length}.png'))
                    plt.close()
                    
                    if(args.futurex_length!=0):
                        fig, axes = plt.subplots(len(train_dataset.drought_indices_names)+1, 1, figsize=(14, 25)) 
                        for task_id in range(len(train_dataset.drought_indices_names)):
                            all_attri = np.array(inter_results[task_id]['future'])
                            local_attri = all_attri[right][0]
                            for idx in range(local_attri.shape[0]) :
                                axes[task_id].plot( local_attri[idx],label = futurexlabels[idx].split('_')[0], color=colors[idx])
                            axes[task_id].set_xlabel(y_labels[task_id])
                            axes[task_id].set_ylabel('future Attribution Value')
                            axes[task_id].legend(fontsize='large')
                        plt.tight_layout()
                        plt.savefig(os.path.join(args.save_path, f'interpretation/{epoch}/future_{right}_right_{batch_test_idx}_{full_length}.png'))
                        plt.close()
            
            
        if not args.no_progress:
            p_bar.close()
                    
        print(f"Test: Total Loss::{test_all_losses.avg}", end="")
        for i, test_loss_avg in enumerate(test_losses):
            print("{} Loss: {}. ".format(train_dataset.drought_indices_names[i].split('_')[0], test_loss_avg.avg), end="")   
        print()

        if epoch % args.log_interval == 0:
            model_filename = os.path.join(args.save_path, f'model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), model_filename )
                        
            fig, axes = plt.subplots(len(train_dataset.drought_indices_names)+1, 1, figsize=(14, 25)) 

            for task_id in task_list:
                random_pair = output_target_pairs[task_id][0]
                Prediction, True_Value = random_pair
                Prediction = Prediction[2]
                True_Value = True_Value[2]
                non_nan_mask  = (~np.isnan(True_Value))
                task_max_value = train_dataset.max_rec[train_dataset.drought_indices_names[task_id]]
                Prediction = np.squeeze(Prediction) * task_max_value.cpu().numpy()
                True_Value = True_Value* task_max_value.cpu().numpy()
                axes[task_id].plot(Prediction, label='Predictions', color='blue', alpha=0.7)
                axes[task_id].plot(True_Value, label='True Values', color='orange', alpha=1)  
                axes[task_id].set_xlabel('Time')
                axes[task_id].set_ylabel('Values')
                axes[task_id].legend()

            plt.tight_layout()
            plt.savefig(os.path.join(args.save_path, f'vis_{epoch}.png'))
            plt.close()
    
        ##################################Validation#######################################
        total_loss = 0
        val_losses = [AverageMeter() for _ in range(len(train_dataset.drought_indices_names))]
        val_all_losses = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(len(val_loader)))
        full_length = len(val_loader)
        
        output_target_pairs = [[] for i in range(len(train_dataset.drought_indices_names))]
        for batch_val_idx, (x_num_static_batch, x_cata_batch, x_num_time_batch, y_batch, location, index, agg_distance, original_index, nldas, futurex) in enumerate(val_loader):

            x_cata_batch = x_cata_batch.to(device)
            current_x_num_static_batch,current_x_num_time_batch_init, y_batch, x_num_static_batch, x_num_time_batch = preprocess_agg(args,x_num_static_batch, x_num_time_batch, y_batch, mean_values,  device)
            if(args.agg):
                f_distance = agg_distance.flatten(start_dim=1).to(device)
                f_distance[:,12] = args.dis
                weighted_time_features = model.agg(current_x_num_static_batch, x_num_static_batch.to(device), current_x_num_time_batch_init.permute(1,2,0), x_num_time_batch.to(device) , distances = f_distance)
                current_x_num_time_batch = weighted_time_features.permute(2,0,1)
            else:
                current_x_num_time_batch = current_x_num_time_batch_init

            futurex_batch_val = torch.flatten(futurex, start_dim=2, end_dim=3).to(device)   
            x_num_time_window_split, y_batch_window_split, start_list, end_list  = create_windows(current_x_num_time_batch, y_batch, input_window=args.input_window, forecast_window=args.single_time+args.unused_time-1,step1=args.window_step0, step2=args.window_step1,  training=False, method = args.sample_method)
            output_figs = [torch.full((x_cata_batch.shape[0],0), float('nan'))  for _ in task_list]
            target_figs = [torch.full((x_cata_batch.shape[0],0), float('nan'))  for _ in task_list]
            
            for window_idx in range(len(x_num_time_window_split)):
                x_num_time_window = x_num_time_window_split[window_idx][:-args.forecast_window,:,:]
                y_batch_window = y_batch_window_split[window_idx]
                with torch.no_grad():
                    dec_inp = torch.zeros_like(current_x_num_time_batch[-args.unused_time:, :, :]).float()
                    futurex_batch_val_time_window = futurex_batch_val[:,:,start_list[window_idx]+args.input_window].permute(2,0,1)
                    for src_dim, tgt_dim in zip(args.futurex_dim_idx, args.decinp_dim_idx):
                        dec_inp[:args.futurex_length, :, tgt_dim] = futurex_batch_val_time_window[:, :, src_dim]
                    
                    dec_inp = torch.cat([x_num_time_window, dec_inp], dim=0).float().to(device)
                    
                    
                    x_cata_batch_need = x_cata_batch.to(device)
                    current_x_num_static_batch_need = current_x_num_static_batch.to(device)
                    x_num_time_window_need = x_num_time_window.to(device)
                    
                    if(torch.isnan(futurex_batch_val_time_window).any()):
                        nan_sample =  torch.isnan(futurex_batch_val_time_window.permute(1,0,2)).any(dim=(1, 2)).to(device)
                        valid_indices = (~nan_sample).nonzero(as_tuple=True)[0]
                        
                        x_num_time_window_need = x_num_time_window_need[:,valid_indices]
                        current_x_num_static_batch_need = current_x_num_static_batch_need[valid_indices]
                        x_cata_batch_need = x_cata_batch_need[valid_indices]
                        dec_inp = dec_inp[:,valid_indices]
                        y_batch_window = y_batch_window[:,valid_indices]

                    outputs = model(x_num_time_window_need.to(device), current_x_num_static_batch_need.to(device), x_cata_batch_need.to(device),tgt=dec_inp)
                all_task_loss = 0
                for task_id in task_list:
                    
                    nan_mask_list_batch = nan_mask_list[task_id][original_index].to(device)

                    if(torch.isnan(futurex_batch_val_time_window).any()):
                        nan_mask_list_batch = nan_mask_list_batch[valid_indices]
                        
                    if(args.single_task_name is not None and train_dataset.drought_indices_names[task_id] != args.single_task_name):
                        val_losses[task_id].update(0)
                        continue
                    output_task = outputs[task_id]
                    if(len(output_task.shape)==3):
                        output_task = output_task.squeeze()
                    if(len(output_task.shape)==1):
                        output_task = output_task.unsqueeze(dim=-1)
                    task_nan_mask = nan_mask_list_batch[:, start_list[window_idx]+args.input_window:end_list[window_idx]].to(device)
                    target_task =  y_batch_window[:,:, task_id].float().to(device).permute(1,0)
                    
                    non_nan_mask  = (~(task_nan_mask))[:,-args.unused_time:]
                    if(non_nan_mask.sum()==0):
                        continue
                    task_loss = loss_fn(output_task[:,-args.unused_time:][non_nan_mask], target_task[:,-args.unused_time:][non_nan_mask]).mean()
                    all_task_loss += task_loss
                    val_losses[task_id].update(task_loss.item())       
                if(all_task_loss==0):
                    continue
                val_all_losses.update(all_task_loss.item())    
                total_loss += all_task_loss

            if not args.no_progress:
                description = "val Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Loss: {loss:.4f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_val_idx + 1,
                    iter=len(train_loader),
                    loss=val_all_losses.avg
                )
                for i, val_loss_avg in enumerate(val_losses):
                    description += "{}: {:.4f}. ".format(train_dataset.drought_indices_names[i].split('_')[0], val_loss_avg.avg)
                p_bar.set_description(description)
                p_bar.update()
        if not args.no_progress:
            p_bar.close()

        print(f"val: Total Loss::{val_all_losses.avg}", end="")
        for i, val_loss_avg in enumerate(val_losses):
            print("{} Loss: {}. ".format(train_dataset.drought_indices_names[i].split('_')[0], val_loss_avg.avg), end="")   
        print()
        
        if(best_val_loss > val_all_losses.avg):
            best_val_loss = val_all_losses.avg
            patient = 0
        else:
            patient += 1
        if(patient >= args.early_stop):
            print("early stop!")
            break



        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer, 0.005 is a good choice now')
    parser.add_argument('--epochs', type=int, default=50, help='train rounds')
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
    
    parser.add_argument('--inner_att_heads', type=int, default=2, help='the number if attention heads in transformer')
    parser.add_argument('--no_progress', action='store_true', default=False)
    parser.add_argument('--agg', action='store_true', default=False, help='if use local aggeragation')
    
    parser.add_argument('--full_length', type=int, default=2288, help='time variables full length')

    parser.add_argument('--no_ncld', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--ablate', type=str, nargs='+', default=None, help='需要消融的变量')
    parser.add_argument('--dis', type=float, default=0.8, help='自己对自己的距离')
    parser.add_argument('--no_encoder_mask', action='store_true', default=False)
    parser.add_argument('--use_decoder_mask', action='store_true', default=False)
    parser.add_argument('--input_window', type=int, default=52, help='')
    parser.add_argument('--window_step0', type=int, default=0, help='')
    parser.add_argument('--window_step1', type=int, default=3, help='') 
    parser.add_argument('--unused_time', type=int, default=46, help='')
    parser.add_argument('--sample_ratio', type=float, default=0.1, help='')
    parser.add_argument('--inter', type=str, default='ig', help='')
    parser.add_argument('--fasttest', action='store_true', default=False, help='')
    parser.add_argument('--single_task_name', type=str, default=None, help='单任务的名字')
    parser.add_argument('--single_time', type=int, default=1, help='')
    
    parser.add_argument('--lambda1', type=float, default=1, help='')
    parser.add_argument('--newloss', action='store_true', default=False)
    parser.add_argument('--early_stop', type=int, default=20, help='')
    parser.add_argument('--sample_method', type=str, default='random')
    parser.add_argument('--dropout', type=float, default=0.1, help='')
    parser.add_argument('--mask_path', type=str, default="/fs/scratch/PAS2599/TrainMask_US_v0822_NLDAS.mat", help='') 
    parser.add_argument('--futurex_length', type=int, default=10, help='')
    
    parser.add_argument('--futurex_location', type=str, default="/fs/ess/PAS2599/zhao4243/DL_input/data_v0822_NLDAS_0125d/CONUS_daily_forecast_ERA5/data_pixel/")
    parser.add_argument('--decinp_dim_idx', type=str, default="013456789", help='未来窗口decoder填充的变量维度索引，索引位置是ablate之后剩余所有变量在d-model中的位置，,limited by d_model parameter')
    parser.add_argument('--futurex_dim_idx', type=str, default="012345678", help='未来的变量维度索引，将ablate之后的剩余变量一一索引到specific idx of futurex full 9 variables, limited by the range of 0-8')

    args = parser.parse_args()

    # print(args.ablate)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    run_detail = f'model_{args.model}_lr_{args.lr}_num_layers_{args.num_layers}_tfdim_{args.dim_feedforward}_embedding_dim_{args.embedding_dim}_mlp_dim_{args.mlp_dim}_loss_{args.loss}_d_model_expanded_{args.d_model_expanded}_agg_{args.agg}_single_task_name_{args.single_task_name}_seed_{args.seed}'
    current_time = time.strftime("%Y%m%d-%H%M%S")
    args.save_path = os.path.join(args.save_path, run_detail, current_time)
    print(args.save_path, flush=True)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        os.makedirs(os.path.join(args.save_path, 'interpretation'))
    
    log_file_path = os.path.join(args.save_path, "log.txt")
    sys.stdout = DualWriter(log_file_path)
    args.decinp_dim_idx = list([int(char) for char in args.decinp_dim_idx])
    args.futurex_dim_idx = list([int(char) for char in args.futurex_dim_idx])
    print(args)
    set_seed(args)
    main(args)
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal
