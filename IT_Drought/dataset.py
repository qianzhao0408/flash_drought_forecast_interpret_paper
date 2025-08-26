# 10/10 增加了Mask-file
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import numpy as np
import torch
import mat73
import os
import time
from util import parse_years_from_index
import gc
import copy

# futurex = torch.load('/fs/scratch/PAS2599/forecastx_predictors_norm.pt') #torch.Size([52148, 7, 44, 365])    

class GridDataset(Dataset):
    def __init__(self, nldas, base_input_file,base_target_file,mask_file, min_max, all_nan_mask, idx_list, split='train', use_agg = False, futurex_location = "/fs/ess/PAS2599/zhao4243/DL_input/data_v0822_NLDAS_0125d/CONUS_daily_forecast_ERA5/data_pixel/", futurex_length=10):
        
        self.drought_indices_data = copy.deepcopy(base_target_file)
        self.drought_indices_names = []
        self.max_rec = min_max['max']
        self.min_rec = min_max['min']
        for k, v in self.drought_indices_data.items():
            self.drought_indices_names.append(k)
            
            if(split=='train'):    
                self.drought_indices_data[k] = torch.cat([self.drought_indices_data[k][:,:16], self.drought_indices_data[k][:,22:38]], dim = 1)
            elif(split=='test'):    
                self.drought_indices_data[k] = torch.cat([self.drought_indices_data[k][:,18-1:22], self.drought_indices_data[k][:,40-1:44]], dim = 1)
            elif(split=='val'):    
                self.drought_indices_data[k] = torch.cat([self.drought_indices_data[k][:,16-1:18], self.drought_indices_data[k][:,38-1:40]], dim = 1)
            elif(split=='all'):    
                self.drought_indices_data[k] = self.drought_indices_data[k]
            elif('-' in split):   
                start = int(split.split('-')[0])
                end = int(split.split('-')[1])
                self.drought_indices_data[k] = self.drought_indices_data[k][:,start-1:end]
            
        print(f"drought indices: {self.drought_indices_names}")
        self.predictors_data = copy.deepcopy(base_input_file)
        self.input_cata_names,self.input_time_names, self.input_static_names= [],[],[]
        
        for k, v in self.predictors_data.items():
            if('nlcd' in k):
                self.input_cata_names.append(k)
                continue
            if len(v.shape) >= 3:
                self.input_time_names.append(k)
            if len(v.shape) <= 2:
                self.input_static_names.append(k)
            if('pr_1979_2022.mat_pr' in k):
                self.predictors_data[k] = torch.log(self.predictors_data[k]+5)
                self.max_rec[k] = torch.nan_to_num(self.predictors_data[k], nan=float('-inf')).max()
                self.min_rec[k] = torch.nan_to_num(self.predictors_data[k], nan=float('inf')).min()
            if len(v.shape) >= 3:
                if(split=='train'):    
                    self.predictors_data[k] = torch.cat([self.predictors_data[k][:,:16], self.predictors_data[k][:,22:38]], dim = 1)  
                elif(split=='test'):    
                    self.predictors_data[k] = torch.cat([self.predictors_data[k][:,18-1:22], self.predictors_data[k][:,40-1:44]], dim = 1) #self.predictors_data[k][:,36:]
                elif(split=='val'):    
                    self.predictors_data[k] = torch.cat([self.predictors_data[k][:,16-1:18], self.predictors_data[k][:,38-1:40]], dim = 1) 
                elif(split=='all'):    
                    self.predictors_data[k] = self.predictors_data[k]
                elif('-' in split):    
                    start = int(split.split('-')[0])
                    end = int(split.split('-')[1])
                    self.predictors_data[k] = self.predictors_data[k][:,start-1:end]
            
        
        print(f"time varing attributes: {self.input_time_names}")
        print(f"static attributes: {self.input_static_names}")
        print(f"cata attributes: {self.input_cata_names}")
        mask_file = mask_file
        
        full_mask = ~np.isnan(mask_file['TrainMask_US'])
        # use_mask = np.ones([200,464])
        # full_mask = ~np.isnan(use_mask)
        fullrows, fullcols = np.nonzero(full_mask)
        self.location_matrix = np.full((200, 464), -1)
        for i, (r, c) in enumerate(zip(fullrows, fullcols)):
            self.location_matrix[r, c] = i
        if(split=='train'):    
            self.nldas =  torch.cat([nldas[full_mask].permute(0,2,1)[:,:16], nldas[full_mask].permute(0,2,1)[:,22:38]], dim = 1)   
            # self.futurex = torch.cat([ futurex[:,:,:16]  ,  futurex[:,:,22:38]], dim = 2)
        elif(split=='test'):    
            self.nldas =  torch.cat([nldas[full_mask].permute(0,2,1)[:,18-1:22], nldas[full_mask].permute(0,2,1)[:,40-1:44]], dim = 1) #self.predictors_data[k][:,36:]
            # self.futurex = torch.cat([ futurex[:,:,18-1:22]  ,  futurex[:,:,40-1:44]], dim = 2)
        elif(split=='val'):    
            self.nldas =  torch.cat([nldas[full_mask].permute(0,2,1)[:,16-1:18], nldas[full_mask].permute(0,2,1)[:,38-1:40]], dim = 1) 
            # self.futurex = torch.cat([ futurex[:,:,16-1:18]  ,  futurex[:,:,38-1:40]], dim = 2)
        elif(split=='all'):    
            self.nldas = nldas[full_mask].permute(0,2,1)
            # self.futurex = futurex
        elif('-' in split):    
            start = int(split.split('-')[0])
            end = int(split.split('-')[1])
            self.nldas = nldas[full_mask].permute(0,2,1)[:,start-1:end]
        self.split = split
        self.mask = full_mask
        
        # if(split=='train'):
        #     self.mask = mask_file['TrainMask_US'] == 1.0
        #     self.list = idx_list['train']
        # elif(split=='test'):
        #     self.mask = mask_file['TrainMask_US'] == 2.0
        #     self.list = idx_list['test']
        # elif(split=='all'):
        #     self.mask = full_mask
        
        # self.list = np.array([])
        # count = 0
        # for i in range(585):
        #     for j in range(1386):
        #         if(np.isnan(mask_file['TrainMask_US'][i][j])):
        #             continue
        #         if(self.mask[i][j]==True):
        #             self.list = np.append(self.list, count)
        #         count+=1
            
        selected_indices = np.nonzero(self.mask)
        self.rows, self.cols = selected_indices
        self.ablate_target = None
        

        self.all_nan_mask = all_nan_mask
        self.max_values =  torch.where(torch.isnan(self.predictors_data['nlcd.mat_ncld'].float()), torch.tensor(float('-inf')), self.predictors_data['nlcd.mat_ncld'].float()).max()
        print(self.predictors_data['nlcd.mat_ncld'].shape)
        self.use_agg = use_agg
        if(use_agg):
            self.distance_matrix = np.zeros((5, 5))
            row=2
            col=2
            for i in range(row-2, row+3):
                for j in range(col-2, col+3):
                    distance = np.sqrt((i - row)**2 + (j - col)**2)
                    self.distance_matrix[i-(row-2)][j-(col-2)] = distance      
        self.futurex_location = futurex_location
        self.futurex_max = torch.tensor([319.7644, 96.6200, 104.7924, 24.4684, 7.5144, 18.9595, 370.6965, 58.7602, 0.1665])      
        self.futurex_min = torch.tensor([232.3090, -0.0016, 62.8070, 2.5152e-05, 0., -0.9952, 0.0009, -228.2989, 0.])     
        self.range_vals = [ max(self.futurex_max[futurex_i]  - self.futurex_min[futurex_i], 1e-8)     for futurex_i in range(len(self.futurex_max))   ]
        self.futurex_length = futurex_length
    def get_loc(self,index):
        return torch.tensor([self.rows[index], self.cols[index]])
    

    def __getitem__(self, index):

        original_index = index
        futurex = torch.load(os.path.join(self.futurex_location, f"{index}.pt"), weights_only=True)  # (7, 44, 365, 10)
        for futurex_i in range(len(futurex)):
            futurex[futurex_i] = (futurex[futurex_i] - self.futurex_min[futurex_i]) / ( self.range_vals[futurex_i])
        if(self.split=='train'):    
            futurex = torch.cat([ futurex[:,:16]  ,  futurex[:,22:38]], dim = 1)
        elif(self.split=='test'):    
            futurex = torch.cat([ futurex[:,18-1:22]  ,  futurex[:,40-1:44]], dim = 1)
        elif(self.split=='val'):    
            futurex = torch.cat([ futurex[:,16-1:18]  ,  futurex[:,38-1:40]], dim = 1)
        elif(self.split=='all'):    
            futurex = futurex
            
        if(not self.use_agg):
            x_cata = self.predictors_data['nlcd.mat_ncld'][original_index]
            x_cata[torch.isnan(x_cata)] = -1
            x_num_static = torch.stack([(self.predictors_data[k][original_index] - self.min_rec[k] ) / (self.max_rec[k]- self.min_rec[k]) for k in self.input_static_names])
            x_num_time = torch.stack([(self.predictors_data[k][original_index]- self.min_rec[k] ) / (self.max_rec[k]- self.min_rec[k]) for k in self.input_time_names])
            y = torch.stack([( v[original_index]- self.min_rec[k] ) / (self.max_rec[k]- self.min_rec[k])  for k,v in self.drought_indices_data.items()])
            return x_num_static, x_cata.to(torch.int), x_num_time, y, self.get_loc(index), index, 0,  original_index, self.nldas[index], futurex[:,:,:,:self.futurex_length]
        else:
            agg_distance = torch.full((5, 5), float('inf'))
            row, col = self.get_loc(index)
            neighbors = []
            for i in range(row-2, row+3):
                for j in range(col-2, col+3):
                    if i == row and j == col:
                        agg_distance[i-(row-2)][j-(col-2)] = 0.5
                        neighbors.append(self_location)
                        continue
                    self_location = self.location_matrix[row][col]
                    if(i-2<0 or i+2 >=200 or j-2<0 or j+2>=464): #越界
                        neighbors.append(self_location)
                        continue
                    neigb_location = self.location_matrix[i][j]    
                    if(neigb_location==-1): #不在训练集测试集内
                        neighbors.append(self_location)
                    else:
                        if(self.all_nan_mask[neigb_location]==True): # target全为拟合数据
                            neighbors.append(self_location)
                        else:    
                            neighbors.append(neigb_location)
                            agg_distance[i-(row-2)][j-(col-2)] = self.distance_matrix[i-(row-2)][j-(col-2)]
            x_cata = self.predictors_data['nlcd.mat_ncld'][original_index]
            x_cata[torch.isnan(x_cata)] = -1
            x_num_static = torch.stack([   (self.predictors_data[k][neighbors] - self.min_rec[k]) / (self.max_rec[k] - self.min_rec[k]) for k in self.input_static_names]).permute(1,0)
            x_num_time = torch.stack([(self.predictors_data[k][neighbors]  - self.min_rec[k] ) / (self.max_rec[k] - self.min_rec[k]) for k in self.input_time_names]).permute(1,0,2,3)
            y = torch.stack([(v[neighbors]  - self.min_rec[k] ) / (self.max_rec[k]  - self.min_rec[k] )  for k,v in self.drought_indices_data.items()]).permute(1,0,2,3)
            return x_num_static, x_cata.to(torch.int), x_num_time, y, self.get_loc(index), index, agg_distance, original_index, self.nldas[index], futurex
    
    def __len__(self):
        return self.mask.sum()
    

def load0822(args):
    read_start_time = time.time()
    
    base_input_file = torch.load( os.path.join(args.data_path,"predictors_v0822.pt"), weights_only=True)
    base_target_file = torch.load(os.path.join(args.data_path, 'drought_v0822.pt'), weights_only=True)
    mean_std_max_min_file =  torch.load(os.path.join(args.data_path, 'mean_std_max_min_v0822.pt'), weights_only=True) 
    nan_mask = torch.load(os.path.join(args.data_path, 'drought_nan_mask_v0822.pt'), weights_only=True) 
    # mask_file = None
    mask_file= mat73.loadmat(args.mask_path)
    idx_list = None
    
    print(f"Read input and target spend:{time.time() - read_start_time}")
    
    ablate_targets = []
    if(not args.ablate is None):
        for ablate_item in args.ablate:
            if(ablate_item in base_input_file.keys()):
                del base_input_file[ablate_item]
                # base_input_file[args.ablate] = torch.zeros_like(base_input_file[args.ablate])
                if('nlcd' in ablate_item):
                    base_input_file[ablate_item] = torch.full_like(base_input_file[ablate_item],-1)
            if(ablate_item in base_target_file.keys()):
                # del base_target_file[args.ablate]
                ablate_targets.append(ablate_item)
                # base_target_file[args.ablate] = torch.zeros_like(base_target_file[args.ablate])
            print(f'ablate {ablate_item}')
        
    
    year_index = {}
    
    # #拓展到11年
    # for i,(k, v) in enumerate(base_target_file.items()):
    #     startyear, endyear = parse_years_from_index(k.split('.')[0])
    #     print(startyear, endyear,k)
    #     if(endyear-startyear+1!=11):
    #         mean_values =torch.nanmean(v, dim=1)
    #         repeated_mean_values = mean_values.unsqueeze(1).expand(-1, 11, -1)
    #         new_tensor = repeated_mean_values.clone()
    #         new_tensor[:, startyear-2003:endyear-2003+1 , :] = v
    #         base_target_file[k] = new_tensor
    #     else:
    #         base_target_file[k] = v
    #     year_index[i] = [(startyear-2003)*52, (endyear-2003+1)*52]
    # print(year_index)
    gc.collect()
    
    nan_mask_list = []
    
            
    for k,v in nan_mask.items():
        nan_mask_list.append(v.flatten(start_dim=1))
    del nan_mask
    all_nan_mask = nan_mask_list[0].all(dim=1)
    for i in range(1, len(base_target_file)):
        all_nan_mask = all_nan_mask & nan_mask_list[i].all(dim=1)
    nan_mask_list = torch.stack(nan_mask_list)
    nldas = torch.tensor(mat73.loadmat(os.path.join(args.data_path, 'Drought_idx_NLDAS.mat'))['Drought_idx'])
    gc.collect()
    train_dataset = GridDataset(nldas, base_input_file, base_target_file, mask_file, mean_std_max_min_file, all_nan_mask, idx_list, split='train', use_agg=args.agg, futurex_location=args.futurex_location, futurex_length=args.futurex_length)
    test_dataset = GridDataset(nldas, base_input_file, base_target_file, mask_file, mean_std_max_min_file, all_nan_mask, idx_list, split='test', use_agg=args.agg, futurex_location=args.futurex_location,  futurex_length=args.futurex_length)
    val_dataset = GridDataset(nldas, base_input_file, base_target_file, mask_file, mean_std_max_min_file, all_nan_mask, idx_list, split='val', use_agg=args.agg, futurex_location=args.futurex_location,  futurex_length=args.futurex_length)
    train_dataset.ablate_target = ablate_targets
    test_dataset.ablate_target = ablate_targets
    val_dataset.ablate_target = ablate_targets
    if(len(ablate_targets)>0):
        target_idx_list = [train_dataset.drought_indices_names.index(ablate_target_name) for ablate_target_name in ablate_targets ]
        args.ablate_target_idx = target_idx_list
    else:
        args.ablate_target_idx = None
    print(len(train_dataset), len(test_dataset), len(val_dataset))
    
    batch_size = args.bs
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    
    
    mean_values = {'time':[], 'static':[], 'target':[]}
    for k,v in mean_std_max_min_file['mean'].items():
        if(k in train_dataset.input_time_names):
            mean_values['time'].append(v)
        if(k in train_dataset.input_static_names):
            mean_values['static'].append(v)
        if(k in train_dataset.drought_indices_names):
            mean_values['target'].append(v)
    for k,v in mean_values.items():
        mean_values[k] = torch.stack(v) 
        
    
    return train_dataset, test_dataset, val_dataset, train_loader, test_loader, val_loader, mean_values, nan_mask_list, year_index


def load0822_all(args):
    read_start_time = time.time()
    
    base_input_file = torch.load( os.path.join(args.data_path,"predictors_v0822.pt"), weights_only=True)
    base_target_file = torch.load(os.path.join(args.data_path, 'drought_v0822.pt'), weights_only=True)
    mean_std_max_min_file =  torch.load(os.path.join(args.data_path, 'mean_std_max_min_v0822.pt'), weights_only=True) 
    nan_mask = torch.load(os.path.join(args.data_path, 'drought_nan_mask_v0822.pt'), weights_only=True) 
    mask_file= mat73.loadmat(args.mask_path)
    idx_list = None
    
    print(f"Read input and target spend:{time.time() - read_start_time}")
    
    ablate_targets = []
    if(not args.ablate is None):
        for ablate_item in args.ablate:
            if(ablate_item in base_input_file.keys()):
                del base_input_file[ablate_item]
                # base_input_file[args.ablate] = torch.zeros_like(base_input_file[args.ablate])
                if('nlcd' in ablate_item):
                    base_input_file[ablate_item] = torch.full_like(base_input_file[ablate_item],-1)
            if(ablate_item in base_target_file.keys()):
                # del base_target_file[args.ablate]
                ablate_targets.append(ablate_item)
                # base_target_file[args.ablate] = torch.zeros_like(base_target_file[args.ablate])
            print(f'ablate {ablate_item}')
        
    year_index = {}
    
    #拓展到11年
    # for i,(k, v) in enumerate(base_target_file.items()):
    #     startyear, endyear = parse_years_from_index(k.split('.')[0])
    #     print(startyear, endyear,k)
    #     if(endyear-startyear+1!=11):
    #         mean_values =torch.nanmean(v, dim=1)
    #         repeated_mean_values = mean_values.unsqueeze(1).expand(-1, 11, -1)
    #         new_tensor = repeated_mean_values.clone()
    #         new_tensor[:, startyear-2003:endyear-2003+1 , :] = v
    #         base_target_file[k] = new_tensor
    #     else:
    #         base_target_file[k] = v
    #     year_index[i] = [(startyear-2003)*52, (endyear-2003+1)*52]
    # print(year_index)
    gc.collect()
    
    nan_mask_list = []
    
    # if(not args.ablate is None):
    #     if(args.ablate in nan_mask.keys()):
    #         del nan_mask[args.ablate]
            
    for k,v in nan_mask.items():
        nan_mask_list.append(v.flatten(start_dim=1))
    del nan_mask
    all_nan_mask = nan_mask_list[0].all(dim=1)
    for i in range(1, len(base_target_file)):
        all_nan_mask = all_nan_mask & nan_mask_list[i].all(dim=1)

    nldas = torch.tensor(mat73.loadmat(os.path.join(args.data_path, 'Drought_idx_NLDAS.mat'))['Drought_idx'])
    gc.collect()
    all_dataset = GridDataset(nldas, base_input_file, base_target_file, mask_file, mean_std_max_min_file, all_nan_mask, idx_list, split=args.split, use_agg=args.agg, futurex_length=args.futurex_length)
    all_dataset.ablate_target = ablate_targets
    if(len(ablate_targets)>0):
        target_idx_list = [all_dataset.drought_indices_names.index(ablate_target_name) for ablate_target_name in ablate_targets ]
        args.ablate_target_idx = target_idx_list
    else:
        args.ablate_target_idx = None
    
    print(len(all_dataset))
    
    batch_size = args.bs
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    
    
    mean_values = {'time':[], 'static':[], 'target':[]}
    for k,v in mean_std_max_min_file['mean'].items():
        if(k in all_dataset.input_time_names):
            mean_values['time'].append(v)
        if(k in all_dataset.input_static_names):
            mean_values['static'].append(v)
        if(k in all_dataset.drought_indices_names):
            mean_values['target'].append(v)
    for k,v in mean_values.items():
        mean_values[k] = torch.stack(v) 
    return all_dataset,  all_loader, mean_values, nan_mask_list, year_index