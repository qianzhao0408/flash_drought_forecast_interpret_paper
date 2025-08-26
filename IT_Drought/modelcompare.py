import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch import nn, Tensor
import torch.nn.init as init


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0., max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        # Adjust the cosine computation for odd d_model
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0).transpose(0, 1)

        # pe[:, 0, 0::2] = torch.sin(position * div_term)
        # pe[:, 0, 1::2] = torch.cos(position * div_term)
        # self.register_buffer('pe', pe)
        
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class ExpandDimsMLP(nn.Module):
    def __init__(self, d_model, d_model_expanded, dropout_prob=0.1):
        super(ExpandDimsMLP, self).__init__()
        self.linear = nn.Linear(d_model, d_model_expanded)
        init.xavier_uniform_(self.linear.weight)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
class BatchDistanceAwareAttentionAggregator(nn.Module):
    def __init__(self, static_dim):
        super(BatchDistanceAwareAttentionAggregator, self).__init__()
        self.query = nn.Linear(static_dim, static_dim)
        self.key = nn.Linear(static_dim, static_dim)
        self.softmax = nn.Softmax(dim=1)  # 注意力权重在每个batch内的位置间进行softmax
        self.static_dim = static_dim
        

    def forward(self, current_static_feature, static_features, current_time_feature, time_features, distances, show=False):
        batch_size, num_positions, _ = static_features.size()
        # 将static_features和positions按照batch和位置展平，以便进行距离计算
        # 注意：这里假设positions已经是一个[b, n, 2]形状的张量，其中包含了每个位置的坐标

        # 计算基于特征的注意力得分
        queries = self.query(current_static_feature.view(-1, self.static_dim))
        keys = self.key(static_features.reshape(-1, self.static_dim))
        queries = torch.relu(queries)
        keys = torch.relu(keys)
        # queries = (static_features[:, 12:13, :].view(-1, self.static_dim))
        # keys = (static_features.view(-1, self.static_dim))
        attention_scores = torch.bmm(queries.view(batch_size, 1, -1), 
                                     keys.view(batch_size, -1, num_positions)) / (self.static_dim ** 0.5)
        # print(attention_scores[0])
        
        # 距离因子，避免自身距离为0的问题
        distance_factors = 1 / (distances + 1e-9)
        # 调整注意力得分
        adjusted_attention_scores = attention_scores * distance_factors.unsqueeze(1)
        # print(adjusted_attention_scores[0])
        attention_weights = torch.softmax(adjusted_attention_scores,dim=2)
        neiborweights = torch.cat((attention_weights[:, :, :12], attention_weights[:, :, 13:]), dim=2)
        neighbor_time_features = torch.cat((time_features[:, :12], time_features[:, 13:]),dim=1)
        
        # 对时间特征进行加权聚合
        # 由于时间特征具有额外的时间长度维度，需要对attention_weights进行扩展以匹配时间特征的形状
        weighted_time_features = torch.einsum('bij,bjkl->bikl', neiborweights, neighbor_time_features).squeeze()
        if(show):
            return current_time_feature * attention_weights[:,0, 12].view(attention_weights[:,0, 12].shape[0], 1, 1) + weighted_time_features.squeeze(), attention_weights
        return current_time_feature * attention_weights[:,0, 12].view(attention_weights[:,0, 12].shape[0], 1, 1) + weighted_time_features.squeeze()
        # return weighted_time_features
        
        
    # 打印邻居位置的注意力权重
    def show(self, current_static_feature, static_features, distances):
        batch_size, num_positions, _ = static_features.size()
        # 将static_features和positions按照batch和位置展平，以便进行距离计算
        # 注意：这里假设positions已经是一个[b, n, 2]形状的张量，其中包含了每个位置的坐标

        # 计算基于特征的注意力得分
        queries = self.query(current_static_feature.view(-1, self.static_dim))
        keys = self.key(static_features.view(-1, self.static_dim))
        queries = torch.relu(queries)
        keys = torch.relu(keys)
        # queries = (static_features[:, 12:13, :].view(-1, self.static_dim))
        # keys = (static_features.view(-1, self.static_dim))
        attention_scores = torch.bmm(queries.view(batch_size, 1, -1), 
                                     keys.view(batch_size, -1, num_positions)) / (self.static_dim ** 0.5)
        # print(attention_scores[0])
        
        # 距离因子，避免自身距离为0的问题
        distance_factors = 1 / (distances + 1e-9)
        # 调整注意力得分
        adjusted_attention_scores = attention_scores * distance_factors.unsqueeze(1)
        # print(adjusted_attention_scores[0])
        attention_weights = torch.softmax(adjusted_attention_scores,dim=2)
        
        return attention_weights
    
    
    

class Model_forecast(nn.Module):
    def __init__(self, args,device, num_categories, num_layers, embedding_dim, 
                 d_model=15, 
                 d_model_expanded = 64,
                 num_heads=4, 
                 dim_feedforward=512,
                 mlp_input = 14,
                 mlp_hidden = 16,
                 mlp_dim = 32,
                 dropout = 0.1,
                 num_tasks=1,
                 mask_length = 1196,
                 batch_first = False):
        super(Model_forecast, self).__init__()
        self.args = args
        self.device = device
        self.batch_first = batch_first
        self.embedding_dim = embedding_dim
        self.mask_length = mask_length
        
        # 对于x_num_time使用Transformer
        self.expand_dims = ExpandDimsMLP(d_model, d_model_expanded,dropout_prob=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model_expanded, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model=d_model_expanded, max_len=mask_length)
        self.pos_decoder = PositionalEncoding(d_model=d_model_expanded + embedding_dim + mlp_dim, max_len=mask_length+100)
        
        num_embeddings = num_categories+1
        self.embeddings = nn.Embedding(num_embeddings+1, embedding_dim, padding_idx=num_embeddings)
        self.embeddings.weight.data[num_embeddings] = torch.zeros(embedding_dim)
        self.num_tasks=num_tasks
        # 对于x_num_static使用MLP
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input, mlp_hidden,bias=False),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_dim,bias=False),
        )
        
        self.mlp2 = nn.Linear(d_model, d_model_expanded + embedding_dim + mlp_dim,bias=True)
        
        print(d_model_expanded + embedding_dim + mlp_dim)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model_expanded + embedding_dim + mlp_dim,
            nhead=num_heads  # Adjust the number of heads as needed
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer,
            num_layers=args.decoder_layers  # Adjust the number of layers as needed
        )

        self.mask = self.generate_square_subsequent_mask(mask_length, 0, self.mask_length).to(self.device)
        if(args.agg == True):
            self.agg = BatchDistanceAwareAttentionAggregator(mlp_input)
        self.regressors_to1 = nn.ModuleList([
            nn.Linear(d_model_expanded + embedding_dim + mlp_dim, 1) for _ in range(num_tasks)
        ])
        # self.output_layer = nn.Linear(feature_dim, num_targets)

    def generate_square_subsequent_mask(self, sz, window_start=0, window_end=80):
        indices = torch.arange(sz).unsqueeze(0) - torch.arange(sz).unsqueeze(1)
        mask = torch.where((indices >= 0) & (indices <= window_end), 0.0, float('-inf'))
        return mask
    
    
class Model_forecast_window(Model_forecast):
    def __init__(self, args,device, num_categories, num_layers, embedding_dim, 
                 d_model=15, 
                 d_model_expanded = 64,
                 num_heads=4, 
                 dim_feedforward=512,
                 mlp_input = 14,
                 mlp_hidden = 16,
                 mlp_dim = 32,
                 dropout = 0.1,
                 num_tasks=1,
                 mask_length = 1196,
                 batch_first = False):
        super(Model_forecast_window, self).__init__(args,device, num_categories, num_layers, embedding_dim, d_model,
                 d_model_expanded,
                 num_heads, 
                 dim_feedforward,
                 mlp_input,
                 mlp_hidden,
                 mlp_dim,
                 dropout,
                 num_tasks,
                 mask_length,
                 batch_first)
        
        if(args.no_encoder_mask):
            self.mask = None
        if(args.use_decoder_mask):
            self.mask_dec = self.generate_square_subsequent_mask(mask_length+args.forecast_window, 0, self.mask_length).to(self.device)
            
    def forward(self, x_num_time, x_num_static, x_cata, tgt=None,task_idx=None, target_idx=None, local_batch_first=False):
        
        # 解释变量时候调整维度
        if(self.batch_first or local_batch_first):
            x_num_time = x_num_time.permute(2,0,1)
            tgt = tgt.permute(2,0,1)

        # 示意图中，transformer encoder前面的layer，拓展时间变量维度，同时建立相互联系
        x_num_time_predictors = self.expand_dims(x_num_time)
        # 时间变量加位置编码
        x_position_embeded = self.pos_encoder(x_num_time_predictors)
        # transformer encoder
        x_time_representaiton = self.transformer_encoder(x_position_embeded, self.mask) #(1196, batch_size, 7)
        
        # 2层MLP 拓展静态变量
        x_static_representaton = self.mlp(x_num_static)  # （batch_size, 64）
        
        # nan变量的embedding填充为0
        zero_idx = self.embeddings.padding_idx
        x_cata[x_cata==-1] = zero_idx
        embed = self.embeddings(x_cata)  # categorical变量
        
        # 调整表征维度，避免出现维度为1，不对齐
        if self.embedding_dim==1:
            x_cata_embedded = torch.squeeze(embed, dim=2)
        else:
            x_cata_embedded = torch.squeeze(embed)

        # 静态变量嵌入到时间变量中
        x_static_representaton = x_static_representaton.unsqueeze(0).repeat(x_time_representaiton.size(0), 1, 1)
        x_cata_embedded = x_cata_embedded.unsqueeze(0).repeat(x_time_representaiton.size(0), 1, 1)
        if(self.args.no_ncld):
            combined = torch.cat([x_time_representaiton, torch.zeros_like(x_cata_embedded).to(self.device) ,x_static_representaton], dim=2)
        else:
            combined = torch.cat([x_time_representaiton, x_cata_embedded, x_static_representaton], dim=2)  # concatenate along feature dimension
       
        # decoder的输入，拓展时间变量维度，同时建立相互联系
        tgt = self.mlp2(tgt)
        pe_tgt = self.pos_decoder(tgt)

        # decoder 
        decoded = self.transformer_decoder(pe_tgt, combined)[-self.args.unused_time:].permute(1,0,2)
        
        if(not task_idx is None): # 用于解释，输出
            return  self.regressors_to1[task_idx](decoded)[:,target_idx:target_idx+1]
        else:   
            outputs = [(regressor(decoded)) for regressor in self.regressors_to1]
            return outputs

    

class Model_forecast_window_singlemlpfuturex(Model_forecast):
    def __init__(self, args,device, num_categories, num_layers, embedding_dim, 
                 d_model=15, 
                 d_model_expanded = 64,
                 num_heads=4, 
                 dim_feedforward=512,
                 mlp_input = 14,
                 mlp_hidden = 16,
                 mlp_dim = 32,
                 dropout = 0.1,
                 num_tasks=1,
                 mask_length = 1196,
                 batch_first = False,
                 futurex_complex=False):
        super(Model_forecast_window_singlemlpfuturex, self).__init__(args,device, num_categories, num_layers, embedding_dim, d_model,
                 d_model_expanded,
                 num_heads, 
                 dim_feedforward,
                 mlp_input,
                 mlp_hidden,
                 mlp_dim,
                 dropout,
                 num_tasks,
                 mask_length,
                 batch_first)
        
        if(args.no_encoder_mask):
            self.mask = None
        if(args.use_decoder_mask):
            self.mask_dec = self.generate_square_subsequent_mask(mask_length+args.forecast_window, 0, self.mask_length).to(self.device)
        if(futurex_complex):
            self.mlp_futurex = nn.Sequential(
                nn.Linear(d_model, d_model_expanded + embedding_dim + mlp_dim),
                nn.LeakyReLU(),
                nn.Linear(d_model_expanded + embedding_dim + mlp_dim, d_model_expanded + embedding_dim + mlp_dim),
                nn.LeakyReLU(),
            )
        else:
            self.mlp_futurex = nn.Linear(d_model, d_model_expanded + embedding_dim + mlp_dim)  
        self.warmup_regressor = nn.Linear(d_model_expanded , 1)
    def forward(self, x_num_time, x_num_static, x_cata, tgt=None,task_idx=None, target_idx=None, local_batch_first=False):
        
        # 解释变量时候调整维度
        if(self.batch_first or local_batch_first):
            x_num_time = x_num_time.permute(2,0,1)
            tgt = tgt.permute(2,0,1)

        # 示意图中，transformer encoder前面的layer，拓展时间变量维度，同时建立相互联系
        x_num_time_predictors = self.expand_dims(x_num_time)
        # 时间变量加位置编码
        x_position_embeded = self.pos_encoder(x_num_time_predictors)
        # transformer encoder
        x_time_representaiton = self.transformer_encoder(x_position_embeded, self.mask) #(1196, batch_size, 7)
        
        # 2层MLP 拓展静态变量
        x_static_representaton = self.mlp(x_num_static)  # （batch_size, 64）
        
        # nan变量的embedding填充为0
        zero_idx = self.embeddings.padding_idx
        x_cata[x_cata==-1] = zero_idx
        embed = self.embeddings(x_cata)  # categorical变量
        
        # 调整表征维度，避免出现维度为1，不对齐
        if self.embedding_dim==1:
            x_cata_embedded = torch.squeeze(embed, dim=2)
        else:
            x_cata_embedded = torch.squeeze(embed)

        # 静态变量嵌入到时间变量中
        x_static_representaton = x_static_representaton.unsqueeze(0).repeat(x_time_representaiton.size(0), 1, 1)
        x_cata_embedded = x_cata_embedded.unsqueeze(0).repeat(x_time_representaiton.size(0), 1, 1)
        if(self.args.no_ncld):
            combined = torch.cat([x_time_representaiton, torch.zeros_like(x_cata_embedded).to(self.device) ,x_static_representaton], dim=2)
        else:
            combined = torch.cat([x_time_representaiton, x_cata_embedded, x_static_representaton], dim=2)  # concatenate along feature dimension
       
        # decoder的输入，拓展时间变量维度，同时建立相互联系
        tgt_history = self.mlp2(tgt[:-self.args.unused_time])
        tgt_futurex = self.mlp_futurex(tgt[-self.args.unused_time:])
        tgt = torch.cat((tgt_history, tgt_futurex))
        pe_tgt = self.pos_decoder(tgt)


        # decoder 
        decoded = self.transformer_decoder(pe_tgt, combined)[-self.args.unused_time:].permute(1,0,2)
        
        if(not task_idx is None): # 用于解释，输出
            return  self.regressors_to1[task_idx](decoded)[:,target_idx:target_idx+1]
        else:   
            outputs = [(regressor(decoded)) for regressor in self.regressors_to1]
            return outputs

    def forward_warmup(self, future_x):
        tgt_futurex = self.mlp_futurex(future_x).permute(1,0,2)
        return self.warmup_regressor(tgt_futurex)
    
class Model_forecast_window_regressorfuturex(Model_forecast):
    def __init__(self, args,device, num_categories, num_layers, embedding_dim, 
                 d_model=15, 
                 d_model_expanded = 64,
                 num_heads=4, 
                 dim_feedforward=512,
                 mlp_input = 14,
                 mlp_hidden = 16,
                 mlp_dim = 32,
                 dropout = 0.1,
                 num_tasks=1,
                 mask_length = 1196,
                 batch_first = False,
                 futurex_complex=False):
        super(Model_forecast_window_regressorfuturex, self).__init__(args,device, num_categories, num_layers, embedding_dim, d_model,
                 d_model_expanded,
                 num_heads, 
                 dim_feedforward,
                 mlp_input,
                 mlp_hidden,
                 mlp_dim,
                 dropout,
                 num_tasks,
                 mask_length,
                 batch_first)
        
        if(args.no_encoder_mask):
            self.mask = None
        if(args.use_decoder_mask):
            self.mask_dec = self.generate_square_subsequent_mask(mask_length+args.forecast_window, 0, self.mask_length).to(self.device)
        if(futurex_complex):
            self.mlp_futurex = nn.Sequential(
                nn.Linear(d_model, d_model_expanded),
                nn.LeakyReLU(),
                nn.Linear(d_model_expanded , d_model_expanded),
                nn.LeakyReLU(),
            )
        else:
            self.mlp_futurex = nn.Linear(d_model, d_model_expanded )  
        self.warmup_regressor = nn.Linear(d_model_expanded , 1)
        self.regressors_to1 = nn.ModuleList([
            nn.Linear(d_model_expanded + embedding_dim + mlp_dim + d_model_expanded, 1) for _ in range(num_tasks)
        ])
    def forward(self, x_num_time, x_num_static, x_cata, tgt=None,task_idx=None, target_idx=None, local_batch_first=False):
        
        # 解释变量时候调整维度
        if(self.batch_first or local_batch_first):
            x_num_time = x_num_time.permute(2,0,1)
            tgt = tgt.permute(2,0,1)

        # 示意图中，transformer encoder前面的layer，拓展时间变量维度，同时建立相互联系
        x_num_time_predictors = self.expand_dims(x_num_time)
        # 时间变量加位置编码
        x_position_embeded = self.pos_encoder(x_num_time_predictors)
        # transformer encoder
        x_time_representaiton = self.transformer_encoder(x_position_embeded, self.mask) #(1196, batch_size, 7)
        
        # 2层MLP 拓展静态变量
        x_static_representaton = self.mlp(x_num_static)  # （batch_size, 64）
        
        # nan变量的embedding填充为0
        zero_idx = self.embeddings.padding_idx
        x_cata[x_cata==-1] = zero_idx
        embed = self.embeddings(x_cata)  # categorical变量
        
        # 调整表征维度，避免出现维度为1，不对齐
        if self.embedding_dim==1:
            x_cata_embedded = torch.squeeze(embed, dim=2)
        else:
            x_cata_embedded = torch.squeeze(embed)

        # 静态变量嵌入到时间变量中
        x_static_representaton = x_static_representaton.unsqueeze(0).repeat(x_time_representaiton.size(0), 1, 1)
        x_cata_embedded = x_cata_embedded.unsqueeze(0).repeat(x_time_representaiton.size(0), 1, 1)
        if(self.args.no_ncld):
            combined = torch.cat([x_time_representaiton, torch.zeros_like(x_cata_embedded).to(self.device) ,x_static_representaton], dim=2)
        else:
            combined = torch.cat([x_time_representaiton, x_cata_embedded, x_static_representaton], dim=2)  # concatenate along feature dimension
       
        # decoder的输入，拓展时间变量维度，同时建立相互联系
        tgt_history = self.mlp2(tgt)
        tgt_futurex = self.mlp_futurex(tgt[-self.args.unused_time:]).permute(1,0,2)
        pe_tgt = self.pos_decoder(tgt_history)


        # decoder 
        decoded = self.transformer_decoder(pe_tgt, combined)[-self.args.unused_time:].permute(1,0,2)
        decoded = torch.cat((decoded,tgt_futurex),dim=2)
        if(not task_idx is None): # 用于解释，输出
            return  self.regressors_to1[task_idx](decoded)[:,target_idx:target_idx+1]
        else:   
            outputs = [(regressor(decoded)) for regressor in self.regressors_to1]
            return outputs

    def forward_warmup(self, future_x):
        tgt_futurex = self.mlp_futurex(future_x).permute(1,0,2)
        return self.warmup_regressor(tgt_futurex)