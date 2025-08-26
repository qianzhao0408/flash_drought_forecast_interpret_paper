from modelcompare import *

def build_model(args,device, num_tasks = 3):
    if(args.model=='ours'):
        print('build transformer based')
        model = Model_forecast(args,device, num_categories = 96, 
                    num_layers = args.num_layers, 
                    d_model = args.d_model,
                    d_model_expanded = args.d_model_expanded,
                    embedding_dim = args.embedding_dim,
                    num_heads = args.inner_att_heads,
                    dim_feedforward = args.dim_feedforward,
                    mlp_input=args.mlp_input,
                    mlp_hidden = args.mlp_hidden,
                    mlp_dim = args.mlp_dim,
                    num_tasks= num_tasks,
                    mask_length  = args.full_length,
                    ).to(device)
    elif(args.model=='window'):
        print('build transformer-window based')
        model = Model_forecast_window(args,device, num_categories = 96, 
                    num_layers = args.num_layers, 
                    d_model = args.d_model,
                    d_model_expanded = args.d_model_expanded,
                    embedding_dim = args.embedding_dim,
                    num_heads = args.inner_att_heads,
                    dim_feedforward = args.dim_feedforward,
                    mlp_input=args.mlp_input,
                    mlp_hidden = args.mlp_hidden,
                    mlp_dim = args.mlp_dim,
                    dropout=args.dropout,
                    num_tasks= num_tasks,
                    mask_length  = args.input_window,
                    ).to(device)
    elif(args.model=='window_single_futurex_module'):
        model = Model_forecast_window_singlemlpfuturex(args,device, num_categories = 96, 
                    num_layers = args.num_layers, 
                    d_model = args.d_model,
                    d_model_expanded = args.d_model_expanded,
                    embedding_dim = args.embedding_dim,
                    num_heads = args.inner_att_heads,
                    dim_feedforward = args.dim_feedforward,
                    mlp_input=args.mlp_input,
                    mlp_hidden = args.mlp_hidden,
                    mlp_dim = args.mlp_dim,
                    dropout=args.dropout,
                    num_tasks= num_tasks,
                    mask_length  = args.input_window,
                    ).to(device)
    elif(args.model=='window_single_futurex_module_complex'):
        model = Model_forecast_window_singlemlpfuturex(args,device, num_categories = 96, 
                    num_layers = args.num_layers, 
                    d_model = args.d_model,
                    d_model_expanded = args.d_model_expanded,
                    embedding_dim = args.embedding_dim,
                    num_heads = args.inner_att_heads,
                    dim_feedforward = args.dim_feedforward,
                    mlp_input=args.mlp_input,
                    mlp_hidden = args.mlp_hidden,
                    mlp_dim = args.mlp_dim,
                    dropout=args.dropout,
                    num_tasks= num_tasks,
                    mask_length  = args.input_window,
                    futurex_complex = True,
                    ).to(device)
    elif(args.model=='window_regressorfuturex'):
        model = Model_forecast_window_regressorfuturex(args,device, num_categories = 96, 
                    num_layers = args.num_layers, 
                    d_model = args.d_model,
                    d_model_expanded = args.d_model_expanded,
                    embedding_dim = args.embedding_dim,
                    num_heads = args.inner_att_heads,
                    dim_feedforward = args.dim_feedforward,
                    mlp_input=args.mlp_input,
                    mlp_hidden = args.mlp_hidden,
                    mlp_dim = args.mlp_dim,
                    dropout=args.dropout,
                    num_tasks= num_tasks,
                    mask_length  = args.input_window,
                    futurex_complex = True,
                    ).to(device)
    
    
    return model
