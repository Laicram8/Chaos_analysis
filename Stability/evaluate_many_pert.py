import  os
import  torch 
import  numpy               as np 
import  matplotlib.pyplot   as plt 
import  utils.plt_rc_setup

from    utils.NNs.RNNs  import LSTMs
from    utils.NNs.Transformer import Transformer2
from    utils.pp        import l2norm_error
from    utils.config    import LSTM         as cfg
from    torch.optim     import lr_scheduler
from    tqdm            import tqdm
from    copy            import deepcopy
from    utils.plot      import colorplate   as cc
from    utils.Solver    import LorenzSystem

device = ("cuda:0" if torch.cuda.is_available() else "cpu" )
print(f"INFO: The training will be undertaken on {device}")

base_dir        = "../data/04_lorenz_prediction/"
out_dir         = "../results/04_lorenz_prediction/"
save_fig_to     = out_dir + "fig/result/"
if not os.path.exists(save_fig_to): os.makedirs(save_fig_to)
load_model_from = base_dir + "model/"
load_test_data  = base_dir + "test_data/"


base_dir        = "data/04_lorenz_prediction/"
out_dir         = "results/04_lorenz_prediction/"

test_data_name = 'test_data/' +  "test_prediction.npz"

### IMPORT MODELS

# Load Model 
lstm_case = f"LSTM_{cfg.in_dim}in_{cfg.d_model}d_{cfg.next_step}next_{cfg.nmode}n"\
                + f"_{cfg.embed}emb_{cfg.num_layer}ly_{cfg.hidden_size}hz_{cfg.out_act}outact_{cfg.Epoch}Epoch"


LSTM = LSTMs(    cfg.nmode,nmode=cfg.nmode,
                    hidden_size=cfg.hidden_size,
                    num_layer=cfg.num_layer,
                    is_output=cfg.is_output,
                    out_dim=cfg.next_step,
                    out_act=cfg.out_act)

lstm_stat_dict = torch.load(load_model_from + "pre_" +  lstm_case + ".pt", map_location= device)

LSTM.load_state_dict(lstm_stat_dict)
LSTM.to(device)
LSTM.eval()
print("INFO The state dict of lstm has been loaded")


# Transformer
from utils.config import Attn as cfg 
self_case = f"selfAttn_{cfg.is_sparse}sparse_{cfg.offset}off_{cfg.in_dim}in_{cfg.d_model}d_{cfg.next_step}next_{cfg.nmode}n"\
                + f"_{cfg.embed}emb_{cfg.num_head}h_{cfg.num_block}nb_{cfg.proj_dim}ff_{cfg.act_proj}ffact_{cfg.out_act}outact_{cfg.Epoch}Epoch"

self_stat_dict = torch.load(load_model_from + "pre_" +  self_case + '.pt', map_location= device)
SELF  = Transformer2(
                    cfg.in_dim,cfg.d_model,cfg.nmode,cfg.num_head,"self",
                    is_sparse=cfg.is_sparse, offset= cfg.offset,
                    embed=cfg.embed,num_block=cfg.num_block,is_res_attn=cfg.is_res_attn,
                    proj_dim=cfg.proj_dim,act_proj=cfg.act_proj,
                    is_res_proj=cfg.is_res_proj,
                    is_output=cfg.is_output,out_dim=cfg.next_step,out_act=cfg.out_act          
                        )

SELF.load_state_dict(self_stat_dict)
SELF.to(device)
SELF.eval()
print("INFO The state dict of self has been loaded")

sparse_case = f"easyAttn_{True}sparse_{cfg.offset}off_{cfg.in_dim}in_{cfg.d_model}d_{cfg.next_step}next_{cfg.nmode}n"\
                + f"_{cfg.embed}emb_{cfg.num_head}h_{cfg.num_block}nb_{cfg.proj_dim}ff_{cfg.act_proj}ffact_{cfg.out_act}outact_{cfg.Epoch}Epoch"
sparse_stat_dict = torch.load(load_model_from + "pre_" +  sparse_case + '.pt', map_location= device)

SPARSE  = Transformer2(
                    cfg.in_dim,cfg.d_model,cfg.nmode,cfg.num_head,"easy",
                    True, offset= cfg.offset,
                    embed=cfg.embed,num_block=cfg.num_block,is_res_attn=cfg.is_res_attn,
                    proj_dim=cfg.proj_dim,act_proj=cfg.act_proj,
                    is_res_proj=cfg.is_res_proj,
                    is_output=cfg.is_output,out_dim=cfg.next_step,out_act=cfg.out_act          
                        )

SPARSE.load_state_dict(sparse_stat_dict)
SPARSE.to(device)
SPARSE.eval()
print("INFO The state dict of sparse has been loaded")

easy_case = f"easyAttn_{False}sparse_{cfg.offset}off_{cfg.in_dim}in_{cfg.d_model}d_{cfg.next_step}next_{cfg.nmode}n"\
                + f"_{cfg.embed}emb_{cfg.num_head}h_{cfg.num_block}nb_{cfg.proj_dim}ff_{cfg.act_proj}ffact_{cfg.out_act}outact_{cfg.Epoch}Epoch"


easy_stat_dict = torch.load(load_model_from + "pre_" +  easy_case + '.pt', map_location= device)

EASY  = Transformer2(
                    cfg.in_dim,cfg.d_model,cfg.nmode,cfg.num_head,"easy",
                    False, offset= cfg.offset,
                    embed=cfg.embed,num_block=cfg.num_block,is_res_attn=cfg.is_res_attn,
                    proj_dim=cfg.proj_dim,act_proj=cfg.act_proj,
                    is_res_proj=cfg.is_res_proj,
                    is_output=cfg.is_output,out_dim=cfg.next_step,out_act=cfg.out_act          
                        )

EASY.load_state_dict(easy_stat_dict)
EASY.to(device)
EASY.eval()
print("INFO The state dict of easy has been loaded")


## We GENERATE N_test timeseries for further testing and also perturbed ones

np.random.seed(1500)
N_test = 100
perturb_start = 4 # timestep 400
predict_start = 5
pertur_norm = 2e-6 # each coorwise (normal dist)

init = 6
init_x = init + np.random.randn(N_test)
init_y = init + np.random.randn(N_test)
init_z = init + np.random.randn(N_test)
slover = LorenzSystem()

X = []; Y = []; Z = []
Xp = []; Yp = []; Zp = []
for i in tqdm(range(len(init_x))):
    
    x0, y0, z0 = slover.Solve(init_state = [init_x[i],init_y[i],init_z[i]],
                         end_time = perturb_start, time_res = 0.01)
    
    x1, y1, z1 = slover.Solve(init_state =
                              [x0[-1],
                               y0[-1],
                               z0[-1]],
                         end_time = 100 - perturb_start + 0.01, time_res = 0.01)
    
    # Perturbated
    x2, y2, z2 = slover.Solve(init_state =
                              [x0[-1] + pertur_norm * np.random.normal(),
                               y0[-1] + pertur_norm * np.random.normal(),
                               z0[-1] + pertur_norm * np.random.normal()],
                         end_time = 100 - perturb_start + 0.01, time_res = 0.01)
    
    x = np.concatenate((x0[:-1], x1), axis = 0)
    y = np.concatenate((y0[:-1], x1), axis = 0)
    z = np.concatenate((z0[:-1], x1), axis = 0)
    
    xp = np.concatenate((x0[:-1], x2), axis = 0)
    yp = np.concatenate((y0[:-1], x2), axis = 0)
    zp = np.concatenate((z0[:-1], x2), axis = 0)
    
    X.append(x)
    Y.append(y)
    Z.append(z)
    
    Xp.append(xp)
    Yp.append(yp)
    Zp.append(zp)

X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)
test_data = np.stack([X,Y,Z],axis=-1)

Xp = np.array(Xp)
Yp = np.array(Yp)
Zp = np.array(Zp)
test_data_p = np.stack([Xp,Yp,Zp],axis=-1)

print(f"INFO: Test data generated, shape = {test_data.shape} and perturbed: {test_data_p.shape}")
seq_len = np.max([test_data.shape[0],test_data.shape[1]])


## Generate prediction

preds = deepcopy(test_data)
for i in tqdm(range(int(predict_start*100),
                    seq_len - cfg.next_step)):
    feature = preds[:,i-cfg.in_dim:i,:]
    x = torch.from_numpy(feature)
    x = x.float().to(device)
    pred = LSTM(x)
    pred = pred.cpu().detach().numpy()
    preds[:,i+cfg.next_step,:] = pred[:,0,:]
lstm_p = preds

predsp = deepcopy(test_data_p)
for i in tqdm(range(int(predict_start*100),
                    seq_len-cfg.next_step)):
    feature = predsp[:,i-cfg.in_dim:i,:]
    x = torch.from_numpy(feature)
    x = x.float().to(device)
    predp = LSTM(x)
    predp = predp.cpu().detach().numpy()
    predsp[:,i+cfg.next_step,:] = predp[:,0,:]
lstm_pp = predsp

preds = deepcopy(test_data)
for i in tqdm(range(int(predict_start*100),
                    seq_len-cfg.next_step)):
    feature = preds[:,i-cfg.in_dim:i,:]
    x = torch.from_numpy(feature)
    x = x.float().to(device)
    pred = EASY(x)
    pred = pred.cpu().detach().numpy()
    preds[:,i+cfg.next_step,:] = pred[:,0,:]
easy_p = preds

predsp = deepcopy(test_data_p)
for i in tqdm(range(int(predict_start*100),
                    seq_len-cfg.next_step)):
    feature = predsp[:,i-cfg.in_dim:i,:]
    x = torch.from_numpy(feature)
    x = x.float().to(device)
    predp = EASY(x)
    predp = predp.cpu().detach().numpy()
    predsp[:,i+cfg.next_step,:] = predp[:,0,:]
easy_pp = predsp

preds = deepcopy(test_data)
for i in tqdm(range(int(predict_start*100),
                    seq_len-cfg.next_step)):
    feature = preds[:,i-cfg.in_dim:i,:]
    x = torch.from_numpy(feature)
    x = x.float().to(device)
    pred = SPARSE(x)
    pred = pred.cpu().detach().numpy()
    preds[:,i+cfg.next_step,:] = pred[:,0,:]
sparse_p = preds

predsp = deepcopy(test_data_p)
for i in tqdm(range(int(predict_start*100),
                    seq_len-cfg.next_step)):
    feature = predsp[:,i-cfg.in_dim:i,:]
    x = torch.from_numpy(feature)
    x = x.float().to(device)
    predp = SPARSE(x)
    predp = predp.cpu().detach().numpy()
    predsp[:,i+cfg.next_step,:] = predp[:,0,:]
sparse_pp = predsp

preds = deepcopy(test_data)
for i in tqdm(range(int(predict_start*100),
                    seq_len-cfg.next_step)):
    feature = preds[:,i-cfg.in_dim:i,:]
    x = torch.from_numpy(feature)
    x = x.float().to(device)
    pred = SELF(x)
    pred = pred.cpu().detach().numpy()
    preds[:,i+cfg.next_step,:] = pred[:,0,:]
self_p = preds

predsp = deepcopy(test_data_p)
for i in tqdm(range(int(predict_start*100),
                    seq_len-cfg.next_step)):
    feature = predsp[:,i-cfg.in_dim:i,:]
    x = torch.from_numpy(feature)
    x = x.float().to(device)
    predp = SELF(x)
    predp = predp.cpu().detach().numpy()
    predsp[:,i+cfg.next_step,:] = predp[:,0,:]
self_pp = predsp

del preds, predsp

test_data = [   
    test_data,
    test_data,
    test_data,
    test_data
]
test_data_p = [   
    test_data_p,
    test_data_p,
    test_data_p,
    test_data_p
]
preds = [   
    easy_p,
    sparse_p,
    self_p,
    lstm_p
]
predsp = [   
    easy_pp,
    sparse_pp,
    self_pp,
    lstm_pp
]
colors =[
    cc.red,
    cc.yellow,
    cc.blue,
    cc.cyan,
]
labels= [
    "Ground Truth", 
    "Easy-Attn",
    "Sparse-Easy",
    "Self-Attn",
    "LSTM"
]

np.savez_compressed(
    load_test_data + "test_prediction_many_pert.npz",
    test_data = test_data,
    test_data_p = test_data_p,
    preds = preds,
    predsp = predsp,
    colors = colors,
    labels = labels
)
