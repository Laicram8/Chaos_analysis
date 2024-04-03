"""
Visualisation of Lyapunov Exponent 
@yuningw
"""
import os
import numpy as np 
from utils.NNs.RNNs import LSTMs
from utils.NNs.Transformer import Transformer2
from utils.pp import l2norm_error
from utils.config import LSTM as cfg
import matplotlib.pyplot as plt 
import utils.plt_rc_setup
from tqdm import tqdm
from copy import deepcopy
from utils.plot import colorplate as cc
import argparse
import scipy.io as sio 

"""
base_dir        = ""
out_dir         = "../results/04_lorenz_prediction/"
save_fig_to     = out_dir + "fig/result/"
if not os.path.exists(save_fig_to): os.makedirs(save_fig_to)
load_model_from = base_dir + "model/"
load_test_data  = base_dir + "test_data/"


base_dir        = "data/04_lorenz_prediction/"
out_dir         = "results/04_lorenz_prediction/"

test_data_name = 'test_data/' +  "test_prediction.npz"
"""
base_dir        = "../data/04_lorenz_prediction/"
out_dir         = "../results/04_lorenz_prediction/"
save_fig_to     = out_dir + "fig/result/"
load_test_data  = base_dir + "test_data/"

def Vis_Lyap(data,
             datap,
             fig = None,
             axs = None,
             color = None,
            ):

    if fig==None:
        fig, axs = plt.subplots(1,1,figsize=(16,6))

    perturb_step = 400 + 1
    fit_step     = [500, 1200]
    plot_end    = 3000
    
    dif     = np.linalg.norm(data - datap, axis = -1)
    dif     = np.mean(dif, axis = 0)
    logdif  = np.log(dif)
    
    # find last fit_step
    index = np.argmax(dif > 1)
    fit_step[1] = index if dif[index] > 1 else np.argmax(dif)
    
    ts          = np.arange(0,10000)
    tss         = np.arange(fit_step[0], fit_step[1])/100
    logdif      = logdif[fit_step[0]:fit_step[1]]
    
    dif[:perturb_step] = np.nan
    dif[:perturb_step] = np.nan

    slope_p = np.polyfit(tss, logdif, 1)
    
    ts       = np.linspace(0,100, int(100/0.01))
    
    axs.plot(ts[perturb_step:plot_end],
             dif[perturb_step:plot_end],
             lw=2, c=color, alpha = 0.8)
    axs.plot(tss, 
             np.exp(slope_p[1]+ slope_p[0]*tss),
             '--', c=color)
    
    return fig, axs, slope_p

if __name__ == '__main__':
    
    # Load data
    data = np.load(f"{load_test_data}/test_prediction_many_pert.npz")
    test_data = data['test_data'][0]
    test_data_p = data['test_data_p'][0]
    preds  = data['preds']
    predsp = data['predsp']
    colors = data['colors']
    labels = data['labels']
    
    fig, axs, lp = Vis_Lyap(test_data,
                            test_data_p,
                            fig=None,
                            axs=None,
                            color=cc.black
                            )  
    axs.text(20.00, 3.5**(-2), f'$\lambda_{{ODE}} =\ $'+str(round(lp[0],3)), color=cc.black, fontsize=20)   

    for i in range(4):
        
        fig, axs, lp = Vis_Lyap(preds[i],
                                predsp[i],
                                fig=fig,
                                axs=axs,
                                color=colors[i]
                                )   
        axs.text(20.00, 3.5**(-i-3), f"$\lambda_{{{labels[i+1]}}} =\ $" +str(round(lp[0],3)), color=colors[i], fontsize=20)   
    
    axs.set(
        xlim=(-0.10,35.00), 
        yscale='log', 
        ylim=(1e-6,1e2))
    axs.set_xlabel('$\delta t$',fontsize=30)
    axs.set_ylabel('$|\delta \mathbf{A}(t)|$',fontsize=30)

    fig.savefig(f'{save_fig_to}Lyap_Lorenz.pdf',bbox_inches='tight',dpi=500)
    fig.savefig(f'{save_fig_to}Lyap_Lorenz.jpg',bbox_inches='tight',dpi=500)
