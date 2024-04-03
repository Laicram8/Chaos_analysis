###############################
Sanchis-Agudo stats
###################
def l2norm_relative_error(g,p):
    """
    L2-norm relative error 

    Args:
        g   :   Gound Truth 
        p   :   Predictions

    returns :   Error
    """
    import numpy as np 
    error = np.linalg.norm((g-p),   axis=0)\
            /np.linalg.norm((g),    axis=0)
    
    return error.mean()*100


#-------------------------------------
def MSE(g,p,if_normalise=True):
    """Compute the MSE"""
    import numpy as np 

    if if_normalise:
        error = np.mean(((g-p)/g)**2)
    else:
        error = np.mean((g-p)**2)
    return error


#----------------------------------------
def hist_PDF(data,
            varname,
            fig=None,
            axs = None,
            color = None):
    """
    Plot the histogram PDF 
    """
    import numpy as np 
    import matplotlib.pyplot as plt 
    from utils import colorplate as cc 

    # std_dev = np.std(data)
    # n       = len(data)
    # bin_width = 3.5 * std_dev * n**(-1/3)
    # num_bins = int(np.ceil((np.max(data) - np.min(data)) / bin_width))
    
    num_bins  = 10
    hist, bins = np.histogram(data-data.mean(),bins=num_bins,density=True)
    
    if fig==None or axs == None:
        fig, axs = plt.subplots(1,1,figsize=(6,4),sharex=True)
    
    if color == None:
        color = cc.blue
        alpha = 1
    else:
        alpha  = 0.8
    axs.stairs(hist,bins,
            fill=True,
            alpha=alpha,
            color=color)
    axs.set_xlabel(varname,fontsize=16)
    axs.set_ylabel('PDF',fontsize=16)

    return fig, axs 

def Zero_Cross(data,postive_dir = True):    
    """
    Function to find the cross section betwee positive and negative of a 1D Vector
    Args:
        data        : 1D numpy arrary, object data
        postive_dir : bool, Find the positive direction or negative
    
    Returns:
        cross_idx   : Indices of the cross-section points 
        x_values    : Estimation of the position of the zero crossing 
                        within the interval between crossings_idx 
                        and crossings_idx_next

    """
    import numpy as np
    zero_crossings = np.where( np.diff(np.signbit(data)) )
    if (postive_dir):
        wherePos = np.where( np.diff(data) > 0 )
    else:
        wherePos = np.where( np.diff(data) < 0 )

    cross_idx      = np.intersect1d(zero_crossings, wherePos)
    cross_idx_next = cross_idx +1

    x_values       = cross_idx - data[list(cross_idx)]/\
                                        (data[list(cross_idx_next)] - data[list(cross_idx)])


    return cross_idx, x_values



#--------------------------------------------------------
def Intersection(data,planeNo = 0,postive_dir = True):
    """
    Compute the intersections of time-series data w.r.t each temporal mode

    Args:
        data        :   A 2D numpy array has shape of [Ntime, Nmodes]
        planeNo     :   Integar, the No. of plane to compute the intersections
        postive_dir :   bool, choose which direction     

    Returns:
        InterSec    : The intersection data in numpy array
    """
    import numpy as np 
    import sys
    if len(data.shape) !=2:
        print("The data should have 2 dimensions")
        sys.exit()

    SeqLen, Nmode               = data.shape[0],data.shape[-1]
    zero_cross, x_value = Zero_Cross(data        = data[:,planeNo], 
                                    postive_dir = postive_dir)

    # Create InterSec to store the results
    InterSec = np.zeros((zero_cross.shape[0],Nmode))
    for mode in range(0,Nmode):
        InterSec[:,mode] = np.interp(x_value, np.arange(SeqLen), data[:,mode])
        

    return InterSec , x_values



#--------------------------------------------------------
def PDFfigure(x, y, N = 1, fig = None, axs = None, color = 'k', levels = 5):
    
    import scipy.stats as st
    import numpy as np

    xmin = -50
    xmax =  50
    ymin = -50
    ymax =  50

    if fig is None:
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))

        xx, yy = np.mgrid[xmin:xmax:N, ymin:ymax:N]

        positions = np.vstack([xx.ravel(), yy.ravel()])
        values    = np.vstack([x, y])
        kernel    = st.gaussian_kde(values)
        f         = np.reshape(kernel(positions).T, xx.shape)

        cset = axs.contour(xx, yy, f,
                       colors = color,
                       levels = levels)
        axs.set_aspect(0.5)
        axs.set_xlim(xmin, xmax)
        axs.set_ylim(ymin, ymax)

        axs.set_title('Probability density function')
        axs.set_xlabel('x', fontsize = "large")
        axs.set_ylabel('y', fontsize = "large")

    # fig.tight_layout()
    return fig, axs

#-------------------------------------------------
def PDF(InterSecX,InterSecY,
        xmin = -1,xmax = 1,x_grid = 50,
        ymin = -1,ymax = 1,y_grid = 50):

    """
    Compute the joint PDF of X and Y 
    Args:
        InterSecX   : numpy array of data 1
        InterSecY   : numpy array of data 2

        xmin, xmax, x_grid  :   The limitation of InterSecX and number of grid to be plot for contour 
        ymin, ymax, y_grid  :   The limitation of InterSecY and number of grid to be plot for contour 

    Returns:
        xx, yy: The meshgrid of InterSecX and InterSecY according to the limitation and number of grids
        pdf   : The joint pdf of InterSecX and InterSecY 
    """
    import numpy as np 
    import scipy.stats as st 
    # Create meshgrid acorrding 
    xx, yy = np.mgrid[xmin:xmax:1j*x_grid, ymin:ymax:1j*y_grid]

    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([InterSecX, InterSecY])
    kernel = st.gaussian_kde(values)
    pdf = np.reshape(kernel(positions).T, xx.shape)

    return xx,yy,pdf
#----------------------------------------------------
def param_arc_length(data, dist_units = 1) :
    import numpy as np 
    from tqdm import tqdm

    new_data = []
    dist_act   = 0
    last_point = data[0]

    for i in tqdm(range(len(data)-1)):

        last = data[i]
        next = data[i+1]
        dif  = next - last
        norm = np.linalg.norm(dif)

        while dist_act < norm:
            point_add = last + dist_act * dif/norm
            dist_act += dist_units
            new_data.append(point_add)
        dist_act -= norm
    
    return(np.array(new_data))


#--------------------------------------------------------
def vis_single_PDF(InterSec,
                planeNo,
                secNo, 
                xmin = -1,xmax = 1,x_grid = 50,
                ymin = -1,ymax = 1,y_grid = 50,
                color = None,
                fig = None, axs = None):
    """"
    Visualise a single intersection for Poincare Map 
    
    """

    import matplotlib.pyplot as plt 
    from matplotlib.lines import Line2D 
    import numpy as np 
    from lib.utils import colorplate as cc 


    if fig == None and axs ==None:
        fig, axs = plt.subplots(1,1,figsize=(6,6))
    if color == None: color = cc.black

    xx, yy = np.mgrid[xmin:xmax:x_grid, ymin:ymax:y_grid]
    axs.contour(
                    xx,
                    yy,
                    InterSec[:,planeNo],
                    colors = color,
                    levels= 5,
                    linewidths = 2.5, 
                    linestyles = 'dashed',
                    alpha = 0.8,
                    )
    
    axs.plot([0.05, 0.95], [0.95, 0.95], color=color, linewidth=2)
    axs.set_aspect(0.5)
    axs.set_xlim(-0.12, 0.12)
    axs.set_ylim(0, 0.5)

    axs.set_ylabel('$a_{}$'.format(secNo), fontsize="large")
    axs.set_title('Section '+ f'\$a_{planeNo} < 0$')
    proxy = [Line2D([0], [0], color='red', linestyle='-', linewidth=2),Line2D([0], [0], color='black', linestyle='dashed', linewidth=2) ]
    fig.legend(proxy, ["Predcition","Reference"])
    
    return fig, axs 

#--------------------------------------------------------
def vis_Joint_PDF(InterSec,
                planeNo,
                secNo1, 
                secNo2, 
                xmin = -1,xmax = 1,x_grid = 50,
                ymin = -1,ymax = 1,y_grid = 50,
                color = None,
                fig = None, axs = None):
    """"
    Visualise a single intersection for Poincare Map 
    
    """

    import matplotlib.pyplot as plt 
    from matplotlib.lines import Line2D 
    import numpy as np 
    from lib.utils import colorplate as cc 


    if fig == None and axs ==None:
        fig, axs = plt.subplots(1,1,figsize=(6,6))
    if color == None: color = cc.black

    
    
    xx, yy, pdf = PDF(InterSec[:,secNo1],
                      InterSec[:,secNo2],
                        xmin,xmax,x_grid,
                        ymin,ymax,y_grid)
    
    axs.contour(xx,yy,pdf,
                colos = color)

    axs.plot([0.05, 0.95], [0.95, 0.95], color=color, linewidth=2)
    axs.set_aspect(0.5)
    axs.set_xlim(-0.12, 0.12)
    axs.set_ylim(0, 0.5)

    axs.set_ylabel('$a_{}$'.format(secNo2), fontsize="large")
    axs.set_xlabel('$a_{}$'.format(secNo1), fontsize="large")
    axs.set_title('Section '+ f'\$a_{planeNo} < 0$')
    proxy = [Line2D([0], [0], color='red', linestyle='-', linewidth=2),Line2D([0], [0], color='black', linestyle='dashed', linewidth=2) ]
    fig.legend(proxy, ["Predcition","Reference"])
    
    return fig, axs 




def Power_Specturm_Density(signals,
                        fs,
                        window_size,
                        fig = None,
                        axs = None,
                        color = None):
    """
    Apply the welch method for analysisng the PSD in the frequency domian 
    """
    from scipy.signal import welch
    import matplotlib.pyplot as plt 
    from utils import colorplate as cc
    if fig == None and axs == None: fig, axs = plt.subplots(1,1,figsize=(6,4))
    if color == None: color = cc.black

    f, Pxx_den = welch(signals, fs, nperseg=window_size)
    Pxx_den /= Pxx_den.max()
    axs.plot(f, Pxx_den,lw =2.5, c = color)

    # axs.set_xlim([0, 1])
    axs.set_xlabel(r"$Hz$")
    # axs.set_ylim([0,1200])
    annot_max(f,Pxx_den,axs)

    return fig, axs  
