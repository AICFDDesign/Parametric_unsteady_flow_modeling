
"""
The Visualisation of time-series prediction results

@yuningw
"""

from lib.runners import latentRunner, PvaeRunner
from lib.pp_space import get_Ek, get_Ek_t
import matplotlib.pyplot as plt


#-----------------------------------------------------------
class colorplate:
    """
    Color in HTML format for plotting 
    """

    red     = "r"
    blue    = "b" 
    yellow  = "y" 
    cyan    = "c" 
    black   = "k" 
    orange  = "orange"

#-----------------------------------------------------------
def vis_temporal_Prediction(
                            pvae         :PvaeRunner,
                            predictor   :latentRunner,
                            model_type,
                            train_time,
                            test_time,
                            case        = 0,
                            if_loss     = False, 
                            if_evo      = False,
                            if_window   = False,
                            if_pmap_s   = False,
                            if_pmap_all = False,
                            if_reconstruc = False,
                            if_reconstruc_one_case = True,
                            if_snapshot = False
                            ):
    """
    Visualisation of the temporal-dynamics prediction results 

    Args:

        vae         : (lib.runners.vaeRunner) The module for employed VAE     
    
        predictor   : (lib.runners.latentRunner) The module for latent-space temporal-dynamics predictions

        model_type  : (str) The type of model used (easy/self/lstm)
        
        if_loss     : (bool) If plot the loss evolution 
        
        if_evo      : (bool) If plot the temporal evolution of latent mode
        
        if_window   : (bool) If plot the l2-norm error horizon
        
        if_pmap_s   : (bool) If plot the single Poincare Map
        
        if_pmap_all : (bool) If plot all the Poincare Maps

        if_snapshot : (bool) If plot the flow field reconsturction 

    """

    from  pathlib import Path
    from lib.init import pathsBib
    import numpy as np 
    import scipy.io
    from lib.pp_time import make_physical_prediction, make_physical_prediction_all_case
    from configs.vae import VAE_config as cfg
    
    figPath     = pathsBib.fig_path + model_type + '/'
    case_name   = 'Latent_results_' + predictor.filename
    datPath     = pathsBib.res_path + case_name + '.mat'
    
    print("#"*30)
    print(f"Start visualisation:\nSave Fig to:{figPath}\nLoad data from:{datPath}")
    # 'Preds':Preds, 
    # 'test_data':test_data,
    # 'Preds_gen':Preds_gen,
    # 'gen_data':gen_data,
    # 'window_error':window_error,
    # 'window_relerror':window_relerror,
    # 'window_error_gen':window_error_gen,
    # 'window_relerror_gen':window_relerror_gen,
    # 'InterSec_pred':np.array(InterSec_pred, dtype=object),
    # 'InterSec_test':np.array(InterSec_test, dtype=object),
    # 'InterSec_gen':np.array(InterSec_gen, dtype=object),
    # 'InterSec_pred_gen':np.array(InterSec_pred_gen, dtype=object),})
    try:
        d           = scipy.io.loadmat(datPath)
        test_data          = d['test_data']
        Preds              = d['Preds']
        Preds_gen          = d['Preds_gen']
        gen_data           = d['gen_data']
        window_error       = d['window_error']
        window_relerror    = d['window_relerror']
        window_error_gen   = d['window_error_gen']
        window_relerror_gen= d['window_relerror_gen']
        InterSec_pred      = d['InterSec_pred']
        InterSec_test      = d['InterSec_test']
        InterSec_gen       = d['InterSec_gen']
        InterSec_pred_gen  = d['InterSec_pred_gen']
        InterSec_pred_gen  = np.squeeze(InterSec_pred_gen)
        InterSec_pred      = np.squeeze(InterSec_pred)
        InterSec_test      = np.squeeze(InterSec_test)
        InterSec_gen       = np.squeeze(InterSec_gen)
       
        # p           = d['p']
        # e           = d['e']
        # pmap_g      = d['pmap_g']
        # pmap_p      = d['pmap_p']

    except:
        print(f"ERROR: FAILD loading data")

    Path(figPath).mkdir(exist_ok=True)

    if if_loss:
        plot_loss(predictor.history, save_file= figPath + "loss_" + case_name + '.png' )
        print(f'INFO: Loss Evolution Saved!')

    if if_evo and (test_data[case,:,:].any() !=None) and (Preds[case,:,:].any() != None):
        plot_signal(test_data[case,:,:],test_data[case,:,:],            save_file=figPath + f"test_signal_case{case}_" + case_name + '.png' )
        plot_signal3D_all(test_data,Preds,gen_data,Preds_gen, cfg.train, cfg.test, model_type = model_type,
                        save_file=figPath + f"test_signal3D_all" + case_name + '.png' )

        print(f"INFO: Test Prediction Evolution Saved!")
    if if_evo and (gen_data[case,:,:].any() !=None) and (Preds_gen[case,:,:].any() != None):
        plot_signal(gen_data[case,:,:],Preds_gen[case,:,:],            save_file=figPath + f"gen_signal_case{case}_" + case_name + '.png' )
        print(f"INFO: Gen Prediction Evolution Saved!")

    if if_window and (window_relerror[case,:].any() != None):
        # plot_pred_horizon_error(window_relerror[case,:], colorplate.blue, save_file=figPath + f"test_Rell2err_case{case}_" + case_name + '.png' )
        plot_pred_horizon_error_all(window_relerror, save_file=figPath + f"test_Rell2err_all_" + case_name + '.png', case=cfg.train)
        plot_pred_horizon_error_all(window_relerror_gen, save_file=figPath + f"gen_Rell2err_all_" + case_name + '.png', case=cfg.test)
        print(f"INFO: Relative l2-norm error Horizion Saved!")
    if if_window and (window_relerror_gen[case,:].any() != None):
        # plot_pred_horizon_error(window_relerror_gen[case,:], colorplate.red, save_file=figPath + f"gen_Rell2err_case{case}_" + case_name + '.png' )
        print(f"INFO: Gen Relative l2-norm error Horizion Saved!")

    ## Poincare Map 
    planeNo = 0
    postive_dir = True
    lim_val = 2.5  # Limitation of x and y bound when compute joint pdf
    grid_val = 50
    i = 1; j = 2 

    if if_pmap_s and (InterSec_test[case+1].any() != None) and (InterSec_pred[case+1].any() != None):
        print(InterSec_pred[case+1],InterSec_test[case+1],)
        print(InterSec_pred[case+1].shape,InterSec_test[case+1].shape)
        plotSinglePoincare( planeNo, i, j, 
                        InterSec_pred[case+1],InterSec_test[case+1],
                        lim_val, grid_val,
                        save_file = figPath + f'Test_Pmap_{i}_{j}_case{case+1}' + case_name + '.png')
        print(f"INFO: Single Poincare Map of {i}, {j} Saved!")
    
    if if_pmap_all and (InterSec_test[case].any() != None) and (InterSec_pred[case].any() != None):
        plotCompletePoincare(predictor.config.latent_dim,planeNo, 
                        InterSec_pred[case], InterSec_test[case],
                        lim_val, grid_val,
                        save_file = None, 
                        dpi       = 300)
        print(f"INFO: Complete Poincare Map Saved!")

    if if_reconstruc and (Preds[case,:,:].any() != None) and (test_data[case,:,:].any() != None):
        VAErec, pred = make_physical_prediction_all_case(vae=pvae,pred_latent=Preds,true_latent=test_data,device=pvae.device)
        VAErec_gen, pred_gen = make_physical_prediction_all_case(vae=pvae,pred_latent=Preds_gen,true_latent=gen_data,device=pvae.device)
        fname = figPath +  f"reconstruc_{predictor.filename}.mat"
        cases_te  = np.size(cfg.train)
        cases_gen = np.size(cfg.test)
        time_te  = test_data.shape[1]
        time_gen = gen_data.shape[1]
        Ecum_te = np.zeros((cases_te, time_te))
        Ecum_gen = np.zeros((cases_gen, time_gen))
        Esnap_te = np.zeros((cases_te, time_te))
        Esnap_gen = np.zeros((cases_gen, time_gen))
        for k in range(cases_te):
            for i in range(time_te):
                Ecum_te[k,i] = get_Ek(pred[k,:i+1,:,:], pvae.d1_te[k, :i+1,:,:])
                Esnap_te[k,i] = get_Ek(pred[k,i,:,:], pvae.d1_te[k,i,:,:])
        for k in range(cases_gen):
            for i in range(time_gen):
                Ecum_gen[k,i] = get_Ek(pred_gen[k,:i+1,:,:], pvae.d1_gen[k, :i+1,:,:])
                Esnap_gen[k,i] = get_Ek(pred_gen[k,i,:,:], pvae.d1_gen[k,  i,:,:])
        scipy.io.savemat(fname, 
                        {'Ecum_te':Ecum_te, 'Esnap_te':Esnap_te,
                         'Ecum_gen':Ecum_gen, 'Esnap_gen':Esnap_gen})
        print(f"INFO: Test and Gen Prediction Reconstruction Done!")
        plot_Ecum_all(Ecum_te, save_file=figPath + f"test_Ecum_all_" + case_name + '.png', case=cfg.train)
        plot_Ecum_all(Ecum_gen, save_file=figPath + f"gen_Ecum_all_" + case_name + '.png', case=cfg.test)
        
    if if_reconstruc_one_case and (Preds[case,:,:].any() != None) and (test_data[case,:,:].any() != None):
        
        VAErec_gen, pred_gen = make_physical_prediction_all_case(vae=pvae,pred_latent=Preds_gen,true_latent=gen_data,device=pvae.device)
        fname = figPath +  f"One_GenCase_reconstruc_{predictor.filename}.mat"
        k=2
        Rec_from_VAE_T_gen = pred_gen[k,:,:,:]
        scipy.io.savemat(fname, 
                        {'Rec_from_VAE_T_gen':Rec_from_VAE_T_gen, })
        print(f"INFO:Gen Prediction Reconstruction at Case {cfg.test[k]+1} Done!")

        # VAErec_test, pred_test = make_physical_prediction_all_case(vae=pvae,pred_latent=Preds,true_latent=test_data,device=pvae.device)
        # fname = figPath +  f"One_TestCase_reconstruc_{predictor.filename}.mat"
        # k=3
        # Rec_from_VAE_T_test = pred_test[k,:,:,:]
        # scipy.io.savemat(fname, 
        #                 {'Rec_from_VAE_T_test':Rec_from_VAE_T_test, })
        # print(f"INFO:Test Prediction Reconstruction at Case {cfg.train[k]+1} Done!")


        

    if if_snapshot and (Preds[case,:,:].any() != None) and (test_data[case,:,:].any() != None):
        cases_te = Preds.shape[0]
        for case in range(cases_te): 
            stepPlot     = int(predictor.config.in_dim + 1) # Here we test the prediction purely based on the predicted variables 
            VAErec, pred = make_physical_prediction(vae=pvae,pred_latent=Preds[case,:stepPlot+1,:],true_latent=test_data[case,:stepPlot+1,:],device=pvae.device)

            test_w = pvae.d1_te[case,:stepPlot+1,:,:].reshape(-1, 1, 200, 200)
            print(test_w.shape, VAErec.shape, pred.shape)
            
            predFieldFigure(test_w,VAErec,pred,
                            stepPlot  = stepPlot,
                            model_name= model_type,
                            save_file = figPath + f"test_recSnapShot_case{cfg.train[case]+1}_" + case_name + '.png')
            print(f"INFO: Case {cfg.train[case]+1}: Test Reconstruted Snapshot at {stepPlot} Saved!")

        cases_gen = Preds_gen.shape[0]
        for case in range(cases_gen):
            VAErec_gen, pred_gen = make_physical_prediction(vae=pvae,pred_latent=Preds_gen[case,:stepPlot+1,:],true_latent=gen_data[case,:stepPlot+1,:],device=pvae.device)
            
            gen_w = pvae.d1_gen[case,:stepPlot+1,:,:].reshape(-1, 1, 200, 200)
            print(gen_w.shape, VAErec_gen.shape, pred_gen.shape)
            
            predFieldFigure(gen_w,VAErec_gen,pred_gen,
                            stepPlot  = stepPlot,
                            model_name= model_type,
                            save_file = figPath + f"gen_recSnapShot_case{cfg.test[case]+1}_" + case_name + '.png')
            print(f"INFO: Case {cfg.test[case]+1}: Gen Reconstruted Snapshot at {stepPlot} Saved!")
    return
    # if if_snapshot and (Preds[case,:,:].any() != None) and (test_data[case,:,:].any() != None):
    #     VAErec, pred = make_physical_prediction(vae=pvae,pred_latent=Preds[case,:,:],true_latent=test_data[case,:,:],device=pvae.device)

        
    #     stepPlot     = int(predictor.config.in_dim + 1) # Here we test the prediction purely based on the predicted variables 

    #     test_w = pvae.d1_te[case,:,:,:].reshape(-1, 1, 200, 200)
    #     print(test_w.shape, VAErec.shape, pred.shape)
        
    #     predFieldFigure(test_w,VAErec,pred,
    #                     stepPlot  = stepPlot,
    #                     model_name= model_type,
    #                     save_file = figPath + f"test_recSnapShot_case{case}_" + case_name + '.png')
    #     print(f"INFO: Test Reconstruted Snapshot at {stepPlot} Saved!")

    #     VAErec_gen, pred_gen = make_physical_prediction(vae=pvae,pred_latent=Preds_gen[case,:,:],true_latent=gen_data[case,:,:],device=pvae.device)
        
    #     gen_w = pvae.d1_gen[case,:,:,:].reshape(-1, 1, 200, 200)
    #     print(gen_w.shape, VAErec_gen.shape, pred_gen.shape)
        
    #     predFieldFigure(gen_w,VAErec_gen,pred_gen,
    #                     stepPlot  = stepPlot,
    #                     model_name= model_type,
    #                     save_file = figPath + f"gen_recSnapShot_case{case}_" + case_name + '.png')
    #     print(f"INFO: Gen Reconstruted Snapshot at {stepPlot} Saved!")
    # return


#-----------------------------------------------------------
def plot_loss(history, save_file, dpi = 300):
    """
    Plot the loss evolution during training
    Args: 
        history     : A dictionary contains loss evolution in list
        save_file   : Path to save the figure, if None, then just show the plot
        dpi         : The dpi for save the file 

    Returns:
        A fig for loss 

    """
    ##import matplotlib.pyplot as plt 
    import utils.plt_rc_setup
    from utils.figs_time import colorplate as cc 

    fig, axs = plt.subplots(1,1)
    
    axs.semilogy(history["train_loss"], c = cc.blue)
    if len(history["val_loss"]) != 0 :
        axs.semilogy(history["val_loss"], c = cc.orange)
    axs.set_xlabel("Epoch")
    axs.set_ylabel("MSE Loss") 
    axs.legend(["Train Loss","Validation Loss"])
    
    if save_file != None:
        plt.savefig(save_file, bbox_inches='tight', dpi= dpi)


#-----------------------------------------------------------
def plot_signal(test_data, Preds, save_file:None, dpi = 300):
    """
    Plot the temproal evolution of prediction and test data
    Args: 
        test_data   : A numpy array of test data 
        Preds       : A numpy array of prediction data
        
        save_file   : Path to save the figure, if None, then just show the plot
        dpi         : The dpi for save the file 

    Returns:
        A fig for temporal dynamic of ground truth and predictions on test data

    """
    import sys
    
    try: 
        test_data.shape == Preds.shape 
    except:
        print("The prediction and test data must have same shape!")
        sys.exit()
    
    ##import matplotlib.pyplot as plt 
    import utils.plt_rc_setup
    from utils.figs_time import colorplate as cc 

    Nmode = min(test_data.shape[0],test_data.shape[-1])

    fig, axs = plt.subplots(Nmode,1,sharex=True)#,figsize=(16,2.5* Nmode)
    
    for i, ax in enumerate(axs):
        ax.plot(test_data[:,i],c = cc.black)
        ax.plot(Preds[:,i],c = cc.blue)
        ax.set_ylabel(f"Mode {i+1}")
        axs[-1].set_xlabel("t")

    ax.legend(["Ground truth",'Prediction'])
    
    if save_file != None:
        plt.savefig(save_file, bbox_inches='tight', dpi= dpi)

def plot_signal3D_all(test_data, Preds,gen_data,Preds_gen, case_tr, case_gen, model_type, save_file:None):
    import numpy as np

    nn_type = model_type
    
    latent_dim = test_data.shape[-1]
    Cases_tr = np.size(case_tr)
    Cases_gen = np.size(case_gen)
    print(f"cases_tr = {Cases_tr}, cases_gen = {Cases_gen}")
    all_cases = np.concatenate([case_tr, case_gen])
    all_types = np.array(['tr'] * len(case_tr) + ['gen'] * len(case_gen))
    sort_idx = np.argsort(all_cases)
    all_cases_sorted = all_cases[sort_idx]
    all_types_sorted = all_types[sort_idx]

    fig, axes = plt.subplots(3, 5, figsize=(20, 12), subplot_kw={'projection': '3d'})
    axes = axes.flatten()

    tr_idx = 0
    gen_idx = 0
    for i, (case, typ) in enumerate(zip(all_cases_sorted, all_types_sorted)):
        ax = axes[i]
        if typ == 'tr':
            # 找到case在case_tr中的索引
            idx = np.where(case_tr == case)[0][0]
            data = test_data[idx, :, :]
            title = f"Test CASE {case+1}"
            data_pred = Preds[idx, :, :]
        else:
            idx = np.where(case_gen == case)[0][0]
            data = gen_data[idx, :, :]
            title = f"Gen. CASE {case+1}"
            data_pred = Preds_gen[idx, :, :]
        label = 'True'
        label_pred = nn_type
        ax.plot(data[:, 0], data[:, 1], data[:, 2], label=label)
        ax.plot(data_pred[:, 0], data_pred[:, 1], data_pred[:, 2], label=label_pred, linestyle='--')
        ax.set_title(title)
        ax.set_xlabel("m1")
        ax.set_ylabel("m2")
        ax.set_zlabel("m3")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.set_zticks([-2, -1, 0, 1, 2])
        ax.legend()
        
    # 隐藏多余的子图
    for j in range(len(all_cases_sorted), 15):
        fig.delaxes(axes[j])
    if save_file != None:
        plt.savefig(save_file, bbox_inches='tight', dpi= 300)


#-----------------------------------------------------------

def plot_pred_horizon_error(window_err, Color, save_file, dpi=300):
    """
    Viusalize the latent-space prediction horizon error 

    Args:
    
        window_err      :   (NumpyArray) The horizon of l2-norm error of  prediction 

        Color           :   (str) The color for the line 

        save_file   : Path to save the figure, if None, then just show the plot

        dpi         : The dpi for save the file 
        
    """
    #import matplotlib.pyplot as plt 

    fig, axs = plt.subplots(1,1)

    axs.plot(window_err*100, c = Color)
    axs.set_xlabel("Prediction steps")
    axs.set_ylabel(r"$\epsilon$(%)")  

    if save_file != None:
        plt.savefig(save_file, bbox_inches='tight', dpi= dpi)

    return 
def plot_pred_horizon_error_all(window_err, save_file, case, dpi=300):
    """
    Viusalize the latent-space prediction horizon error 

    Args:
    
        window_err      :   (NumpyArray) The horizon of l2-norm error of  prediction 

        Color           :   (str) The color for the line 

        save_file   : Path to save the figure, if None, then just show the plot

        dpi         : The dpi for save the file 
        
    """
    #import matplotlib.pyplot as plt 

    fig, axs = plt.subplots(1,1)
    num = window_err.shape[0]
    for i in range(num):
        axs.plot(window_err[i,:]*100, label=f"Case {case[i]+1}", linewidth=1.5)
        axs.set_xlabel("Prediction steps")
        axs.set_ylabel(r"$\epsilon$(%)")
        axs.legend(loc='upper left', bbox_to_anchor=(1, 1))
        axs.grid(visible=True, markevery=1, color='gainsboro', zorder=1)  

    if save_file != None:
        plt.savefig(save_file, bbox_inches='tight', dpi= dpi)

    return 

def plot_Ecum_all(Ecum, save_file, case, dpi=300):
    """
    Viusalize the latent-space prediction horizon error 

    Args:
    
        window_err      :   (NumpyArray) The horizon of l2-norm error of  prediction 

        Color           :   (str) The color for the line 

        save_file   : Path to save the figure, if None, then just show the plot

        dpi         : The dpi for save the file 
        
    """
    #import matplotlib.pyplot as plt 

    fig, axs = plt.subplots(1,1)
    num = Ecum.shape[0]
    for i in range(num):
        axs.plot(Ecum[i,:], label=f"Case {case[i]+1}", linewidth=1.5)
        axs.set_xlabel("Prediction steps")
        axs.set_ylabel("Cumulative E")
        axs.legend(loc='upper left', bbox_to_anchor=(1, 1))
        axs.grid(visible=True, markevery=1, color='gainsboro', zorder=1)  

    if save_file != None:
        plt.savefig(save_file, bbox_inches='tight', dpi= dpi)

    return 

#-----------------------------------------------------------
def plotSinglePoincare( planeNo, i, j, 
                        InterSec_pred,InterSec_test,
                        lim_val, grid_val,
                        save_file:None, 
                        dpi = 300):
    """
    
    Visualisation of a single Poincare Map for test data and prediction
    
    Args: 

        planeNo     :   (int) The plane no to compute the intersection 

        i           :   (int) The Number of the mode on x-Axis

        j           :   (int) The Number of the mode on y-Axis

        lim_val     :   (float) Limitation of region on the map

        grid_val    :   (int) Number of the mesh grid 

        save_file   :   (str) Path to save the file 

        dpi         :   (int) The dpi for the image 

    """
    #import matplotlib.pyplot as plt 
    import utils.plt_rc_setup
    from utils.figs_time import colorplate as cc 
    from lib.pp_time import PDF

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))

    _, _, pdf_test = PDF(InterSecX=InterSec_test[:, i],
                        InterSecY=InterSec_test[:, j],
                        xmin=-lim_val, xmax=lim_val,
                        ymin=-lim_val, ymax=lim_val,
                        x_grid=grid_val, y_grid=grid_val,
                        )

    xx, yy, pdf_pred = PDF(InterSecX=InterSec_pred[:, i],
                        InterSecY=InterSec_pred[:, j],
                        xmin=-lim_val, xmax=lim_val,
                        ymin=-lim_val, ymax=lim_val,
                        x_grid=grid_val, y_grid=grid_val,
                        )

    axs.contour(xx, yy, pdf_test, colors=cc.black)
    axs.contour(xx, yy, pdf_pred, colors='lightseagreen')
    axs.set_xlim(-lim_val, lim_val)
    axs.text(0.80, 0.08, '$r_{}=0$'.format(planeNo+1),
                transform=axs.transAxes, bbox=dict(facecolor='white', alpha=0.4))
    axs.set_xlabel(f"$r_{i + 1}$", fontsize='large')
    axs.set_ylabel(f"$r_{j + 1}$", fontsize='large')
    axs.set_aspect('equal', "box")
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.grid(visible=True, markevery=1, color='gainsboro', zorder=1)
    if save_file != None:
        plt.savefig(save_file, bbox_inches = 'tight', dpi = dpi)


    return 


#-----------------------------------------------------------
def plotCompletePoincare(Nmodes,planeNo, 
                        InterSec_pred, InterSec_test,
                        lim_val, grid_val,
                        save_file = None, 
                        dpi       = 300):
    """

    Visualisation of whole Poincare Maps for test data and prediction
    

    Args: 

        planeNo     :   (int) The plane no to compute the intersection 

        lim_val     :   (float) Limitation of region on the map

        grid_val    :   (int) Number of the mesh grid 

        save_file   :   (str) Path to save the file 

        dpi         :   (int) The dpi for the image 
        
    
    """    
    #import matplotlib.pyplot as plt 
    import utils.plt_rc_setup
    from utils.figs_time import colorplate as cc 
    from lib.pp_time import PDF


    fig, axs = plt.subplots(Nmodes, Nmodes,
                            figsize=(5 * Nmodes, 5 * Nmodes),
                            sharex=True, sharey=True)

    for i in range(0, Nmodes):
        for j in range(0, Nmodes):
            if i == j or j == planeNo or i == planeNo or j > i:
                axs[i, j].set_visible(False)
                continue

            _, _, pdf_test = PDF(InterSecX=InterSec_test[:, i],
                                InterSecY=InterSec_test[:, j],
                                xmin=-lim_val, xmax=lim_val,
                                ymin=-lim_val, ymax=lim_val,
                                x_grid=grid_val, y_grid=grid_val,
                                )

            xx, yy, pdf_pred = PDF(InterSecX=InterSec_pred[:, i],
                                InterSecY=InterSec_pred[:, j],
                                xmin=-lim_val, xmax=lim_val,
                                ymin=-lim_val, ymax=lim_val,
                                x_grid=grid_val, y_grid=grid_val,
                                )

            axs[i, j].contour(xx, yy, pdf_test, colors=cc.black)
            axs[i, j].contour(xx, yy, pdf_pred, colors='lightseagreen')
            axs[i, j].set_xlim(-lim_val, lim_val)
            axs[i, j].set_xlabel(f"$r_{i + 1}$", fontsize='large')
            axs[i, j].set_ylabel(f"$r_{j + 1}$", fontsize='large')
            axs[i, j].set_aspect('equal', "box")
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].grid(visible=True, markevery=1, color='gainsboro', zorder=1)
    
    if save_file != None:
        plt.savefig(save_file, bbox_inches = 'tight', dpi = dpi)

    return



#-----------------------------------------------------------

def predFieldFigure(true, VAErec, pred, stepPlot, model_name, save_file, dpi=300):
    
    """
    
    Visualise the flow fields reconstructed by the latent-space prediction from the transformer/lstm 

    true        :       (NumpyArray) The ground truth 

    VAErec      :       (NumpyArray) The reconstruction from VAE ONLY 

    pred        :       (NumpyArray) The reconstruction from the prediction of transformer 

    std_data    :       (NumpyArray) Std of flow fields
    
    mean_data   :       (NumpyArray) Mean of flow fields

    model_name  :       (str) The name of the predictor model: easy/self/lstm

    save_file   :       (str) Path to save the file 

    dpi         :       (int) The dpi for the image 
        
    """

    #import matplotlib.pyplot as plt 
    from matplotlib.colors import LinearSegmentedColormap
    bwr = LinearSegmentedColormap.from_list(
        'custom_bwr', 
        ['blue', 'white', 'red']
    )
    fig, ax = plt.subplots(1, 3 ,figsize=(3*8, 1*4),sharex='col', sharey='row')

    Umax = 2.0
    Umin = -2.0
    Vlim = 1

    # From dataset
    true_u  = (true[stepPlot, 0, :, :]).squeeze()
    #true_v  = (true[stepPlot, 1, :, :] * std_data[0, 1, :, :] + mean_data[0, 1, :, :]).squeeze()
    
    vae_u   = (VAErec[stepPlot, 0, :, :]).squeeze()
    #vae_v   = (VAErec[stepPlot, 1, :, :] * std_data[0, 1, :, :] + mean_data[0, 1, :, :]).squeeze()
    
    pred_u  = (pred[stepPlot, 0, :, :]).squeeze()
    #pred_v  = (pred[stepPlot, 1, :, :] * std_data[0, 1, :, :] + mean_data[0, 1, :, :]).squeeze()
    
    im = ax[0].pcolormesh(true_u,
                        cmap=bwr, vmin=Umin, vmax=Umax)
    ax[0].set_title('True $w$')
    fig.colorbar(im, ax=ax[0], shrink=1.0, ticks=([-2, -1, 0, 1, 2]))

    

    # Encoded and decoded
    im = ax[1].pcolormesh(vae_u,
                        cmap=bwr, vmin=Umin, vmax=Umax, shading='gouraud')#, extent=[-9, 87, -14, 14])
    ax[1].set_title(r'$\beta$-VAE' + ' $w$')
    fig.colorbar(im, ax=ax[1], shrink=1.0, ticks=([-2, -1, 0, 1, 2]))



    # Encoded, predicted and decoded
    im = ax[2].pcolormesh(pred_u,
                        cmap=bwr, vmin=Umin, vmax=Umax, shading='gouraud')
    # ax[2].set_title(r'$\beta$-VAE + ' + model_name + ' $w$\n($t+$' + (str(stepPlot) if stepPlot > 1 else "") + '$t_c$)')
    ax[2].set_title(r'$\beta$-VAE + ' + model_name + ' $w$')
    fig.colorbar(im, ax=ax[2], shrink=1.0, ticks=([-2, -1, 0, 1, 2]))



    ax[0].set_xlabel('x/c')
    ax[1].set_xlabel('x/c')
    ax[2].set_xlabel('x/c')
    ax[0].set_ylabel('y/c')

    # fig.set_tight_layout(True)

    if save_file != None:
        plt.savefig(save_file, bbox_inches = 'tight', dpi = dpi)
    plt.close()

    return fig, ax
