"""
Initialisation and setup before running 
"""

class pathsBib: 
    
    data_path       = 'data/'
    model_path      = 'models/'; 
    res_path        = 'res/'
    fig_path        = 'figs/'
    log_path        = 'logs/'
    chekp_path      =  model_path + 'checkpoints/'
    pretrain_path   =  model_path + "pretrained/"

#-------------------------------------------------
def init_env(Re=200):
    """
    A function for initialise the path and data 

    Args:
        Re  :   The corresponding Reynolds number of the case   

    Returns:

        data_file   : (str)  File name
    
    """
    from configs.vae import VAE_config
    assert(Re == VAE_config.Re), print(f"ERROR: Please match the config of vae {VAE_config.Re}!")
    
    is_init_path = init_path()
    
    datafile = None

    if is_init_path:
        is_acquired, datafile = acquire_data(Re)
    else: 
        print(f"ERROR: Init Path failed!")
    
    return datafile


#-------------------------------------------------
def init_path():
    """
    Initialisation of all the paths 

    Returns:
        is_init_path    :   (bool) if initialise success
    """
    import os 
    from pathlib import Path
    
    is_init_path = False
    try:
        print("#"*30)
        print(f"Start initialisation of paths")
        path_list =[i for _,i in pathsBib.__dict__.items() if type(i)==str and "/" in i]
        print(path_list)
        for pth in path_list:
            # if "/" in pth:
            Path(pth).mkdir(exist_ok=True)
            print(f"INIT:\t{pth}\tDONE")
        print("#"*30)
        is_init_path = True
    except:
        print(f"Error: FAILD to inital path, Please check setup for your path!")

    return is_init_path


#-------------------------------------------------
def acquire_data(Re=100):

    """
    Acquisition of dataset from zendo
    
    Args:
        Re  :   The corresponding Reynolds number of the case   

    Returns:
        is_acquired : (bool) A flag for results 
        data_file   : (str)  File name
    """

    import urllib.request
    import os 
    import time
    is_acuqired = False
    datfile     = None
    

    
    if Re == 200:
        datfile = pathsBib.data_path + "Re200_cylinder_ALL_15_500_200_200.mat"
        if os.path.exists(datfile):
            print(f"INFO: Data file: {datfile} exists!")
        else:
            print(f"INFO: Data file: {datfile} not found!")
    else:
        print(f"Error: Re {Re} is not supported, please check the data file!")
        
    
        
        is_acuqired = True

    print("#"*30)
    
    return is_acuqired, datfile

