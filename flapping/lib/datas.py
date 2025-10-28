"""
Functions for loading data

"""
#import h5py
import numpy as np
import scipy.io


###############################
## beta-VAE
###############################

#---------------------------------------------------------------------
def loadData(file, printer=True):
    """
    Read flow field dataset
    
    Args: 
            file    :   (str) Path of database

            printer :   (bool) print the summary of datasets
    
    Returns:

            u_scaled:   (NumpyArray) The scaled data
            
            mean    :   (float) mean of data 
            
            std     :   (float) std of data 

    """

    w = scipy.io.loadmat(file)['W']
    para = scipy.io.loadmat(file)['para']

    if printer:
        print(f'INFO: successfully load data {file}!')
        print('w: ', w.shape)
        print('para: ', para.shape)

    return w, para


#---------------------------------------------------------------------
def get_pvae_DataLoader(d1_train, d2_train, train_index, test_index, para_dim, n_train, device, batch_size):
    """
    make tensor data loader for training

    Args:
        d_train: (NumpyArray) Train DataSet 
        
        n_train  : (int) Training samples

        device  : (str) Device
        
        batch_size: (int) Batch size
        

    Return: 
        train_dl, val_dl: The train and validation DataLoader
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if  'cuda' in str(device):
        d1_tr = d1_train[train_index, :n_train, :,:].reshape(-1, 200, 200)
        d2_tr = d2_train[train_index, :n_train, :].reshape(-1, para_dim)
        d1_ts = d1_train[train_index, n_train:, :,:].reshape(-1, 200, 200)
        d2_ts = d2_train[train_index, n_train:, :].reshape(-1, para_dim)
        print(f"INFO: The train data has been reshaped to {d1_tr.shape, d2_tr.shape}")
        print(f"INFO: The test data has been reshaped to {d1_ts.shape, d2_ts.shape}")

        dataset_train = TensorDataset(torch.from_numpy(d1_tr).to(device), torch.from_numpy(d2_tr).to(device))
        dataset_test  = TensorDataset(torch.from_numpy(d1_ts).to(device), torch.from_numpy(d2_ts).to(device))
        train_dl = DataLoader(dataset=dataset_train,
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=0)
        val_dl = DataLoader(dataset=dataset_test,
                                             batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    else:
        d1_tr = d1_train[train_index, :n_train, :,:].reshape(-1, 200, 200)
        d2_tr = d2_train[train_index, :n_train, :].reshape(-1, para_dim)
        d1_ts = d1_train[train_index, n_train:, :,:].reshape(-1, 200, 200)
        d2_ts = d2_train[train_index, n_train:, :].reshape(-1, para_dim)

        dataset_train = TensorDataset(torch.from_numpy(d1_tr), torch.from_numpy(d2_tr))
        dataset_test  = TensorDataset(torch.from_numpy(d1_ts), torch.from_numpy(d2_ts))
        train_dl = DataLoader(dataset=dataset_train, batch_size=batch_size,
                                               shuffle=True, pin_memory=True, num_workers=4,
                                               persistent_workers=True)
        val_dl = DataLoader(dataset=dataset_test, batch_size=batch_size,
                                             shuffle=False, pin_memory=True, num_workers=4,
                                             persistent_workers=True)

    return train_dl, val_dl



###############################
## Temporal Prediction
###############################

#---------------------------------------------------------------------
def make_Sequence(cfg,data):
    """
    Generate time-delay sequence data 

    Args: 
        cfg: A class contain the configuration of data 
        data: A numpy array follows [Cases* Ntime, Nmode] shape

    Returns:
        X: Numpy array for Input 64
        Y: Numpy array for Output 1
    """

    from tqdm import tqdm 
    import numpy as np 

    # if len(data.shape) <=2:
    #     data    = np.expand_dims(data,0)
    data = data.reshape(-1, cfg.train_t, cfg.latent_dim)
    cases = data.shape[0]
    seqLen      = cfg.in_dim
    nSamples    = (data.shape[1]-seqLen)
    X           = np.zeros((cases, nSamples, seqLen, data.shape[-1]))
    Y           = np.zeros((cases, nSamples, cfg.next_step, data.shape[-1]))
    # Fill the input and output arrays with data
    for i in tqdm(np.arange(data.shape[0])):
        k=0
        for j in np.arange(data.shape[1]-seqLen):#- cfg.next_step):
            X[i, k] = data[i, j :j+seqLen, :]
            Y[i, k] = data[i, j+seqLen :j+seqLen+cfg.next_step, :]
            k = k+1
    # 检查 X 和 Y 是否已填充数据
    #print(Y[0,:,:,:])
    if np.any(X == 0):
        print(f"ERROR: X has not been filled with data!, nums of 0 :{np.sum(X == 0)}")
    else:
        print("INFO: X has been successfully filled with data.")

    if np.any(Y == 0):
        print(f"ERROR: Y has not been filled with data!, nums of 0 :{np.sum(Y == 0)}")
    else:
        print("INFO: Y has been successfully filled with data.")
    print(f"The training data has been generated, has shape of {X.shape, Y.shape}")
    
    print(f"And they are reshaped for later dataloder")

    return X.reshape(-1, seqLen, data.shape[-1]), Y.reshape(-1, cfg.next_step, data.shape[-1])


#---------------------------------------------------------------------
def make_DataLoader(X,y,batch_size,
                    drop_last=False,train_split = 0.8):
    """
    make tensor data loader for training

    Args:
        X: Tensor of features
        y: Tensor of target
        batch_size: Batch size
        drop_last: If drop the last batch which does not have same number of mini batch
        train_split: A ratio of train and validation split 

    Return: 
        train_dl, val_dl: The train and validation DataLoader
    """

    from torch.utils.data import DataLoader, TensorDataset,random_split
    import torch

    
    dataset = TensorDataset(X,y)

    len_d = len(dataset)
    train_size = int(train_split * len_d)
    valid_size = len_d - train_size

    train_d , val_d = random_split(dataset,[train_size, valid_size])
    
    train_dl = DataLoader(train_d,batch_size=batch_size,drop_last=drop_last,shuffle=True)
    val_dl = DataLoader(val_d,batch_size=batch_size,drop_last=drop_last,shuffle=True)
    
    return train_dl, val_dl