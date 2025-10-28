"""
Post-processing and analysis algorithm for beta-VAE in latent space and physic space

"""

import torch
import numpy as np
#import h5py
from init import pathsBib

################################
### Main programme for spatial analysis
###############################
def spatial_Mode(   fname,
                    fname_predw,
                    model, 
                    latent_dim,
                    n_train,
                    n_test,
                    para_dim, 
                    train_data,
                    test_data,
                    gen_data,
                    dataset_train,
                    dataset_test,
                    dataset_gen,
                    device,
                    if_order    = True,
                    if_nlmode   = True,
                    if_Ecumt    = True,
                    if_Ek_t     = True,
                ): 
    """
    The main function for spatial mode analysis and generate the dataset 
        
    Args:

        fname           :   (str) The file name

        latent_dim      :   (int) The latent dimension 

        train_data      :   (NumpyArray) Dataset for training
        
        test_data       :   (NumpyArray) Dataset for test

        dataset_train   :   (dataloader) DataLoader for training data
        
        dataset_test    :   (dataloader) DataLoader for test data

        mean            :   (NumpyArray) The mean of flow database
        
        std             :   (NumpyArray) The std of flow database

        device          :   (str) The device for the computation
    
        order           : (NumpyArray) A array which contains the ranking results

        Ecum            : (NumpyArray) accumlative Energy obtained for each mode
    
        Ecum_test       : (NumpyArray) accumlative Energy obtained for each mode

        NLvalues        : (NumpyArray) The used range of value 

        NLmodes         : (NumpyArray) The non-linear spatial mode

        Ek_t            : (List) A list of enery of each snapshot in dataset

        if_order        : (bool) IF ranking the mode

        if_nlmode       : (bool) IF generate non-linear mode

        if_Ecumt        : (bool) IF compute accumulative energy 
        
        if_Ek_t         : (bool) IF compute evolution of energy 
    
    Returns: 

        if_save         : (bool) If successfully save file

    """
    
    print(f"INFO: Start spatial mode generating")
    if if_order:
        print(f"INFO: Start get_order")
        order, Ecum = get_order(model, latent_dim, n_train,
                                        train_data, 
                                        dataset_train, 
                                         device)
        order_all, Ecum_all = get_order_all(model, latent_dim, n_train,
                                        train_data,
                                        dataset_train, 
                                         device)
        print(f"INFO: RANKING DONE !")        
    else:
        order = None
        Ecum  = None
        order_all = None
        Ecum_all = None
    

    if if_nlmode:
        NLvalues, NLmodes = getNLmodes(model, order[0], latent_dim, device)
        print("INFO: Non-linear mode generated")
    else:
        NLmodes = None
        NLvalues = None
    
    if if_Ecumt: 
        print(f"INFO: Start get_EcumTest")
        Ecum_test = get_EcumTest(model, latent_dim, n_test, test_data, dataset_test, device, order)
        # Ecum_gen = get_EcumTest(model, latent_dim, n_train + n_test, gen_data, dataset_gen, device, order)
        Ecum_gen = Ecum_test
        print('INFO: Test E_cum generated')
    else: 
        Ecum_test = None
        Ecum_gen = None
    
    if if_Ek_t: 
        print('INFO: Start get_Ek_t')
        Ek_tr  = get_Ek_t(model=model, data=train_data, n_data=n_train, device=device)
        Ek_te  = get_Ek_t(model=model, data=test_data, n_data=n_test, device=device)
        # Ek_gen = get_Ek_t(model=model, data=gen_data, n_data=n_train + n_test, device=device)
        Ek_gen = Ek_te

    else:
        Ek_tr = None
        Ek_te = None
        Ek_gen = None
    
    is_save = createModesFile(fname, fname_predw, model, latent_dim, 
                            train_data, test_data, gen_data,
                            dataset_train, dataset_test, dataset_gen,
                            n_train, n_test, para_dim,
                            device,
                            order, Ecum, order_all, Ecum_all, 
                            Ecum_test, Ecum_gen,
                            NLvalues, NLmodes,
                            Ek_tr, Ek_te, Ek_gen,)
    
    if is_save: print("INFO: Successfuly DONE!")

    return is_save

################################
### Basic function for using VAE
###############################

#--------------------------------------------------------
def encode(model, data, device):
    """
    Use encoder to compress flow field into latent space 
    Args: 
        model       :   (nn.Module) Pytorch module for beta-VA
        
        data        :   (DataLoader) DataLoader of data to be encoded 

        device      : (str) The device for the computation

    Returns: 

        means       : (NumpyArray) The mu obtained in latent space     
        
        logvars     : (NumpyArray) The sigma obtained in latent space
    """
    mean_list = []
    logvar_list = []
    with torch.no_grad():
        for batch in data:
            #batch = batch.to(device, non_blocking=True).float()
            w, para = batch
            w , para =w.to(device, non_blocking=True).float(), para.to(device, non_blocking=True).float()
            mean, logvariance = torch.chunk(model.encoder(w)*model.paracoder(para), 2, dim=1)

            mean_list.append(mean.cpu().numpy())
            logvar_list.append(logvariance.cpu().numpy())

    means = np.concatenate(mean_list, axis=0)
    logvars = np.concatenate(logvar_list, axis=0)

    return means, logvars


#--------------------------------------------------------
def decode(model, data, device):
    """
    Use decoder to reconstruct flow field back to physical space 

    Args: 
        model       :   (nn.Module) Pytorch module for beta-VA
        
        data        :   (NumpyArray) The latent vectors required to be reconstructed 

        device      :   (str) The device for the computation

    Returns: 

        rec         :   (NumpyArray) The reconstruction of the flow fields. 

    """
    dataset = torch.utils.data.DataLoader(dataset=torch.from_numpy(data), batch_size=512,
                                        shuffle=False, num_workers=2)
    rec_list = []
    with torch.no_grad():
        for batch in dataset:
            batch = batch.to(device).float()
            rec_list.append(model.decoder(batch).cpu().numpy())

    return np.concatenate(rec_list, axis=0)



def get_samples(model, dataset_train, dataset_test, device):
    """
    
    A function for quickly obtain a restructed flow field for the propose of testing or visualisation

    We obtain snapshot through training and test data, respectively 

    Args: 
        model                :   (nn.Module) Pytorch module for beta-VA
        
        dataset_train        :   (DataLoader) DataLoader of training data 
        
        dataset_test         :   (DataLoader) DataLoader of test data 

        device               :   (str) The device for the computation

    Returns: 

        rec_train            :   (NumpyArray) The reconstruction from training dataset. 
        
        rec_test             :   (NumpyArray) The reconstruction from test dataset. 
        
        true_train           :   (NumpyArray) The corresponding ground truth from training dataset. 
        
        true_test            :   (NumpyArray) The corresponding ground truth from test dataset. 
    """
    with torch.no_grad():
        if dataset_train != None:
            for batch_train in dataset_train:
                batch_train = batch_train.to(device, non_blocking=True).float()
                rec_train, _, _ = model(batch_train)
        
                rec_train = rec_train.cpu().numpy()[-1]
                true_train = batch_train.cpu().numpy()[-1]
        
                break
        else: 
            rec_train   = None
            true_train  = None
        
        if dataset_test != None:
            for batch_test in dataset_test:
                batch_test = batch_test.to(device, non_blocking=True).float()
                rec_test, _, _ = model(batch_test)

                rec_test = rec_test.cpu().numpy()[-1]
                true_test = batch_test.cpu().numpy()[-1]

                break
        else: 
            rec_test    = None
            true_test   = None

        return rec_train, rec_test, true_train, true_test


################################
### Spatial-mode generate and analysis
###############################

#--------------------------------------------------------
def calcmode(model, latent_dim, mode, device, case, para_data):
    """
        
    Generate the non-linear mode with unit vector 

    Args: 
        model       :   (nn.Module) Pytorch module for beta-VA
        
        latent_dim  :   (int) Latent-dimension adpot for beta-VA
        
        mode        :   (int) The indice of the mode zi

    Returns: 

        mode        : (NumpyArray) The spatial mode for zi     
    """

    z_sample = np.zeros((1, latent_dim), dtype=np.float32)

    z_sample[:, mode] = 1
    para_tensor = para_data[case, 0, :].reshape(z_sample.shape[0], -1).to(device).float()
    para_coder,_ = torch.chunk(model.paracoder(para_tensor), 2, dim=1)

    with torch.no_grad():
        mode = model.decoder(torch.from_numpy(z_sample).to(device)).cpu().numpy()
        mode_front = model.decoder(torch.from_numpy(z_sample).to(device)*para_coder).cpu().numpy()
        
    return mode, mode_front


#--------------------------------------------------------
def get_spatial_modes(model, latent_dim, device, case, para_data):

    """
    Algorithm for optain the spatial mode from beta-VAE Decoder. 
    For latent variable i, We use the unit vector v (where vi = 1) as the input to obtain the spatial mode for each latent variables
    Also, we compute the spatial mode use zeros vectors as input 

    Args:

        model           : (torch.nn.Module) The beta-VAE model 

        latent_dim      : (int) The latent dimension we employed

        device          : (str) The device for the computation
    
    Returns:

        modes           : The
        
    """


    with torch.no_grad():
        zero_output = model.decoder(torch.from_numpy(np.zeros((1, latent_dim), dtype=np.float32)).to(device)).cpu().numpy()

    modes = np.zeros((latent_dim, zero_output.shape[1], zero_output.shape[2], zero_output.shape[3]))
    modes_front = np.zeros((latent_dim, zero_output.shape[1], zero_output.shape[2], zero_output.shape[3]))

    for mode in range(latent_dim):
        modes[mode, :, :, :], modes_front[mode, :, :, :] = calcmode(model, latent_dim, mode, device, case, para_data)

    return zero_output, modes, modes_front


#--------------------------------------------------------
def getNLmodes(model, mode, latent_dim, device):
    """
    Algorithm for optain single spatial mode from beta-VAE Decoder. 
    
    For latent variable i, We use the vector v (where vi = 1)  with a value within a range
    as the input to obtain the spatial mode for each latent variables

    Args:

        model           : (torch.nn.Module) The beta-VAE model 

        mode            : (int) The indice of the mode zi

        latent_dim      : (int) The latent dimension we employed

        device          : (str) The device for the computation
    
    Returns:

        NLvalues        : (NumpyArray) The used range of value 

        NLmodes         : (NumpyArray) The non-linear spatial mode
        
    """

    zero_output = decode(model, np.zeros((1, latent_dim), dtype=np.float32), device)
    NLvalues = np.arange(-2, 2.1, .1)
    NLmodes = np.zeros((NLvalues.shape[0], zero_output.shape[1], zero_output.shape[2], zero_output.shape[3]),
                        dtype=np.float32)
    
    for idx, value in enumerate(NLvalues):
        latent = np.zeros((1, latent_dim), dtype=np.float32)
        latent[0, mode] = value
        NLmodes[idx,:,:,:] = decode(model, latent, device)

    return NLvalues, NLmodes


#--------------------------------------------------------
def get_order(  model, latent_dim, n_train,
                data, dataset, 
                 device):
    """
    Algorithm for ranking the obtained spatial modes according to the yield accumlated energy level
    For more detail please check the paper

    Args:

        model           : (torch.nn.Module) The beta-VAE model 

        
        latent_dim      : (int) The latent dimension we employed

        data            : (NumpyArray) The flow database 

        dataset         : (torch.Dataloader) The dataloader of the flow data

        std             : (NumpyArray) The std of flow database

        device          : (str) The device for the computation
    
    Returns:

        m               : (NumpyArray) The ranking result (order) of each mode

        Ecum            : (NumpyArray) accumlative Energy obtained for each mode
    
    """
    import time
    import numpy as np 
    print('#'*30)
    print('Ordering modes')

    modes, _ = encode(model, dataset, device)
    modes_all = modes.reshape(-1, n_train, latent_dim)

    print("modes_all shape:",modes_all.shape)

    w, _ = data.tensors
    cases = modes_all.shape[0]
    w = w.reshape(-1, n_train, 200, 200)
    print("ALL cases: ", cases)
    #m = np.zeros(latent_dim, dtype=int)
    m = np.zeros((cases,latent_dim), dtype=int)
    #Ecum = []
    Ecum = np.zeros((cases,latent_dim), dtype=np.float32)
    
    
    for k in range(cases):
        print('-'*15)
        print(f"Case {k}:")
        partialModes = np.zeros_like(modes_all[k,:,:], dtype=np.float32)
        n = np.arange(latent_dim)
        for i in range(latent_dim):
            Eks = []
            for j in n:  # for mode in remaining modes
                start = time.time()
                print(m[k,:i+1], j, end="")
                partialModes *= 0
                #if m[k,:i].size > 0:  # 检查 m[k,:i] 是否为空
                #print(partialModes[:, m[k,:i+2]].shape, modes_all[k,:, m[k,:i+2]].shape)
                partialModes[:, m[k,:i]] = modes_all[k,:, m[k,:i]].T
                #partialModes[:, m[k,:i]] = modes_all[k,:, m[k,:i]]
                partialModes[:, j] = modes_all[k,:, j]
                u_pred = decode(model, partialModes, device)
                Eks.append(get_Ek(w[k,:,:,:], u_pred))
                elapsed = time.time() - start
                print(f' : Ek={Eks[-1]:.4f}, elapsed: {elapsed:.2f}s')
            Eks = np.array(Eks).squeeze()
            ind = n[np.argmax(Eks)]
            m[k,i] = ind
            n = np.delete(n, np.argmax(Eks))
            Ecum[k,i]=np.max(Eks)
            print('Adding: ', ind, ', Ek: ', np.max(Eks))
            print('#'*30)
        print(f"Rank finished, the rank is {m[k,:]}")
        print(f"Cumulative Ek is {Ecum[k,:]}")

    return np.array(m), np.array(Ecum)
def get_order_all(  model, latent_dim, n_train,
                data, dataset, 
                 device):
    """
    Algorithm for ranking the obtained spatial modes according to the yield accumlated energy level
    For more detail please check the paper

    Args:

        model           : (torch.nn.Module) The beta-VAE model 

        
        latent_dim      : (int) The latent dimension we employed

        data            : (NumpyArray) The flow database 

        dataset         : (torch.Dataloader) The dataloader of the flow data

        std             : (NumpyArray) The std of flow database

        device          : (str) The device for the computation
    
    Returns:

        m               : (NumpyArray) The ranking result (order) of each mode

        Ecum            : (NumpyArray) accumlative Energy obtained for each mode
    
    """
    import time
    import numpy as np 
    print('#'*30)
    print('Ordering modes')

    modes, _ = encode(model, dataset, device)

    print(modes.shape)
    w,_ = data.tensors
    m = np.zeros(latent_dim, dtype=int)
    n = np.arange(latent_dim)

    Ecum = []
    partialModes = np.zeros_like(modes, dtype=np.float32)

    for i in range(latent_dim):
        Eks = []
        for j in n:  # for mode in remaining modes
            start = time.time()
            print(m[:i], j, end="")
            partialModes *= 0
            partialModes[:, m[:i]] = modes[:, m[:i]]
            partialModes[:, j] = modes[:, j]
            u_pred = decode(model, partialModes, device)
            Eks.append(get_Ek(w, u_pred))
            elapsed = time.time() - start
            print(f' : Ek={Eks[-1]:.4f}, elapsed: {elapsed:.2f}s')
        Eks = np.array(Eks).squeeze()
        ind = n[np.argmax(Eks)]
        m[i] = ind
        n = np.delete(n, np.argmax(Eks))
        Ecum.append(np.max(Eks))
        print('Adding: ', ind, ', Ek: ', np.max(Eks))
        print('#'*30)
    Ecum = np.array(Ecum)
    print(f"Rank finished, the rank is {m}")
    print(f"Cumulative Ek is {Ecum}")

    return np.array(m), Ecum



################################
### Assessment on Energy
###############################
#--------------------------------------------------------
def get_Ek(original, rec):
    
    """
    Calculate energy percentage reconstructed
    
    Args:   
            original : (NumpyArray) The ground truth 

            rec      : (NumpyArray) The reconstruction from decoder

    Returns:  

            The energy percentage for construction. Note that it is the Ek/100 !!
    """

    import numpy as np 

    original = original.cpu().numpy() if isinstance(original, torch.Tensor) else original
    rec = rec.cpu().numpy() if isinstance(rec, torch.Tensor) else rec

    w_original = original.reshape(-1,1,200,200)[:, 0, :, :]
    TKE_real = w_original ** 2 

    w_rec = rec.reshape(-1,1,200,200)[:, 0, :, :]
    

    return 1 - np.sum((w_original - w_rec) ** 2) / np.sum(TKE_real)



#--------------------------------------------------------
def get_Ek_t(model, data, n_data, device):
    """
    
    Get the Reconstructed energy for snapshots

    Args:

        model           : (torch.nn.Module) The beta-VAE model 
        
        data            : (NumpyArray) The flow database 

        device          : (str) The device for the computation
    
    Returns:

        Ek_t            : (List) A list of enery of each snapshot in dataset
    
    
    """

    dataloader = torch.utils.data.DataLoader(dataset=data, 
                                            batch_size=1,
                                            shuffle=False, 
                                            pin_memory=True, 
                                            num_workers=2)

    rec_list = []
    with torch.no_grad():
        for batch in dataloader:
            #batch = batch.to(device).float()
            w, para = batch
            w, para = w.to(device).float(), para.to(device).float()
            rec, _, _ = model((w, para))
            rec_list.append(rec.cpu().numpy())

    rec = np.concatenate(rec_list, axis=0)
    rec = rec.reshape(-1, n_data, 200, 200)

    print("shpae of rec:",rec.shape)
    #print("shape of w:",w.shape,"shape of para:",para.shape)

    w_all, _ = data.tensors 
    w_all = w_all.reshape(-1, n_data, 200, 200)

    Ek_t = np. zeros((rec.shape[0],rec.shape[1]), dtype=np.float32)
    for k in range(rec.shape[0]):
        for i in range(rec.shape[1]):
            Ek_t[k,i] = get_Ek(w_all[k,i,:,:], rec[k, i,:,:])

    return Ek_t





#--------------------------------------------------------
def get_EcumTest(model, latent_dim, n_test, data, dataset, device, order):

    """
    Get the accumlative energy of test database 

    Args:

        model           : (torch.nn.Module) The beta-VAE model 

        
        latent_dim      : (int) The latent dimension we employed

        data            : (NumpyArray) The flow database 

        dataset         : (torch.Dataloader) The dataloader of the flow data

        std             : (NumpyArray) The std of flow database

        device          : (str) The device for the computation
    
        order           : (NumpyArray) A array which contains the ranking results

    Returns:

        Ecum            : (NumpyArray) accumlative Energy obtained for each mode
    
    """

    modes, _ = encode(model, dataset, device)
    modes_all = modes.reshape(-1, n_test, latent_dim)
    print("modes_all shape:", modes_all.shape)
    w,_ = data.tensors
    w = w.reshape(-1, n_test, 200, 200)
    cases = modes_all.shape[0]
    print("All cases: ", cases)

    Ecum = np.zeros((cases, latent_dim), dtype=np.float32)
    for k in range(cases):
        print('-'*15)
        print(f"Case {k}:")
        for i in range(latent_dim):
            partialModes = np.zeros_like(modes_all[k,:,:], dtype=np.float32)
            partialModes[:, order[k,:i+1]] = modes_all[k, :, order[k,:i+1]].T
            u_pred = decode(model, partialModes, device)
            Ecum[k,i]=get_Ek(w[k,:,:,:], u_pred)
            print(order[k,:i+1], Ecum[k,-1])

    return np.array(Ecum)





################################
### I/O 
###############################
#--------------------------------------------------------
def createModesFile(fname,
                    fname_predw, 
                    model, 
                    latent_dim, 
                    train_data,test_data, gen_data,
                    dataset_train, dataset_test, dataset_gen,
                    n_train, n_test, para_dim,
                    device, 
                    order, Ecum, order_all, Ecum_all, 
                    Ecum_test, Ecum_gen,
                    NLvalues, NLmodes, 
                    Ek_tr, Ek_te, Ek_gen):
    """
    
    Function for integrating all the obtained results and save it as fname

    Args: 

        fname           :   (str) The file name

        latent_dim      :   (int) The latent dimension 

        dataset_train   :   (dataloader) DataLoader for training data
        
        dataset_test    :   (dataloader) DataLoader for test data

        mean            :   (NumpyArray) The mean of flow database
        
        std             :   (NumpyArray) The std of flow database

        device          :   (str) The device for the computation
    
        order           : (NumpyArray) A array which contains the ranking results

        Ecum            : (NumpyArray) accumlative Energy obtained for each mode
    
        Ecum_test       : (NumpyArray) accumlative Energy obtained for each mode

        NLvalues        : (NumpyArray) The used range of value 

        NLmodes         : (NumpyArray) The non-linear spatial mode

        Ek_t            : (List) A list of enery of each snapshot in dataset

    Returns:

        is_save         : (bool)

    """
    import scipy.io
    
    if_save = False

    print(f"Start post-processing")


    means_train, stds_train  =   encode(model, dataset_train, device)
    print('INFO: Latent Variable Train Generated')
    means_test, stds_test    =   encode(model, dataset_test, device)
    print('INFO: Latent Variable Test Generated')
    means_gen, stds_gen      =   encode(model, dataset_gen, device)
    print('INFO: Latent Variable Gen Generated')

    w_train, para_train = train_data.tensors
    w_test, _  = test_data.tensors
    w_gen, _     = gen_data.tensors

    output_train = decode(model, means_train, device)
    output_test  = decode(model, means_test, device)
    output_gen   = decode(model, means_gen, device)

    w_train = w_train.reshape(-1, n_train, 200, 200)
    para_train = para_train.reshape(-1, n_train, para_dim)
    w_test  = w_test.reshape(-1, n_test, 200, 200)
    w_gen   = w_gen.reshape(-1, n_train + n_test, 200, 200)
    output_train = output_train.reshape(-1, n_train, 200, 200)
    output_test  = output_test.reshape(-1, n_test, 200, 200)
    output_gen   = output_gen.reshape(-1, n_train + n_test, 200, 200)

    w_train = w_train.cpu().numpy() if isinstance(w_train, torch.Tensor) else w_train
    w_test  = w_test.cpu().numpy() if isinstance(w_test, torch.Tensor) else w_test
    w_gen   = w_gen.cpu().numpy() if isinstance(w_gen, torch.Tensor) else w_gen
    output_train = output_train.cpu().numpy() if isinstance(output_train, torch.Tensor) else output_train
    output_test  = output_test.cpu().numpy() if isinstance(output_test, torch.Tensor) else output_test
    output_gen   = output_gen.cpu().numpy() if isinstance(output_gen, torch.Tensor) else output_gen

    RMSE_w_train_nm = np.sqrt(np.mean((w_train - output_train) ** 2, axis=(1)))
    RMSE_w_train = np.mean(RMSE_w_train_nm, axis=(1,2)).reshape(-1,1)
    RMSE_w_test_nm = np.sqrt(np.mean((w_test - output_test) ** 2, axis=(1)))
    RMSE_w_test = np.mean(RMSE_w_test_nm, axis=(1,2)).reshape(-1,1)
    RMSE_w_gen_nm = np.sqrt(np.mean((w_gen - output_gen) ** 2, axis=(1)))
    RMSE_w_gen = np.mean(RMSE_w_gen_nm, axis=(1,2)).reshape(-1,1)
    print("RMSE of w_train: ", RMSE_w_train)
    print("RMSE of w_test: ", RMSE_w_test)
    print("RMSE of w_gen: ", RMSE_w_gen)
    print("average RMSE of w_train: ", np.mean(RMSE_w_train))
    print("average RMSE of w_test: ", np.mean(RMSE_w_test))
    print("average RMSE of w_gen: ", np.mean(RMSE_w_gen))
    # RL2_w_test_nm = np.sqrt(np.mean(((w_test - output_test) / w_test) ** 2, axis=(1)))
    # RL2_w_test = np.mean(RL2_w_test_nm, axis=(1,2)).reshape(-1,1)
    # RL2_w_gen_nm = np.sqrt(np.mean(((w_gen - output_gen) / w_gen) ** 2, axis=(1)))
    # RL2_w_gen = np.mean(RL2_w_gen_nm, axis=(1,2)).reshape(-1,1)

    
    zero_output, modes, modes_front = get_spatial_modes(model, latent_dim, device, case=0, para_data=para_train)
    print('INFO: Spatial mode generated')

    if order is None:
        order = np.arange(latent_dim)


    scipy.io.savemat(fname, {
        'vector_train': means_train,
        'vector_test': means_test,
        'vector_gen': means_gen,
        'stds_vector_train': stds_train,
        'stds_vector_test': stds_test,
        'stds_vector_gen': stds_gen,
        'order': order,
        'Ecum': Ecum,
        'order_all': order_all,
        'Ecum_all': Ecum_all,
        'Ecum_test': Ecum_test,
        'Ecum_gen': Ecum_gen,
        'Ek_tr': Ek_tr,
        'Ek_te': Ek_te,
        'Ek_gen': Ek_gen,
        'modes': modes,
        'modes_front': modes_front,
        'zero_output': zero_output,
    })
    print(f"INFO: Post-processing results has been saved as dataset: {fname}.mat")
    scipy.io.savemat(pathsBib.data_path + f'latent_data{latent_dim}.mat', {
        'vector_train': means_train,
        'vector_test': means_test,
        'vector_gen': means_gen,
    })
    print(f"INFO: Laten data has been saved at: {pathsBib.data_path}latent_data{latent_dim}.mat")
    scipy.io.savemat(fname_predw, {
        'w_train': w_train,
        'w_test': w_test,
        'w_gen': w_gen,
        'output_train': output_train,
        'output_test': output_test,
        'output_gen': output_gen,
        'RMSE_w_train_nm': RMSE_w_train_nm,
        'RMSE_w_train': RMSE_w_train,
        'RMSE_w_test_nm': RMSE_w_test_nm,
        'RMSE_w_test': RMSE_w_test,
        'RMSE_w_gen_nm': RMSE_w_gen_nm,
        'RMSE_w_gen': RMSE_w_gen,
        # 'RL2_w_test_nm': RL2_w_test_nm,
        # 'RL2_w_gen_nm': RL2_w_gen_nm,
        # 'RL2_w_train': RL2_w_train,
        # 'RL2_w_test': RL2_w_test,
        # 'RL2_w_gen': RL2_w_gen
    })
    print(f"INFO: pVAE pred w results has been saved as dataset: {fname_predw}.mat")


    if_save = True
    
    return if_save


    
    

    