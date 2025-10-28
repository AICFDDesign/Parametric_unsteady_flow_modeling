class lstm_config:
    """
    A class of config for LSTM Predictor
    """
    from configs.vae import VAE_config 

    in_dim      = 32
    d_model     = 32
    next_step   = 1
    nmode       = VAE_config.latent_dim
    latent_dim  = VAE_config.latent_dim

    train_t     = 400
    test_t      = 100

    num_layer   = 4
    embed       = None
    
    hidden_size = 128

    is_output   = True
    out_act     = None

    Epoch       = 200
    Batch_size  = 128
    lr          = 1e-3

    train_split = 0.8 
    val_split   = 0.2 
    num_train   = 135000

    early_stop = True

    if early_stop == True:
        patience  = 50
    else:
        patience  = 0 




