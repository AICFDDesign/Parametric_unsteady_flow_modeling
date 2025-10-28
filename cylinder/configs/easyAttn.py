class easyAttn_config:
    """
    A class of configuration of Transformer predictor 
    """
    from configs.vae import VAE_config 
    
    
    in_dim      = 32
    out_dim     = 32 # The output sequence length
    d_model     = 32

    train_t     = 400
    test_t      = 100

    latent_dim  = VAE_config.latent_dim

    time_proj   = 32 # The projection on time, which is used for new embedding stragtegy 
    next_step   = 1 
    nmode       = VAE_config.latent_dim  # Should be consistent as the modes
    

    num_head    = 4
    attn_type   = "easy" 
    embed       = "time" # sin / cos/ time
    num_block   = 4      # Number of layer 

    is_res_attn = True
    is_res_proj = True
    proj_dim    = 128

    act_proj    = "relu"
    is_output   = True
    out_act     = None

    Epoch       = 200
    Batch_size  = 128
    lr          = 1e-3

    train_split = 0.8 
    num_train   = 135000

    early_stop  = True

    if early_stop == True:
        patience  = 50 # 30 or 50
    else:
        patience  = 0 

