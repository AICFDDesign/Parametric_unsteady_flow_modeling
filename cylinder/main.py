# =============================================================================
# Parametric unsteady flow modeling method based on $beta$-variational autoencoders and transformers
# Liang Wang, Ran Bi, Xinshuai Zhang, Tingwei Ji, and Fangfang Xie,
# 2025.10
# =============================================================================

"""
Main program for cylinder case
"""

# =============================================================================
#  Inspired by Alberto Solera-Rico, Carlos Sanmiguel Vila, M. A. Gómez, Yuning Wang, Abdulrahman Almashjary, Scott T. M. Dawson, Ricardo Vinuesa
#  "$\beta$-Variational autoencoders and transformers for reduced-order modelling of fluid flow." , Nature Communications (2024) 15:1361,
#  https://github.com/KTH-FlowAI/beta-Variational-autoencoders-and-transformers-for-reduced-order-modelling-of-fluid-flows
# =============================================================================


import      torch
import      numpy as np
import      argparse
from        lib             import init
from        lib.runners     import PvaeRunner, latentRunner
from        utils.figs_time import vis_temporal_Prediction
from        utils.figs      import vis_bvae


parser = argparse.ArgumentParser()
parser.add_argument('-ae',default="para", type=str,   help="beta-VAE with para branch NN ")
parser.add_argument('-nn',default="easy", type=str,   help="Choose the model for time-series prediction: easy, self OR lstm")
parser.add_argument('-re',default=200,     type=int,   help="200 OR 100, Choose corresponding Reynolds number for the case")
parser.add_argument('-m1', default="train", type=str,   help='Switch the mode between train, infer and run')
parser.add_argument('-m2', default="train", type=str,   help='Switch the mode between train, infer and run')
parser.add_argument('-t', default="val",  type=str,    help='The type of saved model: pre/val/final')
args  = parser.parse_args()

device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(255)


if __name__ == "__main__":

    print("#"*30)
    print(f"Using device: {device}")
    print(f"Beta-VAE Running mode: {args.m1}")
    print(f"Temporal prediction Running mode: {args.m2}")
    print(f"Reynolds number: {args.re}")
    print(f"Beta-VAE model type: {args.ae}")
    print(f"Laten Model type: {args.nn}")
    print(f"Time-series prediction model: {args.nn}")
    print("#"*30)
    ## Env INIT
    print("------------------------------ BEnv INIT ------------------------------")
    datafile = init.init_env(args.re)


    # Beta-VAE
    print("------------------------------ Beta-VAE ------------------------------")
    pbvae   = PvaeRunner(device,datafile)
    if args.m1 == 'train':
        pbvae.train()
    elif args.m1 == 'infer':
        pbvae.infer(args.t)
    elif args.m1 == 'run':
        pbvae.run()



    # Time-series prediction runner 
    print("------------------------------ Time-series prediction ------------------------------")
    lruner = latentRunner(args.nn,device)
    if args.m2 == 'train':
        lruner.train()
    elif args.m2 == 'infer':
        lruner.infer(args.t)
    elif args.m2 == 'run':
        lruner.train()
        lruner.infer(args.t)

    # Visualization

    # vis_bvae(init.pathsBib.res_path + "modes_" + pbvae.filename + ".mat",
    #         init.pathsBib.log_path + pbvae.filename)
    # vis_temporal_Prediction(pvae=pbvae, predictor=lruner, model_type=args.nn, train_time=pbvae.config.n_train, test_time=pbvae.config.n_test, case=1)
