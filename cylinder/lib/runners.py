"""
Runners for the VAE and temporal-dynamic prediction in latent space 
"""

import os 
import time
from pathlib import Path
import h5py
import numpy as np
import torch 
from torch          import nn

from lib.init       import pathsBib
from lib.train      import * 
from lib.model      import * 
from lib.pp_time    import * 
from lib.pp_space   import spatial_Mode , sindy_rec_analysis
from lib.datas      import * 
import scipy.io
import matplotlib.pyplot as plt

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


####################################################
### RUNNER for beta-VAE
####################################################

class vaeRunner(nn.Module):
    def __init__(self, device, datafile) -> None:
        """
        A runner for beta-VAE

        Args:

            device          :       (Str) The device going to use
            
            datafile        :       (Str) Path of training data
        """
        
        from configs.vae import VAE_config as cfg 
        from configs.nomenclature import Name_VAE
        
        super(vaeRunner,self).__init__()
        print("#"*30)

        self.config     = cfg
        self.filename   = Name_VAE(self.config)
        
        self.datafile   = datafile

        self.device     = device

        self.model      = get_vae(self.config.latent_dim)
        
        self.model.to(device)

        self.fmat       =  '.pth.tar'

        self.loss_history = []
        self.loss_test_history = []
        self.MSE_history = []
        self.KLD_history = []
        self.MSE_test_history = []
        self.KLD_test_history = []
        self.LR_history = []

        print(f"INIT betaVAE, device: {device}")
        print(f"Case Name:\n {self.filename}")
#-------------------------------------------------
    def run(self):
        self.train()
        self.infer(model_type='final')

#-------------------------------------------------
    def train(self):
        print("#"*30)
        print("INFO: Start Training ")
        self.get_data()
        self.compile()
        self.fit()
        self.train_dl   = None
        self.val_dl     = None
        print(f"INFO: Training finished, cleaned the data loader")
        print("#"*30)

#-------------------------------------------------
    def infer(self, model_type):
        print("#"*30)
        self.load_pretrain_model(model_type=model_type)
        print("INFO: Model has been loaded!")
        
        self.get_test_data()
        print("INFO: test data has been loaded!")
        self.post_process()

        print(f"INFO: Inference ended!")
        print("#"*30)


#-------------------------------------------------

    def get_data(self): 
        """
        
        Generate the DataLoader for training 

        """
        
        # datafile = 
        try:
        # 加载数据
            u_scaled, self.mean, self.std = loadData(self.datafile)
        except Exception as e:
            print(f"ERROR: Failed to load data from {self.datafile}. Exception: {e}.check the file path")
            return

        u_scaled, self.mean, self.std = loadData(self.datafile)

        if self.config.downsample > 1:
            u_scaled        = u_scaled[::self.config.downsample]
        n_total             = u_scaled.shape[0]
        self.n_train        = n_total - self.config.n_test
        print(f"INFO: Data Summary: N train: {self.n_train:d}," + \
                f"N test: {self.config.n_test:d},"+\
                f"N total {n_total:d}")
        
        self.train_dl, self.val_dl = get_pvae_DataLoader(    d_train=u_scaled,
                                                            n_train=self.n_train,
                                                            device= self.device,
                                                            batch_size= self.config.batch_size)
        print( f"INFO: Dataloader generated, Num train batch = {len(self.train_dl)} , " +\
                f"Num val batch = {len(self.val_dl)}")
        



#-------------------------------------------------
    def compile(self):
        """
        
        Compile the optimiser, schedulers and loss function for training

        
        """

        from torch.optim import lr_scheduler
        
        print("#"*30)
        print(f"INFO: Start Compiling")

        encoder_params = list(self.model.encoder.parameters())
        decoder_params = list(self.model.decoder.parameters())

        # get optimizer
        self.opt = torch.optim.Adam(
            [   {'params': encoder_params, 'weight_decay': self.config.encWdecay},
                {'params': decoder_params, 'weight_decay': self.config.decWdecay}], 
                lr=self.config.lr, weight_decay=0)
        
        self.opt_sch = lr_scheduler.OneCycleLR(self.opt, 
                                            max_lr=self.config.lr,
                                            total_steps=self.config.epochs, 
                                            div_factor=2, 
                                            final_div_factor=self.config.lr/self.config.lr_end, 
                                            pct_start=0.2)

        self.beta_sch = betaScheduler(  startvalue =self.config.beta_init,
                                        endvalue      =self.config.beta,
                                        warmup        =self.config.beta_warmup)#warmup linear growth, and then keep constant as self.config.beta

        print(f"INFO: Compiling Finished!")


#-------------------------------------------------

    def fit(self):
        """

        Training beta-VAE
        
        """
        from torch.utils.tensorboard import SummaryWriter
        from utils.io import save_checkpoint

        print(f"Training {self.filename}")
        logger = SummaryWriter(log_dir=pathsBib.log_path + self.filename)

        bestloss = 1e6
        loss = 1e6

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            beta = self.beta_sch.getBeta(epoch, prints=False)
            loss, MSE, KLD, elapsed, collapsed = train_epoch(model=self.model,
                                                                        data=self.train_dl,
                                                                        optimizer=self.opt,
                                                                        beta=beta,
                                                                        device=self.device)
            self.model.eval()
            loss_test, MSE_test, KLD_test, elapsed_test = test_epoch(model=self.model,
                                                                                data=self.val_dl,
                                                                                beta=beta,
                                                                                device=self.device)

            self.opt_sch.step()

            printProgress(epoch=epoch,
                                    epochs=self.config.epochs,
                                    loss=loss,
                                    loss_test=loss_test,
                                    MSE=MSE,
                                    KLD=KLD,
                                    elapsed=elapsed,
                                    elapsed_test=elapsed_test,
                                    collapsed=collapsed)

            self.loss_history.append(loss)
            self.loss_test_history.append(loss_test)
            self.MSE_history.append(MSE)
            self.KLD_history.append(KLD)
            self.MSE_test_history.append(MSE_test)
            self.KLD_test_history.append(KLD_test)
            self.LR_history.append(self.opt_sch.get_last_lr()[0])

            if (loss_test < bestloss and epoch > 100):
                bestloss = loss_test
                checkpoint = {'state_dict': self.model.state_dict(), 'optimizer_dict': self.opt.state_dict()}
                ckp_file = f'{pathsBib.chekp_path}/{self.filename}_bestVal' + self.fmat
                save_checkpoint(state=checkpoint, path_name=ckp_file)
                print(f'## Checkpoint. Epoch: {epoch}, test loss: {loss_test}, saving checkpoint {ckp_file}')

        checkpoint = {'state_dict': self.model.state_dict(), 'optimizer_dict': self.opt.state_dict()}
        ckp_file = f'{pathsBib.chekp_path}/{self.filename}_final' + self.fmat
        save_checkpoint(state=checkpoint, path_name=ckp_file)
        print(f'Checkpoint. Final epoch, loss: {loss}, test loss: {loss_test}, saving checkpoint {ckp_file}')

        loss_history_path = f'{pathsBib.log_path+ self.filename}/VAE_loss_history.mat'
        scipy.io.savemat(loss_history_path, {'loss_history': self.loss_history,
                                                'loss_test_history': self.loss_test_history,
                                                'MSE_history': self.MSE_history,
                                                'KLD_history': self.KLD_history,
                                                'MSE_test_history': self.MSE_test_history,
                                                'KLD_test_history': self.KLD_test_history,
                                                'LR_history': self.LR_history})
        print(f"INFO: Loss history saved to {loss_history_path}")
        self.plot_vae_loss()

#-------------------------------------------------
    def plot_vae_loss(self):
        """
        Plot the loss history and save the figure.
        """
        plt.figure()
        plt.plot(self.loss_history, label='train', color='blue')
        plt.plot(self.loss_test_history, label='test', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.title('Loss History')
        plt.legend()
        plot_path = f'{pathsBib.log_path+ self.filename}/loss_history.png'
        plt.savefig(plot_path)
        plt.close()

        plt.figure()
        plt.plot(self.MSE_history, label='train', color='blue')
        plt.plot(self.MSE_test_history, label='test', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.yscale('log')
        plt.title('MSE Loss History')
        plt.legend()
        plot_path = f'{pathsBib.log_path+ self.filename}/MSE_loss_history.png'
        plt.savefig(plot_path)
        plt.close()

        plt.figure()
        plt.plot(self.KLD_history, label='train', color='blue')
        plt.plot(self.KLD_test_history, label='test', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('KLD Loss')
        plt.yscale('log')
        plt.title('KLD Loss History')
        plt.legend()
        plot_path = f'{pathsBib.log_path+ self.filename}/KLD_loss_history.png'
        plt.savefig(plot_path)
        plt.close()

        plt.figure()
        plt.plot(self.LR_history, label='train', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.title('Learning Rate History')
        plt.legend()
        plot_path = f'{pathsBib.log_path+ self.filename}/LR_history.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"INFO: Loss, MSE_loss,KLD_loss, LR figures saved to {pathsBib.log_path+ self.filename}")


#-------------------------------------------------

    def load_pretrain_model(self,model_type='pre'):
        """

        Load the pretrained model for beta VAE

        Args: 

            model_type  : ['pre', 'val','final']  (str) Choose from pre-trained, best valuation and final model 
        
        """
        
        model_type_all = ['pre','val','final']
        assert(model_type in model_type_all), print('ERROR: No type of the model matched')

        if      model_type == 'pre':    model_path = pathsBib.pretrain_path + self.filename               + self.fmat
        elif    model_type == 'val' :   model_path = pathsBib.chekp_path    + self.filename + '_bestVal' + self.fmat
        elif    model_type == 'final' : model_path = pathsBib.chekp_path    + self.filename + '_final'    + self.fmat
        

        try:
            ckpoint = torch.load(model_path, map_location= self.device)
            
        except:
            print("ERROR: Model NOT found!")
            exit()
        stat_dict   = ckpoint['state_dict']
        self.model.load_state_dict(stat_dict)
        print(f'INFO: the state dict has been loaded!')

#-------------------------------------------------

    def get_test_data(self):
        """
        
        Generate the DataLoder for test 

        """
        from torch.utils.data import DataLoader
        
        u_scaled, self.mean, self.std = loadData(self.datafile)
        
        u_scaled            = u_scaled[::self.config.downsample]
        n_total             = u_scaled.shape[0]
        self.n_train        = n_total - self.config.n_test
        
        print(f"INFO: Data Summary: N train: {self.n_train:d}," + \
                f"N test: {self.config.n_test:d},"+\
                f"N total {n_total:d}")
        
        self.train_d, self.test_d = u_scaled[:self.n_train] ,u_scaled[self.n_train:]

        self.train_dl        = DataLoader(torch.from_numpy(self.train_d), 
                                        batch_size=1,
                                        shuffle=False, 
                                        pin_memory=True, 
                                        num_workers=2)
        self.test_dl       = DataLoader(torch.from_numpy(self.test_d), 
                                        batch_size=1,
                                        shuffle=False, 
                                        pin_memory=True, 
                                        num_workers=2)
        
        print(f"INFO: Dataloader generated, Num Test batch = {len(self.test_dl)}")
        


#-------------------------------------------------
    def post_process(self):

        """

        Post-processing for Beta-VAE 

        """

        assert (self.test_dl != None), print("ERROR: NOT able to do post-processing without test data!")

        fname = pathsBib.res_path + "modes_" + self.filename
        
        if_save_spatial = spatial_Mode(fname,
                                    model=self.model,latent_dim=self.config.latent_dim,
                                    train_data=self.train_d,test_data=self.test_d,
                                    dataset_train=self.train_dl,dataset_test=self.test_dl,
                                    mean = self.mean, std = self.std,
                                    device= self.device,
                                    if_order= True,
                                    if_nlmode= True,
                                    if_Ecumt= True,
                                    if_Ek_t= True
                                    )
        if if_save_spatial: 
            print(f"INFO: Spatial Modes finished!")
        else:
            print(f'ERROR: Spatial modes has not saved!')
        
####################################################
### RUNNER for beta-VAE
####################################################

class PvaeRunner(nn.Module):
    def __init__(self, device, datafile) -> None:
        """
        A runner for beta-VAE

        Args:

            device          :       (Str) The device going to use
            
            datafile        :       (Str) Path of training data
        """
        
        from configs.vae import VAE_config as cfg 
        from configs.nomenclature import Name_pVAE
        
        super(PvaeRunner,self).__init__()
        print("#"*30)

        self.config     = cfg
        self.filename   = Name_pVAE(self.config)
        
        self.datafile   = datafile

        self.device     = device

        self.model      = get_pvae(self.config.latent_dim,self.config.input_para_dim)
        
        self.model.to(device)

        self.fmat       =  '.pth.tar'

        self.loss_history = []
        self.loss_test_history = []
        self.MSE_history = []
        self.KLD_history = []
        self.MSE_test_history = []
        self.KLD_test_history = []
        self.LR_history = []
        NumPara = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"INFO: The model has been generated, num of parameter is {NumPara}")

        print(f"INIT betaVAE, device: {device}")
        print(f"Case Name:\n {self.filename}")
#-------------------------------------------------
    def run(self):
        self.train()
        self.infer(model_type='final')

#-------------------------------------------------
    def train(self):
        print("#"*30)
        print("INFO: Start Training ")
        self.get_data()
        self.compile()
        self.fit()
        self.train_dl   = None
        self.val_dl     = None
        print(f"INFO: Training finished, cleaned the data loader")
        print("#"*30)

#-------------------------------------------------
    def infer(self, model_type):
        print("#"*30)
        self.load_pretrain_model(model_type=model_type)
        print("INFO: Model has been loaded!")
        
        self.get_test_data()
        print("INFO: test data has been loaded!")
        self.post_process()

        print(f"INFO: Inference ended!")
        print("#"*30)

#-------------------------------------------------
    def infer_sindy_rec(self, model_type, sindy_filename='sindy_rec'):
        """
        Infer the SINDy reconstruction from the pre-trained model
        """
        print("#"*30)
        self.load_pretrain_model(model_type=model_type)
        print("INFO: Model has been loaded!")
        
        self.get_test_data()
        print("INFO: test data has been loaded!")
        
        self.sindy_rec(sindy_filename=sindy_filename)

        print(f"INFO: SINDy reconstruction ended!")
        print("#"*30)


#-------------------------------------------------

    def get_data(self): 
        """
        
        Generate the DataLoader for training 

        """
        
        # datafile = 
        try:
        # 加载数据
            w, para = loadData(self.datafile)
        except Exception as e:
            print(f"ERROR: Failed to load data from {self.datafile}. Exception: {e}.check the file path")
            return

        if self.config.downsample > 1:
            w        = w[:,::self.config.downsample,:,:]
        n_total             = w.shape[1]
        self.n_train        = n_total - self.config.n_test
        print(f"INFO: Data Summary: N train: {self.n_train:d}," + \
                f"N test: {self.config.n_test:d},"+\
                f"N total {n_total:d}")
        
        self.train_dl, self.val_dl = get_pvae_DataLoader(   d1_train=w, d2_train=para,
                                                            train_index=self.config.train,
                                                            test_index=self.config.test,
                                                            para_dim=self.config.input_para_dim,
                                                            n_train=self.n_train,
                                                            device= self.device,
                                                            batch_size= self.config.batch_size)
        print( f"INFO: Dataloader generated, Num train batch = {len(self.train_dl)} , " +\
                f"Num val batch = {len(self.val_dl)}")
        



#-------------------------------------------------
    def compile(self):
        """
        
        Compile the optimiser, schedulers and loss function for training

        
        """

        from torch.optim import lr_scheduler
        
        print("#"*30)
        print(f"INFO: Start Compiling")

        encoder_params = list(self.model.encoder.parameters())
        decoder_params = list(self.model.decoder.parameters())
        paracoder_params = list(self.model.paracoder.parameters())

        # get optimizer
        self.opt = torch.optim.Adam(
            [   {'params': encoder_params, 'weight_decay': self.config.encWdecay},
                {'params': decoder_params, 'weight_decay': self.config.decWdecay},
                {'params': paracoder_params, 'weight_decay': self.config.paraWdecay}], 
                lr=self.config.lr, weight_decay=0)
        
        self.opt_sch = lr_scheduler.OneCycleLR(self.opt, 
                                            max_lr=self.config.lr,
                                            total_steps=self.config.epochs, 
                                            div_factor=2, 
                                            final_div_factor=self.config.lr/self.config.lr_end, 
                                            pct_start=0.2)

        self.beta_sch = betaScheduler(  startvalue =self.config.beta_init,
                                        endvalue      =self.config.beta,
                                        warmup        =self.config.beta_warmup)#warmup linear growth, and then keep constant as self.config.beta

        print(f"INFO: Compiling Finished!")


#-------------------------------------------------

    def fit(self):
        """

        Training beta-VAE
        
        """
        # from torch.utils.tensorboard import SummaryWriter
        from utils.io import save_checkpoint

        print(f"Training {self.filename}")
        # logger = SummaryWriter(log_dir=pathsBib.log_path + self.filename)

        bestloss = 1e6
        loss = 1e6
        time_start = time.time()

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            beta = self.beta_sch.getBeta(epoch, prints=False)
            loss, MSE, KLD, elapsed, collapsed = train_epoch(model=self.model,
                                                                        data=self.train_dl,
                                                                        optimizer=self.opt,
                                                                        beta=beta,
                                                                        device=self.device)
            self.model.eval()
            loss_test, MSE_test, KLD_test, elapsed_test = test_epoch(model=self.model,
                                                                                data=self.val_dl,
                                                                                beta=beta,
                                                                                device=self.device)

            self.opt_sch.step()

            printProgress(epoch=epoch,
                                    epochs=self.config.epochs,
                                    loss=loss,
                                    loss_test=loss_test,
                                    MSE=MSE,
                                    KLD=KLD,
                                    elapsed=elapsed,
                                    elapsed_test=elapsed_test,
                                    collapsed=collapsed)

            # logger.add_scalar('General loss/Total', loss, epoch)
            # logger.add_scalar('General loss/MSE', MSE, epoch)
            # logger.add_scalar('General loss/KLD', KLD, epoch)
            # logger.add_scalar('General loss/Total_test', loss_test, epoch)
            # logger.add_scalar('General loss/MSE_test', MSE_test, epoch)
            # logger.add_scalar('General loss/KLD_test', KLD_test, epoch)
            # logger.add_scalar('Optimizer/LR', self.opt_sch.get_last_lr()[0], epoch)
            self.loss_history.append(loss)
            self.loss_test_history.append(loss_test)
            self.MSE_history.append(MSE)
            self.KLD_history.append(KLD)
            self.MSE_test_history.append(MSE_test)
            self.KLD_test_history.append(KLD_test)
            self.LR_history.append(self.opt_sch.get_last_lr()[0])

            if (loss_test < bestloss and epoch > 100):
                bestloss = loss_test
                checkpoint = {'state_dict': self.model.state_dict(), 'optimizer_dict': self.opt.state_dict()}
                ckp_file = f'{pathsBib.chekp_path}/{self.filename}_bestVal' + self.fmat
                save_checkpoint(state=checkpoint, path_name=ckp_file)
                print(f'## Checkpoint. Epoch: {epoch}, test loss: {loss_test}, saving checkpoint {ckp_file}')

        time_end = time.time()
        print(f'Elapsed time: %.2f seconds' % (time_end - time_start))
        checkpoint = {'state_dict': self.model.state_dict(), 'optimizer_dict': self.opt.state_dict()}
        ckp_file = f'{pathsBib.chekp_path}/{self.filename}_final' + self.fmat
        save_checkpoint(state=checkpoint, path_name=ckp_file)
        print(f'Checkpoint. Final epoch, loss: {loss}, test loss: {loss_test}, saving checkpoint {ckp_file}')

        Path(f'{pathsBib.log_path+ self.filename}/').mkdir(exist_ok=True)
        print(f"INFO: Create the log path {pathsBib.log_path+ self.filename}/")
        loss_history_path = f'{pathsBib.log_path+ self.filename}/VAE_loss_history.mat'
        scipy.io.savemat(loss_history_path, {'loss_history': self.loss_history,
                                                'loss_test_history': self.loss_test_history,
                                                'MSE_history': self.MSE_history,
                                                'KLD_history': self.KLD_history,
                                                'MSE_test_history': self.MSE_test_history,
                                                'KLD_test_history': self.KLD_test_history,
                                                'LR_history': self.LR_history})
        print(f"INFO: Loss history saved to {loss_history_path}")
        self.plot_pvae_loss()

#-------------------------------------------------
    def plot_pvae_loss(self):
        """
        Plot the loss history and save the figure.
        """
        plt.figure()
        plt.plot(self.loss_history, label='train', color='blue')
        plt.plot(self.loss_test_history, label='test', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.title('Loss History')
        plt.legend()
        plot_path = f'{pathsBib.log_path+ self.filename}/loss_history.png'
        plt.savefig(plot_path)
        plt.close()

        plt.figure()
        plt.plot(self.MSE_history, label='train', color='blue')
        plt.plot(self.MSE_test_history, label='test', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.yscale('log')
        plt.title('MSE Loss History')
        plt.legend()
        plot_path = f'{pathsBib.log_path+ self.filename}/MSE_loss_history.png'
        plt.savefig(plot_path)
        plt.close()

        plt.figure()
        plt.plot(self.KLD_history, label='train', color='blue')
        plt.plot(self.KLD_test_history, label='test', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('KLD Loss')
        plt.yscale('log')
        plt.title('KLD Loss History')
        plt.legend()
        plot_path = f'{pathsBib.log_path+ self.filename}/KLD_loss_history.png'
        plt.savefig(plot_path)
        plt.close()

        plt.figure()
        plt.plot(self.LR_history, label='train', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.title('Learning Rate History')
        plt.legend()
        plot_path = f'{pathsBib.log_path+ self.filename}/LR_history.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"INFO: Loss, MSE_loss,KLD_loss, LR figures saved to {pathsBib.log_path+ self.filename}")


#-------------------------------------------------

    def load_pretrain_model(self,model_type='pre'):
        """

        Load the pretrained model for beta VAE

        Args: 

            model_type  : ['pre', 'val','final']  (str) Choose from pre-trained, best valuation and final model 
        
        """
        
        model_type_all = ['pre','val','final']
        assert(model_type in model_type_all), print('ERROR: No type of the model matched')

        if      model_type == 'pre':    model_path = pathsBib.pretrain_path + self.filename               + self.fmat
        elif    model_type == 'val' :   model_path = pathsBib.chekp_path    + self.filename + '_bestVal' + self.fmat
        elif    model_type == 'final' : model_path = pathsBib.chekp_path    + self.filename + '_final'    + self.fmat
        

        try:
            ckpoint = torch.load(model_path, map_location= self.device)
            
        except:
            print("ERROR: Model NOT found!")
            exit()
        stat_dict   = ckpoint['state_dict']
        self.model.load_state_dict(stat_dict)
        print(f'INFO: the state dict has been loaded!')

#-------------------------------------------------

    def get_test_data(self):
        """
        
        Generate the DataLoder for test 

        """
        from torch.utils.data import DataLoader, TensorDataset
        
        w, para = loadData(self.datafile)
        
        w            = w[:,::self.config.downsample,:,:]
        para         = para[:,::self.config.downsample,:]
        n_total             = w.shape[1]
        self.n_train        = n_total - self.config.n_test
        
        print(f"INFO: Data Summary: N train: {self.n_train:d}," + \
                f"N test: {self.config.n_test:d},"+\
                f"N total {n_total:d}")
        
        self.d1_tr, self.d2_tr = w[self.config.train, :self.n_train, :,:], para[self.config.train, :self.n_train,:]
        self.d1_te, self.d2_te = w[self.config.train, self.n_train:, :,:], para[self.config.train, self.n_train:,:]
        self.d1_gen, self.d2_gen = w[self.config.test, :, :,:], para[self.config.test, :,:]
        
        self.train_d   = TensorDataset(torch.from_numpy(self.d1_tr).reshape(-1, 1, 200, 200)     , torch.from_numpy(self.d2_tr).reshape(-1, self.config.input_para_dim))
        self.test_d    = TensorDataset(torch.from_numpy(self.d1_te).reshape(-1, 1, 200, 200)     , torch.from_numpy(self.d2_te).reshape(-1, self.config.input_para_dim))
        self.gen_d     = TensorDataset(torch.from_numpy(self.d1_gen).reshape(-1, 1, 200, 200) , torch.from_numpy(self.d2_gen).reshape(-1, self.config.input_para_dim))

        self.train_dl  = DataLoader(self.train_d, 
                                    batch_size=1,
                                    shuffle=False, 
                                    pin_memory=True, 
                                    num_workers=2)
        self.test_dl   = DataLoader(self.test_d, 
                                    batch_size=1,
                                    shuffle=False, 
                                    pin_memory=True, 
                                    num_workers=2)
        self.gen_dl    = DataLoader(self.gen_d, 
                                    batch_size=1,
                                    shuffle=False, 
                                    pin_memory=True, 
                                    num_workers=2)
        
        print(f"INFO: Dataloader generated,Num Train batch = {len(self.train_dl)}, Num Test batch = {len(self.test_dl)}, Num Generalization batch = {len(self.gen_dl)}")
        


#-------------------------------------------------
    def post_process(self):

        """

        Post-processing for Beta-VAE 

        """

        assert (self.test_dl != None), print("ERROR: NOT able to do post-processing without test data!")

        fname = pathsBib.res_path + "modes_" + self.filename + ".mat"
        fname_predw = pathsBib.res_path + "pVAEpredw_" + self.filename + ".mat"

        if Path(fname).exists():
            print(f"INFO (debug mode): File {fname} exists.")
        else:
            print(f"INFO (debug mode): File {fname} does not exist.")

        if Path(fname_predw).exists():
            print(f"INFO (debug mode): File {fname_predw} exists.")
        else:
            print(f"INFO (debug mode): File {fname_predw} does not exist.")

        if (Path(fname_predw).exists() and Path(fname).exists()):
            print(f"INFO (debug mode): Both {fname} and \n {fname_predw} have been generated, skip postprocess!")
            return
        
        if_save_spatial = spatial_Mode(fname,fname_predw,
                                    model=self.model,latent_dim=self.config.latent_dim,
                                    n_train=self.n_train,n_test=self.config.n_test,para_dim=self.config.input_para_dim,
                                    train_data=self.train_d,test_data=self.test_d,gen_data=self.gen_d,
                                    dataset_train=self.train_dl,dataset_test=self.test_dl,dataset_gen=self.gen_dl,
                                    device= self.device,
                                    if_modes= True,
                                    if_order= True,
                                    if_nlmode= True,
                                    if_Ecumt= True,
                                    if_Ek_t= True,
                                    )
        if if_save_spatial: 
            print(f"INFO: Spatial Modes and pVAE pred results finished!")
        else:
            print(f'ERROR: Spatial modes has not saved!')

#-------------------------------------------------
    def sindy_rec(self, sindy_filename):

        """

        Post-processing for Beta-VAE 

        """

        assert (self.test_dl != None), print("ERROR: NOT able to do post-processing without test data!")

        fname = pathsBib.res_path + "sindy_rec_.mat"

        if Path(fname).exists():
            print(f"INFO (debug mode): File {fname} exists.")
        else:
            print(f"INFO (debug mode): File {fname} does not exist.")

        
        Ek_sindy = sindy_rec_analysis(sindy_filename,
                                    model=self.model,
                                    n_train=self.n_train,
                                    train_data=self.train_d,
                                    device= self.device,
                                    )
        scipy.io.savemat(fname,
                        {'Ek_sindy': Ek_sindy,
                         })
        print(f"INFO: SINDy reconstruction finished and saved to {fname}")
        

####################################################
### RUNNER for Temporal-dynamics Prediction
####################################################


class latentRunner(nn.Module): 
    def __init__(self,name,device):
        """
        A runner for latent space temporal-dynmaics prediction

        Args:

            name            :       (str) The model choosed for temporal-dynamics prediction 

            device          :       (Str) The device going to use
            
        """

        super(latentRunner,self).__init__()
        print("#"*30)
        print(f"INIT temporal predictor: {name}, device: {device}")
        self.device = device
        self.model,self.filename, self.config = get_predictors(name)
        
        self.NumPara = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.fmat   = '.pt'
        print(f"INFO: The model has been generated, num of parameter is {self.NumPara}")
        print(f"Case Name:\n {self.filename}")


#-------------------------------------------------

    def train(self):
        print("#"*30)
        print("INFO: Start Training ")
        self.get_data()
        self.compile()
        self.fit()
        self.train_dl   = None
        self.val_dl     = None
        print(f"INFO: Training finished, cleaned the data loader")
        print("#"*30)

#-------------------------------------------------
    def infer(self, model_type = 'pre',
            if_window=True):
        """
        
        Inference and evaluation of the model 

        Args: 

            model_type: (str) The type of model to load 

            if_window : (str) If compute the sliding-widnow error 

            if_pmap : (str) If compute the Poincare Map 
        
        """
        
        print("#"*30)
        print("INFO: Start post-processing")
        # self.com
        self.load_pretrain_model(model_type=model_type)

        self.post_process(if_window)
        print(f"INFO: Inference ended!")
        print("#"*30)

#-------------------------------------------------


    def get_data(self):
        """
        Get the latent space variable data for training and validation
        """ 
        try: 
            mat = scipy.io.loadmat(pathsBib.data_path + f"latent_data{self.config.latent_dim}.mat")
            data_train   = mat['vector_train']
            data_test    = mat['vector_test']
            data_gen     = mat['vector_gen']
        except:
            print(f"Error: DataBase not found, please check path or keys")

        X,Y = make_Sequence(self.config,data=data_train)
        self.train_dl, self.val_dl = make_DataLoader(torch.from_numpy(X),torch.from_numpy(Y),
                                                    batch_size=self.config.Batch_size,
                                                    drop_last=False, 
                                                    train_split=self.config.train_split)
        print(f"INFO: DataLoader Generated!")
        del mat, X, Y

#-------------------------------------------------

    def compile(self): 
        """
        Compile the model with optimizer, scheduler and loss function
        """
        self.loss_fn =   torch.nn.MSELoss()
        self.opt     =   torch.optim.Adam(self.model.parameters(),lr = self.config.lr, eps=1e-7)
        self.opt_sch =  [  
                        torch.optim.lr_scheduler.ExponentialLR(optimizer=self.opt, gamma= (1 - 0.01)) 
                        ]

#-------------------------------------------------

    def fit(self): 
        """
        Training Model, we use the fit() function 
        """

        s_t = time.time()
        history = fitting(  self.device, 
                            self.model,
                            self.train_dl, 
                            self.loss_fn,
                            self.config.Epoch,
                            self.opt,
                            self.val_dl, 
                            scheduler=self.opt_sch,
                            if_early_stop=self.config.early_stop,
                            patience=self.config.patience)
        e_t = time.time()
        cost_time = e_t - s_t
        
        print(f"INFO: Training FINISH, Cost Time: {cost_time:.2f}s")
        
        check_point = { "model":self.model.state_dict(),
                        "history":history,
                        "time":cost_time}
        
        save_model_name = pathsBib.model_path + self.filename + self.fmat
        torch.save(check_point,save_model_name)
        scipy.io.savemat(f'{pathsBib.log_path}laten_loss_history_dim{self.config.latent_dim}_{self.filename}.mat',
                        {'loss_history':history['train_loss'],
                         'val_loss_history':history['val_loss'],})
        print(f"INFO: The checkpoints have been saved at: {save_model_name}")
        print(f"INFO: The loss history has been saved at: {pathsBib.log_path}laten_loss_history_dim{self.config.latent_dim}_{self.filename}.mat")
        self.plot_laten_loss(history)
        print(f"INFO: The Loss plot at {pathsBib.log_path}laten_loss_history_dim{self.config.latent_dim}_{self.filename}.png")
    
    def plot_laten_loss(self, history):
        """
        Plot the latenrunner loss history and save the figure.
        """
        plt.figure()
        plt.plot(history['train_loss'], label='train', color='blue')
        plt.plot(history['val_loss'], label='test', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.title('Loss History')
        plt.legend()
        plot_path = f'{pathsBib.log_path}laten_loss_history_dim{self.config.latent_dim}_{self.filename}.png'
        plt.savefig(plot_path,dpi=300)
        plt.close()

#-------------------------------------------------


    def load_pretrain_model(self,model_type='pre'):
        """

        Load the pretrained model for beta VAE

        Args: 

            model_type  : ['pre', 'val','final']  (str) Choose from pre-trained, best valuation and final model 
        
        """
        
        model_type_all = ['pre','val','final']
        assert(model_type in model_type_all), print('ERROR: No type of the model matched')

        if      model_type == 'pre':    model_path = pathsBib.pretrain_path + self.filename               + self.fmat
        elif    model_type == 'val' :   model_path = pathsBib.model_path    + self.filename               + self.fmat
        elif    model_type == 'final' : model_path = pathsBib.chekp_path    + self.filename + '_final'    + self.fmat
        try:
            ckpoint = torch.load(model_path, map_location= self.device)
            
        except:
            print("ERROR: Model NOT found!")
            exit()
        stat_dict   = ckpoint['model']

        self.model.load_state_dict(stat_dict)
        self.history = ckpoint['history']

        
        print(f'INFO: the state dict has been loaded!')
        print(self.model.eval)


#-------------------------------------------------

    def post_process(self,if_window=True):
        """
        Post Processing of the temporal-dynamics predcition 
        Args:
            
            if_window   :   (bool) If compute the sliding-window error 

            if_pmap     :   (bool) If compute the Poincare Map 
        """ 
        import scipy.io
        from tqdm import tqdm
        
        fname = pathsBib.res_path + 'Latent_results_'+self.filename + ".mat"

        if (Path(fname).exists()):
            print(f"INFO (debug mode): {fname} have been generated, skip postprocess!")
            return
        
        try: 
            mat        = scipy.io.loadmat(pathsBib.data_path + f"latent_data{self.config.latent_dim}.mat")
            test_data   = np.array(mat['vector_test'])
            gen_data    = np.array(mat['vector_gen'])
        except:
            print(f"Error: DataBase not found, please check path or keys or try run the vae first")

        print(f"INFO: Test data loaded, SIZE = {test_data.shape}")
        Preds = make_Prediction(test_data   = test_data, 
                                model       = self.model,
                                device      = self.device,
                                in_dim      = self.config.in_dim,
                                next_step   = self.config.next_step,
                                laten_dim   = self.config.latent_dim,
                                Num_t       = self.config.test_t,)
        Preds_gen = make_Prediction(test_data   = gen_data,
                                model       = self.model,
                                device      = self.device,
                                in_dim      = self.config.in_dim,
                                next_step   = self.config.next_step,
                                laten_dim   = self.config.latent_dim,
                                Num_t       = self.config.test_t+self.config.train_t,)
        test_data = test_data.reshape(-1, self.config.test_t, self.config.latent_dim)
        gen_data  = gen_data.reshape(-1, self.config.test_t+self.config.train_t, self.config.latent_dim)
        if if_window: 
            print(f"Begin to compute the sliding window error")
            window_error, window_relerror = Sliding_Window_Error(test_data, 
                                                self.model, 
                                                self.device, 
                                                self.config.in_dim,
                                                window=50)
            window_error_gen, window_relerror_gen = Sliding_Window_Error(gen_data,
                                                self.model, 
                                                self.device, 
                                                self.config.in_dim,
                                                window=200)
            print(f"Sliding window error computed")
        else: 
            window_error, window_relerror = np.nan, np.nan
            window_error_gen, window_relerror_gen = np.nan, np.nan
        
        scipy.io.savemat(
                        pathsBib.res_path + 'Latent_results_'+self.filename + ".mat",
                        {'Preds':Preds, 
                        'test_data':test_data,
                        'Preds_gen':Preds_gen,
                        'gen_data':gen_data,
                        'window_error':window_error,
                        'window_relerror':window_relerror,
                        'window_error_gen':window_error_gen,
                        'window_relerror_gen':window_relerror_gen,
                        })
        print(f"INFO: The Latent results have been saved at {pathsBib.res_path + 'Latent_results_'+self.filename + '.mat'}")
        
