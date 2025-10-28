import h5py
import time
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.extmath import randomized_svd
# from configs.vae import VAE_config as cfg



class POD:
    def __init__(self, datafile, n_train, n_test, re, n_modes=100, delta_t=1) -> None:
        """
        A runner for POD

        Args:

            datafile        :       (Str) Path of training data
            n_test          :       (int) Number of test timesteps (not used for POD)
            re              :       (int) Reynolds number
            path            :       (str) Path to save POD results
            n_modes         :       (int) Number of POD modes to calculate
            delta_t         :       (int) Steps between snapshots to use
        """

        self.datafile = datafile
        self.n_modes = n_modes
        self.delta_t = delta_t
        self.n_train = n_train
        self.n_test = n_test
        self.re = re

        

    def load_data(self, path, case=10):
        self.case = case    
        self.casename = f'POD_of_VAE_T_rec_nmodes{self.n_modes}_case{self.case+1}.mat'
        self.filename = path + self.casename
        print(f"POD file name:\n {self.filename}")
        
        w = scipy.io.loadmat(self.datafile)['Rec_from_VAE_T_gen']
        w = w.reshape(-1, self.n_train+self.n_test, 200, 200)

        #u_scaled = u_scaled[::self.delta_t]

        n_total = w.shape[1]
        #self.n_train = n_total - self.n_test
        print(f"N train: {self.n_train:d}, N test: {self.n_test:d}, N total {n_total:d}")
        print(f'u_scaled {w.shape}')

        self.u_train = w[0,:self.n_train,:,:]
        self.u_test = w[0,self.n_train:, :,:]
        # self.u_train = w[cfg.train,:self.n_train,:,:].reshape(-1, 200, 200)
        # self.u_test = w[cfg.test,self.n_train:, :,:].reshape(-1, 200, 200)

    def get_POD(self):

        try:
            self.load_POD()
            print('POD loaded from file')
        except:
            print('Calculating POD')
            self.calc_POD()

    def load_POD(self):
        d = scipy.io.loadmat(self.filename)
        self.temporal_modes = d['tm']
        self.spatial_modes = d['sm']
        self.eigVal = d['eig']

    def calc_POD(self):

        u_train_flat = self.u_train.reshape(self.u_train.shape[0], -1)
        u_test_flat = self.u_test.reshape(self.u_test.shape[0], -1)

        print(f'POD u_train: {u_train_flat.shape}')
        print(f'POD u_test: {u_test_flat.shape}')

        print(f'U shape: {u_train_flat.shape}')
        # C matrix
        print('Calc C matrix', end="")
        start = time.time()
        C = u_train_flat.T.dot(u_train_flat)
        C = C / (self.n_train - 1)
        print(f': {(time.time() - start):.1f}s')
        print(f'C shape: {C.shape}')

        # SVD
        print('Calc SVD', end="")
        start = time.time()
        self.spatial_modes, self.eigVal, _ = randomized_svd(C, n_components=self.n_modes, random_state=0)
        print(f': {(time.time() - start):.1f}s')
        print(f'spatial_modes {self.spatial_modes.shape}')

        self.temporal_modes = u_train_flat.dot(self.spatial_modes)

        print(f'temporal_modes shape: {self.temporal_modes.shape}')
        print(f'spatial_modes shape: {self.spatial_modes.shape}')

        scipy.io.savemat(
            self.filename,
            {
                'tm': self.temporal_modes,
                'sm': self.spatial_modes,
                'eig': self.eigVal
            }
        )


# Test code
if __name__ == "__main__":
    # data
    datafile = "../figs/easy/One_GenCase_reconstruc_easyAttn_32in_32dmodel_1next_4dim_timeemb_4h_4nb_128ff_reluact_Noneoutact_200Epoch_135000N_TrueES_50P.mat"

    re = 200
    delta_t = 1
    n_modes = 10
    n_test = 100 // delta_t
    POD_train = POD(datafile, 400, n_test, re, n_modes, delta_t)
    for i in range(7, 8):
        print(f'Case {i+1}/15')
        POD_train.load_data(path='../res/POD_rec/', case=i)
        POD_train.get_POD()
