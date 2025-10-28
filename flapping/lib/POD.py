import h5py
import time
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.extmath import randomized_svd
from configs.vae import VAE_config as cfg


class POD:
    def __init__(self, datafile, n_train, n_test, re, path, n_modes=100, delta_t=1, case=0) -> None:
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
        self.case = case    
        self.casename = f'POD_Re{self.re}_dt{self.delta_t}_ntest{self.n_test}_nmodes{self.n_modes}_case{case}.mat'
        self.filename = path + self.casename

        print(f"POD file name:\n {self.filename}")

    def load_data(self):
        # load data
        # with h5py.File(self.datafile, 'r') as f:
        #     u_scaled = f['UV'][:]
        #     mean = f['mean'][:]
        #     std = f['std'][()]
        w = scipy.io.loadmat(self.datafile)['W']
        w = w.reshape(-1, self.n_train+self.n_test, 200, 200)

        #u_scaled = u_scaled[::self.delta_t]

        n_total = w.shape[1]
        #self.n_train = n_total - self.n_test
        print(f"N train: {self.n_train:d}, N test: {self.n_test:d}, N total {n_total:d}")
        print(f'u_scaled {w.shape}')

        self.u_train = w[self.case,:self.n_train,:,:]
        self.u_test = w[self.case,self.n_train:, :,:]
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
        # np.savez_compressed(
        #     file=self.filename,
        #     tm=self.temporal_modes,
        #     sm=self.spatial_modes,
        #     eig=self.eigVal
        # )

    def eval_POD(self):
        from lib.pp_space import get_Ek

        self.Ek_nm = np.zeros(self.n_modes)

        for nm in range(1, self.n_modes + 1):
            u_train_rec = self.temporal_modes[:, :nm].dot(self.spatial_modes[:, :nm].T).reshape(self.u_train.shape)

            self.Ek_nm[nm - 1] = get_Ek(self.u_train, u_train_rec)
            # print(f'POD train E = {self.Ek_nm[nm-1]:.4f}, {nm} modes')

        print(f'E POD: {self.Ek_nm}')
        data_dict = scipy.io.loadmat(self.filename)
        data_dict['Ek'] = self.Ek_nm
        scipy.io.savemat(
            self.filename,
            data_dict
        )


# Test code
if __name__ == "__main__":
    # data
    datafile = "../data/Re200_cylinder_ALL_15_500_200_200.mat"

    re = 200
    delta_t = 1
    n_modes = 10
    n_test = 100 // delta_t

    POD = POD(datafile, n_test, re, '../res/', n_modes, delta_t)
    POD.load_data()
    POD.get_POD()
    POD.eval_POD()
