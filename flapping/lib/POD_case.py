import h5py
import time
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.extmath import randomized_svd
from matplotlib.colors import LinearSegmentedColormap
bwr = LinearSegmentedColormap.from_list(
    'custom_bwr', 
    ['blue', 'white', 'red']
)


def annot_max(x, y, ax=None):
        import numpy as np

        xmax = x[np.argmax(y)]
        ymax = y.max()
        text = "$f/f_c={:.3f}$".format(xmax)
        if not ax:
            ax = plt.gca()
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
        kw = dict(xycoords='data', textcoords="axes fraction",
                arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
        ax.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)

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
        self.casename = f'POD_Re{self.re}_dt{self.delta_t}_ntest{self.n_test}_nmodes{self.n_modes}_case{self.case}.mat'
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
        from pp_space_POD import get_Ek

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

    def plotCompleteModes_POD(self, path):
        from scipy import signal
        shape = [self.spatial_modes.shape[1],
                -1,
                self.u_train.shape[1],
                self.u_train.shape[2]]
        sm = np.swapaxes(self.spatial_modes, 0, 1).reshape(shape)
    
        fig, ax = plt.subplots(self.n_modes, 2, figsize=(10, self.n_modes * 2.5), sharex='col')
        latent_dim = self.temporal_modes.shape[1]
        temporalModes = self.temporal_modes.reshape(-1, self.n_train, latent_dim)
        #self.case = 0
        # for mode in range(self.n_modes):
        order = np.array(range(self.n_modes))

        for i, mode in np.ndenumerate(order):
            print(i, mode)
            i=i[0]
            Uplot = sm[mode, 0, :, :]

            Ulim = max(abs(Uplot.flatten()))

            # im = ax[i, 0].imshow(Uplot, cmap=bwr, vmin=-Ulim, vmax=Ulim,
            #                     extent=[-9, 87, -14, 14])

            im = ax[i, 0].pcolormesh(Uplot, alpha=None, shading='gouraud',
                    norm=None, cmap=bwr, vmin=-Ulim, vmax=Ulim, )
            
            # ax[mode,0].set_title('Mode ' + str(mode) + ', u')
            ax[i, 0].set_ylabel('y/c')
            fig.colorbar(im, ax=ax[i, 0], shrink=1.0, aspect=10)


            ##使用 Welch 方法 计算给定时间序列的 功率谱密度（Power Spectral Density, PSD），以分析信号在频域上的能量分布
            
            f, Pxx_den = signal.welch(temporalModes[0, :, mode], 1, nperseg=48, scaling='density')
            Pxx_den_normalized = (Pxx_den - Pxx_den.min()) / (Pxx_den.max() - Pxx_den.min())
            ax[i, 1].plot(f, Pxx_den_normalized, color='lightseagreen')
            ax[i, 1].axis(xmin=0, xmax=.33)
            annot_max(f, Pxx_den_normalized, ax=ax[i, 1])
            ax[i, 1].grid(color='whitesmoke', zorder=1)
            ax[i, 1].spines['top'].set_visible(False)
            ax[i, 1].spines['right'].set_visible(False)
            ax[i, 1].set_title('Mode ' + str(i + 1))
            if i == (self.n_modes - 1):
                ax[i, 0].set_xlabel('$x/c$')
                ax[i, 1].set_xlabel('$f/f_c$')

        plt.savefig(path + f'modes_case{self.case}.png', format='png', bbox_inches="tight", dpi=300)
        print("INFO: Figure saved to: ", path + f'modes_case{self.case}.png')    
    def plotTemporalSeries(self, path, tag):

        latent_dim = self.temporal_modes.shape[1]
        modes_all = self.temporal_modes.reshape(-1, self.n_train, latent_dim)
        Cases = modes_all.shape[0]
        # print(f"The number of All cases = {Cases}")
        fig, ax = plt.subplots(latent_dim, 1, figsize=(16, latent_dim * 2.0), sharex='col')

        for i in range(latent_dim):
            for k in range(Cases):
                ax[i].plot(modes_all[k, :500, i])#, label='Mode ' + str(i))
            ax[i].set_title('Mode ' + str(i+1))
            ax[i].set_xlim(0, self.n_train)
            #ax[i].axis(ymin=-2.5, ymax=2.5)

            #ax[i].legend(loc='upper right')
            ax[i].grid()
        ax[latent_dim - 1].set_xlabel('Time step')

        plt.savefig(path + f'series_{tag}_case{self.case}.png', format='png', bbox_inches="tight", dpi=300)
        print("INFO: Figure saved to: ", path + f'series_{tag}_case{self.case}.png')
    def plotEcum_POD(self, path):

        latent_dim = self.Ek_nm.shape[0]
        x = np.arange(1, latent_dim + 1)  # 横坐标：1 到 latent_dim

        plt.figure()
        plt.plot(x, self.Ek_nm, label='Mean(train)', color='blue')
        # plt.plot(x, mean_Ecum_test, label='Mean(test)', color='orange')
        # plt.fill_between(x, mean_Ecum_test - std_Ecum_test, mean_Ecum_test + std_Ecum_test, color='orange', alpha=0.13)
        plt.xticks(x)
        plt.yticks(np.arange(0, 1.2, 0.2))
        plt.ylim(0,1.0)
        plt.xlabel('Number of modes')
        plt.ylabel('Cumulative Ek')
        plt.grid(True)
        # plt.legend()
        plt.savefig(path + f'Ecum_case{self.case}.png', format='png', bbox_inches="tight", dpi=300)
        print("INFO: Figure saved to: ", path + f'Ecum_case{self.case}.png')


from pathlib import Path
# Test code
if __name__ == "__main__":
    # data
    datafile = "../data/Re200_cylinder_ALL_15_500_200_200.mat"

    re = 200
    delta_t = 1
    n_modes = 10
    n_test = 100 // delta_t
    n_train = 400

    path = '../figs/POD/'
    Path(path).mkdir(exist_ok=True)
    for i in range(15):
        print(f"Case {i}")
        pod_instance = POD(datafile, n_train, n_test, re, '../res/', n_modes, delta_t, case=i)
        pod_instance.load_data()
        pod_instance.get_POD()
        pod_instance.eval_POD()
        pod_instance.plotCompleteModes_POD(path)
        pod_instance.plotTemporalSeries(path, 'train')
        pod_instance.plotEcum_POD(path)
        print(i, "done")
