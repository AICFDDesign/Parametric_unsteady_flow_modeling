"""
Function supports visualisation

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
bwr = LinearSegmentedColormap.from_list(
    'custom_bwr', 
    ['blue', 'white', 'red']
)

# --------------------------------------------------------

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


# --------------------------------------------------------

def plotCompleteModes(modes, temporalModes, n_times, numberModes, fs, order, path, case=0):
    """
    Plot the obtained spatial modes and temporal evolution of the latent variables in frequency domain, using welch method

    Args:

        modes           : (NumpyArray)   Spatial modes

        temporal_modes  : (NumpyArray)   Latent varibles from VAE

        numberModes     : (int) Number of modes to be ploted

        fs              : (int) Sampling frequency of welch metod

        order           : (list/NumpyArray) The ranked results of modes accroding to energy level 

        path            : (str) Path to Save figure
    """
    from scipy import signal
    import numpy as np

    fig, ax = plt.subplots(numberModes, 2, figsize=(10, numberModes * 1.8), sharex='col')
    latent_dim = temporalModes.shape[1]
    temporalModes = temporalModes.reshape(-1, n_times, latent_dim)
    #case = 0
    # for mode in range(numberModes):

    for i, mode in np.ndenumerate(order):
        print(i, mode)
        i=i[1]
        Uplot = modes[mode, 0, :, :]

        Ulim = max(abs(Uplot.flatten()))

        # im = ax[i, 0].imshow(Uplot, cmap=bwr, vmin=-Ulim, vmax=Ulim,
        #                     extent=[-9, 87, -14, 14])

        im = ax[i, 0].pcolormesh(Uplot, alpha=None, shading='gouraud',
                   norm=None, cmap=bwr, vmin=-Ulim, vmax=Ulim,)
        
        # ax[mode,0].set_title('Mode ' + str(mode) + ', u')
        ax[i, 0].set_ylabel('y/c')
        fig.colorbar(im, ax=ax[i, 0], shrink=1.0, aspect=10)


        ##使用 Welch 方法 计算给定时间序列的 功率谱密度（Power Spectral Density, PSD），以分析信号在频域上的能量分布
        
        f, Pxx_den = signal.welch(temporalModes[case, :, mode], fs, nperseg=48, scaling='density')
        Pxx_den_normalized = (Pxx_den - Pxx_den.min()) / (Pxx_den.max() - Pxx_den.min())
        ax[i, 1].plot(f, Pxx_den_normalized, color='lightseagreen')
        ax[i, 1].axis(xmin=0, xmax=.33)
        annot_max(f, Pxx_den_normalized, ax=ax[i, 1])
        ax[i, 1].grid(color='whitesmoke', zorder=1)
        ax[i, 1].spines['top'].set_visible(False)
        ax[i, 1].spines['right'].set_visible(False)
        ax[i, 1].set_title('Mode ' + str(i + 1))
        if i == (numberModes - 1):
            ax[i, 0].set_xlabel('$x/c$')
            ax[i, 1].set_xlabel('$f/f_c$')

    plt.savefig(path + f'modes_case{case}.png', format='png', bbox_inches="tight", dpi=300)
    print("INFO: Figure saved to: ", path + f'modes_case{case}.png')

def plotCompleteModes_POD(modes, temporalModes, n_times, numberModes, fs, order, path, case=0):
    """
    Plot the obtained spatial modes and temporal evolution of the latent variables in frequency domain, using welch method

    Args:

        modes           : (NumpyArray)   Spatial modes

        temporal_modes  : (NumpyArray)   Latent varibles from VAE

        numberModes     : (int) Number of modes to be ploted

        fs              : (int) Sampling frequency of welch metod

        order           : (list/NumpyArray) The ranked results of modes accroding to energy level 

        path            : (str) Path to Save figure
    """
    from scipy import signal
    import numpy as np

    fig, ax = plt.subplots(numberModes, 2, figsize=(10, numberModes * 2.5), sharex='col')
    latent_dim = temporalModes.shape[1]
    temporalModes = temporalModes.reshape(-1, n_times, latent_dim)
    #case = 0
    # for mode in range(numberModes):

    for i, mode in np.ndenumerate(order):
        print(i, mode)
        i=i[0]
        Uplot = modes[mode, 0, :, :]

        Ulim = max(abs(Uplot.flatten()))

        # im = ax[i, 0].imshow(Uplot, cmap=bwr, vmin=-Ulim, vmax=Ulim,
        #                     extent=[-9, 87, -14, 14])

        im = ax[i, 0].pcolormesh(Uplot, alpha=None, shading='gouraud',
                   norm=None, cmap=bwr, vmin=-Ulim, vmax=Ulim, )
        
        # ax[mode,0].set_title('Mode ' + str(mode) + ', u')
        ax[i, 0].set_ylabel('y/c')
        fig.colorbar(im, ax=ax[i, 0], shrink=1.0, aspect=10)


        ##使用 Welch 方法 计算给定时间序列的 功率谱密度（Power Spectral Density, PSD），以分析信号在频域上的能量分布
        
        f, Pxx_den = signal.welch(temporalModes[case, :, mode], fs, nperseg=48, scaling='density')
        Pxx_den_normalized = (Pxx_den - Pxx_den.min()) / (Pxx_den.max() - Pxx_den.min())
        ax[i, 1].plot(f, Pxx_den_normalized, color='lightseagreen')
        ax[i, 1].axis(xmin=0, xmax=.33)
        annot_max(f, Pxx_den_normalized, ax=ax[i, 1])
        ax[i, 1].grid(color='whitesmoke', zorder=1)
        ax[i, 1].spines['top'].set_visible(False)
        ax[i, 1].spines['right'].set_visible(False)
        ax[i, 1].set_title('Mode ' + str(i + 1))
        if i == (numberModes - 1):
            ax[i, 0].set_xlabel('$x/c$')
            ax[i, 1].set_xlabel('$f/f_c$')

    plt.savefig(path + f'modes_case{case}.png', format='png', bbox_inches="tight", dpi=300)
    print("INFO: Figure saved to: ", path + f'modes_case{case}.png')    


# --------------------------------------------------------

def correlationMatrix(temporalModes, order, path):
    """
    Visualisation of the correlation matrix to demonstrate the orthogonality 

    Args:   

        temporalModes   :   (NumpyArray) Latent variables encoded by VAE

        order           : (list/NumpyArray) The ranked results of modes accroding to energy level 

        path            : (str) Path to Save figure
    
    """
    import pandas as pd
    import seaborn as sns
    import numpy as np

    df = pd.DataFrame(temporalModes[:, order].reshape(-1, temporalModes.shape[-1]))

    # Create the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle; True = do NOT show
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(4, 3))

    # Generate a colormap
    cmap = sns.color_palette("icefire", as_cmap=True)

    n_modes = temporalModes.shape[1]

    axis_labels = np.arange(1, n_modes + 1).tolist()
    axis_labels = list(map(str, axis_labels))
    if n_modes >= 20:
        for i in range(0, n_modes, 2):
            axis_labels[i] = ''

    # Draw the heatmap with the mask and correct aspect ratio
    # More details at https://seaborn.pydata.org/generated/seaborn.heatmap.html
    sns.heatmap(
        np.abs(corr),  # The data to plot
        # mask=mask,     # Mask some cells
        cmap=cmap,  # What colors to plot the heatmap as
        annot=False,  # Should the values be plotted in the cells?
        fmt=".2f",
        vmax=1,  # The maximum value of the legend. All higher vals will be same color
        vmin=0,  # The minimum value of the legend. All lower vals will be same color
        # center=0.75,      # The center value of the legend. With divergent cmap, where white is
        square=True,  # Force cells to be square
        linewidths=0.5,  # Width of lines that divide cells
        # cbar_kws={"shrink": .5}  # Extra kwargs for the legend; in this case, shrink by 50%
        # norm=LogNorm()
        xticklabels=axis_labels,
        yticklabels=axis_labels,
    )
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.savefig(path + 'matrix.png', format='png', bbox_inches="tight",dpi = 300)
    print("INFO: Figure saved to: ", path + 'matrix.png')


# --------------------------------------------------------

def plotTemporalSeries(modes, n_time, case, path, tag):
    """
    
    Visualize the temproal evolution of the latent variables 

    modes   :   (NumpyArray) The latent variables from VAE

    path    :   (str) Path to Save Figure
    
    """
    latent_dim = modes.shape[1]
    modes_all = modes.reshape(-1, n_time, latent_dim)
    Cases = np.size(case)
    print(f"The number of All cases = {Cases}")
    fig, ax = plt.subplots(latent_dim, 1, figsize=(16, latent_dim * 2.0), sharex='col')

    for i in range(latent_dim):
        for k in range(Cases):
            if i == 0:
                ax[i].plot(modes_all[k, :500, i], label='CASE ' + str(case[k]+1))
            else:
                ax[i].plot(modes_all[k, :500, i])
        ax[i].set_title('Mode ' + str(i+1))
        ax[i].set_xlim(0, n_time)
        #ax[i].axis(ymin=-2.5, ymax=2.5)

        #ax[i].legend(loc='upper right')
        ax[i].grid()
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol = 11)
    ax[latent_dim - 1].set_xlabel('Time step')

    plt.savefig(path + f'series_{tag}.png', format='png', bbox_inches="tight", dpi=300)
    print("INFO: Figure saved to: ", path + f'series_{tag}.png')

# --------------------------------------------------------

def plotTemporal3DSeries(modes, n_time, case, path, tag):
    """
    
    Visualize the temproal evolution of the latent variables 

    modes   :   (NumpyArray) The latent variables from VAE

    path    :   (str) Path to Save Figure
    
    """
    latent_dim = modes.shape[1]
    modes_all = modes.reshape(-1, n_time, latent_dim)
    Cases = np.size(case)
    print(f"The number of All cases = {Cases}")
    ax = plt.figure().add_subplot(projection='3d')
    for k in range(Cases):
        ax.plot(modes_all[k, :, 0],modes_all[k, :, 1],modes_all[k, :, 2], label='CASE ' + str(case[k]+1),lw=1)
    ax.set_xlabel("m1")
    ax.set_ylabel("m2")
    ax.set_zlabel("m3")
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol = 6)
    plt.savefig(path + f'series_3D_{tag}.png', format='png', bbox_inches="tight", dpi=300)
    print("INFO: Figure saved to: ", path + f'series_3D_{tag}.png')

def plotTemporal3DSeries_combine(modes_tr, modes_gen, n_tr, n_gen, case_tr, case_gen, path):
    
    latent_dim = modes_tr.shape[1]
    modes_tr_all = modes_tr.reshape(-1, n_tr, latent_dim)
    Cases_tr = np.size(case_tr)
    modes_gen_all = modes_gen.reshape(-1, n_gen, latent_dim)
    Cases_gen = np.size(case_gen)
    print(f"cases_tr = {Cases_tr}, cases_gen = {Cases_gen}")
    ax = plt.figure().add_subplot(projection='3d')
    for k in range(6):
        ax.plot(modes_tr_all[k, :, 0],modes_tr_all[k, :, 1],modes_tr_all[k, :, 2], label='CASE ' + str(case_tr[k]+1),lw=1)
    for k in range(2):
        ax.plot(modes_gen_all[k, :, 0],modes_gen_all[k, :, 1],modes_gen_all[k, :, 2], label='CASE ' + str(case_gen[k]+1),lw=1)
    ax.set_xlabel("m1")
    ax.set_ylabel("m2")
    ax.set_zlabel("m3")
    ax.legend(loc='upper left', bbox_to_anchor=(-0.05, 1), ncol = 4)
    plt.savefig(path + f'series_3D_2S.png', format='png', bbox_inches="tight", dpi=300)
    print("INFO: Figure saved to: ", path + f'series_3D_2S.png')
    plt.close()
    ax = plt.figure().add_subplot(projection='3d')
    for k in range(6,Cases_tr):
        ax.plot(modes_tr_all[k, :, 0],modes_tr_all[k, :, 1],modes_tr_all[k, :, 2], label='CASE ' + str(case_tr[k]+1),lw=1)
    for k in range(2,Cases_gen):
        ax.plot(modes_gen_all[k, :, 0],modes_gen_all[k, :, 1],modes_gen_all[k, :, 2], label='CASE ' + str(case_gen[k]+1),lw=1)
    ax.set_xlabel("m1")
    ax.set_ylabel("m2")
    ax.set_zlabel("m3")
    ax.legend(loc='upper left', bbox_to_anchor=(-0.05, 1), ncol = 4)
    plt.savefig(path + f'series_3D_P+S.png', format='png', bbox_inches="tight", dpi=300)
    print("INFO: Figure saved to: ", path + f'series_3D_P+S.png')
def plotTemporal3DSeries_15single(modes_tr, modes_gen, n_tr, n_gen, case_tr, case_gen, path):
    latent_dim = modes_tr.shape[1]
    modes_tr_all = modes_tr.reshape(-1, n_tr, latent_dim)
    Cases_tr = np.size(case_tr)
    modes_gen_all = modes_gen.reshape(-1, n_gen, latent_dim)
    Cases_gen = np.size(case_gen)
    print(f"cases_tr = {Cases_tr}, cases_gen = {Cases_gen}")
    all_cases = np.concatenate([case_tr, case_gen])
    all_types = np.array(['tr'] * len(case_tr) + ['gen'] * len(case_gen))
    sort_idx = np.argsort(all_cases)
    all_cases_sorted = all_cases[sort_idx]
    all_types_sorted = all_types[sort_idx]

    fig, axes = plt.subplots(3, 5, figsize=(20, 12), subplot_kw={'projection': '3d'})
    axes = axes.flatten()

    tr_idx = 0
    gen_idx = 0
    for i, (case, typ) in enumerate(zip(all_cases_sorted, all_types_sorted)):
        ax = axes[i]
        if typ == 'tr':
            # 找到case在case_tr中的索引
            idx = np.where(case_tr == case)[0][0]
            data = modes_tr_all[idx, :, :]
            label = f"Train CASE {case+1}"
        else:
            idx = np.where(case_gen == case)[0][0]
            data = modes_gen_all[idx, :, :]
            label = f"Gen. CASE {case+1}"
        ax.plot(data[:, 0], data[:, 1], data[:, 2], label=label)
        ax.set_title(label)
        ax.set_xlabel("m1")
        ax.set_ylabel("m2")
        ax.set_zlabel("m3")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.set_zticks([-2, -1, 0, 1, 2])
        #ax.legend()
    # 隐藏多余的子图
    for j in range(len(all_cases_sorted), 15):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(path + 'series_3D_15cases.png', format='png', bbox_inches="tight", dpi=300)
    print("INFO: Figure saved to: ", path + 'series_3D_15cases.png')

def plotTemporalPDF(modes, n_time, case, path, tag):
    """
    
    Visualize the temproal evolution of the latent variables 

    modes   :   (NumpyArray) The latent variables from VAE

    path    :   (str) Path to Save Figure
    
    """
    from scipy import stats
    from scipy.stats import gaussian_kde
    latent_dim = modes.shape[1]
    modes_all = modes.reshape(-1, n_time, latent_dim)
    Cases = np.size(case)
    print(f"The number of All cases = {Cases}")
    fig, ax = plt.subplots(1,latent_dim,  figsize=(latent_dim * 5,5))

    for i in range(latent_dim):
        ymax = 0
        for k in range(Cases):
            #y=stats.norm.pdf(loc=0,scale=1,x=modes_all[k, :, i])
            x = modes_all[k, :, i]
            if np.max(x)>ymax:
                ymax = np.max(x)
            kde = stats.gaussian_kde(x)
            x_sorted = np.sort(x)
            y = kde(x_sorted)
            if i == 0:
                ax[i].plot(x_sorted,y, label='CASE ' + str(case[k]+1))
            else:
                ax[i].plot(x_sorted,y)
        ax[i].set_title('Mode ' + str(i+1))
        #ax[i].set_xlim(0, n_time)
        ax[i].axis(ymin=0, ymax=ymax)

        #ax[i].legend(loc='upper right')
        ax[i].grid()
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol = 11)
    #ax[latent_dim - 1].set_xlabel('Time step')

    plt.savefig(path + f'series_PDF_{tag}.png', format='png', bbox_inches="tight", dpi=300)
    print("INFO: Figure saved to: ", path + f'series_PDF_{tag}.png')    


# --------------------------------------------------------
def plotEcum(Ecum, Ecum_test, Ecum_gen, path):
    """
    Show the accumlative energy level 

    Ecum    :   (NumpyArray) Obtained energy level 

    path    :   (str) Path to Save Figure

    """
    import numpy as np

    mean_Ecum = np.mean(Ecum, axis=0)
    std_Ecum = np.std(Ecum, axis=0)
    mean_Ecum_test = np.mean(Ecum_test, axis=0)
    std_Ecum_test = np.std(Ecum_test, axis=0)
    mean_Ecum_gen = np.mean(Ecum_gen, axis=0)
    std_Ecum_gen = np.std(Ecum_gen, axis=0)

    latent_dim = Ecum.shape[1]
    x = np.arange(1, latent_dim + 1)  # 横坐标：1 到 latent_dim

    plt.figure()
    plt.plot(x, mean_Ecum, label='Mean(train)', color='blue')
    plt.fill_between(x, mean_Ecum - std_Ecum, mean_Ecum + std_Ecum, color='blue', alpha=0.08)
    # plt.plot(x, mean_Ecum_test, label='Mean(test)', color='orange')
    # plt.fill_between(x, mean_Ecum_test - std_Ecum_test, mean_Ecum_test + std_Ecum_test, color='orange', alpha=0.13)
    plt.plot(x, mean_Ecum_gen, label='Mean(gen.)', color='green')
    plt.fill_between(x, mean_Ecum_gen - std_Ecum_gen, mean_Ecum_gen + std_Ecum_gen, color='green', alpha=0.08)
    plt.xticks(x)
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.ylim(0,1.0)
    plt.xlabel('Number of modes')
    plt.ylabel('Cumulative Ek')
    plt.grid(True)
    plt.legend()
    plt.savefig(path + 'Ecum_with_std.png', format='png', bbox_inches="tight", dpi=300)
    print("INFO: Figure saved to: ", path + 'Ecum_with_std.png')

    # fig = plt.figure()
    # plt.plot(np.arange(1, Ecum.shape[1] + 1), Ecum)
    # plt.xlabel('Number of modes')
    # plt.ylabel('Cumulative Ek')
    # plt.grid()

    # plt.savefig(path + 'Ecum.png', format='png', bbox_inches="tight", dpi=300)
    # print("INFO: Figure saved to: ", path + 'Ecum.png')

def plotEcum_POD(Ecum, path):
    """
    Show the accumlative energy level 

    Ecum    :   (NumpyArray) Obtained energy level 

    path    :   (str) Path to Save Figure

    """
    import numpy as np

    latent_dim = Ecum.shape[0]
    x = np.arange(1, latent_dim + 1)  # 横坐标：1 到 latent_dim

    plt.figure()
    plt.plot(x, Ecum, label='Mean(train)', color='blue')
    # plt.plot(x, mean_Ecum_test, label='Mean(test)', color='orange')
    # plt.fill_between(x, mean_Ecum_test - std_Ecum_test, mean_Ecum_test + std_Ecum_test, color='orange', alpha=0.13)
    plt.xticks(x)
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.ylim(0,1.0)
    plt.xlabel('Number of modes')
    plt.ylabel('Cumulative Ek')
    plt.grid(True)
    # plt.legend()
    plt.savefig(path + 'Ecum.png', format='png', bbox_inches="tight", dpi=300)
    print("INFO: Figure saved to: ", path + 'Ecum.png')


# --------------------------------------------------------
def plotNLmodeField(modes, values, path, order):
    """
    Visualize the non-linear mode

    Args:

        modes   :   (NumpyArray) The spatial modes using decoder.

        values  :   (float) A non-zero value as the element in latent vector

        path    :   (str) Path to Save figure
    
    """
    import numpy as np

    #valuesToPlot = np.array([-2.,  -1.,  0., 1., 2.])
    valuesToPlot = np.array([-2., -1.5,  -1., -0.5,  0., 0.5, 1., 1.5, 2.])
    values = values.reshape(-1)

    # Find the indices of values_array elements in main_array
    indices = []
    for value in valuesToPlot:
        matching_indices = np.where(np.abs(values - value) < 1e-3)[0]
        indices.extend(matching_indices)

    indices_array = np.array(indices)
    print(indices_array)
    values = values[indices_array]
    print(values)

    
    latent_dim = order.size
    print(latent_dim,valuesToPlot.size)

    fig, ax = plt.subplots(valuesToPlot.size, latent_dim,  figsize=(5*latent_dim,2.5*valuesToPlot.size ), sharex='col', sharey='row')

    
    for idx, value in enumerate(values):
        for i in range(latent_dim):
            #print(idx, value)
            Wlim = max(abs(modes[i,indices_array , :, :].flatten()))
            Wplot = modes[i,indices_array[idx],  :, :]

            im = ax[idx, i].pcolormesh(Wplot, alpha=None, shading='gouraud',
                   norm=None, cmap=bwr, vmin=-Wlim, vmax=Wlim, )
            print(idx,value,i)
            ax[0, i].set_title('Mode {}'.format(i+1), fontsize=11)
            # ax[0, i].set_title('$s_1 = {}$'.format(round(value, 1)))
            ax[idx, 0].set_ylabel('$s_i = {}$'.format(round(value, 1))+'\n'+'y/c')
            
            ax[valuesToPlot.size-1, i].set_xlabel('x/c')
        #fig.text(0.02, 0.965 - idx * .184, '$s_i = {}$'.format(round(value, 1)))#, fontsize=11, ha="center")
        fig.colorbar(im, ax=ax[idx, -1])#, shrink=0.7, aspect=10)

    fig.set_tight_layout(True)

    plt.savefig(path + 'NLfield.png', format='png', bbox_inches="tight",dpi=300)
    print("INFO: Figure saved to: ", path + 'NLfield.png')

    # --------------------------------------------------------
def plotPARAmodeField(modes, path, cases_index, tag):
    """
    Visualize the non-linear mode

    Args:

        modes   :   (NumpyArray) The spatial modes using decoder.

        path    :   (str) Path to Save figure

        tag     :   (str) train or gen
    
    """
    import numpy as np

    cases = modes.shape[1]
    latent_dim = modes.shape[0]
    print('latene_dim and cases:',latent_dim,cases)

    fig, ax = plt.subplots(cases, latent_dim,  figsize=(5*latent_dim,2.5*cases ), sharex='col', sharey='row')

    
    for idx, value in enumerate(cases_index):
        for i in range(latent_dim):
            #print(idx, value)
            Wlim = max(abs(modes[i,idx , :, :].flatten()))
            Wplot = modes[i,idx,  :, :]

            im = ax[idx, i].pcolormesh(Wplot, alpha=None, shading='gouraud',
                   norm=None, cmap=bwr, vmin=-Wlim, vmax=Wlim, )
            print(idx,i)
            ax[0, i].set_title('Mode {}'.format(i+1), fontsize=11)
            # ax[0, i].set_title('$s_1 = {}$'.format(round(value, 1)))
            ax[idx, 0].set_ylabel('CASE {}'.format(value+1)+'\n'+'y/c')
            
            ax[cases-1, i].set_xlabel('x/c')
        #fig.text(0.02, 0.965 - idx * .184, '$s_i = {}$'.format(round(value, 1)))#, fontsize=11, ha="center")
        fig.colorbar(im, ax=ax[idx, -1])#, shrink=0.7, aspect=10)

    fig.set_tight_layout(True)

    plt.savefig(path + 'Parafield_{}.png'.format(tag), format='png', bbox_inches="tight",dpi=300)
    print("INFO: Figure saved to: ", path + 'Parafield_{}.png'.format(tag))


def vis_bvae(modes_file, training_file):
    """
    Visualisation of the beta-VAE results 

    Args:   
        modes_file      :   The file saves the post-processing results of VAE 

        training_file   : The history and log of training the model
    
    """
    import h5py
    from lib.init import pathsBib
    from configs.vae import VAE_config as cfg
    from pathlib import Path
    import scipy.io

    path = pathsBib.fig_path + 'pbVAE/'
    Path(path).mkdir(exist_ok=True)

    data = scipy.io.loadmat(modes_file)
    temporalModes = data['vector_train']
    temporalModes_test = data['vector_test']
    temporalModes_gen = data['vector_gen']
    order = data['order']
    Ecum = data['Ecum']
    order_all = data['order_all']
    Ecum_all = data['Ecum_all']
    Ecum_test = data['Ecum_test']
    Ecum_gen = data['Ecum_gen']
    modes = data['modes']
    modes_para_train = data['modes_para_train']
    modes_para_gen = data['modes_para_gen']
    zero_output = data['zero_output']
    Ek_t_tr = data['Ek_tr']
    Ek_t_te = data['Ek_te']
    Ek_t_gen = data['Ek_gen']
    NLvalues = data['NLvalues']
    NLmodes = data['NLmodes']
    # with h5py.File(modes_file, 'r') as f:
    #     print("Keys: %s" % f.keys())
    #     temporalModes = f['vector'][:, :]
    #     temporalModes_test = f['vector_test'][:, :]
    #     modes = f['modes'][:, :]
    #     order = f['order'][:]
    #     Ecum = f['Ecum'][:]
    #     NLvalues = f['NLvalues'][:]
    #     NLmodes = f['NLmodes'][:]

    # Re40 case is sampled at 1tc, Re100 case is sampled at tc/5
    if 'Re200' in modes_file:
        fs = 1
    else:
        fs = 5

    plotPARAmodeField(modes_para_train, path, cfg.train, tag='train')
    plotPARAmodeField(modes_para_gen, path, cfg.test, tag='gen')
    plotNLmodeField(NLmodes, NLvalues, path, order_all)
    plotCompleteModes(modes, temporalModes, cfg.n_train, modes.shape[0], fs, order_all, path, case=0)
    plotTemporalSeries(temporalModes, cfg.n_train, cfg.train, path, tag='train')
    plotTemporalSeries(temporalModes_gen, cfg.n_train+cfg.n_test, cfg.test, path, tag='gen')
    plotTemporal3DSeries(temporalModes, cfg.n_train, cfg.train, path, tag='train')
    plotTemporal3DSeries(temporalModes_gen, cfg.n_train+cfg.n_test, cfg.test, path, tag='gen')
    plotTemporal3DSeries_combine(temporalModes, temporalModes_gen, cfg.n_train, cfg.n_train+cfg.n_test, cfg.train, cfg.test, path)
    plotTemporal3DSeries_15single(temporalModes, temporalModes_gen, cfg.n_train, cfg.n_train+cfg.n_test, cfg.train, cfg.test, path)
    plotTemporalPDF(temporalModes, cfg.n_train, cfg.train, path, tag='train')
    plotTemporalPDF(temporalModes_gen, cfg.n_train+cfg.n_test, cfg.test, path, tag='gen')
    correlationMatrix(temporalModes, order_all, path)
    plotEcum(Ecum, Ecum_test, Ecum_gen, path)

def vis_pod(POD):
    """
    Visualisaton of POD results 

    Args:

        POD : (lib.POD.POD) The running object for implmenting POD 
    
    """

    import h5py
    from lib.init import pathsBib
    from pathlib import Path
    from configs.vae import VAE_config as cfg

    path = pathsBib.fig_path + 'POD/'
    Path(path).mkdir(exist_ok=True)

    # Re40 case is sampled at 1tc, Re100 case is sampled at tc/5
    if POD.re==200:
        fs = 1
    else:
        fs = 5

    shape = [POD.spatial_modes.shape[1],
            -1,
            POD.u_train.shape[1],
            POD.u_train.shape[2]]
    sm = np.swapaxes(POD.spatial_modes, 0, 1).reshape(shape)

    plotCompleteModes_POD(sm,
                    POD.temporal_modes,
                    cfg.n_train,
                    POD.n_modes,
                    fs,
                    np.array(range(POD.n_modes)),
                    path,case=0)

    plotTemporalSeries(POD.temporal_modes, cfg.n_train, path, tag='train')
    plotEcum_POD(POD.Ek_nm, path)
