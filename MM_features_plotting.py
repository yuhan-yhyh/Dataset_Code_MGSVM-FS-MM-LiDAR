# Goal: This script is to plot the MM features for the four material classes: concrete, plastic, stone, and wood.

""" ********************************************   1. Import packages   ******************************************** """
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

""" ********************************************   2. Define functions  ******************************************** """
### 2.1 Output filenames
def file_name_partial(y_axis_vec):
    filename_index = '_'
    for i, tit in enumerate(y_axis_vec):
        if tit == 'd [mm]':
            tit = 'd'
        if i > 0:
            filename_index = filename_index + '+'
        filename_index = filename_index + tit
    filename_index = filename_index + '_SC2'
    print('Modality combination: ',filename_index[1:-4])
    return filename_index

### 2.1 Plot MM features
def plotting_MM_features(fig_tags):
    font = {'family':'Arial',
            'size':'10',
            'weight':'normal'}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(3,4, figsize=(8,5))
    for ind, dim in enumerate(dim_vec):
        y_axis = y_axis_vec[ind]
        if y_axis == 'd [mm]':
            ylim = [-2,2]
        elif y_axis == 'DoLP':
            ylim = [0, 1]
        elif y_axis == 'R':
            ylim = [0, 1]

        indices = [id for id, ele in enumerate(y) if ele == '1']
        mean_value = X[indices,ind*WL.shape[0]:(ind+1)*WL.shape[0]].mean(axis = 0)
        std_value = X[indices,ind*WL.shape[0]:(ind+1)*WL.shape[0]].std(axis = 0)
        axs[ind,0].plot(WL, mean_value,'k-')
        axs[ind,0].fill_between(WL, mean_value + std_value, mean_value - std_value, facecolor = 'k',alpha=0.2)
        axs[ind,0].set_ylabel(y_axis)
        axs[ind,0].set_ylim(ylim)
        axs[ind,0].set_yticks(np.linspace(ylim[0],ylim[1],num=5))
        axs[ind,0].set_yticklabels(np.linspace(ylim[0],ylim[1],num=5))
        if ind == 0:
            axs[ind,0].set_title(material_class_name[0])
        if ind == 2:
            axs[ind,0].set_xlim([550, 950])
            axs[ind,0].set_xticks(np.linspace(600,900,num=4).astype(int))
            axs[ind,0].set_xticklabels(np.linspace(600,900,num=4).astype(int))
            axs[ind,0].set_xlabel(r'$\lambda$ [nm]')
        else:
            axs[ind,0].set_xlim([550, 950])
            axs[ind,0].set_xticks(np.linspace(600,900,num=4).astype(int))
            axs[ind,0].set_xticklabels([])

        indices = [id for id, ele in enumerate(y) if ele == '2']
        mean_value = X[indices,ind*WL.shape[0]:(ind+1)*WL.shape[0]].mean(axis = 0)
        std_value = X[indices,ind*WL.shape[0]:(ind+1)*WL.shape[0]].std(axis = 0)
        axs[ind,1].plot(WL, mean_value,'k-')
        axs[ind,1].fill_between(WL, mean_value + std_value, mean_value - std_value, facecolor = 'k',alpha=0.2)
        axs[ind,1].set_yticklabels([])
        axs[ind,1].set_ylim(ylim)
        axs[ind,1].set_yticks(np.linspace(ylim[0],ylim[1],num=5))
        if ind == 0:
            axs[ind,1].set_title(material_class_name[1])
        if ind == 2:
            axs[ind,1].set_xlim([550, 950])
            axs[ind,1].set_xticks(np.linspace(600,900,num=4).astype(int))
            axs[ind,1].set_xticklabels(np.linspace(600,900,num=4).astype(int))
            axs[ind,1].set_xlabel(r'$\lambda$ [nm]')
        else:
            axs[ind,1].set_xlim([550, 950])
            axs[ind,1].set_xticks(np.linspace(600,900,num=4).astype(int))
            axs[ind,1].set_xticklabels([])

        indices = [id for id, ele in enumerate(y) if ele == '3']
        mean_value = X[indices,ind*WL.shape[0]:(ind+1)*WL.shape[0]].mean(axis = 0)
        std_value = X[indices,ind*WL.shape[0]:(ind+1)*WL.shape[0]].std(axis = 0)
        axs[ind,2].plot(WL, mean_value,'k-')
        axs[ind,2].fill_between(WL, mean_value + std_value, mean_value - std_value, facecolor = 'k',alpha=0.2)
        axs[ind,2].set_yticklabels([])
        axs[ind,2].set_ylim(ylim)
        axs[ind,2].set_yticks(np.linspace(ylim[0],ylim[1],num=5))
        if ind == 0:
            axs[ind,2].set_title(material_class_name[2])
        if ind == 2:
            axs[ind,2].set_xlim([550, 950])
            axs[ind,2].set_xticks(np.linspace(600,900,num=4).astype(int))
            axs[ind,2].set_xticklabels(np.linspace(600,900,num=4).astype(int))
            axs[ind, 2].set_xlabel(r'$\lambda$ [nm]')
        else:
            axs[ind,2].set_xlim([550, 950])
            axs[ind,2].set_xticks(np.linspace(600,900,num=4).astype(int))
            axs[ind,2].set_xticklabels([])

        indices = [id for id, ele in enumerate(y) if ele == '4']
        mean_value = X[indices,ind*WL.shape[0]:(ind+1)*WL.shape[0]].mean(axis = 0)
        std_value = X[indices,ind*WL.shape[0]:(ind+1)*WL.shape[0]].std(axis = 0)
        axs[ind,3].plot(WL, mean_value,'k-')
        axs[ind,3].fill_between(WL, mean_value + std_value, mean_value - std_value, facecolor = 'k',alpha=0.2)
        axs[ind,3].set_yticklabels([])
        axs[ind,3].set_ylim(ylim)
        axs[ind,3].set_yticks(np.linspace(ylim[0],ylim[1],num=5))
        if ind == 0:
            axs[ind,3].set_title(material_class_name[3])
        if ind == 2:
            axs[ind,3].set_xlim([550, 950])
            axs[ind,3].set_xticks(np.linspace(600,900,num=4).astype(int))
            axs[ind,3].set_xticklabels(np.linspace(600,900,num=4).astype(int))
            axs[ind,3].set_xlabel(r'$\lambda$ [nm]')
        else:
            axs[ind,3].set_xlim([550, 950])
            axs[ind,3].set_xticks(np.linspace(600,900,num=4).astype(int))
            axs[ind,3].set_xticklabels([])


        plt.subplots_adjust(wspace=0, hspace=0.2)
    fig.savefig('Fig_3dims_'+fig_tags+'.png', format='png', dpi=600)

""" ********************************************   3. Start the main script   ******************************************** """

""" ---------------- 3.1 Load the MM features ---------------- """
# The folder saving the MM dataset
subfolder_dataset_MM_features = "Dataset_MM_features"
# Load central wavelengths of 28 spectral channels
filepath_WL = Path(subfolder_dataset_MM_features+'/WL'+'.csv')
WL = pd.read_csv(filepath_WL, sep=";").to_numpy()[:, 1:].reshape(-1).astype(int)
# Create the string column headers for MM features (WL)
WL_headers = ["%.1f" % x for x in WL]
# Reflectance spectra
filepath_R = Path(subfolder_dataset_MM_features+'/R'+'.csv')
MM_R_spectra = pd.read_csv(filepath_R, sep=";", usecols = WL_headers).to_numpy()
# Distance spectra
filepath_d = Path(subfolder_dataset_MM_features+'/d'+'.csv')
MM_d_spectra = pd.read_csv(filepath_d, sep=";", usecols = WL_headers).to_numpy()
# Degree of linear polarization spectra
filepath_DoLP = Path(subfolder_dataset_MM_features+'/DoLP'+'.csv')
MM_DoLP_spectra = pd.read_csv(filepath_DoLP, sep=";", usecols = WL_headers).to_numpy()

# Load labels for material classes
y = pd.read_csv(filepath_DoLP, sep=";", usecols=['Material class']).to_numpy().reshape(-1).astype(str)

# Material classes
material_class_ID = [1, 2, 3, 4]
material_class_name = ['concrete', 'plastic', 'stone', 'wood']

""" ---------------- 3.2 Plot the MM features in the form of R(lambda) d(lambda) DoLP(lamda) ---------------- """
X = np.concatenate((MM_R_spectra, MM_d_spectra, MM_DoLP_spectra), axis=1)
y_axis_vec = ['R', 'd [mm]', 'DoLP']
dim_vec = ['Reflectance', 'Distance', 'Degree of linear polarization']
plotting_MM_features('MM_features')
plt.show()


