# Goal: This script is to implement the feature selection for the classification among four material classes: concrete, plastic, stone, and wood.

""" ********************************************   1. Import packages   ******************************************** """
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import matplotlib
import cvxpy as cp
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from mrmr import mrmr_classif
from sklearn.inspection import permutation_importance

"""********************************************   2. Define functions  ********************************************"""

""" --------------------- 2.1 Functional functions ---------------------"""
""" 2.1.1 Output filenames """
def file_name_partial(y_axis_vec):
    filename_index = '_'
    for i, tit in enumerate(y_axis_vec):
        if tit == 'd [mm]':
            tit = 'd'
        if i > 0:
            filename_index = filename_index + '+'
        filename_index = filename_index + tit
    filename_index = filename_index + '_SC2'
    print('########################################################## Modality combination: ',filename_index[1:-4], '##########################################################')
    return filename_index


"""--------------------- 2.3 Feature selection methods ---------------------"""
""" 2.3.1 MGSVM FS """


## 2.3.1.1 Output coefficient matrix (w_mat_opt) and bias vector (b_vec_opt)
def FS_MGSVM(X, y, C, SDs_select):
    ### required parameters from data
    n_sig = len(SDs_select)  # Nr signature dimensions (Nr modalities)
    n_feat = X.shape[1]  # Nr of features
    n_sc = int(n_feat / n_sig)  # Nr spectral channels
    n_data = X.shape[0]  # Nr data points
    n_classes = np.unique(y).shape[0]  # Nr classes

    ### Train the model

    # i) Invoke variables
    b_vec = cp.Variable(shape=n_classes)
    w_mat = cp.Variable(shape=[n_sc, n_sig * n_classes])
    xi = cp.Variable(shape=[n_data, n_classes], nonneg=True)

    # ii) Constraints
    cons = []
    for k in range(n_classes):
        for l in range(n_data):
            if y[l] == k:
                pass
            else:
                ind_y = (y[l]).astype(int)
                cons = cons + [(cp.vec(w_mat[:, ind_y * n_sig:(ind_y + 1) * n_sig]) - cp.vec(
                    w_mat[:, k * n_sig:(k + 1) * n_sig])).T @ X[l, :] + b_vec[ind_y] - b_vec[k] >= 2 - xi[l, k]]

    # iii) Create problem
    obj_fun = cp.mixed_norm(w_mat, 2, 1) + C * np.ones(n_data * n_classes).T @ cp.vec(xi)
    opt_prob = cp.Problem(cp.Minimize(obj_fun), constraints=cons)

    # iv) Solve and assemble
    opt_prob.solve(verbose=False)
    w_mat_opt = w_mat.value
    b_vec_opt = b_vec.value

    return w_mat_opt, b_vec_opt


## 2.3.1.2 Plot the coefficient matrix when spectral channels at "indices_WL" are selected. This function is applied in the function "FS_MGSVM_search"
def MGSVM_coefficients_plotting(w_mat_opt, indices_WL, fig_save, fig_tags):
    ### Plot absolute values of coefficients
    fig, axs = plt.subplots(len(material_class_ID), 1, figsize=(15, 8))
    plt.subplots_adjust(hspace=0.5)
    for k in range(0, len(material_class_ID)):
        data = w_mat_opt[:, k * len(SDs_select):(k + 1) * len(SDs_select)]
        numOfCols = WL.shape[0]
        numOfRows = len(SDs_select)
        xpos = np.arange(0, numOfCols, 1)
        ypos = np.arange(0, numOfRows, 1)
        Z = np.abs(np.transpose(data))
        m = axs[k].imshow(Z, vmin=0, cmap='Greys', aspect=0.5)
        axs[k].set_title('Class ' + str(k + 1))
        axs[k].set_ylabel('Modal \n dimension')
        axs[k].set_yticks(ypos)
        axs[k].set_xticks(xpos)
        sc_label_ticks = []
        for sc in range(0, len(WL)):
            sc_label_ticks.append(('SC' + str(sc + 1)))
        modal_label_ticks = []
        for i, tit in enumerate(y_axis_vec):
            if tit == 'd [mm]':
                tit = 'd'
            modal_label_ticks.append(tit)
        axs[k].set_yticklabels(modal_label_ticks)
        if k == len(material_class_ID) - 1:
            axs[k].set_xticklabels(sc_label_ticks)
            axs[k].set_xlabel('Spectral dimension')
        else:
            axs[k].set_xticklabels([])
        axs[k].hlines(y=np.arange(0, numOfRows) + 0.5, xmin=np.full(numOfRows, 0) - 0.5,
                      xmax=np.full(numOfRows, numOfCols) - 0.5, color="black", linewidth=0.1)
        axs[k].vlines(x=np.arange(0, numOfCols) + 0.5, ymin=np.full(numOfCols, 0) - 0.5,
                      ymax=np.full(numOfCols, numOfRows) - 0.5, color="black", linewidth=0.1)
        # Text coefficient values
        for i, ii in enumerate(xpos):
            for j, jj in enumerate(ypos):
                if round(Z[j, i], 2) > 1e-3:
                    text_number = round(Z[j, i], 2)
                    text_number = np.abs(text_number)
                    axs[k].text(xpos[i], ypos[j], text_number, ha='center', va='center', size=13, color='r')
                else:
                    text_number = round(Z[j, i], 2)
                    text_number = np.abs(text_number)
                    axs[k].text(xpos[i], ypos[j], text_number, ha='center', va='center', size=13, color= [0.8,0.8,0.8])
    cbar = plt.colorbar(m, ax=axs, shrink=0.5, aspect=80, pad=0.1, orientation='horizontal')
    cbar.set_label('Absolute values of coefficients')
    plt.suptitle('Importance (absolute values of coefficients) of different spectral channels, '
                 'optical modalities, and material classes \n  Select ' + 'spectral channels (SCs): ' + str(
        indices_WL + 1))
    if fig_save == "True":
        fig.savefig('Fig_Abs_Coeff_' + fig_tags + '_Nr_SCs_' + str(len(indices_WL)) + '.png', format='png', dpi=600)

    ### Plot selected and non-selected coefficients
    fig, axs = plt.subplots(len(material_class_ID), 1, figsize=(15, 8))
    plt.subplots_adjust(hspace=0.5)
    for k in range(0, len(material_class_ID)):
        data = w_mat_opt[:, k * len(SDs_select):(k + 1) * len(SDs_select)]
        numOfCols = WL.shape[0]
        numOfRows = len(SDs_select)
        xpos = np.arange(0, numOfCols, 1)
        ypos = np.arange(0, numOfRows, 1)
        ZZ = np.abs(data)
        indicies_selected_coefficients = np.nonzero(ZZ > 1e-3)
        Z = np.zeros(ZZ.shape)
        Z[indicies_selected_coefficients] = 1
        Z = np.transpose(Z)
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", [[1, 1, 1], [1, 0, 0]])
        m = axs[k].imshow(Z, vmin=0, cmap=colormap, aspect=0.5)
        axs[k].set_title('Class ' + str(k + 1))
        axs[k].set_ylabel('Modal  \n dimension')
        axs[k].set_yticks(ypos)
        axs[k].set_xticks(xpos)
        sc_label_ticks = []
        for sc in range(0, len(WL)):
            sc_label_ticks.append(('SC' + str(sc + 1)))
        modal_label_ticks = []
        for i, tit in enumerate(y_axis_vec):
            if tit == 'd [mm]':
                tit = 'd'
            modal_label_ticks.append(tit)
        axs[k].set_yticklabels(modal_label_ticks)
        if k == len(material_class_ID) - 1:
            axs[k].set_xticklabels(sc_label_ticks)
            axs[k].set_xlabel('Spectral dimension')
        else:
            axs[k].set_xticklabels([])
        axs[k].hlines(y=np.arange(0, numOfRows) + 0.5, xmin=np.full(numOfRows, 0) - 0.5,
                      xmax=np.full(numOfRows, numOfCols) - 0.5, color="black", linewidth=0.1)
        axs[k].vlines(x=np.arange(0, numOfCols) + 0.5, ymin=np.full(numOfCols, 0) - 0.5,
                      ymax=np.full(numOfCols, numOfRows) - 0.5, color="black", linewidth=0.1)
    cbar = plt.colorbar(m, ax=axs, shrink=0.5, aspect=80, pad=0.1, orientation='horizontal')
    cbar.set_ticks(ticks=[0, 1], labels=['Non-selected', 'Selected'])
    cbar.set_label('Non-selected and selected coefficients')
    plt.suptitle('Non-selected and selected coefficients of different spectral channels, '
                 'optical modalities, and material classes \n  Select ' + 'spectral channels (SCs): ' + str(
        indices_WL + 1))



    if fig_save == "True":
        fig.savefig('Fig_Selected_Coeff_' + fig_tags + '_Nr_SCs_' + str(len(indices_WL)) + '.png', format='png',
                    dpi=600)
    # plt.close('all')


## 2.3.1.3 Sear the proper value of the parameter "C" (the scalar trade-off parameter balancing between training loss and regularization) for a specific target number of selected spectral channels "Nr_WL_target"
def FS_MGSVM_search(C_set, C_set_previous, C_set_next, dir_set, dir_nex, X_train, y_train, k_classes, Nr_WL_target,
                    learning_iterations, intern_learning_iterations):
    ''' # dir_set: available information for the direction given by C_set
            # 'indirect': no information, need to check C_set
            # 'smaller': C_set makes Nr_WL<Nr_WL_target
            # 'larger': C_set makes Nr_WL>Nr_WL_target
    # dir_nex: available information for the direction given by C_nex
            # 'indirect': no information, need to check C_nex
            # 'smaller': C_nex makes Nr_WL<Nr_WL_target
            # 'larger': C_nex makes Nr_WL>Nr_WL_target'''
    w_opt_WL_output = np.array([])
    b_opt_WL_output = np.array([])

    if dir_set == 'indirect':
        learning_iterations += 1
        intern_learning_iterations += 1
        # print('Learning iterations : ', learning_iterations, '          Intern learning iterations: ', intern_learning_iterations)
        # print('Previous, current, next Cs: ', C_set_previous, C_set, C_set_next)

        # collection of indices where coefficients are non-zero
        # Implement the MGSVM
        w_mat_opt, b_vec_opt = FS_MGSVM(X_train, y_train, C_set, SDs_select)
        w_vec_opt_norm = np.abs(w_mat_opt).sum(axis=1)
        indices_WL = np.ravel(np.array(np.nonzero(w_vec_opt_norm > 1e-3)))
        # print('-----> Importance', w_vec_opt_norm.reshape(int(X_train.shape[1]/len(SDs_select)), 1))
        # print('-----> non-zero', np.nonzero(w_vec_opt_norm >1e-3))
        # print(np.shape(indices_WL)[0], ' With the target number ', Nr_WL_target)

        if np.shape(indices_WL)[0] == Nr_WL_target:
            dir_set_new = 'indirect'
            dir_nex_new = 'indirect'
            w_opt_WL_output = w_mat_opt[indices_WL, :]
            b_opt_WL_output = b_vec_opt
            if plot_coefficient == 'True':
                MGSVM_coefficients_plotting(w_mat_opt, indices_WL, plot_coefficient, filename_index)

        elif np.shape(indices_WL)[0] > Nr_WL_target:
            if intern_learning_iterations > 50:
                indices_WL = np.ravel(np.array(np.argsort(w_vec_opt_norm)[::-1][:Nr_WL_target]))
                # print('-----> Importance', w_vec_opt_norm.reshape(int(X_train.shape[1]/len(SDs_select)), 1))
                # print('-----> non-zero', np.nonzero(w_vec_opt_norm >1e-3))
                dir_set_new = 'indirect'
                dir_nex_new = 'indirect'
                w_opt_WL_output = w_mat_opt[indices_WL, :]
                b_opt_WL_output = b_vec_opt
                # print('Dead loops lead to: ',np.shape(indices_WL)[0], ' With the target number ', Nr_WL_target)
                if plot_coefficient == 'True':
                    MGSVM_coefficients_plotting(w_mat_opt, indices_WL, plot_coefficient, filename_index)

            else:
                dir_set = 'larger'
                divider_num = int(np.shape(indices_WL)[0] - Nr_WL_target)

        elif np.shape(indices_WL)[0] < Nr_WL_target:
            if intern_learning_iterations > 50:
                # collection of indices where coefficients are non-zero
                w_mat_opt, b_vec_opt = FS_MGSVM(X_train, y_train, C_set_next, SDs_select)
                indices_WL = np.ravel(np.array(np.argsort(w_vec_opt_norm)[::-1][:Nr_WL_target]))
                # print('-----> Importance', w_vec_opt_norm.reshape(int(X_train.shape[1]/len(SDs_select)), 1))
                # print('-----> non-zero', np.nonzero(w_vec_opt_norm >1e-3))
                dir_set_new = 'indirect'
                dir_nex_new = 'indirect'
                w_opt_WL_output = w_mat_opt[indices_WL, :]
                b_opt_WL_output = b_vec_opt
                # print('Dead loops lead to: ', np.shape(indices_WL)[0], ' With the target number ', Nr_WL_target)
                if plot_coefficient == 'True':
                    MGSVM_coefficients_plotting(w_mat_opt, indices_WL, plot_coefficient, filename_index)

            else:
                dir_set = 'smaller'
    else:
        divider_num = 1

    if dir_set == 'larger':
        # C_pre should give Nr_WL<Nr_WL_target
        C_divider = np.power(10, np.linspace(np.log10(C_set_previous), np.log10(C_set), num=2 + divider_num)[1])
        # C_divider as the new C_set_next
        w_opt_WL_output, b_opt_WL_output, C_intern, indices_WL, learning_iterations, dir_set_new, dir_nex_new, intern_learning_iterations = FS_MGSVM_search(
            C_divider, C_set_previous, C_set, 'indirect', 'larger', X_train, y_train, k_classes, Nr_WL_target,
            learning_iterations, intern_learning_iterations)
        # new C value can give the target number of wavelengths
        C_set = C_intern

    elif dir_set == 'smaller':
        if dir_nex == 'indirect':
            # Try the next C value, to see if it can give the feature number larger than the desired one
            learning_iterations += 1
            intern_learning_iterations += 1
            # print('Learning iterations : ', learning_iterations, '          Intern learning iterations: ', intern_learning_iterations)
            # Implement the MGSVM
            w_mat_opt_nex, b_vec_opt_nex = FS_MGSVM(X_train, y_train, C_set_next, SDs_select)
            w_vec_opt_norm_nex = np.abs(w_mat_opt_nex).sum(axis=1)
            indices_WL_nex = np.ravel(np.array(np.nonzero(w_vec_opt_norm_nex > 1e-3)))
            if np.shape(indices_WL_nex)[0] == Nr_WL_target:
                C_set = C_set_next
                indices_WL = indices_WL_nex
                dir_set_new = 'smaller'
                dir_nex_new = 'indirect'
                # print('C_nex return, ', indices_WL)
                # print(np.shape(indices_WL)[0] ,' With the target number ',Nr_WL_target)
                w_opt_WL_output = w_mat_opt_nex[indices_WL, :]
                b_opt_WL_output = b_vec_opt_nex

                if plot_coefficient == 'True':
                    MGSVM_coefficients_plotting(w_mat_opt_nex, indices_WL, plot_coefficient, filename_index)

            elif np.shape(indices_WL_nex)[0] > Nr_WL_target:
                dir_nex = 'larger'
                divider_num = int(np.shape(indices_WL_nex)[0] - Nr_WL_target)
            elif np.shape(indices_WL_nex)[0] < Nr_WL_target:
                # print('C_set and C_set_next are: ',C_set, C_set_next,' ----> ',np.shape(indices_WL_nex)[0])
                dir_nex = 'smaller'
        else:
            divider_num = 1

        if dir_nex == 'smaller':
            dir_set_new = 'smaller'
            dir_nex_new = 'indirect'
            indices_WL = np.array([])
            # print('C_set and C_set_next are: ',C_set, C_set_next,' ----> empty array ', indices_WL.shape[0])
        elif dir_nex == 'larger':
            C_divider = np.power(10, np.linspace(np.log10(C_set), np.log10(C_set_next), num=2 + divider_num)[1])
            w_opt_WL_output, b_opt_WL_output, C_intern, indices_WL, learning_iterations, dir_set_new, dir_nex_new, intern_learning_iterations = FS_MGSVM_search(
                C_divider, C_set, C_set_next, 'indirect', 'larger', X_train, y_train, k_classes, Nr_WL_target,
                learning_iterations, intern_learning_iterations)
            C_set = C_intern
    return w_opt_WL_output, b_opt_WL_output, C_set, indices_WL, learning_iterations, dir_set_new, dir_nex_new, intern_learning_iterations


"""********************************************   3. Examples of implement the MGSVM feature selection (FS) algorithm   ********************************************"""

"""-------------------------   3.0 Load data (X,y)   -------------------------"""
### Example：Load the MM featres (R(lambda), d(lambda), DoLP(lambda)) as the matrix "X", and the labels of material classes as the vector "y"
# The folder saving the MM dataset
higher_directory = str(Path().absolute().parent.parent)
subfolder_dataset_MM_features = higher_directory + '\Dataset_MM_features'
# Load central wavelengths of 28 spectral channels
filepath_WL = Path(subfolder_dataset_MM_features + '/WL' + '.csv')
WL = pd.read_csv(filepath_WL, sep=";").to_numpy()[:, 1:].reshape(-1).astype(int)
# Create the string column headers for MM features (WL)
WL_headers = ["%.1f" % x for x in WL]
# Reflectance spectra
filepath_R = Path(subfolder_dataset_MM_features + '/R' + '.csv')
MM_R_spectra = pd.read_csv(filepath_R, sep=";", usecols=WL_headers).to_numpy()
# Distance spectra
filepath_d = Path(subfolder_dataset_MM_features + '/d' + '.csv')
MM_d_spectra = pd.read_csv(filepath_d, sep=";", usecols=WL_headers).to_numpy()
# Degree of linear polarization spectra
filepath_DoLP = Path(subfolder_dataset_MM_features + '/DoLP' + '.csv')
MM_DoLP_spectra = pd.read_csv(filepath_DoLP, sep=";", usecols=WL_headers).to_numpy()
# Labels for material classes (4 classes)
M_class_label = pd.read_csv(filepath_DoLP, sep=";", usecols=['Material class']).to_numpy().reshape(-1).astype(str)
material_class_ID = [1, 2, 3, 4]

# The modality combination R+d+DoLP
SDs_select = [1, 2, 3]
dim_vec = ['Reflectance', 'Distance', 'Degree of linear polarization']
y_axis_vec = ['R', 'd [mm]', 'DoLP']
# Get file name
filename_index = file_name_partial(y_axis_vec)
# Assign the MM features to the matrix "X"
X = np.concatenate((MM_R_spectra, MM_d_spectra, MM_DoLP_spectra), axis=1)
# Assign the MM features to the vector "y"
y = np.array(M_class_label).astype(int)-1


"""-------------------------   3.1 Prepare parameters for MGSVM FS  -------------------------"""

""" ----- 3.1.1 Set the target number of selected feature groups Nr_FG_target  ----- """
### Example: Set the target number of selected spectral channels "Nr_WL_target"
# Target to select 3 spectral channels (feature groups), in which all the optical modalities (joint features/feature dimensions) contribute to the multiclass classification task most effectively
Nr_WL_target = 3
Nr_FG_target = Nr_WL_target
# n_sig = len(SDs_select)  # Nr signature dimensions (Number of joint features in each feature group)
# n_feat = X.shape[1]  # Nr of features (Number of total features)
# n_sc = int(n_feat / n_sig)  # Nr spectral channels (Number of feature groups)

""" ----- 3.1.2 Set the searching range "C_vec" for the penalty parameter "C" ----- """

''' Note: The speed of the MGSVM FS depends on the range and step size of the vector "C", 
the data (X,y) and the target number of selected grouped features. '''
### Example "C_vec"
C_vec = np.logspace(-2.5, 2, num=30)
C_vec = np.concatenate((np.array([1e-4]), C_vec), axis=0)
C_vec = np.concatenate((C_vec, np.logspace(2.1, 3, num=8)), axis=0)

""" ----- 3.1.3 Prepare initial parameters for MGSVM FS ----- """
# No information about whether the first element in the vector "C_vec" will give larger or smaller number of selected feature groups
dir_set = 'indirect'
# No information about whether the second element in the vector "C_vec" will give larger or smaller number of selected feature groups
dir_nex = 'indirect'
# Initial value of recording the iterations to find the suitable value "C" to output the target number of selected feature groups
learning_iterations = 0
# Lists for recording the values of the coefficient matrix "W_opt_WL_output_1_list" and the bias vector "b_opt_WL_output_1_list"
w_opt_WL_output_1_list = []
b_opt_WL_output_1_list = []
### Select plotting / not plotting the coefficients in MGSVM FS algorithms
plot_coefficient = "True"

""" ----- 3.1.4 Start the searching loop for MGSVM FS ----- """
for i, C_set in enumerate(C_vec[1:-1]):
    i = i + 1
    # print(i)
    intern_learning_iterations = 0
    w_opt_WL_output_1, b_opt_WL_output_1, C_new, indices_WL, learning_iterations, dir_set, dir_next, intern_learning_iterations = FS_MGSVM_search(
        C_set, C_vec[i - 1], C_vec[i + 1], dir_set, dir_nex, X, y, len(y), Nr_FG_target,
        learning_iterations, intern_learning_iterations)
    if np.shape(indices_WL)[0] == Nr_WL_target:
        print('MGSVM FS-------->  ', np.shape(indices_WL)[0], ' wavelengths are selected.')
        break


plt.show()

