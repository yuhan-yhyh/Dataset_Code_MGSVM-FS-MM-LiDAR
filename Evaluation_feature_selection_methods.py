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

""" 2.1.2 Save data of evaluation metrics """
def save_scores_masks_stability():
    ### Peroformance metrics
    ## MGSVM FS
    np.save('F1_lsvc_kfold_WL'+filename_index+'_MGSVM', F1_lsvc_MGSVM_kfold)
    np.save('F1_lsvc_mean_WL'+filename_index+'_MGSVM', F1_lsvc_MGSVM_kfold_means)
    np.save('F1_lsvc_std_WL'+filename_index+'_MGSVM', F1_lsvc_MGSVM_kfold_stds)
    np.save('F1_svc_rbf_kfold_WL'+filename_index+'_MGSVM', F1_svc_rbf_MGSVM_kfold)
    np.save('F1_svc_rbf_mean_WL'+filename_index+'_MGSVM', F1_svc_rbf_MGSVM_kfold_means)
    np.save('F1_svc_rbf_std_WL'+filename_index+'_MGSVM', F1_svc_rbf_MGSVM_kfold_stds)
    ## random FS
    np.save('F1_lsvc_kfold_WL'+filename_index+'_random', F1_lsvc_rand_loop_kfold)
    np.save('F1_lsvc_mean_WL'+filename_index+'_random', F1_lsvc_rand_loop_kfold_means)
    np.save('F1_lsvc_std_WL'+filename_index+'_random', F1_lsvc_rand_loop_kfold_stds)
    np.save('F1_svc_rbf_kfold_WL'+filename_index+'_random', F1_svc_rbf_rand_loop_kfold)
    np.save('F1_svc_rbf_mean_WL'+filename_index+'_random', F1_svc_rbf_rand_loop_kfold_means)
    np.save('F1_svc_rbf_std_WL'+filename_index+'_random', F1_svc_rbf_rand_loop_kfold_stds)
    ## RF_MDPA FS
    np.save('F1_lsvc_kfold_WL'+filename_index+'_RF_MDPA', F1_lsvc_RF_MDPA_kfold)
    np.save('F1_lsvc_mean_WL'+filename_index+'_RF_MDPA', F1_lsvc_RF_MDPA_kfold_means)
    np.save('F1_lsvc_std_WL'+filename_index+'_RF_MDPA', F1_lsvc_RF_MDPA_kfold_stds)
    np.save('F1_svc_rbf_kfold_WL'+filename_index+'_RF_MDPA', F1_svc_rbf_RF_MDPA_kfold)
    np.save('F1_svc_rbf_mean_WL'+filename_index+'_RF_MDPA', F1_svc_rbf_RF_MDPA_kfold_means)
    np.save('F1_svc_rbf_std_WL'+filename_index+'_RF_MDPA', F1_svc_rbf_RF_MDPA_kfold_stds)
    ## MRMR FS
    np.save('F1_lsvc_kfold_WL' + filename_index + '_mrmr', F1_lsvc_mrmr_kfold)
    np.save('F1_lsvc_mean_WL' + filename_index + '_mrmr', F1_lsvc_mrmr_kfold_means)
    np.save('F1_lsvc_std_WL' + filename_index + '_mrmr', F1_lsvc_mrmr_kfold_stds)
    np.save('F1_svc_rbf_kfold_WL' + filename_index + '_mrmr', F1_svc_rbf_mrmr_kfold)
    np.save('F1_svc_rbf_mean_WL' + filename_index + '_mrmr', F1_svc_rbf_mrmr_kfold_means)
    np.save('F1_svc_rbf_std_WL' + filename_index + '_mrmr', F1_svc_rbf_mrmr_kfold_stds)

    ### masks
    ## MGSVM FS
    np.save('Mask_WL' + filename_index + '_MGSVM', masks_MGSVM_kfold)
    ## random FS
    np.save('Mask_WL' + filename_index + '_random', masks_rand_loop_kfold)
    ## RF_MDPA FS
    np.save('Mask_WL' + filename_index + '_RF_MDPA', masks_RF_MDPA_kfold)
    ## MRMR FS
    np.save('Mask_WL' + filename_index + '_mrmr', masks_mrmr_kfold)

    ### stability index
    ## MGSVM FS
    np.save('I_stab_WL' + filename_index + '_MGSVM', I_s_MGSVM)
    ## random FS
    np.save('I_stab_WL' + filename_index + '_random', I_s_random)
    ## RF_MDPA FS
    np.save('I_stab_WL' + filename_index + '_RF_MDPA', I_s_RF_MDPA)
    ## MRMR FS
    np.save('I_stab_WL' + filename_index + '_mrmr', I_s_mrmr)

"""--------------------- 2.2 Evaluation framework (Cross-validation) ---------------------"""
def cross_validation(X, y, K_fold):
    # Select spectral channels

    time_start = time.time()
    ### Indices of the selected features
    indices_fs_matrix_MGSVM_kfold = []
    indices_fs_matrix_RF_MDPA_kfold = []
    indices_fs_matrix_mrmr_kfold = []

    ### masks
    masks_rand_loop_kfold = []
    masks_MGSVM_kfold = []
    masks_RF_MDPA_kfold = []
    masks_mrmr_kfold = []

    ### Scores
    ## MGSVM FS
    # Linear SVM
    F1_lsvc_MGSVM_kfold = []
    # RBF-SVM
    F1_svc_rbf_MGSVM_kfold = []
    ## random FS
    # Linear SVM
    F1_lsvc_rand_loop_kfold = []
    # RBF-SVM
    F1_svc_rbf_rand_loop_kfold = []
    ## RF_MDPA FS
    # Linear SVM
    F1_lsvc_RF_MDPA_kfold = []
    # RBF-SVM
    F1_svc_rbf_RF_MDPA_kfold = []
    ## MRMR FS
    # Linear SVM
    F1_lsvc_mrmr_kfold = []
    # RBF-SVM
    F1_svc_rbf_mrmr_kfold = []

    y = np.array(y)
    skf = StratifiedKFold(n_splits = K_fold, shuffle=True)
    kk = 0
    for train_index, test_index in skf.split(X, y):
        kk += 1
        print("*************************  fold ", kk,"  ************************")
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

        print('******** Random feature selection ********')
        t_start = time.time()
        ### randomly select features
        masks_rand_loop = []
        F1_lsvc_rand_loop = []
        F1_svc_rbf_rand_loop = []
        for i in np.array(range(1,WL.shape[0]+1)):
            index = np.random.choice(WL.shape[0], i, replace=False)
            indices = np.array([])
            for ind in range(0, len(SDs_select)):
                indices = np.append(indices, np.add(WL.shape[0]*ind, index)).astype(int)
                mask_rand_loop = np.zeros(X_train.shape[1])
                mask_rand_loop[indices] = 1
            masks_rand_loop.append(mask_rand_loop)
            # print('Indices [random FS]: ', indices)
            #-----> Predict and score
            X_selected_train = X_train[:,indices]
            X_selected_test = X_test[:,indices]
            ### linear SVM
            lsvc = LinearSVC(C=100, dual=False, tol=1e-5, max_iter=1e3)
            lsvc.fit(X_selected_train, y_train)
            y_predict_lsvc = lsvc.predict(X_selected_test)
            F1_lsvc = f1_score(np.ravel(y_test), y_predict_lsvc, average='macro')
            ### RBF-SVM
            svc_rbf = SVC(C=100, gamma=1, kernel='rbf', tol=1e-5, max_iter=1e7)
            svc_rbf.fit(X_selected_train, y_train)
            y_predict_svc_rbf = svc_rbf.predict(X_selected_test)
            F1_svc_rbf = f1_score(np.ravel(y_test), y_predict_svc_rbf, average='macro')
            F1_lsvc_rand_loop.append(F1_lsvc)
            F1_svc_rbf_rand_loop.append(F1_svc_rbf)
        masks_rand_loop_kfold.append(np.array(masks_rand_loop))
        F1_lsvc_rand_loop_kfold.append(np.array(F1_lsvc_rand_loop))
        F1_svc_rbf_rand_loop_kfold.append(np.array(F1_svc_rbf_rand_loop))
        # print('Escaped time for random FS is: ', time.time()-t_start, ' s')

        print('******** RF_MDPA feature selection ********')
        t_start = time.time()
        ### RF_MDPA features selection (RF_MDPA FS)
        indices_fs_matrix, masks,  F1_lsvc, F1_svc_rbf = FS_RF_MDPA(
            X_train, X_test, y_train, y_test, SDs_select)
        indices_fs_matrix_RF_MDPA_kfold.append(indices_fs_matrix)
        masks_RF_MDPA_kfold.append(masks)
        F1_lsvc_RF_MDPA_kfold.append(F1_lsvc)
        F1_svc_rbf_RF_MDPA_kfold.append(F1_svc_rbf)
        # print('Escaped time for RF_MDPA FS is: ', time.time() - t_start, ' s')

        print('******** MRMR feature selection ********')
        t_start = time.time()
        ### MRMR features selection (MRMR FS)
        indices_fs_matrix, masks,  F1_lsvc, F1_svc_rbf = FS_MRMR(
            X_train, X_test, y_train, y_test, SDs_select)
        indices_fs_matrix_mrmr_kfold.append(indices_fs_matrix)
        masks_mrmr_kfold.append(masks)
        F1_lsvc_mrmr_kfold.append(F1_lsvc)
        F1_svc_rbf_mrmr_kfold.append(F1_svc_rbf)
        # print('Escaped time for MRMR FS is: ', time.time() - t_start, ' s')

        print('******** MGSVM feature selection ********')
        t_start = time.time()
        ### MGSVM features selection (MGSVM FS)
        indices_fs_matrix, masks,  F1_lsvc, F1_svc_rbf = FS_MGSVM_multiclass_loop(
            X_train, X_test, y_train, y_test, len(material_class_ID), SDs_select)
        indices_fs_matrix_MGSVM_kfold.append(indices_fs_matrix)
        masks_MGSVM_kfold.append(masks)
        F1_lsvc_MGSVM_kfold.append(F1_lsvc)
        F1_svc_rbf_MGSVM_kfold.append(F1_svc_rbf)
       # print('Escaped time for MGSVM FS is: ', time.time() - t_start, ' s')

    ### Calculate mean and std values of the performance metrics
    F1_lsvc_MGSVM_kfold = np.array(F1_lsvc_MGSVM_kfold)
    F1_lsvc_MGSVM_kfold_means = F1_lsvc_MGSVM_kfold.mean(axis=0)
    F1_lsvc_MGSVM_kfold_stds = F1_lsvc_MGSVM_kfold.std(axis=0)
    F1_svc_rbf_MGSVM_kfold = np.array(F1_svc_rbf_MGSVM_kfold)
    F1_svc_rbf_MGSVM_kfold_means = F1_svc_rbf_MGSVM_kfold.mean(axis=0)
    F1_svc_rbf_MGSVM_kfold_stds = F1_svc_rbf_MGSVM_kfold.std(axis=0)

    F1_lsvc_rand_loop_kfold = np.array(F1_lsvc_rand_loop_kfold)
    F1_lsvc_rand_loop_kfold_means = F1_lsvc_rand_loop_kfold.mean(axis=0)
    F1_lsvc_rand_loop_kfold_stds = F1_lsvc_rand_loop_kfold.std(axis=0)
    F1_svc_rbf_rand_loop_kfold = np.array(F1_svc_rbf_rand_loop_kfold)
    F1_svc_rbf_rand_loop_kfold_means = F1_svc_rbf_rand_loop_kfold.mean(axis=0)
    F1_svc_rbf_rand_loop_kfold_stds = F1_svc_rbf_rand_loop_kfold.std(axis=0)

    F1_lsvc_RF_MDPA_kfold = np.array(F1_lsvc_RF_MDPA_kfold)
    F1_lsvc_RF_MDPA_kfold_means = F1_lsvc_RF_MDPA_kfold.mean(axis=0)
    F1_lsvc_RF_MDPA_kfold_stds = F1_lsvc_RF_MDPA_kfold.std(axis=0)
    F1_svc_rbf_RF_MDPA_kfold = np.array(F1_svc_rbf_RF_MDPA_kfold)
    F1_svc_rbf_RF_MDPA_kfold_means = F1_svc_rbf_RF_MDPA_kfold.mean(axis=0)
    F1_svc_rbf_RF_MDPA_kfold_stds = F1_svc_rbf_RF_MDPA_kfold.std(axis=0)

    F1_lsvc_mrmr_kfold = np.array(F1_lsvc_mrmr_kfold)
    F1_lsvc_mrmr_kfold_means = F1_lsvc_mrmr_kfold.mean(axis=0)
    F1_lsvc_mrmr_kfold_stds = F1_lsvc_mrmr_kfold.std(axis=0)
    F1_svc_rbf_mrmr_kfold = np.array(F1_svc_rbf_mrmr_kfold)
    F1_svc_rbf_mrmr_kfold_means = F1_svc_rbf_mrmr_kfold.mean(axis=0)
    F1_svc_rbf_mrmr_kfold_stds = F1_svc_rbf_mrmr_kfold.std(axis=0)

    # print('The calculation duration for one iteration of cross-validation is: ', time.time() - time_start, ' s')
    return (masks_rand_loop_kfold, masks_MGSVM_kfold, indices_fs_matrix_MGSVM_kfold, masks_RF_MDPA_kfold, masks_mrmr_kfold,
            F1_lsvc_MGSVM_kfold, F1_lsvc_MGSVM_kfold_means, F1_lsvc_MGSVM_kfold_stds,
            F1_svc_rbf_MGSVM_kfold, F1_svc_rbf_MGSVM_kfold_means, F1_svc_rbf_MGSVM_kfold_stds,
            F1_lsvc_rand_loop_kfold, F1_lsvc_rand_loop_kfold_means, F1_lsvc_rand_loop_kfold_stds,
            F1_svc_rbf_rand_loop_kfold, F1_svc_rbf_rand_loop_kfold_means, F1_svc_rbf_rand_loop_kfold_stds,
            F1_lsvc_RF_MDPA_kfold, F1_lsvc_RF_MDPA_kfold_means, F1_lsvc_RF_MDPA_kfold_stds,
            F1_svc_rbf_RF_MDPA_kfold, F1_svc_rbf_RF_MDPA_kfold_means, F1_svc_rbf_RF_MDPA_kfold_stds,
            F1_lsvc_mrmr_kfold, F1_lsvc_mrmr_kfold_means, F1_lsvc_mrmr_kfold_stds,
            F1_svc_rbf_mrmr_kfold, F1_svc_rbf_mrmr_kfold_means, F1_svc_rbf_mrmr_kfold_stds)

"""--------------------- 2.3 Feature selection methods ---------------------"""
""" 2.3.1 MGSVM FS """

## 2.3.1.1 Output coefficient matrix (w_mat_opt) and bias vector (b_vec_opt)
def FS_MGSVM( X, y, C, SDs_select):
    ### required parameters from data
    n_sig = len(SDs_select)   # Nr signature dimensions (Nr modalities)
    n_feat = X.shape[1]    # Nr of features
    n_sc = int(n_feat/n_sig)   # Nr spectral channels
    n_data = X.shape[0] # Nr data points
    n_classes = np.unique(y).shape[0] # Nr classes
    
    ### Train the model
    
    # i) Invoke variables
    b_vec = cp.Variable(shape=n_classes)
    w_mat = cp.Variable(shape=[n_sc, n_sig*n_classes])
    xi = cp.Variable(shape=[n_data, n_classes], nonneg=True)
    
    # ii) Constraints
    cons = []
    for k in range(n_classes):
        for l in range(n_data):
            if y[l] == k:
                pass
            else:
                ind_y = (y[l]).astype(int)
                cons = cons+[(cp.vec(w_mat[:, ind_y*n_sig:(ind_y+1)*n_sig])-cp.vec(w_mat[:, k*n_sig:(k+1)*n_sig])).T@X[l, :]+b_vec[ind_y]-b_vec[k] >= 2-xi[l, k]]
    
    # iii) Create problem
    obj_fun = cp.mixed_norm(w_mat,2,1) + C*np.ones(n_data*n_classes).T@cp.vec(xi)
    opt_prob = cp.Problem(cp.Minimize(obj_fun), constraints=cons)
    
    # iv) Solve and assemble
    opt_prob.solve(verbose=False)
    w_mat_opt = w_mat.value
    b_vec_opt = b_vec.value

    return w_mat_opt, b_vec_opt

## 2.3.1.2 Plot the coefficient matrix when spectral channels at "indices_WL" are selected. This function is applied in the function "FS_MGSVM_search"
def MGSVM_coefficients_plotting(w_mat_opt, indices_WL, fig_save, fig_tags):
    ### Plot absolute values of coefficients
    fig, axs = plt.subplots( len(material_class_ID), 1, figsize=(15,8))
    plt.subplots_adjust(hspace=0.5)
    for k in range(0,len(material_class_ID)):
        data = w_mat_opt[:, k*len(SDs_select):(k+1)*len(SDs_select)]
        numOfCols = WL.shape[0]
        numOfRows = len(SDs_select)
        xpos = np.arange(0, numOfCols, 1)
        ypos = np.arange(0, numOfRows, 1)
        Z = np.abs(np.transpose(data))
        m = axs[k].imshow(Z, vmin=0, cmap='Greys', aspect=0.5)
        axs[k].set_title('Class '+str(k+1))
        axs[k].set_ylabel('Modal \n dimension')
        axs[k].set_yticks(ypos)
        axs[k].set_xticks(xpos)
        sc_label_ticks = []
        for sc in range(0, len(WL)):
            sc_label_ticks.append(('SC'+str(sc+1)))
        modal_label_ticks = []
        for i, tit in enumerate(y_axis_vec):
            if tit == 'd [mm]':
                tit = 'd'
            modal_label_ticks.append(tit)
        axs[k].set_yticklabels(modal_label_ticks)
        if k == len(material_class_ID)-1:
            axs[k].set_xticklabels(sc_label_ticks)
            axs[k].set_xlabel('Spectral dimension')
        else:
            axs[k].set_xticklabels([])
        axs[k].hlines(y=np.arange(0, numOfRows) + 0.5, xmin=np.full(numOfRows, 0) - 0.5, xmax=np.full(numOfRows, numOfCols) - 0.5, color="black", linewidth=0.1)
        axs[k].vlines(x=np.arange(0, numOfCols) + 0.5, ymin=np.full(numOfCols, 0) - 0.5, ymax=np.full(numOfCols, numOfRows) - 0.5, color="black", linewidth=0.1)

    cbar = plt.colorbar(m, ax=axs, shrink=0.5, aspect=80, pad=0.1, orientation = 'horizontal')
    cbar.set_label('Absolute values of coefficients')
    plt.suptitle('Importance (absolute values of coefficients) of different spectral channels, '
                 'optical modalities, and material classes \n  Select ' + 'spectral channels (SCs): '+ str(indices_WL+1))
    if fig_save == "True":
        fig.savefig('Fig_Abs_Coeff_' + fig_tags + '_Nr_SCs_'+str(len(indices_WL))+'.png', format='png', dpi=600)
        
    ### Plot selected and non-selected coefficients
    fig, axs = plt.subplots( len(material_class_ID), 1, figsize=(15,8))
    plt.subplots_adjust(hspace=0.5)
    for k in range(0,len(material_class_ID)):
        data = w_mat_opt[:, k*len(SDs_select):(k+1)*len(SDs_select)]
        numOfCols = WL.shape[0]
        numOfRows = len(SDs_select)
        xpos = np.arange(0, numOfCols, 1)
        ypos = np.arange(0, numOfRows, 1)
        ZZ= np.abs(data)
        indicies_selected_coefficients = np.nonzero(ZZ>1e-3)
        Z = np.zeros(ZZ.shape)
        Z[indicies_selected_coefficients] = 1
        Z = np.transpose(Z)
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", [[1, 1, 1], [1, 0 , 0]])
        m = axs[k].imshow(Z, vmin=0, cmap=colormap, aspect=0.5)
        axs[k].set_title('Class '+str(k+1))
        axs[k].set_ylabel('Modal  \n dimension')
        axs[k].set_yticks(ypos)
        axs[k].set_xticks(xpos)
        sc_label_ticks = []
        for sc in range(0, len(WL)):
            sc_label_ticks.append(('SC'+str(sc+1)))
        modal_label_ticks = []
        for i, tit in enumerate(y_axis_vec):
            if tit == 'd [mm]':
                tit = 'd'
            modal_label_ticks.append(tit)
        axs[k].set_yticklabels(modal_label_ticks)
        if k == len(material_class_ID)-1:
            axs[k].set_xticklabels(sc_label_ticks)
            axs[k].set_xlabel('Spectral dimension')
        else:
            axs[k].set_xticklabels([])
        axs[k].hlines(y=np.arange(0, numOfRows) + 0.5, xmin=np.full(numOfRows, 0) - 0.5,
                      xmax=np.full(numOfRows, numOfCols) - 0.5, color="black", linewidth=0.1)
        axs[k].vlines(x=np.arange(0, numOfCols) + 0.5, ymin=np.full(numOfCols, 0) - 0.5,
                  ymax=np.full(numOfCols, numOfRows) - 0.5, color="black", linewidth=0.1)
    cbar = plt.colorbar(m, ax=axs, shrink=0.5, aspect=80, pad=0.1, orientation = 'horizontal')
    cbar.set_ticks(ticks=[0, 1], labels=['Non-selected', 'Selected'])
    cbar.set_label('Non-selected and selected coefficients')
    plt.suptitle('Non-selected and selected coefficients of different spectral channels, '
                 'optical modalities, and material classes \n  Select ' + 'spectral channels (SCs): '+ str(indices_WL+1))
    if fig_save == "True":
        fig.savefig('Fig_Selected_Coeff_' + fig_tags + '_Nr_SCs_'+str(len(indices_WL))+'.png', format='png', dpi=600)
    plt.close('all')

## 2.3.1.3 Sear the proper value of the parameter "C" (the scalar trade-off parameter balancing between training loss and regularization) for a specific target number of selected spectral channels "Nr_WL_target"
def FS_MGSVM_search(C_set, C_set_previous, C_set_next, dir_set, dir_nex, X_train, y_train, k_classes, Nr_WL_target, learning_iterations, intern_learning_iterations):

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
        indices_WL = np.ravel(np.array(np.nonzero(w_vec_opt_norm >1e-3)))
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
        C_divider = np.power(10, np.linspace(np.log10(C_set_previous),np.log10(C_set), num=2+divider_num)[1])
        # C_divider as the new C_set_next
        w_opt_WL_output, b_opt_WL_output, C_intern, indices_WL, learning_iterations, dir_set_new, dir_nex_new, intern_learning_iterations = FS_MGSVM_search(C_divider, C_set_previous, C_set,'indirect', 'larger', X_train, y_train, k_classes, Nr_WL_target, learning_iterations, intern_learning_iterations)
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
            indices_WL_nex = np.ravel(np.array(np.nonzero(w_vec_opt_norm_nex >1e-3)))
            if np.shape(indices_WL_nex)[0] == Nr_WL_target:
                C_set = C_set_next
                indices_WL = indices_WL_nex
                dir_set_new = 'smaller'
                dir_nex_new = 'indirect'
              # print('C_nex return, ', indices_WL)
              # print(np.shape(indices_WL)[0] ,' With the target number ',Nr_WL_target)
                w_opt_WL_output = w_mat_opt_nex[indices_WL,:]
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
            C_divider = np.power(10, np.linspace(np.log10(C_set),np.log10(C_set_next), num=2+divider_num)[1])
            w_opt_WL_output, b_opt_WL_output, C_intern, indices_WL, learning_iterations, dir_set_new, dir_nex_new, intern_learning_iterations = FS_MGSVM_search(C_divider, C_set,C_set_next, 'indirect','larger',X_train, y_train, k_classes, Nr_WL_target, learning_iterations, intern_learning_iterations)
            C_set = C_intern
    return w_opt_WL_output, b_opt_WL_output, C_set, indices_WL, learning_iterations, dir_set_new, dir_nex_new, intern_learning_iterations

## 2.3.1.4 Loop through a vector of potential values of the parameter "C" to achieve all numbers of spectral channels to be selected. In this case, the target number of selected spectral channels ranges from 1 to the maximum number of available spectral channels
def FS_MGSVM_multiclass_loop(X_train, X_test, y_train, y_test, k_classes, SDs_select):
    # Penalty parameter C vector
    C_vec = np.logspace(-3,2, num=30)
    C_vec = np.concatenate((np.array([1e-4]), C_vec),axis=0)
    C_vec = np.concatenate((C_vec, np.logspace(2.1,3, num=8)),axis=0)
    # print(C_vec)

    n_sig = len(SDs_select)   # Nr signature dimensions
    n_feat = X_train.shape[1]    # Nr of features
    n_sc = int(n_feat/n_sig)   # Nr spectral channels
    y_train = y_train.astype(int) - 1
    y_test = y_test.astype(int) - 1
    # Traget number of selected featur groups
    Nr_WL_target = 1
    # The corresponding indices of the C_vec when we have the target selected feature number
    i_fs = np.array([])
    # Indices of the selected features
    indices_fs_matrix = []
    # masks
    masks = []
    ### Scores
    # linear SVM
    F1_lsvc = []
    # RBF-SVM
    F1_svc_rbf = []


    dir_set = 'indirect'
    dir_nex = 'indirect'
    learning_iterations = 0
    w_opt_WL_output_1_list = []
    b_opt_WL_output_1_list = []
    for i, C_set in enumerate(C_vec[1:-1]):
        i = i+1
        # print(i)
        intern_learning_iterations = 0
        w_opt_WL_output_1, b_opt_WL_output_1, C_new, indices_WL,learning_iterations, dir_set, dir_next, intern_learning_iterations = FS_MGSVM_search(C_set, C_vec[i-1], C_vec[i+1], dir_set, dir_nex,X_train, y_train, k_classes, Nr_WL_target, learning_iterations,intern_learning_iterations)
        if np.shape(indices_WL)[0] == Nr_WL_target:
            print('MGSVM FS-------->  ',np.shape(indices_WL)[0], ' wavelengths are selected.')
            Nr_WL_target += 1
            i_fs = np.append(i_fs, i)
            C_vec[i] = C_new
            # Get the mask and index of features
            indices_fs = np.array([])
            for ind in range(0,n_sig):
                indices_fs = np.append(indices_fs, indices_WL + ind*n_sc).astype(int)
            mask = np.zeros(X_train.shape[1], dtype=bool)
            mask[indices_fs] = True
            masks.append(mask)
            indices_fs_matrix.append(indices_fs)
            # Collect old coefficients
            w_opt_WL_output_1_list.append(w_opt_WL_output_1)
            b_opt_WL_output_1_list.append(b_opt_WL_output_1)

            # print('Indices [MGSVM FS]: ', indices_fs)

            # -----> Predict and score
            X_test_selected = X_test[:, indices_fs]
            X_train_selected = X_train[:, indices_fs]
            ### linear SVM
            lsvc = LinearSVC(C=100, dual=False, tol=1e-5, max_iter=1e3)
            lsvc.fit(X_train_selected, y_train)
            y_predict_lsvc = lsvc.predict(X_test_selected)
            F1_lsvc.append(f1_score(np.ravel(y_test), y_predict_lsvc, average='macro'))
            ### RBF-SVM
            svc_rbf = SVC(C=100, gamma=1, kernel='rbf', tol=1e-5, max_iter=1e7)
            svc_rbf.fit(X_train_selected, y_train)
            y_predict_svc_rbf = svc_rbf.predict(X_test_selected)
            F1_svc_rbf.append(f1_score(np.ravel(y_test), y_predict_svc_rbf, average='macro'))

        if np.shape(indices_WL)[0] == (n_sc):
            # print('Reach **********************', np.shape(indices_WL)[0], ' = ', n_sc)
            break
        elif (np.shape(indices_WL)[0] < (n_sc)) and (i == C_vec[1:-1].shape[0]):
            # raise Exception('The step is too small, please change the C_vec!')
            print('The step of the vector C_vec is too small, please change the C_vec!')
            # Implement the binary classification
            w_opt_WL_output_1, b_opt_WL_output_1 = FS_MGSVM(X_train, y_train, C_vec[-1], SDs_select)
            w_opt_WL_output_norm_1 = np.abs(w_opt_WL_output_1).sum(axis=1)
            indices_WL_end = np.ravel(np.array(np.argsort(w_opt_WL_output_norm_1)[::-1][:n_sc]))

            ii_target = Nr_WL_target
            for Nr_WL_target in np.arange(ii_target, n_sc+1):
                indices_WL = indices_WL_end[:Nr_WL_target]
                print('MGSVM FS-------->  ',np.shape(indices_WL)[0], ' wavelengths are selected.')
                # Get the mask and index of features
                indices_fs = np.array([])
                for ind in range(0,n_sig):
                    indices_fs = np.append(indices_fs, indices_WL + ind*n_sc).astype(int)
                mask = np.zeros(X_train.shape[1], dtype=bool)
                mask[indices_fs] = True
                masks.append(mask)
                indices_fs_matrix.append(indices_fs)
                # Collect old coefficients
                w_opt_WL_output_1_list.append(w_opt_WL_output_1[indices_WL,:])
                b_opt_WL_output_1_list.append(b_opt_WL_output_1)

                # print('Indices [MGSVM FS]: ', indices_fs)

                # -----> Predict and score
                X_test_selected = X_test[:, indices_fs]
                X_train_selected = X_train[:, indices_fs]
                ### linear SVM
                lsvc = LinearSVC(C=100, dual=False, tol=1e-5, max_iter=1e3)
                lsvc.fit(X_train_selected, y_train)
                y_predict_lsvc = lsvc.predict(X_test_selected)
                F1_lsvc.append(f1_score(np.ravel(y_test), y_predict_lsvc, average='macro'))
                ### RBF-SVM
                svc_rbf = SVC(C=100, gamma=1, kernel='rbf', tol=1e-5, max_iter=1e7)
                svc_rbf.fit(X_train_selected, y_train)
                y_predict_svc_rbf = svc_rbf.predict(X_test_selected)
                F1_svc_rbf.append(f1_score(np.ravel(y_test), y_predict_svc_rbf, average='macro'))

    ### Feature mask of the selected feature subset
    masks = np.array(masks)
    ### linear SVM performence metrics
    F1_lsvc = np.array(F1_lsvc)
    ### RBF-SVM performence metrics
    F1_svc_rbf = np.array(F1_svc_rbf)

    return indices_fs_matrix, masks,  F1_lsvc, F1_svc_rbf

""" 2.3.2 RF-MDPA FS """
def FS_RF_MDPA(X_train, X_test, y_train, y_test, dim_vec):
    """Credit: Source code and examples can be found under
    https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py"""


    ### Final variables to return
    # List of feature index at each target number of wavelengths
    indices_fs_matrix = []
    # Mask list at each target number of wavelengths
    masks = []

    ### Scores
    # linear SVM
    F1_lsvc = []
    # RBF-SVM
    F1_svc_rbf = []

    ### Get the feature ranking by RF_MDPA feature selection algorithm
    # Define the RF classifier
    rf_classifer = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    rf_classifer.fit(X_train, y_train)
    # Get permutation importance from the RF classifier
    result = permutation_importance(
        rf_classifer, X_train, y_train, n_repeats=10, random_state=42, n_jobs=2
    )
    # Get the ranking of feature indices
    idx = result.importances_mean.argsort()
    idx = idx[::-1]
    idx = np.array(idx)
    # print('IDX:', idx)

    # Target number of WL
    nr_target_WL = 1
    for idx_stop_index in np.arange(X.shape[1]):
        index_WL_selected = np.unique(idx[:idx_stop_index+1]%WL.shape[0])
        # index array of selected featres
        index_fs_selected = np.array([])
        if len(index_WL_selected) == nr_target_WL:
            print('RF_MDPA FS------> ',len(index_WL_selected),' WLs are selected.')
            for ind in np.arange(len(dim_vec)):
                index_fs_selected = np.append(index_fs_selected, ind*WL.shape[0]+index_WL_selected)
            index_fs_selected = index_fs_selected.astype(int)
            # print(nr_target_WL*(ind+1), 'features are selected: ', index_fs_selected)
            nr_target_WL = nr_target_WL + 1

            ### Get the list of feature index at each target number of wavelengths
            indices_fs_matrix.append(index_fs_selected)
            # print('RF_MDPA append here', len(indices_fs_matrix))
            ### Get the list of masks at each target number of wavelengths
            mask = np.zeros(X_train.shape[1], dtype=bool)
            mask[index_fs_selected] = True
            masks.append(mask)

            ### -----> Predict and score
            X_test_selected = X_test[:, index_fs_selected]
            X_train_selected = X_train[:, index_fs_selected]
            ### linear SVM
            lsvc = LinearSVC(C=100, dual=False, tol=1e-5, max_iter=1e3)
            lsvc.fit(X_train_selected, y_train)
            y_predict_lsvc = lsvc.predict(X_test_selected)
            F1_lsvc.append(f1_score(np.ravel(y_test), y_predict_lsvc, average='macro'))
            ### RBF-SVM
            svc_rbf = SVC(C=100, gamma=1, kernel='rbf', tol=1e-5, max_iter=1e7)
            svc_rbf.fit(X_train_selected, y_train)
            y_predict_svc_rbf = svc_rbf.predict(X_test_selected)
            F1_svc_rbf.append(f1_score(np.ravel(y_test), y_predict_svc_rbf, average='macro'))

            continue
    return indices_fs_matrix, masks,  F1_lsvc, F1_svc_rbf

""" 2.3.3 MRMR FS """
def FS_MRMR(X_train, X_test, y_train, y_test, dim_vec):
    """Credit: Source code and refernece can be found under https://github.com/smazzanti/mrmr"""

    ### Final variables to return
    # List of feature index at each target number of wavelengths
    indices_fs_matrix = []
    # Mask list at each target number of wavelengths
    masks = []

    ### Scores
    # linear SVM
    F1_lsvc = []
    # RBF-SVM
    F1_svc_rbf = []

    # Get the feature ranking by MRMR feature selection algorithm
    X_training= pd.DataFrame(X_train)
    y_training = pd.Series(y_train)
    idx = mrmr_classif(X=X_training, y=y_training, K=X.shape[1])
    idx = np.array(idx)
    # print('IDX:', idx)

    # Target number of WL
    nr_target_WL = 1
    for idx_stop_index in np.arange(X.shape[1]):
        index_WL_selected = np.unique(idx[:idx_stop_index+1]%WL.shape[0])
        # index array of selected featres
        index_fs_selected = np.array([])
        if len(index_WL_selected) == nr_target_WL:
            nr_target_WL = nr_target_WL + 1
            print('MRMR FS------> ',len(index_WL_selected),' WLs are selected.')
            for ind in np.arange(len(dim_vec)):
                index_fs_selected = np.append(index_fs_selected, ind*WL.shape[0]+index_WL_selected)
            index_fs_selected = index_fs_selected.astype(int)
            # print(nr_target_WL*(ind+1), 'features are selected: ', index_fs_selected)

            ### Get the list of feature index at each target number of wavelengths
            indices_fs_matrix.append(index_fs_selected)
            # print('MRMR append here', len(indices_fs_matrix))
            ### Get the list of masks at each target number of wavelengths
            mask = np.zeros(X_train.shape[1], dtype=bool)
            mask[index_fs_selected] = True
            masks.append(mask)

            ### -----> Predict and score
            X_test_selected = X_test[:, index_fs_selected]
            X_train_selected = X_train[:, index_fs_selected]
            ### linear SVM
            lsvc = LinearSVC(C=100, dual=False, tol=1e-5, max_iter=1e3)
            lsvc.fit(X_train_selected, y_train)
            y_predict_lsvc = lsvc.predict(X_test_selected)
            F1_lsvc.append(f1_score(np.ravel(y_test), y_predict_lsvc, average='macro'))

            ### RBF-SVM
            svc_rbf = SVC(C=100, gamma=1, kernel='rbf', tol=1e-5, max_iter=1e7)
            svc_rbf.fit(X_train_selected, y_train)
            y_predict_svc_rbf = svc_rbf.predict(X_test_selected)
            F1_svc_rbf.append(f1_score(np.ravel(y_test), y_predict_svc_rbf, average='macro'))

            continue
    return indices_fs_matrix, masks,  F1_lsvc, F1_svc_rbf

"""--------------------- 2.4 Kuncheva's stability index (one evaluation metric) ---------------------"""
def stability_measure(masks_kfold):
    # Convert the masks from feature mask to wavelength mask
    masks_WL = []
    for k in range(0, len(masks_kfold)):
        masks_f = np.array(masks_kfold[k]).T
        masks_WL.append(masks_f[0:int(masks_f.shape[0]/len(SDs_select)), :])
    # Consistency index for two subsets (Kuncheva's instability measure) I_c
    I_c_vec = np.array([])
    for i in range(0, len(masks_WL)-1):
        for j in range(i+1, len(masks_WL)):
            # overlap/intersection of two feature subsets
            r = np.count_nonzero(np.array(masks_WL[i], dtype=int) + np.array(masks_WL[j], dtype=int) == 2, axis=0)
            # number of total features: n
            n = np.array(masks_WL[j], dtype=int).shape[0]
            # fixed subset size (cardinality): k
            k = np.arange(1, np.array(masks_WL[j], dtype=int).shape[1]+1)
            I_c = (r[:-1]*n-k[:-1]*k[:-1])/(k[:-1]*(n-k[:-1]))
            I_c_vec = np.append(I_c_vec, I_c)
    # Stability index I_s
    I_c_matrix = I_c_vec.reshape((int(I_c_vec.shape[0]/(r.shape[0]-1))), r.shape[0]-1)
    I_s = 2/(len(masks_kfold)*(len(masks_kfold)-1))*I_c_matrix.sum(axis=0)
    return I_s

"""********************************************   3. Start the main script   ********************************************"""

"""-------------------------   3.0 Parameters to define   -------------------------"""
### Select the modality combination to be examined
# # Example 1: All seven modality combinations
# SDs = [ [3],[1], [2],[1, 2], [1, 3], [2, 3], [1, 2, 3]]
# dim_vecs = [['DoLP'],['R'],['d'],['R','d'],['R','DoLP'],['d','DoLP'],['R','d','DoLP']]
# # Example 2: The modality combination R+d+DoLP
# SDs = [[1, 2, 3]]
# dim_vecs = [['R','d','DoLP']]
# Example 3: The modality combination d+DoLP
SDs = [[2,3]]
dim_vecs = [['d+DoLP']]

### Select the number of the folds for cross validation
K_fold = 10
### Select plotting / not plotting the coefficients in MGSVM FS algorithms
plot_coefficient = "False"

### Implement and evaluate feature selection algorithms for all the selected modality combinations
time_initial = time.time()
for SDs_select_input in SDs:
    # Select the signature dimension: reflectance, distance, DoLP
    SDs_select = SDs_select_input  # 1: reflectance spectra, 2: distance spectra, 3: DoLP spectra
    """-------------------------   3.1 Import and define dataset   -------------------------  """
    ### Define parameters related to material classes
    # Material classes
    material_class_ID = [1, 2, 3, 4]
    material_class_name = ['concrete', 'plastic', 'stone', 'wood']
    Nr_material_class = len(material_class_ID)
    # Material subclasses
    material_subclass_ID = [11, 12, 13, 21, 23, 31, 32, 41, 42, 43]
    material_subclass_name = ['standard concrete', 'fiber concrete', 'light-weight concrete', 'PE', 'PVC',
                              'sandstone', 'limestone', 'spruce', 'beech', 'fir']
    Nr_material_subclass = len(material_subclass_ID)
    # Material samples
    Nr_SamplePerSubclass = 3 # Number of material samples per material subclasses
    Nr_material_samples = Nr_material_subclass*Nr_SamplePerSubclass
    # Measurement on the whole collection of material samples
    Nr_MeasPerSample = 20 # Number of measurement per material sample (Each measurement on different surface positions for a specific material sample)
    Nr_Meas_tot =  Nr_MeasPerSample * Nr_material_samples
    # Materal labels for each measurement
    M_class_label = []
    # The folder saving the MM dataset
    subfolder_dataset_MM_features = "Dataset_MM_features"
    # Load central wavelengths of 28 spectral channels
    filepath_WL = Path(subfolder_dataset_MM_features + '/WL' + '.csv')
    WL = pd.read_csv(filepath_WL, sep=";").to_numpy()[:, 1:].reshape(-1).astype(int)
    # Create the string column headers for MM features (WL)
    WL_headers = ["%.1f" % x for x in WL]

    ### Load the multimodal multispectral (MM) features and labels for each measurement
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

    ### Assign the MM features to "X"
    if SDs_select == [1]:
        X = MM_R_spectra
    elif SDs_select == [2]:
        X = MM_d_spectra
    elif SDs_select == [3]:
        X = MM_DoLP_spectra
    elif SDs_select == [1, 2]:
        X = np.concatenate((MM_R_spectra, MM_d_spectra), axis=1)
    elif SDs_select == [1, 3]:
        X = np.concatenate((MM_R_spectra, MM_DoLP_spectra), axis=1)
    elif SDs_select == [2, 3]:
        X = np.concatenate((MM_d_spectra, MM_DoLP_spectra), axis=1)
    elif SDs_select == [1, 2, 3]:
        X = np.concatenate((MM_R_spectra, MM_d_spectra, MM_DoLP_spectra), axis=1)
    else:
        print("Oops!  That was not a valid number.  Please input 1, 2 or 3 to select a certain signature dimension (reflectance, distance, DoLP). ")

    ### Assign the classification labels to "y"
    y = M_class_label

    ### Define some useful vectors
    if SDs_select == [1]:
        y_axis_vec = ['R']
        ylim_vec = [[0, 1.2]]
        dim_vec = ['Reflectance']
    elif SDs_select == [2]:
        y_axis_vec = ['d [mm]']
        ylim_vec = [[-3, 3]]
        dim_vec = ['Distance']
    elif SDs_select == [3]:
        y_axis_vec = ['DoLP']
        ylim_vec = [[0, 1.2]]
        dim_vec = ['Degree of linear polarization']
    elif SDs_select == [1, 2]:
        y_axis_vec = ['R', 'd [mm]']
        ylim_vec = [[0, 1.2], [-3, 3]]
        dim_vec = ['Reflectance', 'Distance']
    elif SDs_select == [1, 3]:
        y_axis_vec = ['R', 'DoLP']
        ylim_vec = [[0, 1.2], [0, 1.2]]
        dim_vec = ['Reflectance', 'Degree of linear polarization']
    elif SDs_select == [2, 3]:
        y_axis_vec = ['d [mm]', 'DoLP']
        ylim_vec = [[-3, 3], [0, 1.2]]
        dim_vec = ['Distance', 'Degree of linear polarization']
    elif SDs_select == [1, 2, 3]:
        y_axis_vec = ['R', 'd [mm]', 'DoLP']
        ylim_vec = [[0, 1.2], [-3, 3], [0, 1.2]]
        dim_vec = ['Reflectance', 'Distance', 'Degree of linear polarization']

    """ -------------------------   3.2 Get file name   ------------------------- """
    filename_index = file_name_partial(y_axis_vec)

    """-------------------------   3.3 Cross-Validation to retrun accuracies and masks   ------------------------- """
    # Cross-Validation to retrun accuracies and masks
    (masks_rand_loop_kfold, masks_MGSVM_kfold, indices_fs_matrix_MGSVM_kfold, masks_RF_MDPA_kfold, masks_mrmr_kfold,
     F1_lsvc_MGSVM_kfold, F1_lsvc_MGSVM_kfold_means, F1_lsvc_MGSVM_kfold_stds,
     F1_svc_rbf_MGSVM_kfold, F1_svc_rbf_MGSVM_kfold_means, F1_svc_rbf_MGSVM_kfold_stds,
     F1_lsvc_rand_loop_kfold, F1_lsvc_rand_loop_kfold_means, F1_lsvc_rand_loop_kfold_stds,
     F1_svc_rbf_rand_loop_kfold, F1_svc_rbf_rand_loop_kfold_means, F1_svc_rbf_rand_loop_kfold_stds,
     F1_lsvc_RF_MDPA_kfold, F1_lsvc_RF_MDPA_kfold_means, F1_lsvc_RF_MDPA_kfold_stds,
     F1_svc_rbf_RF_MDPA_kfold, F1_svc_rbf_RF_MDPA_kfold_means, F1_svc_rbf_RF_MDPA_kfold_stds,
     F1_lsvc_mrmr_kfold, F1_lsvc_mrmr_kfold_means, F1_lsvc_mrmr_kfold_stds,
     F1_svc_rbf_mrmr_kfold, F1_svc_rbf_mrmr_kfold_means, F1_svc_rbf_mrmr_kfold_stds) = cross_validation(X, y, K_fold)
    """ -------------------------   3.4 Calculate stability   ------------------------- """
    # Calculate stability
    I_s_MGSVM = stability_measure(masks_MGSVM_kfold)
    I_s_random = stability_measure(masks_rand_loop_kfold)
    I_s_RF_MDPA = stability_measure(masks_RF_MDPA_kfold)
    I_s_mrmr = stability_measure(masks_mrmr_kfold)
    """ -------------------------   3.5 Save scores, masks, and stabilities   ------------------------- """
    # Save scores, masks, and stabilities
    save_scores_masks_stability()

print('The total runing time for this script is: ' + str(time.time() - time_initial) + ' s')
print('The evaluation of different feature selection algorithms is done! :)')

plt.show()
