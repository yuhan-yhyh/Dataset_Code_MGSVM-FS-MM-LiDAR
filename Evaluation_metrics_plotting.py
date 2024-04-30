# Goal: This script is to plot the final evaluation metrics.
# ********************************************   1. Import packages   ********************************************
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ********************************************   2. Define functions  ********************************************
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

def load_scores_masks_stability():
    ### Peroformance metrics

    ## MGSVM FS
    F1_lsvc_MGSVM_kfold = np.load('Results_evaluation_metrics\F1_lsvc_kfold_WL' + filename_index + '_MGSVM.npy')
    F1_lsvc_MGSVM_kfold_means = np.load('Results_evaluation_metrics\F1_lsvc_mean_WL' + filename_index + '_MGSVM.npy')
    F1_lsvc_MGSVM_kfold_stds = np.load('Results_evaluation_metrics\F1_lsvc_std_WL' + filename_index + '_MGSVM.npy')

    F1_svc_rbf_MGSVM_kfold = np.load('Results_evaluation_metrics\F1_svc_rbf_kfold_WL' + filename_index + '_MGSVM.npy')
    F1_svc_rbf_MGSVM_kfold_means = np.load(
        'Results_evaluation_metrics\F1_svc_rbf_mean_WL' + filename_index + '_MGSVM.npy')
    F1_svc_rbf_MGSVM_kfold_stds = np.load(
        'Results_evaluation_metrics\F1_svc_rbf_std_WL' + filename_index + '_MGSVM.npy')

    ## random FS
    F1_lsvc_rand_loop_kfold = np.load('Results_evaluation_metrics\F1_lsvc_kfold_WL' + filename_index + '_random.npy')
    F1_lsvc_rand_loop_kfold_means = np.load(
        'Results_evaluation_metrics\F1_lsvc_mean_WL' + filename_index + '_random.npy')
    F1_lsvc_rand_loop_kfold_stds = np.load('Results_evaluation_metrics\F1_lsvc_std_WL' + filename_index + '_random.npy')

    F1_svc_rbf_rand_loop_kfold = np.load(
        'Results_evaluation_metrics\F1_svc_rbf_kfold_WL' + filename_index + '_random.npy')
    F1_svc_rbf_rand_loop_kfold_means = np.load(
        'Results_evaluation_metrics\F1_svc_rbf_mean_WL' + filename_index + '_random.npy')
    F1_svc_rbf_rand_loop_kfold_stds = np.load(
        'Results_evaluation_metrics\F1_svc_rbf_std_WL' + filename_index + '_random.npy')

    ## RF_MDPA FS
    F1_lsvc_RF_MDPA_kfold = np.load('Results_evaluation_metrics\F1_lsvc_kfold_WL' + filename_index + '_RF_MDPA.npy')
    F1_lsvc_RF_MDPA_kfold_means = np.load(
        'Results_evaluation_metrics\F1_lsvc_mean_WL' + filename_index + '_RF_MDPA.npy')
    F1_lsvc_RF_MDPA_kfold_stds = np.load('Results_evaluation_metrics\F1_lsvc_std_WL' + filename_index + '_RF_MDPA.npy')

    F1_svc_rbf_RF_MDPA_kfold = np.load(
        'Results_evaluation_metrics\F1_svc_rbf_kfold_WL' + filename_index + '_RF_MDPA.npy')
    F1_svc_rbf_RF_MDPA_kfold_means = np.load(
        'Results_evaluation_metrics\F1_svc_rbf_mean_WL' + filename_index + '_RF_MDPA.npy')
    F1_svc_rbf_RF_MDPA_kfold_stds = np.load(
        'Results_evaluation_metrics\F1_svc_rbf_std_WL' + filename_index + '_RF_MDPA.npy')

    ## MRMR FS
    F1_lsvc_mrmr_kfold = np.load('Results_evaluation_metrics\F1_lsvc_kfold_WL' + filename_index + '_mrmr.npy')
    F1_lsvc_mrmr_kfold_means = np.load('Results_evaluation_metrics\F1_lsvc_mean_WL' + filename_index + '_mrmr.npy')
    F1_lsvc_mrmr_kfold_stds = np.load('Results_evaluation_metrics\F1_lsvc_std_WL' + filename_index + '_mrmr.npy')

    F1_svc_rbf_mrmr_kfold = np.load('Results_evaluation_metrics\F1_svc_rbf_kfold_WL' + filename_index + '_mrmr.npy')
    F1_svc_rbf_mrmr_kfold_means = np.load(
        'Results_evaluation_metrics\F1_svc_rbf_mean_WL' + filename_index + '_mrmr.npy')
    F1_svc_rbf_mrmr_kfold_stds = np.load('Results_evaluation_metrics\F1_svc_rbf_std_WL' + filename_index + '_mrmr.npy')

    ### masks
    ## MGSVM FS
    masks_MGSVM_kfold = np.load('Results_evaluation_metrics\Mask_WL' + filename_index + '_MGSVM.npy')
    ## random FS
    masks_rand_loop_kfold = np.load('Results_evaluation_metrics\Mask_WL' + filename_index + '_random.npy')
    ## RF_MDPA FS
    masks_RF_MDPA_kfold = np.load('Results_evaluation_metrics\Mask_WL' + filename_index + '_RF_MDPA.npy')
    ## MRMR FS
    masks_mrmr_kfold = np.load('Results_evaluation_metrics\Mask_WL' + filename_index + '_mrmr.npy')

    ### stability index
    ## MGSVM FS
    I_s_MGSVM = np.load('Results_evaluation_metrics\I_stab_WL' + filename_index + '_MGSVM.npy')
    ## random FS
    I_s_random = np.load('Results_evaluation_metrics\I_stab_WL' + filename_index + '_random.npy')
    ## RF_MDPA FS
    I_s_RF_MDPA = np.load('Results_evaluation_metrics\I_stab_WL' + filename_index + '_RF_MDPA.npy')
    ## MRMR FS
    I_s_mrmr = np.load('Results_evaluation_metrics\I_stab_WL' + filename_index + '_mrmr.npy')

    return (F1_lsvc_MGSVM_kfold, F1_lsvc_MGSVM_kfold_means, F1_lsvc_MGSVM_kfold_stds,
            F1_svc_rbf_MGSVM_kfold, F1_svc_rbf_MGSVM_kfold_means, F1_svc_rbf_MGSVM_kfold_stds,
            F1_lsvc_rand_loop_kfold, F1_lsvc_rand_loop_kfold_means, F1_lsvc_rand_loop_kfold_stds,
            F1_svc_rbf_rand_loop_kfold, F1_svc_rbf_rand_loop_kfold_means, F1_svc_rbf_rand_loop_kfold_stds,
            F1_lsvc_RF_MDPA_kfold, F1_lsvc_RF_MDPA_kfold_means, F1_lsvc_RF_MDPA_kfold_stds,
            F1_svc_rbf_RF_MDPA_kfold, F1_svc_rbf_RF_MDPA_kfold_means, F1_svc_rbf_RF_MDPA_kfold_stds,
            F1_lsvc_mrmr_kfold, F1_lsvc_mrmr_kfold_means, F1_lsvc_mrmr_kfold_stds,
            F1_svc_rbf_mrmr_kfold, F1_svc_rbf_mrmr_kfold_means, F1_svc_rbf_mrmr_kfold_stds,
            masks_rand_loop_kfold, masks_MGSVM_kfold, masks_RF_MDPA_kfold, masks_mrmr_kfold,
            I_s_MGSVM, I_s_random, I_s_RF_MDPA, I_s_mrmr)

def plotting_accuracy_features_stability_2subplots(masks_MGSVM_kfold, scores_MGSVM_kfold_means, scores_MGSVM_kfold_stds,
                                            scores_rand_loop_kfold_means, scores_rand_loop_kfold_stds,
                                            scores_RF_MDPA_kfold_means, scores_RF_MDPA_kfold_stds,
                                            scores_mrmr_kfold_means, scores_mrmr_kfold_stds,
                                            I_s_MGSVM, I_s_rand, I_s_RF_MDPA, I_s_mrmr, filename_index,
                                            classifier_index):

    font = {'family': 'Arial',
            'size': '15',
            'weight': 'normal'}
    matplotlib.rc('font', **font)

    ### ------------ Plot the accuracy and the stability measure ------------

    fig, axs = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(6.65, 4), layout='constrained')
    fig.get_layout_engine().set(w_pad=0.2, h_pad=0.1, hspace=1, wspace=0)
    ### Plot the accuracy
    color_rand = 'k'
    color_MGSVM = np.array([255, 102, 20]) / 255
    color_RF_MDPA = np.array([56, 176, 0]) / 255
    color_mrmr = np.array([51, 102, 204]) / 255

    lins1 = axs.plot(np.array(range(0, WL.shape[0])) + 0.5, scores_rand_loop_kfold_means, "o-",
                     color=color_rand, markersize=4, label='mean-F1 (random)')
    axs.fill_between(np.array(range(0, WL.shape[0])) + 0.5,
                     scores_rand_loop_kfold_means + scores_rand_loop_kfold_stds,
                     scores_rand_loop_kfold_means - scores_rand_loop_kfold_stds, facecolor=color_rand,
                     alpha=0.15)
    lins2 = axs.plot(np.array(range(0, WL.shape[0])) + 0.5, scores_RF_MDPA_kfold_means, 'o-',
                     color=color_RF_MDPA, markersize=4, label='mean-F1 (RF-MDPA)')
    axs.fill_between(np.array(range(0, WL.shape[0])) + 0.5,
                     scores_RF_MDPA_kfold_means + np.array(scores_RF_MDPA_kfold_stds),
                     scores_RF_MDPA_kfold_means - np.array(scores_RF_MDPA_kfold_stds), facecolor=color_RF_MDPA,
                     alpha=0.15)
    lins3 = axs.plot(np.array(range(0, WL.shape[0])) + 0.5, scores_mrmr_kfold_means, 'o-',
                     color=color_mrmr, markersize=4, label='mean-F1 (MRMR)')
    axs.fill_between(np.array(range(0, WL.shape[0])) + 0.5,
                     scores_mrmr_kfold_means + np.array(scores_mrmr_kfold_stds),
                     scores_mrmr_kfold_means - np.array(scores_mrmr_kfold_stds), facecolor=color_mrmr,
                     alpha=0.15)
    lins4 = axs.plot(np.array(range(0, WL.shape[0])) + 0.5, scores_MGSVM_kfold_means, 'o-', color=color_MGSVM,
                     markersize=4, label='mean-F1 (MGSVM)')
    axs.fill_between(np.array(range(0, WL.shape[0])) + 0.5,
                     scores_MGSVM_kfold_means + np.array(scores_MGSVM_kfold_stds),
                     scores_MGSVM_kfold_means - np.array(scores_MGSVM_kfold_stds), facecolor=color_MGSVM,
                     alpha=0.15)

    axs.set_xticks(np.array(range(0, WL.shape[0]))[::3] + 0.5)
    axs.set_xticklabels(np.array(range(0, WL.shape[0]))[::3] + 1)
    axs.set_xlabel('Number of selected spectral channels')
    metric_index = 'F1'
    axs.set_ylabel("F1 score")
    axs.set_ylim([0, 1.02])
    axs.set_yticks(np.linspace(0, 1, num=6))
    axs.set_yticklabels(np.round(np.linspace(0, 1, num=6), 1), color='k')
    axs.set_title(filename_index[1:-4], fontsize=16)
    axs.grid(True, alpha=0.2)

    ### Plot the stability measure
    ax2 = axs.twinx()
    width = 0.25
    bar5 = ax2.bar(np.array(range(0, WL.shape[0] - 1)) + 0.5 - width * 3 / 2, I_s_rand, width, facecolor=color_rand,
                   alpha=1, label='Stability (random)')
    bar6 = ax2.bar(np.array(range(0, WL.shape[0] - 1)) + 0.5 - width / 2, I_s_RF_MDPA, width, facecolor=color_RF_MDPA,
                   alpha=1, label='Stability (RF-MDPA)')
    bar7 = ax2.bar(np.array(range(0, WL.shape[0] - 1)) + 0.5 + width / 2, I_s_mrmr, width, facecolor=color_mrmr,
                   alpha=1, label='Stability (MRMR)')
    bar8 = ax2.bar(np.array(range(0, WL.shape[0] - 1)) + 0.5 + width * 3 / 2, I_s_MGSVM, width, facecolor=color_MGSVM,
                   alpha=1, label='Stability (MGSVM )')
    ax2.set_ylim(-0.2, 4)
    label = ax2.set_ylabel('Stability index', loc='bottom', labelpad=8, size=14)
    ax2.yaxis.set_label_coords(1.08, -0.06)
    ax2.set_yticks(np.concatenate((np.array([-0.2]), np.linspace(0, 1, num=6))))
    ax2.set_yticklabels(np.round(np.concatenate((np.array([-0.2]), np.linspace(0, 1, num=6))), 1), color='k', size=12)
    ax2.grid(True, alpha=0.5, linestyle='--', color='r')

    # add these three lines and show legend
    lins = lins1 + lins2 + lins3 + lins4 + [bar5] + [bar6] + [bar7] + [bar8]
    labs = [l.get_label() for l in lins] + [bar5._label] + [bar6._label] + [bar7._label] + [bar8._label]
    axs.legend(lins, labs, loc='upper right', bbox_to_anchor=(1.01, 0.62), prop={'size': 12.5}, ncol=2,
               columnspacing=0.4, labelspacing=0.1, handletextpad=0.1)
    fig.savefig('Fig_Scor_Fs_Sta' + filename_index + metric_index + classifier_index + '_2.png',
                format='png', dpi=600)



    ### ------------  Plot the Selection probability of MGSVM  ------------
    ### Calculate the selection probability (SP) presented as "mask_XXX_ave"
    ## MGSVM features selection
    masks_MGSVM_sum = np.zeros(shape=np.shape(masks_MGSVM_kfold[0]))
    for k in range(0, len(masks_MGSVM_kfold)):
        masks_MGSVM_sum = np.add(masks_MGSVM_sum, np.array(masks_MGSVM_kfold[k]))
    # Calculate the stability of feature selection
    masks_MGSVM_ave = masks_MGSVM_sum.T / K_fold
    masks_MGSVM_ave = masks_MGSVM_ave[0:int(masks_MGSVM_ave.shape[0] / len(SDs_select)), :]

    fig, axs = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(5.5, 3.5), layout='constrained')
    fig.get_layout_engine().set(w_pad=0, h_pad=0.1, hspace=0, wspace=0)
    colormap = 'gist_heat_r'
    m = axs.pcolormesh(masks_MGSVM_ave, cmap=colormap, edgecolors='w', linewidth=1)
    axs.set_ylabel(r'$\lambda$ [nm]')
    axs.set_xticks(np.array(range(0, WL.shape[0]))[::3] + 0.5)
    axs.set_yticks(np.array(range(0, WL.shape[0]))[::3] + 0.5)
    axs.set_yticklabels(WL[::3].astype(int))
    axs.set_xticks(np.array(range(0, WL.shape[0]))[::3] + 0.5)
    axs.set_xticklabels(np.array(range(0, WL.shape[0]))[::3] + 1)
    axs.set_xlabel('Number of selected spectral channels')
    axs.set_title(filename_index[1:-4], fontsize=16)
    cbar = plt.colorbar(m, ax=axs, shrink=1, aspect=40, pad=0.01)
    cbar.set_label('Selection probability')
    fig.savefig('Fig_ChosenFeatures' + filename_index + '_MGSVM.png', format='png', dpi=600)


def plotting_comparison_2D_pcolormesh_single(score_combi_MGSVM_mean, WL, models, colormap, tags):
    fig, ax = plt.subplots(figsize=(20, 4))
    plt.subplots_adjust(bottom=0.15)
    # MGSVM
    Nr_WL = np.arange(0, WL.shape[0]) + 1
    model = models[::-1]
    data = score_combi_MGSVM_mean[::-1, :]
    numOfCols = WL.shape[0]
    numOfRows = len(models)
    xpos = np.arange(0, numOfCols, 1)
    ypos = np.arange(0, numOfRows, 1)
    X, Y = xpos, ypos
    Z = data
    # Plot the 2D mesh
    m = ax.pcolormesh(X, Y, Z, cmap=colormap, vmin=0.4, vmax=1, edgecolors='w', linewidth=0.05)
    ax.set_xticks(xpos[::1])
    ax.set_yticks(ypos)
    ax.set_xticklabels(Nr_WL[::1])
    ax.set_yticklabels(model)
    ax.set_xlabel('Number of selected spectral channels')
    ax.set_title('MGSVM-FS (' + tags + ')')
    cbar = plt.colorbar(m, ax=ax, shrink=1, pad=0.01)
    cbar.set_label('Averaged accuracy score')
    for i, ii in enumerate(xpos):

        for j, jj in enumerate(ypos):
            if round(Z[j, i], 2) >= 0.9:
                plt.text(xpos[i], ypos[j], round(Z[j, i], 2), ha='center', va='center', size=13, color='k')
            else:
                plt.text(xpos[i], ypos[j], round(Z[j, i], 2), ha='center', va='center', size=13, color='k')
    fig.savefig('Fig_Scor_compar' + colormap + tags + '.png', format='png', dpi=600)


def plotting_accuracy_mean_difference(score_dif_mean, WL, models, colormap, tags):
    fig, ax = plt.subplots(figsize=(20, 4))
    plt.subplots_adjust(bottom=0.15)
    # MGSVM
    Nr_WL = np.arange(0, WL.shape[0]) + 1
    model = models[::-1]
    data = score_dif_mean[::-1, :]
    numOfCols = WL.shape[0]
    numOfRows = len(models)
    xpos = np.arange(0, numOfCols, 1)
    ypos = np.arange(0, numOfRows, 1)
    X, Y = xpos, ypos
    Z = data
    # Plot the 2D mesh
    m = ax.pcolormesh(X, Y, Z, cmap=colormap, vmin=-0.6, vmax=0.6, edgecolors='w', linewidth=0.05)
    ax.set_xticks(xpos[::1])
    ax.set_yticks(ypos)
    ax.set_xticklabels(Nr_WL[::1])
    ax.set_yticklabels(model)
    ax.set_xlabel('Number of selected spectral channels')
    cbar = plt.colorbar(m, ax=ax, shrink=1, pad=0.01)
    cbar.set_label('Mean-F1 improvement')
    for i, ii in enumerate(xpos):

        for j, jj in enumerate(ypos):
            if round(Z[j, i], 2) >= 0.9:
                text_number = round(Z[j, i], 2)
                if text_number > -0.01:
                    text_number = np.abs(text_number)
                plt.text(xpos[i], ypos[j], text_number, ha='center', va='center', size=13, color='k')
            else:
                text_number = round(Z[j, i], 2)
                if text_number > -0.01:
                    text_number = np.abs(text_number)
                plt.text(xpos[i], ypos[j], text_number, ha='center', va='center', size=13, color='k')
    fig.savefig('Fig_Scor_mean_difference' + colormap + tags + '.png', format='png', dpi=600)



# ********************************************   3. Start the main script   ********************************************

### Select the modality combination to be examined
# Example 1: All seven modality combinations
SDs = [ [3],[1], [2],[1, 2], [1, 3], [2, 3], [1, 2, 3]]
dim_vecs = [['DoLP'],['R'],['d'],['R','d'],['R','DoLP'],['d','DoLP'],['R','d','DoLP']]
# # Example 2: The modality combination R+d+DoLP
# SDs = [[1, 2, 3]]
# dim_vecs = [['R','d','DoLP']]

# List of the model names
models = []
for SDs_select_input in SDs:
    # -------------------------   3.0 Parameters to define   -------------------------
    # Select the signature dimension: reflectance, distance, DoLP
    SDs_select = SDs_select_input  # 1: reflectance spectra, 2: distance spectra, 3: DoLP spectra
    # The number of the folds for cross validation
    K_fold = 10
    # -------------------------   3.1 Import and define dataset   -------------------------
    # Material classes
    material_class_ID = [1, 2, 3, 4]
    material_class_name = ['concrete', 'plastic', 'stone', 'wood']
    # Material subclasses
    material_subclass_ID = [11, 12, 13, 21, 23, 31, 32, 41, 42, 43]
    material_subclass_name = ['standard concrete', 'fiber concrete', 'light-weight concrete', 'PE', 'PVC',
                              'sandstone', 'limestone', 'spruce', 'beech', 'fir']
    # Material samples
    Nr_SamplePerSubclass = 3
    rows = np.shape(material_subclass_ID)[0] * Nr_SamplePerSubclass
    material_sample_ID = [None for i in range(rows)]
    material_sample_name = [None for i in range(rows)]
    for i in range(np.shape(material_sample_ID)[0]):
        if (i % Nr_SamplePerSubclass) == 0:
            material_sample_ID[i] = str(material_subclass_ID[i // Nr_SamplePerSubclass]) + 'a'
            material_sample_name[i] = str(material_subclass_name[i // Nr_SamplePerSubclass]) + ' a'
        elif (i % Nr_SamplePerSubclass) == 1:
            material_sample_ID[i] = str(material_subclass_ID[i // Nr_SamplePerSubclass]) + 'b'
            material_sample_name[i] = str(material_subclass_name[i // Nr_SamplePerSubclass]) + ' b'
        elif (i % Nr_SamplePerSubclass) == 2:
            material_sample_ID[i] = str(material_subclass_ID[i // Nr_SamplePerSubclass]) + 'c'
            material_sample_name[i] = str(material_subclass_name[i // Nr_SamplePerSubclass]) + ' c'

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

    # -------------------------   3.2 Get file name   -------------------------
    filename_index = file_name_partial(y_axis_vec)

    # -------------------------   3.3 Load scores, masks, and stabilities   -------------------------
    # Load central wavelengths of different spectral channels
    WL = np.load('Results_evaluation_metrics\WL.npy')
    # Load scores, masks, and stabilities
    (F1_lsvc_MGSVM_kfold, F1_lsvc_MGSVM_kfold_means, F1_lsvc_MGSVM_kfold_stds,
     F1_svc_rbf_MGSVM_kfold, F1_svc_rbf_MGSVM_kfold_means, F1_svc_rbf_MGSVM_kfold_stds,
     F1_lsvc_rand_loop_kfold, F1_lsvc_rand_loop_kfold_means, F1_lsvc_rand_loop_kfold_stds,
     F1_svc_rbf_rand_loop_kfold, F1_svc_rbf_rand_loop_kfold_means, F1_svc_rbf_rand_loop_kfold_stds,
     F1_lsvc_RF_MDPA_kfold, F1_lsvc_RF_MDPA_kfold_means, F1_lsvc_RF_MDPA_kfold_stds,
     F1_svc_rbf_RF_MDPA_kfold, F1_svc_rbf_RF_MDPA_kfold_means, F1_svc_rbf_RF_MDPA_kfold_stds,
     F1_lsvc_mrmr_kfold, F1_lsvc_mrmr_kfold_means, F1_lsvc_mrmr_kfold_stds,
     F1_svc_rbf_mrmr_kfold, F1_svc_rbf_mrmr_kfold_means, F1_svc_rbf_mrmr_kfold_stds,
     masks_rand_loop_kfold, masks_MGSVM_kfold, masks_RF_MDPA_kfold, masks_mrmr_kfold,
     I_s_MGSVM, I_s_random, I_s_RF_MDPA, I_s_mrmr) = load_scores_masks_stability()

    # -------------------------   3.4 Plot scores, masks, and stabilities  -------------------------
    classifier_index = 'lsvc'
    metric_MGSVM_means = F1_lsvc_MGSVM_kfold_means
    metric_MGSVM_stds = F1_lsvc_MGSVM_kfold_stds
    metric_random_means = F1_lsvc_rand_loop_kfold_means
    metric_random_stds = F1_lsvc_rand_loop_kfold_stds
    metric_RF_MDPA_means = F1_lsvc_RF_MDPA_kfold_means
    metric_RF_MDPA_stds = F1_lsvc_RF_MDPA_kfold_stds
    metric_mrmr_means = F1_lsvc_mrmr_kfold_means
    metric_mrmr_stds = F1_lsvc_mrmr_kfold_stds
    plotting_accuracy_features_stability_2subplots(masks_MGSVM_kfold, metric_MGSVM_means, metric_MGSVM_stds,
                                            metric_random_means, metric_random_stds,
                                            metric_RF_MDPA_means, metric_RF_MDPA_stds,
                                            metric_mrmr_means, metric_mrmr_stds, I_s_MGSVM, I_s_random, I_s_RF_MDPA,
                                            I_s_mrmr, filename_index, classifier_index)

    # -------------------------   3.5 Fill accuracy matrix -------------------------
    # Fill accuracy matrix and model list
    models.append(filename_index[1:-4])
    if SDs_select_input == SDs[0]:
        F1_lsvc_combi_MGSVM_mean = F1_lsvc_MGSVM_kfold_means
        F1_lsvc_combi_MGSVM_std = F1_lsvc_MGSVM_kfold_stds
        F1_svc_rbf_combi_MGSVM_mean = F1_svc_rbf_MGSVM_kfold_means
        F1_svc_rbf_combi_MGSVM_std = F1_svc_rbf_MGSVM_kfold_stds

        F1_lsvc_combi_rand_mean = F1_lsvc_rand_loop_kfold_means
        F1_lsvc_combi_rand_std = F1_lsvc_rand_loop_kfold_stds
        F1_lsvc_combi_RF_MDPA_mean = F1_lsvc_RF_MDPA_kfold_means
        F1_lsvc_combi_RF_MDPA_std = F1_lsvc_RF_MDPA_kfold_stds
        F1_lsvc_combi_mrmr_mean = F1_lsvc_mrmr_kfold_means
        F1_lsvc_combi_mrmr_std = F1_lsvc_mrmr_kfold_stds
    else:
        F1_lsvc_combi_MGSVM_mean = np.vstack((F1_lsvc_combi_MGSVM_mean, F1_lsvc_MGSVM_kfold_means))
        F1_lsvc_combi_MGSVM_std = np.vstack((F1_lsvc_combi_MGSVM_std, F1_lsvc_MGSVM_kfold_stds))
        F1_svc_rbf_combi_MGSVM_mean = np.vstack((F1_svc_rbf_combi_MGSVM_mean, F1_svc_rbf_MGSVM_kfold_means))
        F1_svc_rbf_combi_MGSVM_std = np.vstack((F1_svc_rbf_combi_MGSVM_std, F1_svc_rbf_MGSVM_kfold_stds))

        F1_lsvc_combi_rand_mean = np.vstack((F1_lsvc_combi_rand_mean, F1_lsvc_rand_loop_kfold_means))
        F1_lsvc_combi_rand_std = np.vstack((F1_lsvc_combi_rand_std, F1_lsvc_rand_loop_kfold_stds))
        F1_lsvc_combi_RF_MDPA_mean = np.vstack((F1_lsvc_combi_RF_MDPA_mean, F1_lsvc_RF_MDPA_kfold_means))
        F1_lsvc_combi_RF_MDPA_std = np.vstack((F1_lsvc_combi_RF_MDPA_std, F1_lsvc_RF_MDPA_kfold_stds))
        F1_lsvc_combi_mrmr_mean = np.vstack((F1_lsvc_combi_mrmr_mean, F1_lsvc_mrmr_kfold_means))
        F1_lsvc_combi_mrmr_std = np.vstack((F1_lsvc_combi_mrmr_std, F1_lsvc_mrmr_kfold_stds))
    # -------------------------   3.11 Calculate F1 score difference matrix  -------------------------
    if SDs_select_input == SDs[0]:
        F1_lsvc_dif_rand_mean = F1_lsvc_MGSVM_kfold_means - F1_lsvc_rand_loop_kfold_means
        F1_lsvc_dif_rand_std = F1_lsvc_MGSVM_kfold_stds - F1_lsvc_rand_loop_kfold_stds
        F1_lsvc_dif_RF_MDPA_mean = F1_lsvc_MGSVM_kfold_means - F1_lsvc_RF_MDPA_kfold_means
        F1_lsvc_dif_RF_MDPA_std = F1_lsvc_MGSVM_kfold_means - F1_lsvc_RF_MDPA_kfold_stds
        F1_lsvc_dif_mrmr_mean = F1_lsvc_MGSVM_kfold_means - F1_lsvc_mrmr_kfold_means
        F1_lsvc_dif_mrmr_std = F1_lsvc_MGSVM_kfold_means - F1_lsvc_mrmr_kfold_stds
        F1_lsvc_dif_lsvc_svc_rbf_mean = F1_svc_rbf_MGSVM_kfold_means - F1_lsvc_MGSVM_kfold_means
        F1_lsvc_dif_lsvc_svc_rbf_std = F1_lsvc_MGSVM_kfold_means - F1_svc_rbf_MGSVM_kfold_stds
    else:
        F1_lsvc_dif_rand_mean = np.vstack(
            (F1_lsvc_dif_rand_mean, F1_lsvc_MGSVM_kfold_means - F1_lsvc_rand_loop_kfold_means))
        F1_lsvc_dif_rand_std = np.vstack(
            (F1_lsvc_dif_rand_std, F1_lsvc_MGSVM_kfold_stds - F1_lsvc_rand_loop_kfold_stds))
        F1_lsvc_dif_RF_MDPA_mean = np.vstack(
            (F1_lsvc_dif_RF_MDPA_mean, F1_lsvc_MGSVM_kfold_means - F1_lsvc_RF_MDPA_kfold_means))
        F1_lsvc_dif_RF_MDPA_std = np.vstack(
            (F1_lsvc_dif_RF_MDPA_std, F1_lsvc_MGSVM_kfold_stds - F1_lsvc_RF_MDPA_kfold_stds))
        F1_lsvc_dif_mrmr_mean = np.vstack(
            (F1_lsvc_dif_mrmr_mean, F1_lsvc_MGSVM_kfold_means - F1_lsvc_mrmr_kfold_means))
        F1_lsvc_dif_mrmr_std = np.vstack((F1_lsvc_dif_mrmr_std, F1_lsvc_MGSVM_kfold_stds - F1_lsvc_mrmr_kfold_stds))
        F1_lsvc_dif_lsvc_svc_rbf_mean = np.vstack(
            (F1_lsvc_dif_lsvc_svc_rbf_mean, F1_svc_rbf_MGSVM_kfold_means - F1_lsvc_MGSVM_kfold_means))
        F1_lsvc_dif_lsvc_svc_rbf_std = np.vstack(
            (F1_lsvc_dif_lsvc_svc_rbf_std, F1_svc_rbf_MGSVM_kfold_stds - F1_lsvc_MGSVM_kfold_means))

# ********************************************   4. Plot comparison   ********************************************

colormap = 'YlOrRd_r'
# # Plot average F1 scores of MGSVM FS with the linear SVM as the evaluation classifier
# plotting_comparison_2D_pcolormesh_single(F1_lsvc_combi_MGSVM_mean, WL, models, colormap, 'lsvc')
# # Plot average F1 scores of MGSVM FS with a nonlinear SVM (radial basis function) as the evaluation classifier
# plotting_comparison_2D_pcolormesh_single(F1_svc_rbf_combi_MGSVM_mean, WL, models, colormap, 'svc_rbf')

# Plot differences of average F1 scores of MGSVM FS with linear and nonlinear SVMs
if len(SDs) == 7:
    plotting_accuracy_mean_difference(F1_lsvc_dif_lsvc_svc_rbf_mean, WL, models, 'coolwarm', 'lsvc_rbfsvc')

print('All evaluation metrics are plotted! :)')

plt.show()
