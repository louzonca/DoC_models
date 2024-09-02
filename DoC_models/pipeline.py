import numpy as np
import scipy.io as sio
import os
import pickle
# This script runs the entire processing pipeline. 
# Each step is implemented by an external script called here

## --------------------- *Load data*----------------------
# Choose whoich dataset to use
dataset_choice = 'all' #'dragana' # 

# Choose initial parcellation (in [100,1000] and subcorticals in [0,1])
parcel_num = 100
subcorticals = 0

if dataset_choice == 'all':
    from load_preproc import load_data, normalize_data, preprocess_data, encode_patients, pca_reduction
    # Load and preprocess all datatsets (the path to the data is defined in load_preproc.py)
    all_data2_4, all_data2, all_dataLiege, all2_4_y, all2_y, allLiege_y = load_data(parcel_num,subcorticals)
    dict_datasets = dict()
    dict_datasets['Paris2_4'] = all_data2_4
    dict_datasets['Paris2'] = all_data2
    dict_datasets['Liege'] = all_dataLiege
    clues = ['Paris2_4','Paris2','Liege']

    dict_y = dict()
    dict_y['Paris2_4'] = all2_4_y
    dict_y['Paris2'] = all2_y
    dict_y['Liege'] = allLiege_y

    freq_band = [0.01,0.1] #Hz
    all_data_normalizedf, all_data_normalized2f, all_data_normalizedLiegef, all2_4_y, all2_y, allLiege_y = preprocess_data(dict_datasets, dict_y, clues, freq_band)
    all_data_labels = np.concatenate((all2_y,all2_4_y,allLiege_y))

elif dataset_choice == 'dragana':
    from load_preproc_dragana import load_data, normalize_data, preprocess_data, encode_patients
    # Path to the data in load_preproc_dragana.py
    all_data, all_labels, all_ids = load_data(parcel_num,subcorticals)
    freq_band = [0.01, 0.1]
    all_data_preprocessed = preprocess_data(all_data, freq_band)
    all_data_labels = all_labels

## ------------------ *Load or train AE* -----------------
from train_AE_kfold import build_training_test_sets, run_ae_training

# Choose whether to load already trained AE for a specific latent dim or 
# to train new ones for a range of latent dimensions
aechoice ='load_trained' #'train_new' #
lat_dim = 15
latdim_range = range(2,26)#range(7,8)

if aechoice == 'load_trained':
    print(f"Go to next step")

elif aechoice == 'train_new':
    proportion = 1 # ratio between training and test, here 1 because we do k-fold after
    time_wndw = 1 # size of windows to split the original data, for fMRI we generally use 1 timeframe
    if dataset_choice == 'all':
        dataset1, _ = build_training_test_sets(all_data_normalizedf[:,:,20:-1],time_wndw, proportion)
        dataset2, _ = build_training_test_sets(all_data_normalized2f[:,:,20:-1],time_wndw, proportion)
        datasetL, _ = build_training_test_sets(all_data_normalizedLiegef[:,:,20:-1], time_wndw, proportion)
        print(f"dataset 1 (Paris) shape {dataset1.shape}")
        print(f"dataset 2 (Paris) shape {dataset2.shape}")
        print(f"dataset 2 (Liège) shape {datasetL.shape}")

        dataset = np.concatenate((dataset1, dataset2, datasetL))

    elif dataset_choice == 'dragana':
            dataset, _ = build_training_test_sets(all_data_preprocessed[:,:,20:-1],time_wndw, proportion)
            print(f"dataset (Paris, Dragana) shape {dataset.shape}")

    # Initiate
    sz_dat = dataset.shape
    print(f"dataset shape {sz_dat}")

    mean_val_mse = []
    mean_val_acc = []
    std_val_mse = []
    std_val_acc = []

    k = 2 # sz_dat[0] # for K-fold validation (sz_dat[0] for 1 vs all validation) 
    save_dir = f"trained_autoencoders/{k}_fold_validation_{parcel_num}_{subcorticals}_parcellation_{dataset_choice}_data_TEST_A_SUPPRIMER" # location to save the trained AEs
    # Run the training
    run_ae_training(k, save_dir, dataset, latdim_range)


## ----------- *Project data in the latent space* --------
from autoEncoderDense import AutoEncoderDense

load_dir = f"../trained_autoencoders/10_fold_validation_{parcel_num}_{subcorticals}_parcellation_ParisLiegeData"
# {dataset_choice}_data"
encoder_trained = AutoEncoderDense.load_encoder(load_dir, f"encoder_7_{lat_dim}")

##
if dataset_choice == 'all':
    encoded_paris = encode_patients(all_data_normalizedf, lat_dim, encoder_trained)
    encoded_paris2 = encode_patients(all_data_normalized2f, lat_dim, encoder_trained)
    encoded_liege = encode_patients(all_data_normalizedLiegef, lat_dim, encoder_trained)

    # Normalize and center in the latent space
    encoded_parisN = normalize_data(encoded_paris)
    encoded_paris2N = normalize_data(encoded_paris2)
    encoded_liegeN = normalize_data(encoded_liege)
    # Concatenate with same time length for all datasets
    encoded_patients = np.concatenate((encoded_parisN[:,:,54:-1],encoded_paris2N,encoded_liegeN[:,:,119:-1]))


elif dataset_choice == 'dragana':
    encoded_data = encode_patients(all_data_preprocessed, lat_dim, encoder_trained)
    encoded_patients = normalize_data(encoded_data)

# Save the encoded patients for the fitting step (on the cluster to parallelize)
save_pat_dir = f'results_{lat_dim}_{parcel_num}/'
# Create folder if it does no exist
if not os.path.exists(save_pat_dir):
    os.makedirs(save_pat_dir)

#sio.savemat(save_pat_dir + f"encoded_patients_{lat_dim}_{dataset_choice}.mat", mdict={f'encoded_patients':encoded_patients})

## ------------- *Plot the latent dim analysis* ----------
# Figure 1A: Example time-series natural vs latent space (1 per condition)
# Figure 1B: Lat dim vs MSE per init parcellation
# Figure 1C: Lat_dim vs MSE per condition

## ------------- *Fit (or load) node parameters and GEC* ---------
from models_numba import hopf_model, ahp_model, integrate

what_to_do = 'load_fitted' #'run_fit' # 
model = ahp_model
pat_labels = np.concatenate((all2_y,all2_4_y,allLiege_y))
#all_data_labels = np.concatenate((all2_y,all2_4_y,allLiege_y))

if what_to_do == 'run_fit':
    from fitting_numba import process_patient
    patients_data_path = save_pat_dir

    # Run the fitting for one patient 
    pat_id = 10
    model = ahp_model # hopf_model
    # Load encoded data
    patients_data_encoded = sio.loadmat(patients_data_path + f'encoded_patients_{lat_dim}_{dataset_choice}.mat') 
    data_for_fit = patients_data_encoded[f'encoded_patients'][pat_id, :, :]

    fitted_state_eq, fitted_ahp, fitted_sig, ssim_evolution, best_ssim_evol, pat_id = \
        process_patient(pat_id, data_for_fit, model)
    
    from gec_fitting_numba import gec_fit
    Nmax = 25 # Max number of iterations for the GEC fitting
    GEC, best_ssim, best_ssim_evol, ssim_evolution, pat_id = \
            gec_fit(data_for_fit, pat_id, model, fitted_state_eq, fitted_ahp, fitted_sig, Nmax)

elif what_to_do == 'load_fitted':
    from load_and_plot import load_node_param_fitted
    from models_numba import hopf_model, ahp_model, integrate
    if dataset_choice == 'all':
        folder_path = 'results/AHP_OK'  # for hopf_model 'results/allData' # for ahp_model 'results/AHP_OK'
        folder_name = 'results_adapt_loop_node_fitting_'
        folderGEC_name = 'results_adapt_loop_GEC_fitting_'
        pat_list = list(range(0,140))

    elif dataset_choice == 'dragana':
        folder_path = 'results/AHP'  # for ahp_model results/AHP_OK
        folder_name = 'results_adapt_loop_dragana_node_fitting_'
        folderGEC_name = 'results_adapt_loop_dragana_GEC_fitting_'
        pat_list = list(range(0,51))

    model = ahp_model

    maxIter = 6 # for ahp_model maxIter = 6

    state_eq_all, sigma_all, ahp_all, b_ssim_evol_all, ssim_evol_all, df, dfNorm = \
        load_node_param_fitted(folder_path, folder_name, dataset_choice, model, pat_list, all_data_labels, maxIter)
    
    GEC = {}
    ssim_gec = {}
    for pat in pat_list:
        for iteration in range(0,maxIter+1):
            #try:
            with open(f'{folder_path}/{folderGEC_name}{iteration}/GEC_{model.__name__}_pat_{pat}', 'rb') as f:
                GEC[(f'pat_{pat}',f'iter_{iteration}')] = pickle.load(f)
            with open(f'{folder_path}/{folderGEC_name}{iteration}/best_ssim_{model.__name__}_pat_{pat}', 'rb') as f:
                ssim_gec[(f'pat_{pat}',f'iter_{iteration}')] = pickle.load(f)
            #except:
            #    print(f'GEC of patient {pat} iteration {iteration} is missing for {model.__name__}')
  

## ------------- *Plot the fitting results: MBBs* --------
# Figure 2A: Schematic fitting pipeline
# Figure 2B: Example simulated time-series vs data + FC matrices after fitting (1 per condition)
# Figure 2C: Fit quality FC empirical vs simulated per iteration step Hopf model
# Figure 2D: same for AHP model

# Figure 3A: MBBs nodes parameters per condition
# Figure 3B: MBBs GEC (quantify something?)

## ------------------ *Classify in all dim* --------------
from classify import classify_all_dims, classify_gec
num_slices = 1 

if dataset_choice == 'all':
    pat_list = range(140)

    time_slice_siz = int(np.floor(all_data_normalized2f.shape[2]/num_slices))   #all_data_normalizedLiegef
    data_slice_2 = all_data_normalized2f[:, :, 0:time_slice_siz]
    data_slice_2_4 = all_data_normalizedf[:, :, 0:time_slice_siz]
    data_slice_Liege = all_data_normalizedLiegef[:, :, 0:time_slice_siz]
    # Concatenate
    all_data = np.concatenate((data_slice_2, data_slice_2_4, data_slice_Liege))
    # Label all the patients: 1 = CNT, 2 = MCS, 3 = UWS
    all_data_y = np.concatenate((all2_y, all2_4_y, allLiege_y))
    for t_idx in range(1, num_slices):
        data_slice_2 = all_data_normalized2f[:, :, t_idx*time_slice_siz: (t_idx+1)*time_slice_siz]
        data_slice_2_4 = all_data_normalizedf[:, :, t_idx*time_slice_siz: (t_idx+1)*time_slice_siz]
        data_slice_Liege = all_data_normalizedLiegef[:, :, t_idx*time_slice_siz: (t_idx+1)*time_slice_siz]
        # Concatenate
        all_data = np.concatenate((all_data, data_slice_2, data_slice_2_4, data_slice_Liege))
        # Label all the patients: 1 = CNT, 2 = MCS, 3 = UWS
        all_data_y = np.concatenate((all_data_y, all2_y, all2_4_y, allLiege_y))

    print(f"Total dataset shape {all_data.shape}")
    print(f"Total dataset shape Y {all_data_y.shape}")

elif dataset_choice == 'dragana':
    pat_list = range(51)

    all_y = np.empty(all_labels.shape)
    for p in range(len(all_labels)):
        if all_labels[p] == 'VS': 
            all_y[p] = 0
        if all_labels[p] == 'MCS-': 
            all_y[p] = 1
        if all_labels[p] == 'MCS+': 
            all_y[p] = 2
        if all_labels[p] == 'EMCS': 
            all_y[p] = 2
        if all_labels[p] == 'COMA': 
            all_y[p] = 0
    
    time_slice_siz = int(np.floor(all_data_preprocessed.shape[2]/num_slices))
    data_slice = all_data_preprocessed[:, :, 0:time_slice_siz]
    # Concatenate
    all_data_sliced = data_slice
    all_data_y = all_y
    print(f"Iteration 0 Total dataset shape {all_data_sliced.shape}")
    print(f"Iteration 0 Total dataset shape Y {all_data_y.shape}")
    for t_idx in range(1, num_slices):
        data_slice_2 = all_data_preprocessed[:, :, t_idx*time_slice_siz: (t_idx+1)*time_slice_siz]
        # Concatenate
        all_data_sliced = np.concatenate((all_data_sliced, data_slice_2))
        all_data_y = np.concatenate((all_data_y, all_y))

        print(f"Iteration {t_idx} Total dataset shape {all_data_sliced.shape}")
        print(f"Iteration {t_idx} Total dataset shape Y {all_data_y.shape}")

# Classify based on FC, in natural space and for each dimension in latdim_range for AE vs PCA
k = 10
save_dir = 'classification_results_all_data_LOO_300724'
classify_all_dims(all_data, all_data_y, load_dir, num_slices, latdim_range, save_dir, k, pat_list)

gec_path = f"results/results_GEC_fitting_for_classif"
PRECISION_GEC, RECALL_GEC, F1_GEC = classify_gec(GEC, all_data_y, k, pat_list, model, maxIter) 

# Figure 1D: Classification accuracy (FC) vs latent dimension (PCA and AE)
# Figure 3C: Classification accuracy EC vs FC (natural ? and latent space)
## ------------------- *UMAPS and clusters* --------------
from clustering import load_organize_node_data_for_umaps, plot_umaps_groups, cluster_node_data
df = load_organize_node_data_for_umaps(model, X)

# Figure 4A: UMAPs projections (per condition - clustered) Hopf model
it = 2
embedding = plot_umaps_groups(df, model, it, patlist)
# Figure 4B: UMAPs projections (per condition - clustered) AHP model
labels, embedding = cluster_node_data(embedding, it)
# Figure 4C: Clusters compositions % each condition per cluster
# Figure 4D-XX: Clusters vs outcome / etiology / sex / age / evolution
# Figure 4XX: Overlap between two models (same patients in same clusters?)/Compare with direct k-means without UMAPS? 
