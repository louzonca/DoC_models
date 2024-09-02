import numpy as np
import scipy.io as sio
import os
import pickle

from tensorflow.keras import backend
from sklearn.model_selection import KFold

from classification_methods import build_training_test_sets_classif, train_classif, predict_class, eval_accuracy, eval_classif
from autoEncoderDense import AutoEncoderDense
from models_numba import hopf_model, ahp_model, integrate
from preprocessing_methods import normalize_data
from load_preproc import encode_patients, pca_reduction

def classify_all_dims(all_data, all_data_y, load_dir, num_slices, latdim_range, save_dir, k, pat_list):
    # k: for the k-fold
    print(f"Total dataset shape {all_data.shape}")
    print(f"Total dataset shape Y {all_data_y.shape}")

    # Classifier options
    train_proportion = 1 # the split is done with the kfold
    crit = 'FC' # criteria for classifier: FC (classification methods)
    kernel_type = 'poly'
    Cvalue = 0.1

    # K-fold validation datasets
    train_set_all, test_set, train_y_all, test_y = build_training_test_sets_classif(all_data, all_data_y, train_proportion, crit)
    kf = KFold(n_splits=k, shuffle=True, random_state=None)

    # Initiate
    fold_iteration = 1
    PRECISION_FULL_DIM = []
    RECALL_FULL_DIM = []
    ACCURACY_FULL_DIM = []

    PRECISION_AE = np.zeros((np.max(latdim_range), k))
    RECALL_AE = np.zeros((np.max(latdim_range), k))
    ACCURACY_AE = np.zeros((np.max(latdim_range), k))

    PRECISION_PCA = np.zeros((np.max(latdim_range), k))
    RECALL_PCA = np.zeros((np.max(latdim_range), k))
    ACCURACY_PCA = np.zeros((np.max(latdim_range), k))

    #for train_idx, test_idx in kf.split(train_set_all):
    for train_idx, test_idx in kf.split(pat_list):
        # >>>>>>>>>>>>>>>> REMPLACER PAR SPLIT MANUEL ONE VS ALL NO MIX BETWEEN PATIENTS IN TRAIN AND TEST <<<<<<<<<<<<<<<<
        train_set = train_set_all[train_idx, :]
        train_y = train_y_all[train_idx]
        test_set = train_set_all[test_idx, :]
        test_y = train_y_all[test_idx]
        print(f"training set size={train_set.shape}")
        print(f"test set size={test_set.shape}")

        # Train the full dimensional dataset classificator
        full_dim_classifier = train_classif(train_set, train_y, Cvalue, kernel_type)

        # Evaluate the results
        full_dim_pred = predict_class(test_set, full_dim_classifier)
        print(f"full dim predictions {full_dim_pred}")
        print(f"full dim real values {test_y}")
        # Predict results on test_set and evaluate accuracy
        PRECISION_FULL_DIM_k, RECALL_FULL_DIM_k, F1_k = eval_classif(full_dim_pred, test_y)
        ACCURACY_FULL_DIM_k = eval_accuracy(test_set, test_y, full_dim_classifier)
        PRECISION_FULL_DIM.append(PRECISION_FULL_DIM_k)
        RECALL_FULL_DIM.append(RECALL_FULL_DIM_k)
        ACCURACY_FULL_DIM.append(ACCURACY_FULL_DIM_k)

        # Classify and evaluate in the reduced dimension PCA vs AutoEncoder
        for lat_dim in latdim_range:
            print(f"Latent dimension = {lat_dim} fold iteration {fold_iteration}")
            # ------------------------- AutoEncoder ----------------------------------------
            # Load trained Encoder
            encoder_trained = AutoEncoderDense.load_encoder(load_dir, f"encoder_5_{lat_dim}")
            # Encode all data
            data_encoded = encode_patients(all_data, lat_dim, encoder_trained)
            data_encodedN = normalize_data(data_encoded)

            # Build classifier training and test sets in the latent space
            train_setAE_all, test_setAE, train_yAE_all, test_yAE = build_training_test_sets_classif(data_encodedN, all_data_y, train_proportion, crit)
            # Train the latent space classifier
            train_setAE = train_setAE_all[train_idx, :]
            train_yAE = train_yAE_all[train_idx]
            test_setAE = train_setAE_all[test_idx, :]
            test_yAE = train_yAE_all[test_idx]
            latent_space_classifier = train_classif(train_setAE, train_yAE, Cvalue, kernel_type)
            # Predict results on test_set and evaluate accuracy
            lat_dim_pred = predict_class(test_setAE, latent_space_classifier)
            precision, recall, f1 = eval_classif(lat_dim_pred, test_yAE)
            PRECISION_AE[lat_dim-1, fold_iteration-1] = precision
            RECALL_AE[lat_dim-1, fold_iteration-1] = recall
            ACCURACY_AE[lat_dim-1, fold_iteration-1] = eval_accuracy(test_setAE, test_yAE, latent_space_classifier)

            # ---------------------------- PCA ----------------------------------------
            # Reduce to lat_dim
            data_PCA = pca_reduction(all_data, lat_dim)
            # Train the classifier with PCA reduction
            train_setPCA_all, test_setPCA, train_yPCA_all, test_yPCA = build_training_test_sets_classif(data_PCA, all_data_y, train_proportion, crit)
            train_setPCA = train_setPCA_all[train_idx, :]
            train_yPCA = train_yPCA_all[train_idx]
            test_setPCA = train_setPCA_all[test_idx, :]
            test_yPCA = train_yPCA_all[test_idx]
            # print Train the latent space classifier
            PCA_classifier = train_classif(train_setPCA, train_yPCA, Cvalue, kernel_type)
            # Predict results on test_set and evaluate accuracy
            pca_dim_pred = predict_class(test_setPCA, PCA_classifier)
            precision, recall, f1 = eval_classif(pca_dim_pred, test_yPCA)
            PRECISION_PCA[lat_dim - 1, fold_iteration-1] = precision
            RECALL_PCA[lat_dim - 1, fold_iteration-1] = recall
            ACCURACY_PCA[lat_dim - 1, fold_iteration-1] = eval_accuracy(test_setPCA, test_yPCA, PCA_classifier)

        fold_iteration += 1
        print(f"fold iteration # {fold_iteration}  lat dim {lat_dim}")
        backend.clear_session()

    # Save all results
    print(f"precision ae = {PRECISION_AE}")
    print(f"precision pca = {PRECISION_PCA}")
    print(f"precision full = {PRECISION_FULL_DIM}")
    print(f"recall ae = {RECALL_AE}")
    print(f"recall pca = {RECALL_PCA}") 
    print(f"recall full = {RECALL_FULL_DIM}")

    # Create folder if it does not exist 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sio.savemat(save_dir + f"/precision_AE_{crit}.mat", mdict={f'PRECISION_AE': PRECISION_AE})
    sio.savemat(save_dir + f"/recall_AE_{crit}.mat", mdict={f'RECALL_AE': RECALL_AE})
    sio.savemat(save_dir + f"/accuracy_AE_{crit}.mat", mdict={f'ACCURACY_AE': ACCURACY_AE})
    sio.savemat(save_dir + f"/precision_PCA_{crit}.mat", mdict={f'PRECISION_PCA': PRECISION_PCA})
    sio.savemat(save_dir + f"/recall_PCA_{crit}.mat", mdict={f'RECALL_PCA': RECALL_PCA})
    sio.savemat(save_dir + f"/accuracy_PCA_{crit}.mat", mdict={f'ACCURACY_PCA': ACCURACY_PCA})
    sio.savemat(save_dir + f"/precision_full_{crit}.mat", mdict={f'PRECISION_FULL': PRECISION_FULL_DIM})
    sio.savemat(save_dir + f"/recall_full_{crit}.mat", mdict={f'RECALL_FULL': RECALL_FULL_DIM})
    sio.savemat(save_dir + f"/accuracy_full_{crit}.mat", mdict={f'ACCURACY_FULL': ACCURACY_FULL_DIM})


def classify_gec(gec, all_data_y, k, pat_list, model, maxIter):
    # pat_list: list of patients to include for the classifier
    # k: for the k-fold

    # Inititate
    kf = KFold(n_splits=k, shuffle=True, random_state=None)
    fold_iteration = 0

    kernel_type = 'linear' 
    Cvalue = 0.1
    fold_iteration = 0
    gam = 7.5
    degr = 6
    PRECISION_GEC = np.zeros(k,) 
    RECALL_GEC = np.zeros(k,)
    ACCURACY_GEC = np.zeros(k,)
    F1_GEC = np.zeros(k,)

    gec_all = dict()
    for p in pat_list:
        for it in range(maxIter,maxIter+1):
            #with open(os.path.join(gec_path_iter, f'GEC_{model.__name__}_pat_{p}_{it}'), 'rb') as f:
            gec_all[f'patient_{p}_{it}'] = gec[(f'pat_{p}',f'iter_{it}')]

    for train_idx, test_idx in kf.split(pat_list):
        
        train_setGEC_k = [gec_all[f'patient_{i}_{it}'].flatten() for i in train_idx for it in range(maxIter,maxIter+1)]
        train_yGEC_k = [np.int32(lab) for lab in all_data_y[train_idx] for rep in range(maxIter,maxIter+1)]
        #train_yGEC_k = all_data_y[train_idx]
        test_setGEC_k = [gec_all[f'patient_{i}_{it}'].flatten() for i in test_idx for it in range(maxIter,maxIter+1)]
        test_yGEC_k = [np.int32(lab) for lab in all_data_y[test_idx] for rep in range(maxIter,maxIter+1)]
        print(f'train set {len(train_setGEC_k)}')
        print(f'test set {len(test_setGEC_k)}')
        #test_yGEC_k = all_data_y[test_idx]

        latent_space_GEC_classifier = train_classif(train_setGEC_k, train_yGEC_k, Cvalue, kernel_type, gam, degr)
        # Predict results on test_set and evaluate accuracy
        gec_pred = predict_class(test_setGEC_k, latent_space_GEC_classifier)
        print(f'Predictions fold {fold_iteration} {gec_pred}')
        print(f'Real values {fold_iteration} {test_yGEC_k}')
        precision, recall, f1 = eval_classif(gec_pred, test_yGEC_k)
        PRECISION_GEC[fold_iteration] = precision
        RECALL_GEC[fold_iteration] = recall
        F1_GEC[fold_iteration] = f1
        #ACCURACY_GEC[fold_iteration] = eval_accuracy(test_setGEC_k, test_yGEC_k, latent_space_GEC_classifier)
        fold_iteration+=1
    print(f'precision = {np.mean(PRECISION_GEC)} +- {np.std(PRECISION_GEC)} - C = {Cvalue}')
    print(f'recall = {np.mean(RECALL_GEC)} +- {np.std(RECALL_GEC)} - C = {Cvalue}')
    print(f'F1 = {2*np.mean(PRECISION_GEC)*np.mean(RECALL_GEC)/(np.mean(PRECISION_GEC)+np.mean(RECALL_GEC))} +- {2*np.std(PRECISION_GEC)*np.std(RECALL_GEC)/(np.std(PRECISION_GEC)+np.std(RECALL_GEC))} - computed')
    print(f'F1 = {np.mean(F1_GEC)} +- {np.std(F1_GEC)} - from sklearn')
    return PRECISION_GEC, RECALL_GEC , F1_GEC

