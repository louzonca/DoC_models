import scipy.io as sio
from scipy import signal
import matplotlib.pyplot as plt
from autoEncoderDense import AutoEncoderDense
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras import backend

## ------------------------------ *Define training functions* ------------------------------

def build_training_test_sets(dataset, chunk_siz=60, train_prop=0.9):
    dataset = dataset.reshape(dataset.shape + (1,))
    sz_dat = dataset.shape
    print(sz_dat)
    tot_patients = sz_dat[0]
    num_chunks = int(np.floor(sz_dat[2]/chunk_siz))
    num_train_patients = int(np.floor(tot_patients*train_prop))
    x_train = np.empty([num_train_patients*num_chunks, sz_dat[1], chunk_siz, 1])
    x_test = np.empty([(tot_patients-num_train_patients)*num_chunks, sz_dat[1], chunk_siz, 1])
    for chunk_idx in range(0, num_chunks-1):
        # Take num_train_patients patients randomly in time chunk chunk_idx and add them to the training
        patients_to_add = np.random.choice(tot_patients, size=num_train_patients, replace=False)
        other_patients = np.delete(range(0, tot_patients), patients_to_add, axis=0)
        x_train[chunk_idx*num_train_patients:(chunk_idx+1)*num_train_patients, :, :] = \
            dataset[patients_to_add, :, chunk_idx*chunk_siz:(chunk_idx+1)*chunk_siz, :]
        x_test[chunk_idx*(tot_patients-num_train_patients):(chunk_idx+1)*(tot_patients-num_train_patients), :, :] = \
            dataset[other_patients, :, chunk_idx*chunk_siz:(chunk_idx+1)*chunk_siz, :]
    return x_train, x_test


def train(x_train, x_valid, learning_rate, batch_size, epochs, parcels, latent_dim=10, chunk_siz=60):
    autoenc = AutoEncoderDense(
        input_size=[parcels, chunk_siz, 1],
        layers_dim=[int(parcels), int(0.6*parcels), int(0.4*parcels)],#[int(0.75*parcels), int(0.5*parcels), int(0.25*parcels)],
        latent_space_dim=latent_dim
    )
    autoenc.summary()
    autoenc.train(x_train, x_valid, learning_rate, batch_size, epochs)
    return autoenc


# Define constant parameter values for training
LEARNING_RATE = 0.0001
BATCH_SIZE = 20
EPOCHS = 10

## ------------------ *Train AE for each latent dimension* -----------------
def run_ae_training(k, save_dir, dataset, latdim_range=range(2,20)):
    sz_dat = dataset.shape

    #parcels = sz_dat[1]
    parcel_num = sz_dat[1]

    mean_val_mse = []
    mean_val_acc = []
    std_val_mse = []
    std_val_acc = []

    for lat_dim in latdim_range:
        VALIDATION_ACC = []
        VALIDATION_MSE = []

        # K-fold validation datasets
        kf = KFold(n_splits=k, shuffle=True, random_state=None)
        fold_iteration = 1
        for train_idx, test_idx in kf.split(dataset):
            #print(f"Train idx = {train_idx} test idx = {test_idx}")
            x_train = dataset[train_idx, :, :, :]
            x_valid = dataset[test_idx, :, :, :]
            print(f"training set size={x_train.shape}")
            print(f"test set size={x_valid.shape}")

            # Training step # fold_iteration/k

            autoencoder = train(x_train, x_valid, LEARNING_RATE, BATCH_SIZE, EPOCHS, parcel_num, lat_dim, 1)
            autoencoder.save(save_dir, f"model_{fold_iteration}_{lat_dim}")
            autoencoder.save_encoder(save_dir, f"encoder_{fold_iteration}_{lat_dim}")
            autoencoder.save_decoder(save_dir, f"decoder_{fold_iteration}_{lat_dim}")

            autoencoder_trained = AutoEncoderDense.load(save_dir, f"model_{fold_iteration}_{lat_dim}")
            autoencoder_trained.summary()
            weights = autoencoder_trained.get_weights()

            encoder_trained = AutoEncoderDense.load_encoder(save_dir, f"encoder_{fold_iteration}_{lat_dim}")
            encoder_trained.summary()
            encoder_weights = encoder_trained.get_weights()

            # Evaluate the performance of the model - VALIDATION SET
            x_predict = autoencoder_trained.predict(x_valid)
            results = autoencoder_trained.evaluate(x_predict, x_valid)

            print(f"MSE results: {results[1]}")
            print(f"accuracy results: {results[2]}")
            print(f"metrics names {autoencoder_trained.metrics_names}")

            VALIDATION_MSE.append(results[1])
            VALIDATION_ACC.append(results[2])

            backend.clear_session()

            fold_iteration += 1
            print(f"fold iteration # {fold_iteration}  lat dim {lat_dim}")
            print(f"MSE (valid) {VALIDATION_MSE}, lat dim {lat_dim}")

        mean_val_mse.append(np.mean(VALIDATION_MSE))
        mean_val_acc.append(np.mean(VALIDATION_ACC))
        std_val_mse.append(np.std(VALIDATION_MSE))
        std_val_acc.append(np.std(VALIDATION_ACC))

    # Save results evaluation for matlab
    sio.savemat(save_dir + f"/validation_mse_{lat_dim}_{parcel_num}parcellation.mat", mdict={f'validation_MSE_{parcel_num}': mean_val_mse})
    sio.savemat(save_dir + f"/validation_acc_{lat_dim}_{parcel_num}parcellation.mat", mdict={f'validation_acc_{parcel_num}': mean_val_acc})
    sio.savemat(save_dir + f"/std_mse_{lat_dim}_{parcel_num}parcellation.mat", mdict={f'std_MSE_{parcel_num}': std_val_mse})
    sio.savemat(save_dir + f"/std_acc_{lat_dim}_{parcel_num}parcellation.mat", mdict={f'std_acc_{parcel_num}': std_val_acc})

## ------------------------ PLOTS -- TO CHECK --------------------------------
if __name__ == "__main__":

    ## Plot the MSE
    colors_data_points = plt.cm.gist_rainbow(np.linspace(0, 1, 12))
    fig = plt.figure()
    col = 0
    labs = ['100, no subcorticals', '100, with subcorticals', '1000, no subcorticals']
    for prc in [[100, 0]]: #, [100, 1], [1000, 0]]:
        parcel_num = prc[0] 
        subcorticals = prc [1]
        save_dir = f"trained_autoencoders/{k}_fold_validation_{parcel_num}_{subcorticals}_parcellation_ParisDragana_2024_05_07"
        mean_val_mse = sio.loadmat(save_dir + f"/validation_mse_25_{parcel_num}parcellation.mat")
        mean_val_mse = mean_val_mse[f'validation_MSE_{parcel_num}'].T
        std_val_mse = sio.loadmat(save_dir + f"/std_mse_25_{parcel_num}parcellation.mat")
        std_val_mse = std_val_mse[f'std_MSE_{parcel_num}'].T

        plt.plot(latdim_range, mean_val_mse, color=colors_data_points[col], label=labs[col])
        plt.xlabel("Latent dimension")
        plt.xticks(ticks=latdim_range, labels=latdim_range)
        plt.ylabel(f"Mean squared error")
        plt.title(f"{k}-fold validation - parcellation {parcel_num}")
        plt.fill_between(latdim_range, np.squeeze(mean_val_mse-std_val_mse/np.sqrt(k)), np.squeeze(mean_val_mse+std_val_mse/np.sqrt(k)), alpha=0.3, color=colors_data_points[col], ec=None)
        col+=1
    plt.legend()
    #plt.show() 
    #plt.savefig(f"figures/MSE_all_parcelations_dragana_preprocessed.pdf", format="pdf", bbox_inches="tight")

    ## Plot the MSE per group ---------------------------------------------------------
    parcel_num = 100
    subcorticals =0
    proportion = 1
    time_wndw = 1

    latdim_range = range(2, 26, 1)

    k = 10 # for K-fold validation
    save_dir = f"trained_autoencoders/{k}_fold_validation_{parcel_num}_{subcorticals}_parcellation_ParisDragana_2024_05_07_withPreprocessing"

    vs_list = [i for i, x in enumerate(all_labels) if x == 'VS']
    mcsm_list = [i for i, x in enumerate(all_labels) if x == 'MCS-']
    mcsp_list = [i for i, x in enumerate(all_labels) if x == 'MCS+']
    lists = {'VS': vs_list, 'MCS-': mcsm_list, 'MCS+': mcsp_list}
    # Compute for Draganas data            ---
    for gp in ['VS', 'MCS-', 'MCS+']:

        dataset, x_test = build_training_test_sets(all_data[lists[gp],:,20:-1],time_wndw, proportion)
        sz_dat = dataset.shape
        sz_test = x_test.shape
        print(f"dataset shape {sz_dat}") 
        print(f"test set shape {sz_test}")

        mean_val_mse = []
        mean_val_acc = []
        std_val_mse = []
        std_val_acc = []

        for lat_dim in latdim_range:
            VALIDATION_ACC = []
            VALIDATION_MSE = []

            # K-fold validation datasets
            kf = KFold(n_splits=k, shuffle=True, random_state=None)
            fold_iteration = 1
            for train_idx, test_idx in kf.split(dataset):
                x_train = dataset[train_idx, :, :, :]
                x_valid = dataset[test_idx, :, :, :]
                print(f"training set size={x_train.shape}")
                print(f"test set size={x_valid.shape}")

                autoencoder_trained = AutoEncoderDense.load(save_dir, f"model_{fold_iteration}_{lat_dim}")
                autoencoder_trained.summary()
                weights = autoencoder_trained.get_weights()

                encoder_trained = AutoEncoderDense.load_encoder(save_dir, f"encoder_{fold_iteration}_{lat_dim}")
                encoder_trained.summary()
                encoder_weights = encoder_trained.get_weights()

                # Evaluate the performance of the model - VALIDATION SET
                x_predict = autoencoder_trained.predict(x_valid)
                results = autoencoder_trained.evaluate(x_predict, x_valid)

                print(f"MSE results gp {gp}: {results[1]}")
                print(f"accuracy results gp {gp}: {results[2]}")
                print(f"metrics names {autoencoder_trained.metrics_names}")

                VALIDATION_MSE.append(results[1])
                VALIDATION_ACC.append(results[2])

                backend.clear_session()

                fold_iteration += 1
                print(f"fold iteration # {fold_iteration}  lat dim {lat_dim}")
                print(f"MSE (valid) {VALIDATION_MSE}, lat dim {lat_dim}")

            mean_val_mse.append(np.mean(VALIDATION_MSE))
            mean_val_acc.append(np.mean(VALIDATION_ACC))
            std_val_mse.append(np.std(VALIDATION_MSE))
            std_val_acc.append(np.std(VALIDATION_ACC))

        sio.savemat(save_dir + f"/validation_mse_{lat_dim}_{parcel_num}parcellation_group_{gp}.mat", mdict={f'validation_MSE_{parcel_num}': mean_val_mse})
        sio.savemat(save_dir + f"/validation_acc_{lat_dim}_{parcel_num}parcellation_group_{gp}.mat", mdict={f'validation_acc_{parcel_num}': mean_val_acc})
        sio.savemat(save_dir + f"/std_mse_{lat_dim}_{parcel_num}parcellation_group_{gp}.mat", mdict={f'std_MSE_{parcel_num}': std_val_mse})
        sio.savemat(save_dir + f"/std_acc_{lat_dim}_{parcel_num}parcellation_group_{gp}.mat", mdict={f'std_acc_{parcel_num}': std_val_acc})

    ## compute for all datasets together
    cnt_list_paris = [i for i, x in enumerate(all_data_labels) if x == 1 and i < 28]
    mcs_list_paris = [i for i, x in enumerate(all_data_labels) if x == 2 and i < 28]
    uws_list_paris = [i for i, x in enumerate(all_data_labels) if x == 3 and i < 28]

    cnt_list_paris2 = [i-28 for i, x in enumerate(all_data_labels) if x == 1 and 28<= i < 28+32]
    mcs_list_paris2 = [i-28 for i, x in enumerate(all_data_labels) if x == 2 and 28<= i < 28+32]
    uws_list_paris2 = [i-28 for i, x in enumerate(all_data_labels) if x == 3 and 28<= i < 28+32]

    cnt_list_liege = [i-28-32 for i, x in enumerate(all_data_labels) if x == 1 and 28+32<= i]
    mcs_list_liege = [i-28-32 for i, x in enumerate(all_data_labels) if x == 2 and 28+32<= i]
    uws_list_liege = [i-28-32 for i, x in enumerate(all_data_labels) if x == 3 and 28+32<= i]
    for gp in [0, 1, 2, -1]:
        if gp == 0:
            gp_list_paris = cnt_list_paris
            gp_list_paris2 = cnt_list_paris2
            gp_list_liege = cnt_list_liege
        elif gp == 1:
            gp_list_paris = mcs_list_paris
            gp_list_paris2 = mcs_list_paris2
            gp_list_liege = mcs_list_liege
        elif gp == 2:
            gp_list_paris = uws_list_paris
            gp_list_paris2 = uws_list_paris2
            gp_list_liege = uws_list_liege
        
        if gp > 0:  
            dataset1, x_test1 = build_training_test_sets(all_data_normalizedf[gp_list_paris,:,20:-1],time_wndw, proportion)

        dataset2, x_test2 = build_training_test_sets(all_data_normalized2f[gp_list_paris2,:,20:-1],time_wndw, proportion)
        datasetL, x_testL = build_training_test_sets(all_data_normalizedLiegef[gp_list_liege,:,20:-1], time_wndw, proportion)
        #print(f"dataset 1 (Paris) shape {dataset1.shape}")
        print(f"dataset 2 (Paris) shape {dataset2.shape}")
        print(f"dataset 2 (Liège) shape {datasetL.shape}")
        if gp > 0:
            dataset = np.concatenate((dataset1, dataset2, datasetL))
            x_test = np.concatenate((x_test1, x_test2, x_testL))
        else:
            dataset = np.concatenate((dataset2, datasetL))
            x_test = np.concatenate((x_test2, x_testL))

        sz_dat = dataset.shape
        sz_test = x_test.shape
        print(f"dataset shape {sz_dat}") 
        print(f"test set shape {sz_test}")

        mean_val_mse = []
        mean_val_acc = []
        std_val_mse = []
        std_val_acc = []

        for lat_dim in latdim_range:
            VALIDATION_ACC = []
            VALIDATION_MSE = []

            # K-fold validation datasets
            kf = KFold(n_splits=k, shuffle=True, random_state=None)
            fold_iteration = 1
            for train_idx, test_idx in kf.split(dataset):
                x_train = dataset[train_idx, :, :, :]
                x_valid = dataset[test_idx, :, :, :]
                print(f"training set size={x_train.shape}")
                print(f"test set size={x_valid.shape}")

                autoencoder_trained = AutoEncoderDense.load(save_dir, f"model_{fold_iteration}_{lat_dim}")
                autoencoder_trained.summary()
                weights = autoencoder_trained.get_weights()

                encoder_trained = AutoEncoderDense.load_encoder(save_dir, f"encoder_{fold_iteration}_{lat_dim}")
                encoder_trained.summary()
                encoder_weights = encoder_trained.get_weights()

                # Evaluate the performance of the model - VALIDATION SET
                x_predict = autoencoder_trained.predict(x_valid)
                results = autoencoder_trained.evaluate(x_predict, x_valid)

                print(f"MSE results gp {gp}: {results[1]}")
                print(f"accuracy results gp {gp}: {results[2]}")
                print(f"metrics names {autoencoder_trained.metrics_names}")

                VALIDATION_MSE.append(results[1])
                VALIDATION_ACC.append(results[2])

                backend.clear_session()

                fold_iteration += 1
                print(f"fold iteration # {fold_iteration}  lat dim {lat_dim}")
                print(f"MSE (valid) {VALIDATION_MSE}, lat dim {lat_dim}")

            mean_val_mse.append(np.mean(VALIDATION_MSE))
            mean_val_acc.append(np.mean(VALIDATION_ACC))
            std_val_mse.append(np.std(VALIDATION_MSE))
            std_val_acc.append(np.std(VALIDATION_ACC))

        sio.savemat(save_dir + f"/validation_mse_{lat_dim}_{parcel_num}parcellation_group_{gp}.mat", mdict={f'validation_MSE_{parcel_num}': mean_val_mse})
        sio.savemat(save_dir + f"/validation_acc_{lat_dim}_{parcel_num}parcellation_group_{gp}.mat", mdict={f'validation_acc_{parcel_num}': mean_val_acc})
        sio.savemat(save_dir + f"/std_mse_{lat_dim}_{parcel_num}parcellation_group_{gp}.mat", mdict={f'std_MSE_{parcel_num}': std_val_mse})
        sio.savemat(save_dir + f"/std_acc_{lat_dim}_{parcel_num}parcellation_group_{gp}.mat", mdict={f'std_acc_{parcel_num}': std_val_acc})

    ## 
    fig = plt.figure()
    col = 0
    labs = ['VS', 'MCS-', 'MCS+']#['CNT', 'MCS', 'UWS']

    for gp in ['VS', 'MCS-', 'MCS+']:#[0, 1, 2]:
        save_dir = f"trained_autoencoders/{k}_fold_validation_{parcel_num}_{subcorticals}_parcellation_ParisDragana_2024_05_07_withPreprocessing"

        mean_val_mse = sio.loadmat(save_dir + f"/validation_mse_25_{parcel_num}parcellation_group_{gp}.mat")
        mean_val_mse = mean_val_mse[f'validation_MSE_{parcel_num}'].T
        std_val_mse = sio.loadmat(save_dir + f"/std_mse_25_{parcel_num}parcellation_group_{gp}.mat")
        std_val_mse = std_val_mse[f'std_MSE_{parcel_num}'].T

        plt.plot(latdim_range, mean_val_mse, color=colors_gp[col], label=labs[col])
        plt.xlabel("Latent dimension")
        plt.xticks(ticks=latdim_range, labels=latdim_range)
        plt.ylabel(f"Mean squared error")
        plt.fill_between(latdim_range, np.squeeze(mean_val_mse-std_val_mse/np.sqrt(k)), np.squeeze(mean_val_mse+std_val_mse/np.sqrt(k)), alpha=0.3, color=colors_gp[col], ec=None)
        col+=1
    plt.legend()
    #plt.show() 
    #fig.savefig(f"figures/MSE_all_parcelations_per_group_dragana_preprocessed.pdf", format="pdf", bbox_inches="tight")
