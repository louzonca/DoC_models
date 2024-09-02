import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d  
from scipy import signal
from autoEncoderDense import AutoEncoderDense
from PCA import PCAfMRI
# -------------- Load Function -------------------
def load_data(parcel_num,subcorticals):

    # UWS = Unresponsive Wakefulness Syndrome
    # VS = Vegetative State
    # MCS = Minimally Conscious State
    # CNT = Control

    if subcorticals == 1:
        suffix = '_with_subs.mat'
    else:
        suffix = '.mat'
    print(f'Parcellation: {parcel_num}, subcorticals {subcorticals}')
    # Load Paris
    # Schafer *parcel_num* parcelation - TR = 2,4s AND TR = 2s - PARIS -
    path = '../../../00_Data/Coma_Paris'
    cnt_data2_4 = sio.loadmat(path + f'/controls_{parcel_num}/tc_CNT_TR2_4' + suffix)
    cnt_data2_4 = cnt_data2_4[list(cnt_data2_4)[-1]]

    mcs_data2_4 = sio.loadmat(path + f'/MCS_{parcel_num}/tc_MCS_TR2_4' + suffix)
    mcs_data2_4 = mcs_data2_4[list(mcs_data2_4)[-1]]
    mcs_data2 = sio.loadmat(path + f'/MCS_{parcel_num}/tc_MCS_TR2' + suffix)
    mcs_data2 = mcs_data2[list(mcs_data2)[-1]] 

    uws_data2_4 = sio.loadmat(path + f'/UWS_{parcel_num}/tc_UWS_TR2_4' + suffix)
    uws_data2_4 = uws_data2_4[list(uws_data2_4)[-1]]
    uws_data2 = sio.loadmat(path + f'/UWS_{parcel_num}/tc_UWS_TR2' + suffix)
    uws_data2 = uws_data2[list(uws_data2)[-1]]

    all_data2_4 = np.concatenate((cnt_data2_4, mcs_data2_4, uws_data2_4))
    # Labels 1 = CNT, 2 = MCS, 3 = UWS
    all2_4_y = np.concatenate((np.ones(cnt_data2_4.shape[0]), 2*np.ones(mcs_data2_4.shape[0]), 3*np.ones(uws_data2_4.shape[0])))
    all_data2 = np.concatenate((mcs_data2, uws_data2))
    # Labels
    all2_y = np.concatenate((2*np.ones(mcs_data2.shape[0]), 3*np.ones(uws_data2.shape[0])))

    # Load Liege
    # Schafer *parcel_num* parcelation - TR = 2s - LIEGE - 
    path = '../../../00_Data/Coma'
    if parcel_num == 1000 and subcorticals == 0:
        cnt_dataLiege = sio.loadmat(path + f'/tc_coma_schaefer{parcel_num}MNI_gp1.mat')
        cnt_dataLiege = cnt_dataLiege[f'ts_all_schaeffer{parcel_num}_gp1']
        mcs_dataLiege = sio.loadmat(path + f'/tc_coma_schaefer{parcel_num}MNI_gp2.mat')
        mcs_dataLiege = mcs_dataLiege[f'ts_all_schaeffer{parcel_num}_gp2']
        uws_dataLiege = sio.loadmat(path + f'/tc_coma_schaefer{parcel_num}MNI_gp3.mat')
        uws_dataLiege = uws_dataLiege[f'ts_all_schaeffer{parcel_num}_gp3']
    elif parcel_num == 100 and subcorticals == 0:
        cnt_dataLiege = sio.loadmat(path + '/timeseries_all_CNT_liege100.mat')
        cnt_dataLiege = cnt_dataLiege['tc_all_cnt']
        mcs_dataLiege = sio.loadmat(path + '/timeseries_all_MCS_liege100.mat')
        mcs_dataLiege = mcs_dataLiege['tc_all_mcs']
        uws_dataLiege = sio.loadmat(path + '/timeseries_all_UWS_liege100.mat')
        uws_dataLiege = uws_dataLiege['tc_all_uws']
    elif subcorticals == 1:
        cnt_dataLiege = sio.loadmat(path + f'/timeseries_all_CNT_{parcel_num}_with_sub.mat')
        cnt_dataLiege = cnt_dataLiege[f'tc_all_cnt_{parcel_num}_with_sub']
        mcs_dataLiege = sio.loadmat(path + f'/timeseries_all_MCS_{parcel_num}_with_sub.mat')
        mcs_dataLiege = mcs_dataLiege[f'tc_all_mcs_{parcel_num}_with_sub']
        uws_dataLiege = sio.loadmat(path + f'/timeseries_all_UWS_{parcel_num}_with_sub.mat')
        uws_dataLiege = uws_dataLiege[f'tc_all_uws_{parcel_num}_with_sub']
    # Concatenate Liege
    all_data_Liege = np.concatenate((cnt_dataLiege, mcs_dataLiege, uws_dataLiege))
    allLiege_y = np.concatenate(
        (np.ones(cnt_dataLiege.shape[0]), 2 * np.ones(mcs_dataLiege.shape[0]), 3 * np.ones(uws_dataLiege.shape[0])))

    # dim 0 = num subjects / dim 1 = num ROIs / dim 2 = timepoints 
    sz_paris = all_data2_4.shape
    sz_paris2 = all_data2.shape
    sz_liege = all_data_Liege.shape
    print(f'Paris TR 2.4s size: {sz_paris} / Paris TR 2s size: {sz_paris2} / Liege TR 2s size: {sz_liege}')
    return all_data2_4, all_data2, all_data_Liege, all2_4_y, all2_y, allLiege_y

# ------------------ Preprocessing ------------------------
# Normalize and center function
def normalize_data(dataset_to_normalize):
    all_data_norm = dataset_to_normalize
    sz_dataset = dataset_to_normalize.shape
    for patient_num in range(0, sz_dataset[0]):
        patient = dataset_to_normalize[patient_num, :, :]
        norm_patient = patient
        # Get patient's min and max global (over all ROIs)
        max_patient = np.amax(norm_patient)
        min_patient = np.amin(norm_patient)
        if max_patient != 0 or min_patient != 0:
            try:
               for roi_num in range(0, sz_dataset[1]):
                    norm_patient[roi_num, :] = (norm_patient[roi_num, :] - np.mean(norm_patient[roi_num,:]))/(np.max(norm_patient[roi_num,:])-np.min(norm_patient[roi_num,:]))
            except:
                print(f"ROI # {roi_num} is zero for patient {patient_num}")
        all_data_norm[patient_num, :, :] = norm_patient
    return all_data_norm

# Interpolate timestep function
def interpolate_timestep(dataset_to_interpolate, time_vector, new_time_vector):
    interpolated = interp1d(time_vector, dataset_to_interpolate)
    interpolated_dataset = interpolated(new_time_vector)
    return interpolated_dataset

def preprocess_data(dict_datasets, dict_y, clues, freq_band):
    
    # Normalize all datasets
    all_data_normalized = normalize_data(dict_datasets[clues[0]])
    all_data_normalized2 = normalize_data(dict_datasets[clues[1]])
    all_data_normalizedLiege = normalize_data(dict_datasets[clues[2]])
    # Interpolate to TR = 2s the TR=2.4s dataset from Paris 
    dt = 2  # seconds
    time_vec2_4 = np.linspace(0, all_data_normalized.shape[2]*2.4, all_data_normalized.shape[2])
    time_vec2 = np.linspace(0, all_data_normalized.shape[2]*2.4,
                            int(np.floor(all_data_normalized.shape[2]*2.4/dt)))
    all_data_normalized_i = interpolate_timestep(all_data_normalized,  time_vec2_4, time_vec2) 

    # Bandpass filter all datasets
    #freq_band = [0.01, 0.1]#[0.04, 0.07]#
    b, a = signal.butter(6, freq_band, btype='band') 
    all_data_normalizedf = signal.lfilter(b, a, all_data_normalized_i)
    all_data_normalized2f = signal.lfilter(b, a, all_data_normalized2)
    all_data_normalizedLiegef = signal.lfilter(b, a, all_data_normalizedLiege)

    # Remove the recordings with NaN values
    aliveP = np.squeeze(np.argwhere(np.sum(np.isnan(dict_datasets[clues[0]]), axis=(1,2)) == 0))
    patients0 = np.squeeze(np.argwhere(np.sum(dict_datasets[clues[0]], axis=(1,2)) == 0.0 ))
    aliveP = np.setdiff1d(aliveP, patients0)
    all_data_normalized = all_data_normalized[aliveP]
    all_data_normalizedf = all_data_normalizedf[aliveP]
    all2_4_y = dict_y[clues[0]][aliveP]
    #print(f'Paris 2.4s patients OK {aliveP}')

    alive2 = np.squeeze(np.argwhere(np.sum(np.isnan(dict_datasets[clues[1]]), axis=(1,2)) == 0))
    patients0 = np.squeeze(np.argwhere(np.sum(dict_datasets[clues[1]], axis=(1,2)) == 0.0 ))
    alive2 = np.setdiff1d(alive2, patients0)
    all_data_normalized2 = all_data_normalized2[alive2]
    all_data_normalized2f = all_data_normalized2f[alive2]
    all2_y = dict_y[clues[1]][alive2]
    #print(f'Paris 2s patients OK {alive2}')

    alive = np.squeeze(np.argwhere(np.sum(np.isnan(dict_datasets[clues[2]]), axis=(1,2)) == 0))
    patients0 = np.squeeze(np.argwhere(np.sum(dict_datasets[clues[2]], axis=(1,2)) == 0.0 ))
    alive = np.setdiff1d(alive, patients0)
    all_data_normalizedLiege = all_data_normalizedLiege[alive]
    all_data_normalizedLiegef = all_data_normalizedLiegef[alive]
    allLiege_y = dict_y[clues[2]][alive]
    #print(f'Liege 2s patients OK {alive}')

    # Shift above zero (for the AE)
    all_data_normalizedf = all_data_normalizedf+np.abs(np.min(all_data_normalizedf))
    all_data_normalized2f = all_data_normalized2f+np.abs(np.min(all_data_normalized2f))
    all_data_normalizedLiegef = all_data_normalizedLiegef+np.abs(np.min(all_data_normalizedLiegef))

    return all_data_normalizedf, all_data_normalized2f, all_data_normalizedLiegef, all2_4_y, all2_y, allLiege_y
    
# ------------- Encode patients --------------------
# Encode the concatenated data (dim d_opt) & obtain encoded time series for each patient
def encode_patients(data, latent_dim, encoder):
    encoded_patients = np.empty([data.shape[0], latent_dim, data.shape[2]])
    for patient_idx in range(0, data.shape[0]):
        to_encode = data[patient_idx, :, :].transpose()
        to_encode = to_encode.reshape(to_encode.shape + (1,))
        encoded_patient = encoder.predict(to_encode)
        encoded_patients[patient_idx, :, :] = encoded_patient.transpose()
    return encoded_patients

# ------------- PCA ---------------------------------
def pca_reduction(dataset, dimension):
    pca = PCAfMRI(dataset, dimension)
    reduced_dim_pca_dataset = pca.pca_reduction()
    return reduced_dim_pca_dataset
