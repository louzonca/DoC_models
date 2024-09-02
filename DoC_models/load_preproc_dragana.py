import numpy as np
import scipy.io as sio
import pandas as pd
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

    print(f'Parcellation: {parcel_num}, subcorticals {subcorticals}')
    
    # load metadata
    metadata = pd.read_excel('fMRI_DoC_patients_forPD.xlsx')
    duration = 176
    data_dir = "../../../00_Data/DoCParisDragana/Schaefer2018_mat/"
    idx = 0
    nPats = metadata['NEW_CODE'].size
    patients = np.zeros((nPats,parcel_num,duration))
    for pat in metadata['NEW_CODE']:
        patient = sio.loadmat(data_dir + f'{pat}_atlas-Schaefer2018_100_roi-time-series.mat')
        patient = patient['roi_ts'].T
        patients[idx] = patient
        idx+=1
    labels = metadata['state']
    ids = metadata['NEW_CODE']
    return patients, labels, ids
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

def preprocess_data(datasets, freq_band):
    
    # Normalize
    all_data_normalized = normalize_data(datasets)

    # Bandpass filter
    #freq_band = [0.01, 0.1]#[0.04, 0.07]#
    b, a = signal.butter(6, freq_band, btype='band') 
    all_data_normalizedf = signal.lfilter(b, a, all_data_normalized)

    return all_data_normalizedf
    
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
