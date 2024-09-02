from scipy.interpolate import interp1d
import numpy as np
from scipy.signal import spectrogram, convolve2d
from scipy.integrate import cumtrapz
from numba import njit

# FUNCTION1 - normalize and center at 0
def normalize_data(dataset_to_normalize):
    all_data_norm = dataset_to_normalize
    sz_dataset = dataset_to_normalize.shape
    for patient_num in range(0, sz_dataset[0]):
        patient = dataset_to_normalize[patient_num, :, :]
        norm_patient = patient
        # Get patient's min and max global (over all ROIs)
        max_patient = np.amax(norm_patient)
        min_patient = np.amin(norm_patient)
        mean_patient = np.mean(norm_patient)
        std_patient = np.std(norm_patient)
        """
        if max_patient != 0 or min_patient != 0:
            for roi_num in range(0, sz_dataset[1]):
                norm_patient[roi_num, :] = (norm_patient[roi_num, :] - min_patient)/(max_patient - min_patient)
        """
        for roi_num in range(0, sz_dataset[1]):
            norm_patient[roi_num, :] = (norm_patient[roi_num, :]-mean_patient)/std_patient
        all_data_norm[patient_num, :, :] = norm_patient
    return all_data_norm


# FUNCTION2 - Interpolate the timestep
def interpolate_timestep(dataset_to_interpolate, time_vector, new_time_vector):
    interpolated = interp1d(time_vector, dataset_to_interpolate)
    interpolated_dataset = interpolated(new_time_vector)
    return interpolated_dataset


# FUNCTION3 - Extract a specific time segment of the data
def take_time_slice(dataset_total, time_slice_in, time_slice_out):
    dataset_slice = dataset_total[:, :, time_slice_in:time_slice_out]
    return dataset_slice


# FUNCTION4 - Get the Spectral Edge Frequency
def compute_sef(data, fs, percentage=0.95):
    dim, tim = data.shape
    sef = np.zeros(dim,)
    for d in range(0, dim):
        f, t, Sxx = spectrogram(data[d, :], fs, nperseg=12, noverlap=4, scaling='spectrum')
        # SEF95
        int_spectrum = cumtrapz(Sxx.transpose(), f, fs)
        sef_d = np.zeros(t.shape)
        for tIdx in range(0, len(t)):
            sef_d[tIdx] = f[np.where(int_spectrum[tIdx, :] >= percentage*(int_spectrum[tIdx, -1]))[0][0]]
        sef[d] = np.mean(sef_d)
    return sef

# FUNCTION5 - SSIM
#@njit
def ssim(im1, im2, wndw=8, sig=1.5, L=1.0):
    mat1 = np.float64(im1)
    mat2 = np.float64(im2)

    mu1 = convolve2d(mat1, np.ones((wndw, wndw)), mode='valid')/wndw**2
    mu2 = convolve2d(mat2, np.ones((wndw, wndw)), mode='valid')/wndw**2
    sig1_2 = convolve2d(mat1**2, np.ones((wndw, wndw)), mode='valid')/wndw**2 - mu1**2
    sig2_2 = convolve2d(mat2**2, np.ones((wndw, wndw)), mode='valid')/wndw**2 - mu2**2
    sig12 = convolve2d(mat1*mat2, np.ones((wndw, wndw)), mode='valid')/wndw**2 - mu1*mu2

    c1 = (L*0.01)**2
    c2 = (L*0.03)**2

    ssim = np.mean((2*mu1*mu2+c1)*(2*sig12+c2)/((mu1**2+mu2**2+c1)*(sig1_2+sig2_2+c2)))

    return ssim

