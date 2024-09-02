import numpy as np
import scipy.io as sio
import os


class PCAfMRI:

    def __init__(self,
                 data,
                 dimension):
        self.data = data  # Full dimensional data
        self.dimension = dimension  # Projection dimension

        self.dataset_vect = None
        self.cov_matrix = None
        self.sorted_eigenvalues = None
        self.sorted_eigenvectors = None
        self.cum_explained_var = None

        self._compute_pca()  # Method that computes the PCA

    # Public methods: pca_reduction, get_eigenvalues/eigenvectors, compute the percentage of expained variance
    def pca_reduction(self):
        proj_mat = np.matmul(self.sorted_eigenvectors.T, self.dataset_vect)
        reduced_data = proj_mat[0:self.dimension, :]
        # Reshape dataset to slpit all patients (num_patients, dimension, duration)
        sz_dat = self.data.shape
        num_patients = sz_dat[0]
        duration = sz_dat[2]
        reshaped_reduced_data = np.empty([num_patients, self.dimension, duration])
        for patient_idx in range(0, num_patients):
            reshaped_reduced_data[patient_idx, :, :] = reduced_data[:, patient_idx * duration:(patient_idx + 1) * duration]
        return reshaped_reduced_data

    # Save results evaluation for matlab
    def save_pca(self, save_dir):
        # Create folder if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Save the results in Matlab format
        sio.savemat(save_dir + "eigenvalues.mat",
                    mdict={'sorted_eigenvalues': self.sorted_eigenvalues})
        sio.savemat(save_dir + "eigenvalues.mat",
                    mdict={'sorted_eigenvalues': self.sorted_eigenvectors})
        sio.savemat(save_dir + "sorted_eigenvectors.mat",
                    mdict={f'cum_explained_var': self.cum_explained_var})
        sio.savemat(save_dir + "cov_mat.mat",
                    mdict={f'cov_mat': self.cov_mat})

    # Private methods: compute the covariance matrix, sort the eigenvalues and eigenvectors
    # Reshape input dataset: put each patient in a vector
    def _reshape_data_input(self):
        sz_dat = self.data.shape
        num_patients = sz_dat[0]
        num_roi = sz_dat[1]
        duration = sz_dat[2]
        dataset_vect = np.empty([num_roi, num_patients*duration])
        # Put all patients one after another
        for patient_idx in range(0, num_patients):
            dataset_vect[:, patient_idx*duration:(patient_idx+1)*duration] = self.data[patient_idx, :, :]
        return dataset_vect

    def _compute_pca(self):
        self.dataset_vect = self._reshape_data_input()
        # Covariance matrix
        self.cov_mat = np.cov(self.dataset_vect)
        # Eigenvalues and Eigenvectors
        eig_val, eig_vec = np.linalg.eigh(self.cov_mat)
        # Sort the eigenvalues & eigenvectors in descending order
        sorted_index = np.argsort(eig_val)[::-1]
        self.sorted_eigenvalues = eig_val[sorted_index]
        self.sorted_eigenvectors = eig_vec[:, sorted_index]

        explained_var = []
        for i in range(len(self.sorted_eigenvalues)):
            explained_var.append(self.sorted_eigenvalues[i] / np.sum(self.sorted_eigenvalues))
        # Cumulative explained variance
        self.cum_explained_var = [explained_var[0]]
        for i in range(1, len(self.sorted_eigenvalues)):
            self.cum_explained_var.append(self.cum_explained_var[i - 1] + explained_var[i])
