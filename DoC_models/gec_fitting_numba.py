import numpy as np
import scipy.io as sio
import time

from models_numba import hopf_model, ahp_model, integrate
from BOLDHemModel_Stephan2008 import BOLDModel
from preprocessing_methods import ssim, compute_sef

import matplotlib.pyplot as plt

def gec_fit(encoded_patient, patient_id, model, fitted_state_eq, fitted_ahp, fitted_sig, Nmax=500, epsilon=0.01, tau=3):
    # Fitting of the GEC
    # encoded_patient: patient's data encoded in the latent space
    # patient id: anonymized id or number corresponding to patient (for the save)
    # model: ahp_model or hopf_model
    # fitted_state_eq: fitted node resting state values (fixed for hopf, 
    #                  [2]=X_eq and [3]=Y_eq fitted in AHP model)
    # fitted_ahp: AHP parameters or bifurcation parameter (for Hopf), from the fitting step
    # fitted_sig: fitted noise amplitude
    # Nmax: Max number of iterations to fit the GEC
    # espilon: learning rate
    TR = 2 # seconds
    lat_dim, dur_data = encoded_patient.shape
    duration = dur_data*TR # simulations duration in seconds
    sef = compute_sef(encoded_patient, 1/TR, 0.5)
    
    # Initiate
    ssimMax = 0.95 # if ssimMax is reached before Nmax iterations then stop
    iter_num = 0
    
    ssim_evolution = np.zeros(Nmax+1,)
    best_ssim = 1e-8
    ssim_fc_ec = 1e-8
    best_ssim_evol = np.zeros(Nmax+1,)

    Nsim = 5 # Number of simulations for each iteration
    dtsim = 0.05 # seconds

    # Empirical data
    FC_emp = np.corrcoef(encoded_patient, rowvar=True)
    FC_emp_tau = np.corrcoef(np.concatenate((encoded_patient[:,tau:],encoded_patient[:,0:-tau]), axis=0), rowvar=True)
    FC_emp_tau = FC_emp_tau[lat_dim:,:lat_dim]

    GEC = FC_emp # Inititate GEC with FC empirical
    bold_dur = np.int32(duration/dtsim - 20/dtsim)

    while ssim_fc_ec <= ssimMax and iter_num <= Nmax:
        sim_pat_bold = np.zeros((lat_dim, bold_dur, Nsim))
        for i in range(Nsim):
            HH, XX, _, tim = integrate(model, duration, dtsim, \
                                               fitted_state_eq, fitted_ahp, sef, fitted_sig, GEC)
            if model.__name__ == 'ahp_model':
                sim_pat_tmp = HH
            elif model.__name__ == 'hopf_model':
                sim_pat_tmp = XX
            for d in range(lat_dim):
                sim_pat_norm = (sim_pat_tmp[d,:]-np.mean(sim_pat_tmp))/(np.amax(sim_pat_tmp)-np.amin(sim_pat_tmp))
                sim_pat_bold[d,:,i] = BOLDModel(duration, sim_pat_norm, dtsim)
        sim_patient = np.mean(sim_pat_bold, axis=2)

        FC_sim = np.corrcoef(sim_patient, rowvar=True)
        FC_sim_tau = np.corrcoef(np.concatenate((sim_patient[:,tau:],sim_patient[:,0:-tau]), axis=0), rowvar=True)
        FC_sim_tau = FC_sim_tau[lat_dim:,:lat_dim]

        ssim_fc_ec = (ssim(FC_emp, FC_sim)+ssim(FC_emp_tau, FC_sim_tau))/2
        if np.isnan(ssim_fc_ec):
            ssim_fc_ec = -1
        ssim_evolution[iter_num] = ssim_fc_ec
        print(f'current ssim {ssim_fc_ec}, best {best_ssim}')
        if ssim_fc_ec >= best_ssim:
            GEC = GEC + epsilon*(np.absolute(FC_emp-FC_sim) + np.absolute(FC_emp_tau - FC_sim_tau))
            best_ssim = ssim_fc_ec
            best_ssim_evol[iter_num:-1] = best_ssim
        
        # Plot FC for verif
        #fig, ax = plt.subplots(3,2)
        #ax[0,0].imshow(FC_sim - np.diag(np.diag(FC_sim)), extent=[0, 1, 0, 1])
        #ax[0,1].imshow(FC_emp - np.diag(np.diag(FC_emp)), extent=[0, 1, 0, 1])
        #ax[1,0].imshow(FC_sim_tau - np.diag(np.diag(FC_sim_tau)), extent=[0, 1, 0, 1])
        #ax[1,1].imshow(FC_emp_tau - np.diag(np.diag(FC_emp_tau)), extent=[0, 1, 0, 1])
        #ax[2,0].imshow(GEC - np.diag(np.diag(GEC)), extent=[0, 1, 0, 1])
        #ax[0,0].set_title("Simulated FC")
        #ax[0,1].set_title("Empirical FC")
        #ax[2,0].set_title("GEC")
        #plt.show()
        
        iter_num+=1
    return GEC, best_ssim, best_ssim_evol, ssim_evolution, patient_id


## ---------------------- TEST ----------------------------
if __name__ == "__main__":
## --------------------------------------------------------

    pat_id = 10
     
    latent_dim = 15
    initParcellation = 100

    model = hopf_model

    # Load the fitted node parameters
    #fitted_state_eq = np.zeros(latent_dim,)
    #fitted_ahp = -0.02*np.ones(latent_dim,)
    #fitted_sig = 1

    # Load encoded data
    patients_data_encoded = sio.loadmat(f'../results_d_opt_{initParcellation}/encoded_patients_{latent_dim}_dragana.mat') 
    patient_data = patients_data_encoded[f'encoded_patients'][pat_id, :, :]
    
    # Rune the fitting
    t0 = time.time()

    Nmax = 25

    GEC, best_ssim, best_ssim_evol, ssim_evolution, pat_id = \
            gec_fit(patient_data, pat_id, model, fitted_state_eq, fitted_ahp, fitted_sig, Nmax)
    
    print(f'Elapsed time {time.time()-t0}')
