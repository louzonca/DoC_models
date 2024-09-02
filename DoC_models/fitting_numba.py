import numpy as np
#from numba import njit, prange
import os
import pickle
import time
import scipy.io as sio

from BOLDHemModel_Stephan2008 import BOLDModel

from models_numba import hopf_model, ahp_model, integrate
from preprocessing_methods import compute_sef, ssim

import matplotlib.pyplot as plt

# ----- Parameter space exploration ----------------------------

#@njit(parallel=True)
def fit_parameters(model, numIter, state_eq, state_range, bif_or_ahp, bif_range, 
                   t_constants, sig, sig_range, FC,
                   data_for_fit, TR, dtsim, Nsim, pat_id):
    # numIter: number of iterations for parameter space exploration
    # state_eq: initial resting state values, fixed for Hopf model / 
    #           state_eq[1]=Xeq and state_eq[2]=Yeq *to fit* in AHP model
    # state_range: authorized range to explore
    # bif_or_ahp: bifurcation parameter a for Hopf *to fit*/ 
    #             bif_or_ahp[0]=Th, [1]=tmAHP, [2]=tsAHP *to fit*
    # bif_range: authorize range for bif_or_ahp
    # t_constants: extracted from the data for Hopf and t_constants[0]=tau for AHP
    #              t_constants[1]=tf and t_constants[2]=tr fixed
    # sig: noise amplitude *to fit* for both models
    # sig_range: authorized range to explore
    # FC: connectivity matrix, fixed during node parmeter fitting (i.e. this function)
    # data-for-fit: time series of the data of the patient
    # TR: acquisition timestep for the data 
    # dtsim: timestep of simulations
    # Nsim: number of simulations per iteration step (i.e. with fixed parameters)
    # pat_id: id number of the patient

    # Extract info from data
    dim_data, dur_data = data_for_fit.shape
    FC_emp = np.corrcoef(data_for_fit, rowvar=True)

    duration = dur_data*TR # simulations duration (s)

    # Initialize
    ssim_evolution = np.zeros(numIter,)
    best_ssim = 1e-8
    best_ssim_evol = np.zeros(numIter,)

    curr_state_eq = state_eq*1.0
    curr_ahp = bif_or_ahp*1.0
    curr_sig = sig*1.0
    fitted_state_eq = curr_state_eq*1.0
    fitted_ahp = curr_ahp*1.0
    fitted_sig = curr_sig*1.0

    for iter in range(numIter):
        # Run the simulations
        bold_dur = np.int32(duration/dtsim - 20/dtsim)
        bold = np.zeros((dim_data, bold_dur, Nsim))
        # print(f'shape bold {bold.shape}')
        for i in range(Nsim):
            #print(f'sim # {i}, with {model.__name__}')
            HH, XX, YY, tt = integrate(model, duration, dtsim, 
                                       curr_state_eq, curr_ahp, t_constants, curr_sig, FC_emp)
            #print(f'sim {i} is done, now lets compute the BOLD')
            # Compute BOLD response
            if model == hopf_model:
                HH = XX
            for d in range(dim_data):
                bold[d,:,i] = BOLDModel(duration, (HH[d,:]-np.mean(HH))/(np.amax(HH)-np.amin(HH)), dtsim)

            # Average over simulations
            output_sim = np.mean(bold, axis=2)
            
            # Plot for verification
            #fig, ax = plt.subplots(2, 1)
            #ax[0].plot(output_sim.T)
            #ax[1].plot(data_for_fit.T)
            #ax[0].set_xlabel(f"Simulated  mean BOLD signal")
            #ax[1].set_xlabel(f"Data")
            #ax[0].set_title(f"Patient #{pat_id}, iteration {i}")
            #plt.show()

        # Compare to the data
        FC_sim = np.corrcoef(output_sim, rowvar=True)
        
        # Plot FC for verif
        #fig, ax = plt.subplots(1,2)
        #ax[0].imshow(FC_sim - np.diag(np.diag(FC_sim)), extent=[0, 1, 0, 1])
        #ax[1].imshow(FC_emp - np.diag(np.diag(FC_emp)), extent=[0, 1, 0, 1])
        #ax[0].set_title("Simulated FC")
        #ax[1].set_title("Empirical FC")

        ssim_emp_sim = ssim(FC_emp, FC_sim)
        print(f'Current SSIM {ssim_emp_sim}, best {best_ssim}')
        ssim_evolution[iter] = ssim_emp_sim
        #plt.suptitle(f'SSIM {ssim_emp_sim}')
        #plt.show()
        
        # Test and keep best parameter set
        if ssim_emp_sim >= best_ssim:
            best_ssim = ssim_emp_sim
            best_ssim_evol[iter:-1] = best_ssim
            fitted_state_eq = curr_state_eq
            fitted_ahp = curr_ahp
            fitted_sig = curr_sig
        
        # Pick new parameter values
        curr_state_eq = (state_range[1,:]-state_range[0,:])*np.random.random_sample()+state_range[0,:]
        curr_ahp = (bif_range[1,:]-bif_range[0,:])*np.random.random_sample()+bif_range[0,:]
        curr_sig = (sig_range[1]-sig_range[0])*np.random.random_sample()+sig_range[0]

    return fitted_state_eq, fitted_ahp, fitted_sig, ssim_evolution, best_ssim_evol

#@njit(parallel=True)
def process_patient(pat_id, data_for_fit, model):
    
    TR = 2
    sef = compute_sef(data_for_fit, 1/TR, 0.5)
    print(f'SEF = {sef}')
    t_constants = sef
    
    # Fixed parameters
    numIter = 10
    Nsim = 5

    TR = 2
    dtsim = 0.05

    # Initial parameter values and authorized range

    if model.__name__ == 'ahp_model':
        state_eq = np.array([0., 0.088, 1.])
        ahp_par = np.array([-30., 1/0.15, 1/5.])
        
        state_range = np.array([[0, 0.05, 0.5],[0, 0.15, 1]])
        ahp_range = np.array([[-50, 1/0.5, 1/10.],[-10, 1/0.05, 1/2.5]])

        sig = 5 
        sig_range = np.array([0.5, 7.5]) 
    
    elif model.__name__ == 'hopf_model':
        state_eq = np.zeros(15,)
        ahp_par = -0.02*np.array(15,)
        
        state_range = np.array([np.zeros(latent_dim,), np.zeros(latent_dim,)])
        ahp_range = np.array([-0.5*np.ones(latent_dim,), 0.5*np.ones(latent_dim,)])
        
        sig = 5
        sig_range = 1*np.array([0.5, 7.5]) 

    FC_emp = np.corrcoef(data_for_fit, rowvar=True)
    
    # Run the fitting
    fitted_state_eq, fitted_ahp, fitted_sig, ssim_evolution, best_ssim_evol = \
        fit_parameters(model, numIter, state_eq, state_range, ahp_par, ahp_range, \
                   t_constants, sig, sig_range, FC_emp, data_for_fit, TR, dtsim, Nsim, pat_id)

    return fitted_state_eq, fitted_ahp, fitted_sig, ssim_evolution, best_ssim_evol, pat_id


## ---------------------- TEST ----------------------------
if __name__ == "__main__":
## --------------------------------------------------------

    pat_id = 10
     
    latent_dim = 15
    initParcellation = 100

    model = hopf_model
    
    # Load encoded data
    patients_data_encoded = sio.loadmat(f'../results_d_opt_{initParcellation}/encoded_patients_{latent_dim}_dragana.mat') 
    data_for_fit = patients_data_encoded[f'encoded_patients'][pat_id, :, :]
    
    # Rune the fitting
    t0 = time.time()

    fitted_state_eq, fitted_ahp, fitted_sig, ssim_evolution, best_ssim_evol, pat_id = \
        process_patient(pat_id, data_for_fit, model)
    
    print(f'Elapsed time {time.time()-t0}')
