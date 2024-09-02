import numpy as np
from numba import njit
import time
import matplotlib.pyplot as plt
from BOLDHemModel_Stephan2008 import BOLDModel
#from BalloonWindkessel import balloonWindkessel

@njit
def hopf_model(t, H, X, Y, state_eq, bif_param, t_constants, sig, FC, time, w_dt):
    """
    Hopf model, H is always 0, it is just here to make implementation 
    of integration function easier for both models at the same time...
    """
    dim = FC.shape[0]
    eqs = state_eq*1.0
    a = bif_param

    dH = np.zeros(dim,)
    dX = (a - (X - eqs)**2 - (Y - eqs)**2)*(X - eqs) - t_constants*(Y - eqs) \
            + (FC.T @ (X-eqs)) - np.sum(FC,axis=1).T * (X-eqs) + sig*w_dt.T
    dY = (a - (X - eqs)**2 - (Y - eqs)**2)*(Y - eqs) + t_constants*(X - eqs) \
            + (FC.T @ (Y-eqs)) - np.sum(FC,axis=1).T * (Y-eqs) + sig*w_dt.T
    #print(f'Doing hopf model, dh={dH}, dx={dX}, dy={dY}')
    return dH, dX, dY


@njit
def ahp_model(t, H, X, Y, state_eq, ahp_par, t_constants, sig, FC, time, w_dt):
    """
    AHP model
    """

    # Model dimension = number of nodes
    dim = FC.shape[0]

    # Model parameters
    Heq = state_eq[0] # resting state of H -- varies with state
    Xeq = state_eq[1]
    Yeq = state_eq[2]

    Th = ahp_par[0]
    tmAHP = ahp_par[1]
    tsAHP = ahp_par[2]

    tau = 1/t_constants # time constant of H -- varies with state 
    tf = 0.9 #s  #t_constants[1]  # facilitation time constant -- fixed
    tr = 2.9 #s  #t_constants[2]  # depression time constant -- fixed

    K = 0.037 # Hz
    L = 0.028 # Hz

    # Switching thresholds for the AHP dynamics
    Y_AHP = 0.85
    H_AHP = -7.5
    Y_h = 0.5

    # Inititate varying constants
    equilibrium = np.zeros(FC.shape[0])
    time_const = tau*np.ones(FC.shape[0])

    for d in range(dim):
        # case 1: fast dynamics, no hyperpolarization
        if ((1 - Y[d])/tr - L*X[d]*Y[d]*(H[d] - equilibrium[d]) <= 0) or (Y[d] >= Y_AHP and H[d] >= equilibrium[d] + H_AHP):
            equilibrium[d] = Heq
            time_const[d] = tau[d]
            #print(f'.', end=' ')
        # case 2: medium dynamics (t_mAHP) with hyperpolarization
        elif (Y[d] <= Y_h) and ((1 - Y[d])/tr - L*X[d]*Y[d]*(H[d] - (equilibrium[d])) > 0):
            equilibrium[d] = Heq + Th
            time_const[d] = tmAHP
            #print(f'+', end=' ')
        # case 3: slow recovery (t_sAHP) to no hyperpolarization
        elif (H[d] < Heq + H_AHP) or (Y_AHP > Y[d] > Y_h and ((1-Y[d])/tr-L*X[d]*(H[d]-equilibrium[d]) > 0)):
            equilibrium[d] = Heq
            time_const[d] = tsAHP
            #print(f'#' , end=' ')
        #else:
         #   raise Exception(f"uncontrolled state case for region {d} at time {t} with Y={Y[d]}, X={X[d]}, H={H[d]} and H_eq={equilibrium[d]} and trate = {time_const[d]}")

    hplus = np.maximum(H - equilibrium, 0)
    dH = (equilibrium - H + (FC @ hplus) * X * Y) * time_const + \
         (sig * w_dt.T * np.sqrt(time_const))
    dX = (Xeq - X) / tf + K * (1-X) * hplus  # np.multiply(np.ones(N) - X, hplus)
    dY = (Yeq - Y) / tr - L * X * Y * hplus  # np.multiply(np.multiply(hplus, X), Y)

    return dH, dX, dY

@njit
def integrate(model, duration, dt, state_eq, ahp_par, t_constants, sig, FC):
    """ 
    Integration function 
    """

    # Time sampling
    N = int(np.floor(duration / dt))
    time = np.zeros((N, 1))
    for i in range(0, N):
        time[i] = dt * (i - 1)

    # Initiate variables
    dim =  FC.shape[0]
    H = np.zeros((dim, N))
    X = np.zeros((dim, N))
    Y = np.zeros((dim, N))
    H[:, 0] = state_eq[0]*1.0
    X[:, 0] = state_eq[1]*1.0
    Y[:, 0] = state_eq[2]*1.0
    
    # Noise matrix (1/sqrt(dt) correction because of RK4)
    w_dot = np.random.randn(dim, 2*N) / np.sqrt(dt)
    
    # Integrate with RK4
    for i in range(N-1):
        # Compute noise
        w_dt = w_dot[:, 2*i]
        k1h, k1x, k1y = model(time[i], H[:,i], X[:,i], Y[:,i], \
                state_eq, ahp_par, t_constants, sig, FC, time, w_dt)
        w_dt = w_dot[:, 2*i+1]
        k2h, k2x, k2y = model(time[i]+dt/2, H[:,i]+dt*k1h/2, X[:,i]+dt*k1x/2, Y[:,i]+dt*k1y/2, \
                state_eq, ahp_par, t_constants, sig, FC, time, w_dt)
        w_dt = w_dot[:, 2*i+1]
        k3h, k3x, k3y = model(time[i]+dt/2, H[:,i]+dt*k2h/2, X[:,i]+dt*k2x/2, Y[:,i]+dt*k2y/2, \
                state_eq, ahp_par, t_constants, sig, FC, time, w_dt)
        w_dt = w_dot[:, 2*i+2]
        k4h, k4x, k4y = model(time[i]+dt, H[:,i]+dt*k3h, X[:,i]+dt*k3x, Y[:,i]+dt*k3y, \
                state_eq, ahp_par, t_constants, sig, FC, time, w_dt)

        H[:,i+1] = H[:,i] + dt/6*(k1h+2*k2h+2*k3h+k4h)
        X[:,i+1] = X[:,i] + dt/6*(k1x+2*k2x+2*k3x+k4x)
        Y[:,i+1] = Y[:,i] + dt/6*(k1y+2*k2y+2*k3y+k4y)
        #print(f'Hopf t {i}- H={H[:,i]}, X = {X[:,i]}, Y = {Y[:,i]}') 
    return H, X, Y, time
   

## ---------------------- TEST ------------------------
if __name__ == "__main__":
# ----------------------------------------------------
    dt = 0.05
    duration = 100

    FC = np.array([[1., 0., 0.], [0., 1., 0.], [-0., 0.0, 1.]])*5.

    ## --- AHP model ---
    state_eq = np.array([0., 0.08825, 1.])
    ahp_par = np.array([-30., 1/0.15, 1/5.])
    t_constants = np.array([0.05, 0.09, 0.04])

    sig = 5

    #np.random.seed(16)

    t0 = time.time()

    HH, XX, YY, tt = integrate(ahp_model, duration, dt, state_eq, ahp_par, t_constants, sig, FC)

    print(f'elapsed time: {time.time()-t0}')

    fig, axs = plt.subplots(nrows=4, ncols=1)
    axs[0].plot(tt, (HH[0,:]-np.mean(HH))/(np.amax(HH)-np.amin(HH)), color='red')
    axs[0].plot(tt, (HH[1,:]-np.mean(HH))/(np.amax(HH)-np.amin(HH)), color='purple')
    axs[0].plot(tt, (HH[2,:]-np.mean(HH))/(np.amax(HH)-np.amin(HH)), color='green')
    axs[1].plot(tt, XX[0, :], color='red')
    axs[1].plot(tt, XX[1, :], color='purple')
    axs[1].plot(tt, XX[2, :], color='green')
    axs[2].plot(tt, YY[0, :], color='red')
    axs[2].plot(tt, YY[1, :], color='purple')
    axs[2].plot(tt, YY[2, :], color='green')
    #plt.show()

    bold = BOLDModel(duration, (HH[0,:]-np.mean(HH))/(np.amax(HH)-np.amin(HH)), dt)
    axs[3].plot(bold, color='red')
    bold = BOLDModel(duration, (HH[1,:]-np.mean(HH))/(np.amax(HH)-np.amin(HH)), dt)
    axs[3].plot(bold, color='purple')
    bold = BOLDModel(duration, (HH[2,:]-np.mean(HH))/(np.amax(HH)-np.amin(HH)), dt)
    axs[3].plot(bold, color='green')
    plt.show()
    print(f'length bold signal {bold.shape}')

    ## --- Hopf model ---
    state_eq = np.zeros((FC.shape[0],))
    bif_param = 0.45651243 #np.array([-0.02, 0.1, 0.5])
    t_constants = np.array([0.4, 0.6, 0.7])

    sig = 5

    t0 = time.time()

    HH, XX, YY, tt = integrate(hopf_model, duration, dt, state_eq, bif_param, t_constants, sig, FC)

    print(f'elapsed time: {time.time()-t0}')

    fig, axs = plt.subplots(nrows=4, ncols=1)
    axs[0].plot(tt, (XX[0,:]-np.mean(XX))/(np.amax(XX)-np.amin(XX)), color='red')
    axs[0].plot(tt, (XX[1,:]-np.mean(XX))/(np.amax(XX)-np.amin(XX)), color='purple')
    axs[0].plot(tt, (XX[2,:]-np.mean(XX))/(np.amax(XX)-np.amin(XX)), color='green')
    axs[1].plot(tt, YY[0, :], color='red')
    axs[1].plot(tt, YY[1, :], color='purple')
    axs[1].plot(tt, YY[2, :], color='green')
    axs[2].plot(tt, HH[0, :], color='red')
    axs[2].plot(tt, HH[1, :], color='purple')
    axs[2].plot(tt, HH[2, :], color='green')
    bold = BOLDModel(duration, (XX[0,:]-np.mean(XX))/(np.amax(XX)-np.amin(XX)),dt)
    axs[3].plot(bold, color='red')
    bold = BOLDModel(duration, (XX[1,:]-np.mean(XX))/(np.amax(XX)-np.amin(XX)),dt)
    axs[3].plot(bold, color='purple')
    bold = BOLDModel(duration, (XX[2,:]-np.mean(XX))/(np.amax(XX)-np.amin(XX)),dt)
    axs[3].plot(bold, color='green')
    plt.show()
