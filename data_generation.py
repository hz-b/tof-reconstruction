# #!/usr/bin/env python3
# Created by the authors of publication https://www.nature.com/articles/s41566-018-0107-6
# Adapted by David Meier on Apr 24 2024

import os
import h5py
import sys

import numpy as np
from tqdm import trange

from multiprocessing import Pool


def Job(joblist):
    """
    Calculates the spectrogram and the detector images depending on the experimental parameters
    For questions about the calculation refers to Gregor Hartmann

    Parameters
    ----------
    joblist : List
        contains the jobs parameter for the generation.

    Returns
    -------
    None


    """
    np.random.seed(joblist[7])

    N_batch = joblist[0]
    KICK_MIN = joblist[1]
    KICK_MAX = joblist[2]
    PEAKS = joblist[3]
    ELLIPT = joblist[4]
    ELL_TILT = joblist[5]
    PULSE = joblist[6]
    HOT_ENABLED = joblist[8]

    PHASE_STEPS = 80  # 1st dimension of Y
    ENERGY_STEPS = 60  # 2nd dimension of X and Y

    sigmax_max = 7
    sigmay_max = 7

    TILT = (np.pi,)
    ENERGY_GAUSS = 1  # further partial wave properties(tilt and width)

    # Used arrays for partial wave creation
    DANGLE = np.array([np.radians(22.5 * i) for i in range(16)])
    EA = np.array(range(ENERGY_STEPS))  # 2nd axis of X and Y
    KEG_REC = (EA.copy())  # here it is the same, but for code extensions this distinction should be kept
    PHASEG_REC = np.linspace(0, 2 * np.pi, PHASE_STEPS)  # 1st axis of Y in physical values
    ENERGY_GAUSS = 1

    def gauss(x, x0, xw):  # gaussian distribution
        return np.exp(-((x - x0) ** 2) / 2 / (xw / 2.35) ** 2)

    def ef(phase):  # ellipticity function
        return (ELLIPT) ** 2 / (
            (ELLIPT * np.cos(phase - ELL_TILT)) ** 2 + (np.sin(phase - ELL_TILT)) ** 2)

    def sine(ke, kick, phase):  # sinefunction*ellipticity
        return ke + kick * np.cos(DANGLE - phase) * ef(phase)

    def sim(ke, kick, phase):  # simulation of partial wave

        return (
            ANGDIST
            * np.array([gauss(EA, en, ENERGY_GAUSS) for en in sine(ke, kick, phase)]).T
        )

    def create_basis_reconstruction(kick):  # create a basis set for a fixed kick
        Lout = []
        for p in range(len(PHASEG_REC)):
            for k in range(len(KEG_REC)):
                Lout.append(sim(KEG_REC[k], kick, PHASEG_REC[p]))
        return np.array(Lout)

    def transform_YX(spec, basis):  # using a given basis for partial wave adding
        Lrec = np.zeros((ENERGY_STEPS, 16))
        counter = 0
        for p in range(PHASE_STEPS):
            for k in range(ENERGY_STEPS):
                Lrec += spec[k, p] * basis[counter]
                counter += 1
        return Lrec

    def add_gauss(Y, sigmax, sigmay, centerx, centery, intensity):
        Ynew = np.zeros((ENERGY_STEPS, PHASE_STEPS))
        Yadd = Ynew.copy()
        for x in range(PHASE_STEPS):
            for y in range(ENERGY_STEPS):
                Ynew[y, x] += (
                    intensity
                    * gauss(x, PHASE_STEPS // 2, sigmax)
                    * gauss(y, centery, sigmay)
                )
        Yadd = np.append(Ynew[:, centerx:PHASE_STEPS], Ynew[:, 0:centerx], axis=1)
        Y += Yadd
        return Y

    def create_training_data():
        Y = np.zeros((ENERGY_STEPS, PHASE_STEPS))
        kick = np.random.uniform(low=KICK_MIN, high=KICK_MAX, size=None)
        features = PEAKS
        for _ in range(features):
            centerx = int(np.random.rand() * PHASE_STEPS)
            centery = int((np.random.rand()) * (ENERGY_STEPS - kick * 2) + kick)
            sigmax = np.random.rand() * sigmax_max
            sigmay = np.random.rand() * sigmay_max
            intensity = np.random.rand()
            Y = add_gauss(Y, sigmax, sigmay, centerx, centery, intensity)
        if HOT_ENABLED:
            hot_ones=np.random.randint(0,high=15)
            for hotty in range(hot_ones):
                x=np.random.randint(0,high=PHASE_STEPS)
                y=np.random.randint(0,high=ENERGY_STEPS)
                Y[y,x]+=np.random.rand()
            
        
        basis_reconstruction = create_basis_reconstruction(kick)
        X = transform_YX(Y, basis_reconstruction)
        return [X, Y]

    x = []

    for i in trange(N_batch):
        BETA_2 = 2. # np.random.uniform(-1, 2, size=1)
        ANGDIST = 1 + BETA_2 / 2.0 * (0.5 - 0.5 * np.cos(2 * (DANGLE - TILT)) - 1)
        trainer = create_training_data()
        X = np.array(trainer[0])
        x.append(X.flatten())

    if __name__ == "__main__":
        fe = h5py.File(
            "./datasets/sigmaxy_7_peaks_0_20_hot_15/"
            + "N"
            + str(N_batch)
            + "_peaks"
            + str(PEAKS)
            + "_seed"
            + str(joblist[7])
            + ".h5",
            "w",
        )

        fe.create_dataset("x", data=np.array(x), compression="gzip")

        fe.close()
    else:
        return trainer

if __name__ == "__main__":
    # Amount of multithreading tasks/cpus
    Number_Workers = 100

    if not os.path.exists(train_export):
        os.makedirs(train_export)

    Ltodo = []
    # Amount of samples per file
    N = 100000
    files_per_peak = 5
    max_peaks = 20
    init_seed = 42 + int(sys.argv[1]) * files_per_peak * max_peaks
    hot_enabled = True

    # Fixed experimental parameters
    kick_min = 0
    kick_max = 100
    ellipt = 0.73
    elltilt = (90 - 22.5) / 180 * np.pi
    pulse = 30

    for file_nr in range(files_per_peak):
        for peak in range(1, max_peaks + 1):
            Ltodo.append(
                [
                    N,
                    kick_min,
                    kick_max,
                    peak,
                    ellipt,
                    elltilt,
                    pulse,
                    init_seed + file_nr * max_peaks + peak,
                    hot_enabled
                ]
            )


    with Pool(Number_Workers) as p:
        p.map(Job, Ltodo)
