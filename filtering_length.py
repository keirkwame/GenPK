import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
import astropy.units as u

def power_spectrum_model(k, A, n, filtering_length):
    return A * (k ** n) * np.exp(-1. * ((k * filtering_length) ** 2))

if __name__ == "__main__":
    power_spectrum_filename = '/Users/kwame/Simulations/emulator/HighRes512/snapdir_011/PK-by-PART_011-1024-norm'
    box_length_kpc_h = 10000. #* u.kpc
    hubble = 0.6724

    power_spectrum_file = np.loadtxt(power_spectrum_filename)
    k = power_spectrum_file[:, 0] * 2. * np.pi * hubble / box_length_kpc_h
    power_spectrum_dimensionless = ((power_spectrum_file[:, 0] * 1.) ** 3) * power_spectrum_file[:, 1]

    k_cut = 40. / 1000.
    parameter_bounds = (np.array([-np.inf, -np.inf, 0.]), np.inf)
    optimised_parameters, parameter_covariance = spo.curve_fit(power_spectrum_model, k[k<k_cut], power_spectrum_dimensionless[k<k_cut], bounds=parameter_bounds)
    print('A = %e ckpc; n = %e / ckpc; filtering length = %.2f ckpc'%tuple(optimised_parameters))
    print('Paramater covariance =', parameter_covariance)

    plt.figure()
    plt.plot(k * 1000., power_spectrum_dimensionless, label=r'Simulation')
    plt.plot(k[k<k_cut] * 1000., power_spectrum_model(k[k < k_cut], *optimised_parameters), ls='--', label=r'Best-fit model')
    plt.axvline(x = k_cut * 1000., ls=':', color='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel(r'k (1 / cMpc)')
    plt.ylabel(r'Dimensionless power')
    plt.title(r'Real-space flux')
    plt.show()
