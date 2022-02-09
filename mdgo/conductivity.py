# coding: utf-8
# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""
This module implements functions to calculate the ionic conductivity.
"""
from typing import Union

import numpy as np
from tqdm.notebook import tqdm
from scipy import stats
from MDAnalysis import Universe, AtomGroup

__author__ = "Kara Fong, Tingzheng Hou"
__version__ = "1.0"
__maintainer__ = "Tingzheng Hou"
__email__ = "tingzheng_hou@berkeley.edu"
__date__ = "Feb 9, 2021"

"""
Algorithms in this section are adapted from DOI: 10.1051/sfn/201112010 and
http://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft#34222273
"""


def autocorr_fft(x: np.ndarray) -> np.ndarray:
    """Calculates the autocorrelation function using the fast Fourier transform.

    Args:
        x (numpy.array): function on which to compute autocorrelation function

    Returns a numpy.array of the autocorrelation function
    """
    N = len(x)
    F = np.fft.fft(x, n=2 * N)  # 2*N because of zero-padding
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res = (res[:N]).real
    n = N * np.ones(N) - np.arange(0, N)
    return res / n


def msd_fft(r: np.ndarray) -> np.ndarray:
    """Calculates mean square displacement of the array r using the fast Fourier transform.

    Args:
        r (numpy.array): atom positions over time

    Returns a numpy.array containing the mean-squared displacement over time
    """
    N = len(r)
    D = np.square(r).sum(axis=1)
    D = np.append(D, 0)
    S2 = sum([autocorr_fft(r[:, i]) for i in range(r.shape[1])])
    Q = 2 * D.sum()
    S1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m - 1] - D[N - m]
        S1[m] = Q / (N - m)
    return S1 - 2 * S2

def cross_corr(x, y):
    """
    Calculates cross-correlation function of x and y using the 
    fast Fourier transform.
    :param x: array[float], data set 1
    :param y: array[float], data set 2
    :return: cf: array[float], cross-correlation function
    """
    N=len(x)
    F1 = np.fft.fft(x, n=2**(N*2 - 1).bit_length())
    F2 = np.fft.fft(y, n=2**(N*2 - 1).bit_length())
    PSD = F1 * F2.conjugate()
    res = np.fft.ifft(PSD)
    res= (res[:N]).real   
    n=N*np.ones(N)-np.arange(0,N)
    cf = res/n
    return cf

def msd_fft_cross(r, k):
    """
    Calculates "MSD" (cross-correlations) using the fast Fourier transform.
    :param r: array[float], positions of atom type 1 over time
    :param k: array[float], positions of atom type 2 over time
    :return: msd: array[float], "MSD" over time
    """
    N=len(r)
    D=np.multiply(r,k).sum(axis=1) 
    D=np.append(D,0) 
    S2=sum([cross_corr(r[:, i], k[:,i]) for i in range(r.shape[1])])
    S3=sum([cross_corr(k[:, i], r[:,i]) for i in range(k.shape[1])])
    Q=2*D.sum()
    S1=np.zeros(N)
    for m in range(N):
        Q=Q-D[m-1]-D[N-m]
        S1[m]=Q/(N-m)
    msd = S1-S2-S3
    return msd


def calc_cond_msd(
    u: Universe,
    anions: AtomGroup,
    cations: AtomGroup,
    run_start: int,
    cation_charge: Union[int, float] = 1,
    anion_charge: Union[int, float] = -1,
) -> np.ndarray:
    """Calculates the conductivity "mean square displacement" over time

    Note:
       Coordinates must be unwrapped (in dcd file when creating MDAnalysis Universe)
       Ions selections may consist of only one atom per ion, or include all of the atoms
          in the ion. The ion AtomGroups may consist of multiple types of cations/anions.

    Args:
        u: MDAnalysis universe
        anions: MDAnalysis AtomGroup containing all anions
        cations: MDAnalysis AtomGroup containing all cations
        run_start (int): index of trajectory from which to start analysis
        cation_charge (int): net charge of cation
        anion_charge (int): net charge of anion

    Returns a numpy.array containing conductivity "MSD" over time
    """
    # convert AtomGroup into list of molecules
    cation_list = cations.split("residue")
    anion_list = anions.split("residue")
    # compute sum over all charges and positions
    qr = []
    for ts in u.trajectory[run_start:]:
        qr_temp = np.zeros(3)
        for anion in anion_list:
            qr_temp += anion.center_of_mass() * anion_charge
        for cation in cation_list:
            qr_temp += cation.center_of_mass() * cation_charge
        qr.append(qr_temp)
    msd = msd_fft(np.array(qr))
    return msd


def get_beta(
    msd: np.ndarray,
    time_array: np.ndarray,
    start: int,
    end: int,
) -> tuple:
    """Fits the MSD to the form t^(beta) and returns beta. beta = 1 corresponds
    to the diffusive regime.

    Args:
        msd (numpy.array): mean squared displacement
        time_array (numpy.array): times at which position data was collected in the simulation
        start (int): index at which to start fitting linear regime of the MSD
        end (int): index at which to end fitting linear regime of the MSD

    Returns beta (int) and the range of beta values within the region
    """
    msd_slope = np.gradient(np.log(msd[start:end]), np.log(time_array[start:end]))
    beta = np.mean(np.array(msd_slope))
    beta_range = np.max(msd_slope) - np.min(msd_slope)
    return beta, beta_range

def get_slope(f, start, end, times):
    """
    Perform a linear regression.
    :param f: array[float], "MSD" data
    :param start: int, time index at which to start fitting
    :param end: int, time index at which to end fitting
    :param times: array[float], times at which position data was collected in the simulation
    :return lij: float, transport coefficient, i.e., slope of "MSD" in fitting region
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(times[start:end], f[start:end])

    return slope


def choose_msd_fitting_region(
    msd: np.ndarray,
    time_array: np.ndarray,
) -> tuple:
    """Chooses the optimal fitting regime for a mean-squared displacement.
    The MSD should be of the form t^(beta), where beta = 1 corresponds
    to the diffusive regime; as a rule of thumb, the MSD should exhibit this
    linear behavior for at least a decade of time. Finds the region of the
    MSD with the beta value closest to 1.

    Note:
       If a beta value great than 0.9 cannot be found, returns a warning
       that the computed conductivity may not be reliable, and that longer
       simulations or more replicates are necessary.

    Args:
        msd (numpy.array): mean squared displacement
        time_array (numpy.array): times at which position data was collected in the simulation

    Returns at tuple with the start of the fitting regime (int), end of the
    fitting regime (int), and the beta value of the fitting regime (float).
    """
    beta_best = 0  # region with greatest linearity (beta = 1)
    # choose fitting regions to check
    for i in np.logspace(np.log10(2), np.log10(len(time_array) / 10), 10):  # try 10 regions
        start = int(i)
        end = int(i * 10)  # fit over one decade
        beta, beta_range = get_beta(msd, time_array, start, end)
        slope_tolerance = 2  # acceptable level of noise in beta values
        # check if beta in this region is better than regions tested so far
        if (np.abs(beta - 1) < np.abs(beta_best - 1) and beta_range < slope_tolerance) or beta_best == 0:
            beta_best = beta
            start_final = start
            end_final = end
    if beta_best < 0.9:
        print(f"WARNING: MSD is not sufficiently linear (beta = {beta_best}). Consider running simulations longer.")
    return start_final, end_final, beta_best


def conductivity_calculator(
    time_array: np.ndarray,
    cond_array: np.ndarray,
    v: Union[int, float],
    name: str,
    start: int,
    end: int,
    T: Union[int, float],
    units: str = "real",
) -> float:
    """Calculates the overall conductivity of the system

    Args:
        time_array (numpy.array): times at which position data was collected in the simulation
        cond_array (numpy.array): conductivity "mean squared displacement"
        v (float): simulation volume (Angstroms^3)
        name (str): system name
        start (int): index at which to start fitting linear regime of the MSD
        end (int): index at which to end fitting linear regime of the MSD
        units (str): unit system (currently 'real' and 'lj' are supported)

    Returns the overall ionic conductivity (float)
    """
    # Unit conversions
    if units == "real":
        A2cm = 1e-8  # Angstroms to cm
        ps2s = 1e-12  # picoseconds to seconds
        e2c = 1.60217662e-19  # elementary charge to Coulomb
        kb = 1.38064852e-23  # Boltzmann Constant, J/K
        convert = e2c * e2c / ps2s / A2cm * 1000
        cond_units = "mS/cm"
    elif units == "lj":
        kb = 1
        convert = 1
        cond_units = "q^2/(tau sigma epsilon)"
    else:
        raise ValueError("units selection not supported")

    slope, _, _, _, _ = stats.linregress(time_array[start:end], cond_array[start:end])
    cond = slope / 6 / kb / T / v * convert

    print("Conductivity of " + name + ": " + str(cond) + " " + cond_units)

    return cond

def calc_Lii_self(atom_positions, times):
    """ 
    Calculates the "MSD" for the self component for a diagonal transport coefficient (L^{ii}).
    :param atom_positions: array[float,float,float], position of each atom over time.
    Indices correspond to time, ion index, and spatial dimension (x,y,z), respectively.
    :param times: array[float], times at which position data was collected in the simulation
    :return msd: array[float], "MSD" corresponding to the L^{ii}_{self} transport 
    coefficient at each time
    """
    Lii_self = np.zeros(len(times))
    n_atoms = np.shape(atom_positions)[1]
    for atom_num in (range(n_atoms)):
        r = atom_positions[:,atom_num, :]
        msd_temp = msd_fft(np.array(r))
        Lii_self += msd_temp
    msd = np.array(Lii_self)
    return msd

def calc_Lii(atom_positions, times):
    """ 
    Calculates the "MSD" for the diagonal transport coefficient L^{ii}. 
    :param atom_positions: array[float,float,float], position of each atom over time.
    Indices correspond to time, ion index, and spatial dimension (x,y,z), respectively.
    :param times: array[float], times at which position data was collected in the simulation
    :return msd: array[float], "MSD" corresponding to the L^{ii} transport 
    coefficient at each time
    """
    r_sum = np.sum(atom_positions, axis = 1)
    msd = msd_fft(r_sum)
    return np.array(msd)

def calc_Lij(cation_positions, anion_positions, times):
    """
    Calculates the "MSD" for the off-diagonal transport coefficient L^{ij}, i \neq j.
    :param cation_positions, anion_positions: array[float,float,float], position of each 
    atom (anion or cation, respectively) over time. Indices correspond to time, ion index,
    and spatial dimension (x,y,z), respectively.
    :param times: array[float], times at which position data was collected in the simulation
    :return msd: array[float], "MSD" corresponding to the L^{ij} transport coefficient at 
    each time.
    """
    r_cat = np.sum(cation_positions, axis = 1)
    r_an = np.sum(anion_positions, axis = 1)
    msd = msd_fft_cross(np.array(r_cat),np.array(r_an))
    return np.array(msd)

def compute_all_Lij(cation_positions, anion_positions, times, volume, kbT):
    """
    Computes the "MSDs" for all transport coefficients.
    :param cation_positions, anion_positions: array[float,float,float], position of each 
    atom (anion or cation, respectively) over time. Indices correspond to time, ion index,
    and spatial dimension (x,y,z), respectively.
    :param times: array[float], times at which position data was collected in the simulation
    :param volume: float, volume of simulation box
    :return msds_all: list[array[float]], the "MSDs" corresponding to each transport coefficient,
    L^{++}, L^{++}_{self}, L^{--}, L^{--}_{self}, L^{+-}
    """
    volume = volume * 1e-8 #convert to A^2 cm units for conductivity
    msd_self_cation = calc_Lii_self(cation_positions, times)/6/kbT/volume #units of 1/(J-cm)
    msd_self_anion =  calc_Lii_self(anion_positions, times)/6/kbT/volume
    msd_cation = calc_Lii(cation_positions, times)/6/kbT/volume
    msd_anion = calc_Lii(anion_positions, times)/6/kbT/volume
    msd_distinct_catAn = calc_Lij(cation_positions, anion_positions, times)/6/kbT/volume
    msds_all = [msd_self_cation, msd_self_anion, msd_cation, msd_anion,  msd_distinct_catAn]
    return msds_all


