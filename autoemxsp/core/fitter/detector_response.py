#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detector Response Function module.

Provides handling of EDS detector response, including convolution matrices for energy resolution
and incomplete charge collection (ICC).

This module is used by both peaks and background models to accurately simulate detector effects.

Main Class:
    - DetectorResponseFunction: Manages detector response and convolution operations
"""

import os
import json
import time
import warnings
import numpy as np
from pathlib import Path
from scipy.special import erfc
from scipy.integrate import quad, trapezoid
from scipy.optimize import root_scalar

from autoemxsp.utils import print_single_separator, load_msa
import autoemxsp.utils.constants as cnst
import autoemxsp.calibrations as calibs
from autoemxsp.data.Xray_absorption_coeffs import xray_mass_absorption_coeff

parent_dir = str(Path(__file__).resolve().parent.parent.parent)


class DetectorResponseFunction:
    """
    Provides static and class methods for handling the detector's instrumental response.
    
    Includes convolution matrices for energy resolution and incomplete charge collection (ICC).
    This class is initialized with calibration data and is used by both background and peak 
    models to accurately simulate detector effects.
    
    Class Attributes
    ----------------
    det_res_conv_matrix : np.ndarray or None
        Detector resolution convolution matrix (cropped to spectrum limits).
    icc_conv_matrix : np.ndarray or None
        ICC convolution matrix (cropped to spectrum limits).
    det_eff_energy_vals : np.ndarray
        Energy values for detector efficiency curve.
    det_eff_vals : np.ndarray
        Detector efficiency values.
    energy_vals_padding : int
        Padding added to energy_vals for convolution operations.
    """
    
    det_res_conv_matrix = None
    icc_conv_matrix = None
    energy_vals_padding = 30  # Padding added to energy_vals to ensure correct functioning of convolution operation

    @classmethod
    def setup_detector_response_vars(cls, det_ch_offset, det_ch_width, spectrum_lims, microscope_ID, verbose=True):
        """
        Initialize detector response variables for the EDS system.
    
        Loads detector efficiency and convolution matrices for the specified microscope.
        If convolution matrices for the given channel settings do not exist, they are calculated and saved.
    
        Parameters
        ----------
        det_ch_offset : float
            Energy offset for detector channels (in keV).
        det_ch_width : float
            Channel width (in keV).
        spectrum_lims : tuple of int
            (low, high) indices for the usable spectrum region.
        microscope_ID : str
            Identifier for the microscope/calibration directory.
        verbose : bool, optional
            If True, print status messages.
    
        Sets Class Attributes
        ---------------------
        det_eff_energy_vals : np.ndarray
            Energy values for detector efficiency.
        det_eff_vals : np.ndarray
            Detector efficiency values.
        det_res_conv_matrix : np.ndarray
            Detector response convolution matrix (cropped to spectrum_lims).
        icc_conv_matrix : np.ndarray
            ICC convolution matrix (cropped to spectrum_lims).
    
        Notes
        -----
        - Convolution matrices are detector-dependent and cached in JSON for efficiency.
        - If multiple EDS detectors are used, this code should be adapted.
        """
    
        # --- Load EDS detector efficiency spectrum ---
        detector_efficiency_path = os.path.join(
            parent_dir, cnst.XRAY_SPECTRA_CALIBS_DIR, microscope_ID, cnst.DETECTOR_EFFICIENCY_FILENAME
        )
        det_eff_energy_vals, det_eff_vals, metadata = load_msa(detector_efficiency_path)
        if metadata['XUNITS'] == 'eV':
            det_eff_energy_vals /= 1000  # Convert to keV
    
        cls.det_eff_energy_vals = det_eff_energy_vals
        cls.det_eff_vals = det_eff_vals
    
        # --- Load or calculate convolution matrices ---
        conv_matrices_file_path = os.path.join(
            parent_dir, cnst.XRAY_SPECTRA_CALIBS_DIR, microscope_ID, cnst.DETECTOR_CONV_MATRICES_FILENAME
        )
        lock_file_path = conv_matrices_file_path + ".lock"
        conv_mat_key = f"O{det_ch_offset},W{det_ch_width}"
        
        conv_matrices = None

        # 1. FAST PATH: Try to read the file without a lock first.
        # If the file exists and our key is in it, we don't need to lock anything.
        if os.path.exists(conv_matrices_file_path):
            try:
                with open(conv_matrices_file_path, 'r') as file:
                    conv_matrices_dict = json.load(file)
                    conv_matrices = conv_matrices_dict.get(conv_mat_key)
            except Exception:
                pass # Ignore decode errors (another core might be mid-write)

        # 2. SLOW PATH: We need to compute it (or wait for another core to compute it)
        if conv_matrices is None:
            start_wait_time = time.time()
            max_wait_time = 600  # 10 minute timeout for stale locks
            
            # Loop until conv_matrices is successfully populated
            while conv_matrices is None:
                try:
                    # Attempt to exclusively create the lock file
                    with open(lock_file_path, 'x') as f:
                        f.write(f"PID {os.getpid()}")
                    
                    # === WE HAVE THE LOCK ===
                    try:
                        # Reload JSON: another process might have JUST calculated it while we waited
                        conv_matrices_dict = {}
                        if os.path.exists(conv_matrices_file_path):
                            with open(conv_matrices_file_path, 'r') as file:
                                conv_matrices_dict = json.load(file)
                        
                        conv_matrices = conv_matrices_dict.get(conv_mat_key)
                        
                        # If it's STILL missing, we actually do the heavy lifting
                        if conv_matrices is None:
                            if True: # verbose:
                                print(f"Calculating convolution matrices for key {conv_mat_key}...")
                                
                            full_en_vector = [det_ch_offset + j * det_ch_width for j in range(calibs.detector_ch_n)]
                            det_res_conv_matrix = cls._calc_det_res_conv_matrix(full_en_vector, verbose)
                            icc_conv_matrix = cls._calc_icc_conv_matrix(full_en_vector, verbose)
                            
                            conv_matrices_dict[conv_mat_key] = (det_res_conv_matrix.tolist(), icc_conv_matrix.tolist())
                            
                            with open(conv_matrices_file_path, 'w') as file:
                                json.dump(conv_matrices_dict, file)
                            
                            conv_matrices = (det_res_conv_matrix, icc_conv_matrix)
                            
                    finally:
                        # === RELEASE THE LOCK ===
                        if os.path.exists(lock_file_path):
                            os.remove(lock_file_path)
                            
                except FileExistsError:
                    # === ANOTHER CORE HAS THE LOCK ===
                    if time.time() - start_wait_time > max_wait_time:
                        # Lock is stale (the other core crashed). Force delete it.
                        try:
                            os.remove(lock_file_path)
                            start_wait_time = time.time() # Reset timeout
                        except OSError:
                            pass
                    else:
                        # Wait patiently
                        if verbose:
                            print("Waiting for another core to finish computing matrices...")
                        time.sleep(3)
                        
                        # Peek at the file to see if the other core finished our key
                        if os.path.exists(conv_matrices_file_path):
                            try:
                                with open(conv_matrices_file_path, 'r') as file:
                                    conv_matrices = json.load(file).get(conv_mat_key)
                            except Exception:
                                pass # File is currently being written to, loop will retry naturally

        else:
            if verbose:
                print_single_separator()
                print("Detector response convolution matrices loaded")
                
        det_res_conv_matrix, icc_conv_matrix = conv_matrices
    
        # --- Crop matrices to match spectrum limits ---
        low_l = spectrum_lims[0] + 1
        high_l = spectrum_lims[1] + cls.energy_vals_padding // 2 + cls.energy_vals_padding - 1
        cls.det_res_conv_matrix = np.array(det_res_conv_matrix)[low_l:high_l, low_l:high_l]
        cls.icc_conv_matrix = np.array(icc_conv_matrix)[low_l:high_l, low_l:high_l] 

    # =============================================================================
    # Convolution of signal with detector response function
    # =============================================================================
    @classmethod
    def _apply_padding_with_fit(cls, signal):
        """
        Pad a signal at both ends by linear extrapolation, using a fit to the first and last few points.
    
        This method is used to extend the signal array, avoiding edge effects in convolution.
        Padding values are clipped to be non-negative.
    
        Parameters
        ----------
        signal : np.ndarray or list
            The input 1D signal to pad.
    
        Returns
        -------
        padded_signal : np.ndarray
            The input signal with linear-extrapolated padding added at both ends.
    
        Notes
        -----
        - The number of padding points is determined by cls.energy_vals_padding.
        - Linear fits are performed on the first and last 4 points of the signal.
        - Extrapolated values are clipped at zero to avoid negative padding.
        """
        n_pts_fitted = 4  # Number of points used for linear fit at each end
    
        # --- Padding at the beginning (head) ---
        x_head = signal[:n_pts_fitted]
        x_indices_head = np.arange(len(x_head))
        # Linear fit to the first few points
        slope_head, intercept_head = np.polyfit(x_indices_head, x_head, 1)
        # Indices for extrapolation (negative indices for padding before signal)
        extrapolation_indices_head = np.arange(-cls.energy_vals_padding // 2 + 1, 0)
        # Linear extrapolation for padding
        extrapolated_values_head = slope_head * extrapolation_indices_head + intercept_head
        # Ensure no negative values in padding
        extrapolated_values_head = np.clip(extrapolated_values_head, 0, None)
    
        # --- Padding at the end (tail) ---
        x_tail = signal[-n_pts_fitted:]
        x_indices_tail = np.arange(len(x_tail))
        # Linear fit to the last few points
        slope_tail, intercept_tail = np.polyfit(x_indices_tail, x_tail, 1)
        # Indices for extrapolation (beyond signal end)
        extrapolation_indices_tail = np.arange(len(x_tail) + 1, len(x_tail) + cls.energy_vals_padding)
        # Linear extrapolation for padding
        extrapolated_values_tail = slope_tail * extrapolation_indices_tail + intercept_tail
        # Ensure no negative values in padding
        extrapolated_values_tail = np.clip(extrapolated_values_tail, 0, None)
    
        # --- Combine padding and original signal ---
        padded_signal = np.concatenate([extrapolated_values_head, signal, extrapolated_values_tail])
    
        return padded_signal
    
    
    @classmethod
    def _apply_det_response_fncts(cls, signal):
        """
        Apply both detector resolution and ICC convolution functions to a signal,
        with edge padding by linear extrapolation.
    
        The signal is padded at both ends using a linear fit, then convolved sequentially
        with the detector resolution and ICC convolution matrices. The padding is removed
        from the final result.
    
        Parameters
        ----------
        signal : np.ndarray or list
            The input 1D signal to be convolved.
    
        Returns
        -------
        processed_model : np.ndarray
            The signal after both convolutions, trimmed to exclude the padding.
    
        Notes
        -----
        - Padding is performed by fitting and extrapolating the first and last few points.
        - The convolution matrices must be initialized in the class before use.
        - This approach is more accurate than simple replicative padding, but slightly slower.
        """
        # Pad the signal at both ends using linear fit/extrapolation
        padded_signal = cls._apply_padding_with_fit(signal)
    
        # First, convolve with the detector resolution matrix
        model = np.sum(cls.det_res_conv_matrix * padded_signal, axis=1)
    
        # Then, convolve with the ICC convolution matrix
        model = np.sum(cls.icc_conv_matrix * model, axis=1)
    
        # Remove padding from the result to return only the original signal region
        processed_model = model[cls.energy_vals_padding // 2 - 1 : -cls.energy_vals_padding + 1]
    
        return processed_model
    
    # =============================================================================
    # Detector resolution convolution
    # =============================================================================
    @staticmethod
    def _det_sigma(E):
        """
        Calculate the detector Gaussian sigma for a given X-ray energy.
        
        Requires calibration file to be correctly loaded via setup_detector_response_vars.
    
        Based on:
        N.W.M. Ritchie, "Spectrum Simulation in DTSA-II", Microsc. Microanal. 15 (2009) 454–468.
        https://doi.org/10.1017/S1431927609990407
    
        Parameters
        ----------
        E : float or array-like
            X-ray energy in keV.
    
        Returns
        -------
        sigma : float or np.ndarray
            Standard deviation (sigma) of the detector response at energy E.
    
        Notes
        -----
        - Parameters (conv_eff, elec_noise, F) are calibrated on several elements in bulk standards.
        - See Calculation_pars_peak_fwhm.py for details.
        """
        # Calculate detector sigma using calibrated parameters
        sigma = calibs.conv_eff * np.sqrt(calibs.elec_noise**2 + E * calibs.F / calibs.conv_eff)
        return sigma
        
    
    @classmethod
    def _calc_det_res_conv_matrix(cls, energy_vals, verbose = True):
        """
        Calculate the detector resolution convolution matrix.
    
        Each row of the matrix represents the probability distribution (Gaussian)
        for an input energy, accounting for the detector's energy resolution.
        Padding is added to minimize edge effects during convolution.
    
        Parameters
        ----------
        energy_vals : array-like
            Array of energy values (in keV) for which to compute the convolution matrix.
        verbose : bool, optional
            If True, print status messages.
    
        Returns
        -------
        det_res_conv_matrix : np.ndarray
            The detector resolution convolution matrix (energy_vals x energy_vals).
    
        Notes
        -----
        - Padding is applied to account for convolution spillover at the spectrum edges.
        - The Gaussian sigma is energy-dependent and calculated with _det_sigma().
        - Integration is performed for each matrix element to ensure normalization.
        """
    
        if verbose:
            start_time = time.time()
            print("Calculating convolution matrix for detector resolution")
    
        deltaE = energy_vals[5] - energy_vals[4]
        n_intervals = cls.energy_vals_padding
    
        # Extend the energy axis with padding on both sides to avoid edge effects
        left_pad = [energy_vals[0] - deltaE * i for i in range(n_intervals // 2, 0, -1)]
        right_pad = [energy_vals[-1] + deltaE * i for i in range(1, n_intervals)]
        energy_vals_extended = left_pad + list(energy_vals) + right_pad
    
        def gaussian(E, E0, sigma):
            """Normalized Gaussian function."""
            return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 / sigma**2 * (E - E0)**2)
    
        conv_matrix = []
        for i, en in enumerate(energy_vals_extended):
            # Initialize row for this energy; padding prevents signal loss at edges
            g_vals = [0.0 for _ in range(len(energy_vals_extended))]
            sigma = cls._det_sigma(en)
            for j in range(-n_intervals, n_intervals + 1):
                cen_E = en + j * deltaE
                idx = i + j
                if 0 <= idx < len(g_vals):
                    try:
                        # Integrate the Gaussian over the width of the energy bin
                        int_E, _ = quad(lambda E: gaussian(E, en, sigma), cen_E - deltaE / 2, cen_E + deltaE / 2)
                        g_vals[idx] = int_E
                    except Exception:
                        pass  # Ignore integration errors (should be rare)
            conv_matrix.append(g_vals)
    
        det_res_conv_matrix = np.array(conv_matrix).T  # Transpose for correct orientation
        
        if verbose:
            process_time = time.time() - start_time
            print(f"Calculation executed in {process_time:.1f} s")
    
        return det_res_conv_matrix
    
    
    # =============================================================================
    # Incomplete Charge Collection Spectrum Calculations
    # =============================================================================
    @staticmethod
    def get_icc_spectrum(energy_vals, line_en, R_e=50e-7, F_loss=0.27):
        """
        Generate the ICC (Incomplete Charge Collection) smearing function for a given line energy.
    
        Parameters
        ----------
        energy_vals : array-like
            Energy axis (in keV) of detector channels.
        line_en : float
            X-ray line energy (in keV).
        R_e : float, optional
            Effective recombination parameter (cm).
        F_loss : float, optional
            Fractional charge loss parameter.
    
        Returns
        -------
        icc_n_vals_distr : list
            ICC distribution, mapped to detector channel energies.
        """
        icc_e_vals, icc_n_vals = DetectorResponseFunction._icc_fnct(line_en, R_e, F_loss)
        icc_n_vals_distr = DetectorResponseFunction._distribute_icc_over_EDS_channels(
            energy_vals, icc_e_vals, icc_n_vals
        )
        return icc_n_vals_distr
    
    
    @staticmethod
    def _icc_fnct(line_en, R_e=50e-7, F_loss=0.27):
        """
        Calculate the ICC function n(E) for a given X-ray line energy.
        
        ICC model as described in:
        Redus, R. H., & Huber, A. C. (2015). Response Function of Silicon Drift Detectors for Low Energy X-rays.
        In Advances in X-ray Analysis (AXA) (pp. 274–282). International Centre for Diffraction Data (ICDD).
    
        Parameters
        ----------
        line_en : float
            X-ray line energy (in keV).
        R_e : float, optional
            Effective recombination parameter (cm).
        F_loss : float, optional
            Fractional charge loss parameter.
    
        Returns
        -------
        e_vals : list of float
            Energy values (in keV) for the ICC function.
        n_vals : list of float
            ICC function values at those energies.
        """
        # Absorption coefficient of Si at energy line_en
        alpha = xray_mass_absorption_coeff(element='Si', energies=line_en) * calibs.Si_density  # cm^-1
    
        V_tot = 4 * np.pi / 3 * (R_e) ** 3
    
        def V_1(z):
            return np.pi / 3 * (R_e - z) ** 2 * (2 * R_e + z)
    
        def Q(z):
            return (V_tot - F_loss * V_1(z)) / V_tot
    
        def dQ_dz(z):
            return -np.pi * F_loss / V_tot * (z ** 2 - R_e ** 2)
    
        def N(z):
            return 1 - np.exp(-alpha * z)
    
        def dN_dz(z):
            return alpha * np.exp(-alpha * z)
    
        def get_z(Q_val):
            Q_val_rnd = np.clip(Q_val, Q_min, 1)
            solution = root_scalar(lambda z: Q(z) - Q_val_rnd, method='brentq', bracket=[0, R_e])
            return solution.root
    
        def n(x):
            Q_val = x / line_en
            z_val = get_z(Q_val)
            n_val = dN_dz(z_val) * dQ_dz(z_val) ** -1 / line_en
            return n_val
    
        def get_n_at_line_en(integral_rest, last_E, last_n):
            def n_fnct(n_):
                n_ = np.float64(n_)
                res = np.trapz([last_n, n_], [last_E, line_en])
                return res
            guess = (1 - integral_rest) / (line_en - last_E)
            guess_2 = guess * 2
            solution = root_scalar(lambda n_: n_fnct(n_) - (1 - integral_rest), x0=guess, x1=guess_2, method='secant')
            return solution.root
    
        # Calculate left boundary of ICC smearing function
        Q_min = Q(0)
        E_min = line_en * Q_min
    
        e_vals = list(np.linspace(E_min, line_en, 1000))
        e_vals.pop()  # Remove last energy value corresponding to line_en
        n_vals = [n(en) for en in e_vals]
        signal_integral = trapezoid(n_vals, e_vals)
        n_val_at_E = get_n_at_line_en(signal_integral, e_vals[-1], n_vals[-1])
    
        # Update lists with values at line_en
        e_vals.append(line_en)
        n_vals.append(n_val_at_E)
    
        return e_vals, n_vals


    @staticmethod
    def _distribute_icc_over_EDS_channels(eds_en_vals, icc_en_vals, icc_n_vals):
        """
        Distribute the ICC function over EDS detector channels.
    
        Parameters
        ----------
        eds_en_vals : array-like
            Detector channel energy values (in keV).
        icc_en_vals : list
            ICC function energy values (in keV).
        icc_n_vals : list
            ICC function values.
    
        Returns
        -------
        eds_icc_n_vals : list
            ICC values distributed over the detector channels.
        """
        ch_width = eds_en_vals[1] - eds_en_vals[0]
        icc_en_spacing = icc_en_vals[1] - icc_en_vals[0]
    
        # Determine which channels are affected by ICC
        indices_affected = [
            i for i, en in enumerate(eds_en_vals)
            if icc_en_vals[0] - ch_width / 2 < en <= icc_en_vals[-1] + ch_width / 2
        ]
        # Calculate number of points to add on the right side of the list to make it symmetrical. Needed to avoid shifts during convolution
        n_pts_to_center_data = len(indices_affected) - 1
        n_pts_added = 20  # Pad array of energy values for full overlap during convolution
        
        # Check if reference energy is outside the range of energy values. This can happen with characteristic X-rays outside the energy range
        if len(indices_affected) == 0:
            return [1] # ICC convolution is not applied if peak is outside energy range
    
        indices_affected = (
            list(range(indices_affected[0] - n_pts_added, indices_affected[0])) +
            indices_affected +
            list(range(indices_affected[-1] + 1, indices_affected[-1] + n_pts_added + n_pts_to_center_data + 1))
        )
    
        # Remove negative indices, maintaining symmetry
        if indices_affected[0] < 0:
            index_zero = indices_affected.index(0)
            indices_affected = indices_affected[index_zero: len(indices_affected) - index_zero]
    
        # Remove indices beyond dimension of eds_en_vals if needed
        len_eds_en_vals = len(eds_en_vals)
        if indices_affected[-1] >= len_eds_en_vals:
            index_last = indices_affected.index(len_eds_en_vals)
            n_pts_to_remove = len(indices_affected) - index_last + 1
            indices_affected = indices_affected[n_pts_to_remove: len(indices_affected) - n_pts_to_remove]
    
        # Distribute ICC function over the detector channels
        eds_icc_en_vals = [eds_en_vals[i] for i in indices_affected]
        eds_icc_n_vals = []
        for index, en in enumerate(eds_icc_en_vals):
            interval_boundary_left = en - ch_width / 2
            interval_boundary_right = en + ch_width / 2
            indices_to_int = [
                i for i, e in enumerate(icc_en_vals) if interval_boundary_left < e <= interval_boundary_right
            ]
            e_vals_to_int = [icc_en_vals[i] for i in indices_to_int]
            n_vals_to_int = [icc_n_vals[i] for i in indices_to_int]
            if indices_to_int:
                if len(indices_to_int) > 1:
                    # Integrate ICC function over interval corresponding to energy value en
                    eds_icc_n_val = trapezoid(n_vals_to_int, e_vals_to_int)
                else: # Case of only 1 point within the detector channel
                    # There is no full interval of en_spacing width within the current detector channel
                    # The portion of interval within this channel is added on the next steps
                    eds_icc_n_val = 0
                # Add portion of interval shared with left of en, unless at boundary
                if interval_boundary_left < e_vals_to_int[0] and e_vals_to_int[0] > 0 and indices_to_int[0] != 0:
                    extra_i_left = indices_to_int[0] - 1
                    left_int = trapezoid([icc_n_vals[extra_i_left], n_vals_to_int[0]],
                                         [icc_en_vals[extra_i_left], e_vals_to_int[0]])
                    eds_icc_n_val += left_int * (e_vals_to_int[0] - interval_boundary_left) / icc_en_spacing
                # Add portion of interval shared with right of en, unless at boundary
                if interval_boundary_right > e_vals_to_int[-1] and indices_to_int[-1] != len(icc_n_vals) - 1:
                    extra_i_right = indices_to_int[-1] + 1
                    right_int = trapezoid([n_vals_to_int[-1], icc_n_vals[extra_i_right]],
                                          [e_vals_to_int[-1], icc_en_vals[extra_i_right]])
                    eds_icc_n_val += right_int * (interval_boundary_right - e_vals_to_int[-1]) / icc_en_spacing
            else:
                eds_icc_n_val = 0
            eds_icc_n_vals.append(eds_icc_n_val)
    
        return eds_icc_n_vals


    @classmethod
    def _calc_icc_conv_matrix(cls, energy_vals, verbose = True):
        """
        Calculate the ICC convolution matrix for all detector channels.
    
        Parameters
        ----------
        energy_vals : array-like
            Array of energy values (in keV) for which to compute the convolution matrix.
        verbose : bool, optional
            If True, print status messages.
            
        Returns
        -------
        icc_conv_matrix : np.ndarray
            The ICC convolution matrix (energy_vals x energy_vals).
        """
        import sys
        
        if verbose:
            start_time = time.time()
            print("Calculating convolution matrix for incomplete charge collection", file=sys.stderr, flush=True)
    
        deltaE = energy_vals[5] - energy_vals[4]
        n_intervals = cls.energy_vals_padding
    
        # Extend the energy axis with padding on both sides to avoid edge effects
        left_pad = [energy_vals[0] - deltaE * i for i in range(n_intervals // 2, 0, -1)]
        right_pad = [energy_vals[-1] + deltaE * i for i in range(1, n_intervals)]
        energy_vals_extended = left_pad + list(energy_vals) + right_pad
    
        conv_matrix = []
        len_row = len(energy_vals_extended)
        for i, en in enumerate(energy_vals_extended):
            if verbose:
                print(f'{i}\tEnergy: {en * 1000:.1f} eV')
            icc_n_vals = np.zeros([len_row])
            if en > 0:
                icc_spec = DetectorResponseFunction.get_icc_spectrum(
                    energy_vals_extended, en, calibs.R_e_background, calibs.F_loss_background
                )
                if len(icc_spec) == 0:
                    icc_spec = [0]
                icc_n_vals[i] = 1
                icc_n_vals = np.convolve(icc_n_vals, icc_spec, mode='same')
            conv_matrix.append(icc_n_vals)
    
        icc_conv_matrix = np.array(conv_matrix).T
        
        if verbose:
            process_time = time.time() - start_time
            print(f"Calculation executed in {process_time:.1f} s")
    
        return icc_conv_matrix
