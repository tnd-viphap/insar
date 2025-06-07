import os
import time
import numpy as np
from tqdm import tqdm

class PSDS:
    def __init__(self, coh_matrix, slcstack, interfstack, slc_filename, interf_filename, shp, reference_idx,
                 InSAR_path, bronumthre, cohthre, cohthre_slc_filt, InSAR_processor):
        
        self.coh_matrix = coh_matrix
        self.slcstack = slcstack
        self.interfstack = interfstack
        self.slc_filename = slc_filename
        self.interf_filename = interf_filename
        self.shp = shp
        self.reference_idx = reference_idx
        if self.reference_idx is None:
            self.reference_idx = 0
        self.InSAR_path = InSAR_path
        self.BroNumthre = bronumthre
        if self.BroNumthre is None:
            self.BroNumthre = 5
        self.Cohthre = cohthre
        if self.Cohthre is None:
            self.Cohthre = 0.65
        self.Cohthre_slc_filt = cohthre_slc_filt
        self.InSAR_processor = InSAR_processor
      
    def interf_linking(self):
        """
        Python version of the MATLAB Intf_PL function.
        Perform phase linking across coherence matrices.

        Parameters:
        - Coh: np.ndarray, shape (n_slc, n_slc, nlines, nwidths)
        - N_iter: int, number of iterations for phase linking
        - reference: int, index of reference SLC (0-based indexing)

        Returns:
        - phi_PL: np.ndarray, estimated phase, shape (nlines, nwidths, n_slc)
        - Coh_cal: np.ndarray, coherence metric, shape (nlines, nwidths)
        - v_PL: np.ndarray, amplitude estimates, shape (nlines, nwidths, n_slc)
        """
        # print("-> Performing phase linking...")
        n_slc, _, nlines, nwidths = self.coh_matrix.shape

        self.phi_PL = np.zeros((nlines, nwidths, n_slc), dtype=np.float32)
        self.v_PL = np.zeros((nlines, nwidths, n_slc), dtype=np.float32)
        self.Coh_cal = np.zeros((nlines, nwidths), dtype=np.float32)

        for jj in range(nwidths):
            for kk in range(nlines):
                self.W = self.coh_matrix[:, :, kk, jj]
                if not np.all(np.isfinite(self.W)):
                    continue

                phi, temp_coh, v = self._linking()
                self.phi_PL[kk, jj, :] = phi
                self.v_PL[kk, jj, :] = v
                self.Coh_cal[kk, jj] = (np.sum(np.abs(temp_coh)) - n_slc) / (n_slc**2 - n_slc)

    def _linking(self, method=1):
        """
        Python version of MATLAB phase_linking function.
        
        Parameters:
        - W: np.ndarray, complex coherence matrix (N x N)
        - N_iter: int, number of iterations for MLE method
        - reference: int, reference index (0-based)
        - method: int, 1 for EMI, 2 for MLE
        
        Returns:
        - phi: np.ndarray, phase vector (N,)
        - W_cal: np.ndarray, calibrated coherence matrix
        - v_ml: np.ndarray, normalized complex amplitude vector
        """

        N = self.W.shape[0]

        # Spectral regularization
        beta = 0.5
        self.W = (1 - beta) * self.W + beta * np.eye(N)
        
        try:
            W_inv = np.linalg.inv(self.W + 1e-14 * np.eye(N))
        except np.linalg.LinAlgError:
            W_inv = np.linalg.pinv(self.W + 1e-14 * np.eye(N))

        R = self.W * np.abs(W_inv)

        # Avoid NaN or Inf
        R[~np.isfinite(R)] = 1e-14

        if method == 1:
            # EMI method
            eigvals, eigvecs = np.linalg.eig(R)
            min_idx = np.argmin(np.real(eigvals))
            phi_emi = np.angle(eigvecs[:, min_idx])
            phi = phi_emi - phi_emi[self.reference_idx]

        elif method == 2:
            # MLE method
            U, _, _ = np.linalg.svd(self.W + 1e-14 * np.eye(N))
            phi_initial = np.angle(U[:, 0] / U[self.reference_idx, 0])
            phi_mle = phi_initial.copy()
            for _ in range(10):
                for p in range(N):
                    not_p = np.delete(np.arange(N), p)
                    S = R[not_p, p] * np.exp(-1j * phi_initial[not_p])
                    phi_mle[p] = -np.angle(np.sum(S))
                phi_initial = phi_mle.copy()
            phi = phi_mle - phi_mle[self.reference_idx]

        else:
            raise ValueError("Unknown method. Use 1 for EMI, 2 for MLE.")

        # Normalize phase
        phi = np.angle(np.exp(1j * phi))

        # Normalize estimated phases for compression
        v_ml = np.exp(1j * phi)
        v_ml /= np.linalg.norm(v_ml)

        O = np.diag(np.exp(1j * phi))
        W_cal = O.conj().T @ self.W @ O

        return phi, W_cal, v_ml
    
    def interf_filtering(self):
        # print("-> Applying filter strategies on interferogram...")
        self.phi_PL = np.delete(self.phi_PL, self.reference_idx, axis=2)
        mask_coh = self.Coh_cal > self.Cohthre
        mask_PS = self.shp["BroNum"] > self.BroNumthre

        # Ensure masks have the same dimensions
        if mask_PS.shape != mask_coh.shape:
            # Reshape mask_PS to match mask_coh dimensions
            # First handle rows
            if mask_PS.shape[0] != mask_coh.shape[0]:
                mask_PS = np.repeat(mask_PS, mask_coh.shape[0] // mask_PS.shape[0], axis=0)
                if mask_PS.shape[0] < mask_coh.shape[0]:
                    pad_width = ((0, mask_coh.shape[0] - mask_PS.shape[0]), (0, 0))
                    mask_PS = np.pad(mask_PS, pad_width, mode='constant', constant_values=False)
            
            # Then handle columns
            if mask_PS.shape[1] != mask_coh.shape[1]:
                mask_PS = np.repeat(mask_PS, mask_coh.shape[1] // mask_PS.shape[1], axis=1)
                if mask_PS.shape[1] < mask_coh.shape[1]:
                    pad_width = ((0, 0), (0, mask_coh.shape[1] - mask_PS.shape[1]))
                    mask_PS = np.pad(mask_PS, pad_width, mode='constant', constant_values=False)

        # Combine masks
        mask = np.logical_and(mask_PS, mask_coh)

        # Repeat mask along the 3rd axis
        mask = np.repeat(mask[:, :, np.newaxis], len(self.interf_filename), axis=2)

        # Apply the phase update to infstack
        self.interfstack["datastack"][mask] = np.abs(self.interfstack["datastack"][mask]) * np.exp(1j * self.phi_PL[mask])
        
    def slc_despeckle(self):
        # print("-> Reducing speckle noise on SLC stack...")
        # start_time = time.time()
        nlines, nwidths, npages = self.slcstack["datastack"].shape
        self.slcImg_despeckle = np.abs(self.slcstack["datastack"])
        
        CalWin = self.shp["CalWin"]
        RadiusRow = (CalWin[0] - 1) // 2
        RadiusCol = (CalWin[1] - 1) // 2
        slcstack = np.pad(self.slcImg_despeckle.astype(np.float32),
                          ((RadiusRow, RadiusRow), (RadiusCol, RadiusCol), (0, 0)),
                          mode='symmetric')
        
        # Get the actual dimensions of the SHP data
        _, total_pixels = self.shp["PixelInd"].shape
        shp_width = total_pixels // nlines
        
        for ii in range(npages):
            temp = slcstack[:, :, ii]
            for jj in range(nwidths):
                for kk in range(nlines):
                    x_global = jj + RadiusCol
                    y_global = kk + RadiusRow
                    MliWindow = temp[y_global - RadiusRow: y_global + RadiusRow + 1,
                                    x_global - RadiusCol: x_global + RadiusCol + 1]
                    
                    # Calculate the pixel index in the SHP data
                    pixel_idx = kk * shp_width + jj
                    if pixel_idx >= total_pixels:
                        continue
                        
                    # Get the mask for this pixel
                    mask = self.shp["PixelInd"][:, pixel_idx]
                    MliValues = MliWindow.flatten()[mask]
                    self.slcImg_despeckle[kk, jj, ii] = np.mean(MliValues)

        # elapsed = time.time() - start_time
        # print(f"-> DeSpeckling operation completed in {elapsed / 60:.2f} minute(s).")
        
    def slc_filtering(self):
        # print("-> Applying filter strategies on SLC stack...")
        _, _, npages = self.slcstack["datastack"].shape
        mask_coh = self.Coh_cal > self.Cohthre_slc_filt
        mask_PS = self.shp["BroNum"] > self.BroNumthre

        # Ensure masks have the same dimensions
        if mask_PS.shape != mask_coh.shape:
            # Reshape mask_PS to match mask_coh dimensions
            # First handle rows
            if mask_PS.shape[0] != mask_coh.shape[0]:
                mask_PS = np.repeat(mask_PS, mask_coh.shape[0] // mask_PS.shape[0], axis=0)
                if mask_PS.shape[0] < mask_coh.shape[0]:
                    pad_width = ((0, mask_coh.shape[0] - mask_PS.shape[0]), (0, 0))
                    mask_PS = np.pad(mask_PS, pad_width, mode='constant', constant_values=False)
            
            # Then handle columns
            if mask_PS.shape[1] != mask_coh.shape[1]:
                mask_PS = np.repeat(mask_PS, mask_coh.shape[1] // mask_PS.shape[1], axis=1)
                if mask_PS.shape[1] < mask_coh.shape[1]:
                    pad_width = ((0, 0), (0, mask_coh.shape[1] - mask_PS.shape[1]))
                    mask_PS = np.pad(mask_PS, pad_width, mode='constant', constant_values=False)

        # Combine masks
        mask = np.logical_and(mask_coh, mask_PS)
        mask = np.repeat(mask[:, :, np.newaxis], npages, axis=2)
        self.slcstack["datastack"][mask] = np.abs(self.slcImg_despeckle[mask]) * np.exp(1j * np.angle(self.slcstack["datastack"][mask]))

    def run(self):
        self.interf_linking()
        self.interf_filtering()
        self.slc_despeckle()
        self.slc_filtering()
        # self.interf_export(self.InSAR_path + '/diff0', '.psds')
        # self.slc_export(self.InSAR_path +'/rslc', '.psar')
        return self.slcstack["datastack"], self.interfstack["datastack"]