import os
import time
import numpy as np
from tqdm import tqdm

class PSDS:
    def __init__(self, coh_matrix, slcstack, interfstack, shp, reference_idx,
                 InSAR_path, bronumthre, cohthre, cohthre_slc_filt, InSAR_processor):
        
        self.coh_matrix = coh_matrix
        self.slcstack = slcstack
        self.interfstack = interfstack
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
        print("-> Performing phase linking...")
        n_slc, _, nlines, nwidths = self.coh_matrix.shape

        self.phi_PL = np.zeros((nlines, nwidths, n_slc), dtype=np.float32)
        self.v_PL = np.zeros((nlines, nwidths, n_slc), dtype=np.float32)
        self.Coh_cal = np.zeros((nlines, nwidths), dtype=np.float32)

        for jj in range(nwidths):
            for kk in range(nlines):
                self.W = np.squeeze(self.coh_matrix[:, :, kk, jj])
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
            U, _, _ = np.linalg.svd(self.W + 1e-14)
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
        W_cal = O.T @ self.W @ O

        return phi, W_cal, v_ml
    
    def interf_filtering(self):
        print("-> Applying filter strategies on interferogram...")
        self.phi_PL = np.delete(self.phi_PL, self.reference_idx, axis=2)
        mask_coh = self.Coh_cal > self.Cohthre
        mask_PS = self.shp["BroNum"] > self.BroNumthre
        # Combine masks
        mask = np.logical_and(mask_PS, mask_coh)

        # Repeat mask along the 3rd axis
        mask = np.repeat(mask[:, :, np.newaxis], len(self.interfstack["filename"]), axis=2)

        # Apply the phase update to infstack
        self.interfstack["datastack"][mask] = np.abs(self.interfstack["datastack"][mask]) * np.exp(1j * self.phi_PL[mask])
        
    def slc_despeckle(self):
        print("-> Reducing speckle noise on SLC stack...")
        start_time = time.time()
        nlines, nwidths, npages = self.slcstack["datastack"].shape
        self.slcImg_despeckle = np.abs(self.slcstack["datastack"])
        
        CalWin = self.shp["CalWin"]
        RadiusRow = (CalWin[0] - 1) // 2
        RadiusCol = (CalWin[1] - 1) // 2
        slcstack = np.pad(self.slcImg_despeckle.astype(np.float32),
                          ((RadiusRow, RadiusRow), (RadiusCol, RadiusCol), (0, 0)),
                          mode='symmetric')
        
        for ii in tqdm(range(npages), total=npages, desc="-> ADP. DESPECKLING: ", unit="pair"):
            temp = slcstack[:, :, ii]
            num = 0
            for jj in range(nwidths):
                for kk in range(nlines):
                    x_global = jj + RadiusCol
                    y_global = kk + RadiusRow
                    MliWindow = temp[y_global - RadiusRow: y_global + RadiusRow + 1,
                                    x_global - RadiusCol: x_global + RadiusCol + 1]
                    MliValues = MliWindow.flatten()[self.shp["PixelInd"][:, num]]
                    self.slcImg_despeckle[kk, jj, ii] = np.mean(MliValues)
                    num += 1

        elapsed = time.time() - start_time
        print(f"-> DeSpeckling operation completed in {elapsed / 60:.2f} minute(s).")
        
    def slc_filtering(self):
        print("-> Applying filter strategies on SLC stack...")
        _, _, npages = self.slcstack["datastack"].shape
        mask_coh = self.Coh_cal > self.Cohthre_slc_filt
        mask_PS = self.shp["BroNum"] > self.BroNumthre
        mask = np.logical_and(mask_coh, mask_PS)
        mask = np.repeat(mask[:, :, np.newaxis], npages, axis=2)
        
        self.slcstack["datastack"][mask] = np.abs(self.slcImg_despeckle[mask]) * np.exp(1j * np.angle(self.slcstack["datastack"][mask]))
        
    def interf_export(self, path, extension):
        """
        Export interferogram stack to binary files compatible with SAR processors.

        Parameters:
        - infstack: np.ndarray, shape (nlines, nwidths, n_interf), complex64
        - inflist: np.ndarray, shape (n_interf, 2), each row contains [master_id, slave_id]
        - path: str, output directory
        - extension: str, file extension (e.g., '.bin', '.int')
        - InSAR_processor: str, either 'snap' or 'isce'
        """
        print("-> Exporting interferogram...")
        nlines, nwidths, n_interf = self.interfstack["datastack"].shape

        real_index = np.arange(0, nwidths * 2, 2)
        imag_index = np.arange(1, nwidths * 2, 2)
        line_cpx = np.zeros(nwidths * 2, dtype=np.float32)

        master_id = self.slcstack["filename"][self.reference_idx]
        for i in range(n_interf):
            slave_id = self.interfstack["filename"][i]
            if self.InSAR_processor == 'snap':
                filename = os.path.join(path, f"{master_id}_{slave_id}{extension}")
                fid = open(filename, 'wb')
            elif self.InSAR_processor == 'isce':
                isce_dir = os.path.join(path, str(slave_id))
                os.makedirs(isce_dir, exist_ok=True)
                filename = os.path.join(isce_dir, f"isce_minrefdem.int{extension}")
                fid = open(filename, 'wb')
            else:
                raise ValueError("InSAR_processor not supported. Use 'snap' or 'isce'.")

            data = np.squeeze(self.interfstack["datastack"][:, :, i])
            for k in range(nlines):
                line_cpx[real_index] = np.real(data[k, :])
                line_cpx[imag_index] = np.imag(data[k, :])
                fid.write(line_cpx.tobytes())
            fid.close()
        
    def slc_export(self, path, extension):
        print("-> Exporting SLC stack...")
        """
        Export SLC stack to binary files compatible with SAR processors.

        Parameters:
        - slcstack: np.ndarray, shape (nlines, nwidths, n_slc), dtype=complex64
        - slclist: list or np.ndarray of SLC identifiers (length = n_slc)
        - path: str, output directory
        - extension: str, file extension (e.g., '.slc', '.bin')
        - InSAR_processor: str, either 'snap' or 'isce'
        - reference_index: int, index of reference SLC (0-based)
        """
        nlines, nwidths, n_slc = self.slcstack["datastack"].shape
        real_index = np.arange(0, nwidths * 2, 2)
        imag_index = np.arange(1, nwidths * 2, 2)
        line_cpx = np.zeros(nwidths * 2, dtype=np.float32)

        for i in range(n_slc):
            if self.InSAR_processor == 'snap':
                filename = os.path.join(path, f"{self.slcstack['filename'][i]}{extension}")
                fid = open(filename, 'wb')
            elif self.InSAR_processor == 'isce':
                if i == self.reference_idx:
                    out_dir = os.path.join(path, "reference")
                    os.makedirs(out_dir, exist_ok=True)
                    filename = os.path.join(out_dir, f"reference.slc{extension}")
                else:
                    out_dir = os.path.join(path, str(self.slcstack["filename"][i]))
                    os.makedirs(out_dir, exist_ok=True)
                    filename = os.path.join(out_dir, f"secondary.slc{extension}")
                fid = open(filename, 'wb')
            else:
                raise ValueError("Unsupported InSAR_processor. Use 'snap' or 'isce'.")

            data = np.squeeze(self.slcstack["datastack"][:, :, i])
            for k in range(nlines):
                line_cpx[real_index] = np.real(data[k, :])
                line_cpx[imag_index] = np.imag(data[k, :])
                fid.write(line_cpx.tobytes())
            fid.close()

    def run(self):
        self.interf_linking()
        self.interf_filtering()
        self.slc_despeckle()
        self.slc_filtering()
        self.interf_export(self.InSAR_path + '/diff0', '.psds')
        self.slc_export(self.InSAR_path +'/rslc', '.psar')