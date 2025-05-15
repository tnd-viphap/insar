import os
import time
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

class ComSAR:
    def __init__(self, slcstack, interfstack, shp,
                 InSAR_path, bronumthre, cohthre,
                 ministacksize, cohthre_slc_filt, unified_flag, InSAR_processor):
        
        self.slcstack = slcstack
        self.interfstack = interfstack
        self.shp = shp
        self.InSAR_path = InSAR_path
        self.BroNumthre = bronumthre
        self.Cohthre = cohthre
        self.miniStackSize = ministacksize
        self.Cohthre_slc_filt = cohthre_slc_filt
        self.Unified_flag = unified_flag
        if not self.Unified_flag:
            self.Unified_flag = False
        self.InSAR_processor = InSAR_processor
        self.reference_idx = 0
      
    def interf_linking(self, coh, idx=None):
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
        if idx:
            self.reference_idx = idx
        n_slc, _, nlines, nwidths = coh.shape

        phi_PL = np.zeros((nlines, nwidths, n_slc), dtype=np.float32)
        v_PL = np.zeros((nlines, nwidths, n_slc), dtype=np.float32)
        Coh_cal = np.zeros((nlines, nwidths), dtype=np.float32)

        for jj in range(nwidths):
            for kk in range(nlines):
                self.W = coh[:, :, kk, jj]
                if not np.all(np.isfinite(self.W)):
                    continue

                phi, temp_coh, v = self._linking()
                phi_PL[kk, jj, :] = phi
                v_PL[kk, jj, :] = v
                Coh_cal[kk, jj] = (np.sum(np.abs(temp_coh)) - n_slc) / (n_slc**2 - n_slc)
        
        return phi_PL, Coh_cal, v_PL

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

    @staticmethod
    def compute_slc_coherence_chunk(args):
        """
        Process a chunk of pixels for SLC coherence computation.
        
        Parameters:
        - args: tuple containing:
            - ii, ss: pair indices
            - chunk_start, chunk_end: chunk boundaries
            - nlines, nwidths: dimensions
            - RadiusRow, RadiusCol: window parameters
            - m1_initial, m2_initial: master and slave amplitudes
            - data: complex SLC data
            - shp_pixelind: pixel indices for SHP
        
        Returns:
        - tuple: (ii, ss, chunk_start, chunk_end, result)
        """
        ii, ss, chunk_start, chunk_end, nlines, nwidths, RadiusRow, RadiusCol, m1_initial, m2_initial, data, shp_pixelind = args
        
        # Extract chunk data
        chunk_width = chunk_end - chunk_start
        m1_chunk = m1_initial[:, chunk_start:chunk_end]
        m2_chunk = m2_initial[:, chunk_start:chunk_end]
        data_ii = data[:, chunk_start:chunk_end, ii]
        data_ss = data[:, chunk_start:chunk_end, ss]
        
        # Calculate phase difference and interferogram
        Dphi = np.exp(1j * np.angle(data_ii * np.conj(data_ss)))
        Intf = np.sqrt(m1_chunk * m2_chunk) * Dphi
        
        # Pad data for window operations
        m1 = np.pad(m1_chunk, ((RadiusRow, RadiusRow), (RadiusCol, RadiusCol)), mode='symmetric')
        m2 = np.pad(m2_chunk, ((RadiusRow, RadiusRow), (RadiusCol, RadiusCol)), mode='symmetric')
        Intf = np.pad(Intf, ((RadiusRow, RadiusRow), (RadiusCol, RadiusCol)), mode='symmetric')
        
        # Initialize result array
        result = np.zeros((nlines, chunk_width), dtype=np.complex64)
        
        # Process each pixel in the chunk
        for jj in range(chunk_width):
            for kk in range(nlines):
                x_local = jj + RadiusCol
                y_local = kk + RadiusRow
                
                # Extract windows
                m_win = m1[y_local - RadiusRow:y_local + RadiusRow + 1,
                          x_local - RadiusCol:x_local + RadiusCol + 1].flatten()
                s_win = m2[y_local - RadiusRow:y_local + RadiusRow + 1,
                          x_local - RadiusCol:x_local + RadiusCol + 1].flatten()
                i_win = Intf[y_local - RadiusRow:y_local + RadiusRow + 1,
                           x_local - RadiusCol:x_local + RadiusCol + 1].flatten()
                
                # Get pixel indices
                global_pixel_idx = (chunk_start + jj) * nlines + kk
                pix_inds = shp_pixelind[:, global_pixel_idx]
                
                # Calculate coherence
                nu = np.sum(i_win[pix_inds])
                de1 = np.sum(m_win[pix_inds])
                de2 = np.sum(s_win[pix_inds])
                
                result[kk, jj] = nu / np.sqrt(de1 * de2)
        
        return ii, ss, chunk_start, chunk_end, result

    def slc_cov(self, data, shp):
        """
        Estimate coherence matrix from SLC stack and SHP struct using parallel processing.
        
        Parameters:
        - data: 3D numpy array (nlines, nwidths, n_slc)
        - shp: dict with keys 'CalWin' and 'PixelInd'
        
        Returns:
        - Coh: 4D numpy array (n_slc, n_slc, nlines, nwidths), dtype complex64
        """
        print("-> Computing SLC coherence matrix...")
        start_time = time.time()
        
        nlines, nwidths, n_slc = data.shape
        
        # Normalize non-zero values
        mask_nonzero = data != 0
        data[mask_nonzero] = data[mask_nonzero] / np.abs(data[mask_nonzero])
        
        CalWin = shp["CalWin"]
        RadiusRow = (CalWin[0] - 1) // 2
        RadiusCol = (CalWin[1] - 1) // 2
        
        mlistack = np.abs(data)
        Coh = np.zeros((n_slc, n_slc, nlines, nwidths), dtype=np.complex64)
        
        # Set diagonal elements to 1
        for ii in range(n_slc):
            Coh[ii, ii, :, :] = 1.0
        
        # Define chunk size
        chunk_width_multiplier = max(1, 4)  # Adjust based on memory considerations
        chunk_width = chunk_width_multiplier * (2 * RadiusCol + 1)
        num_chunks = (nwidths + chunk_width - 1) // chunk_width
        
        print(f"-> {n_slc * (n_slc-1)//2} pairs detected. Progressing...")
        
        # Process pairs in parallel
        for ii in range(n_slc):
            m1_initial = mlistack[:, :, ii]
            
            for ss in range(ii + 1, n_slc):
                m2_initial = mlistack[:, :, ss]
                
                # Prepare arguments for parallel processing
                args_list = []
                for c in range(num_chunks):
                    chunk_start = c * chunk_width
                    chunk_end = min((c + 1) * chunk_width, nwidths)
                    
                    args_list.append((
                        ii, ss, chunk_start, chunk_end, nlines, nwidths,
                        RadiusRow, RadiusCol, m1_initial, m2_initial, data,
                        shp["PixelInd"]
                    ))
                
                # Process chunks based on their number
                if num_chunks <= 10:
                    for args in tqdm(
                        args_list,
                        desc="   ",
                        ncols=120,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} chunks [{elapsed}<{remaining}]'
                    ):
                        ii_res, ss_res, start, end, result = self.compute_slc_coherence_chunk(args)
                        Coh[ii_res, ss_res, :, start:end] = np.abs(result)
                        Coh[ss_res, ii_res, :, start:end] = np.abs(result).conjugate()
                else:
                    print(f"-> Parallel processing for pair ({ii}, {ss}) with {num_chunks} chunks...")
                    
                    with ProcessPoolExecutor(max_workers=min(8, int(os.cpu_count()))) as executor:
                        futures = [executor.submit(self.compute_slc_coherence_chunk, args) for args in args_list]
                        
                        for future in tqdm(
                            as_completed(futures),
                            total=len(futures),
                            desc="   ",
                            ncols=120,
                            bar_format='   {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} chunks [{elapsed}<{remaining}]'
                        ):
                            ii_res, ss_res, start, end, result = future.result()
                            Coh[ii_res, ss_res, :, start:end] = np.abs(result)
                            Coh[ss_res, ii_res, :, start:end] = np.abs(result).conjugate()
        
        end_time = time.time()
        print(f"-> SLC coherence estimation completed in {(end_time - start_time)/60.0:.2f} minutes\n")
        
        return Coh
    
    def slc_despeckle(self):
        print("-> Reducing speckle noise on SLC stack...")
        start_time = time.time()
        nlines, nwidths, npages = self.slcstack["datastack"].shape
        slcImg_despeckle = np.abs(self.slcstack["datastack"])
        
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
                    slcImg_despeckle[kk, jj, ii] = np.mean(MliValues)
                    num += 1

        elapsed = time.time() - start_time
        print(f"-> DeSpeckling operation completed in {elapsed / 60:.2f} minute(s).")
        return slcImg_despeckle
        
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
        nlines, nwidths, n_interf = self.interfstack["datastack"].shape
        n_slc = n_interf + 1

        # Normalize the interferogram stack
        non_zero_mask = self.interfstack["datastack"] != 0
        interfstack = self.interfstack["datastack"].copy()
        interfstack[non_zero_mask] = interfstack[non_zero_mask] / np.abs(interfstack[non_zero_mask])
        reference_ind = list(set(self.slcstack["filename"]) - set(self.interfstack["filename"]))[0]
        reference_idx = self.slcstack["filename"].index(reference_ind)

        # Build full interferogram stack including reference image
        inf_full = np.zeros((nlines, nwidths, n_slc), dtype=np.complex64)
        if reference_idx > 0:
            inf_full[:, :, :reference_idx] = interfstack[:, :, :reference_idx]
            inf_full[:, :, reference_idx] = np.abs(self.slcstack["datastack"][:, :, reference_idx - 1])
            inf_full[:, :, reference_idx + 1:] = interfstack[:, :, reference_idx:]
        else:
            inf_full[:, :, 0] = np.abs(self.slcstack["datastack"][:, :, 0])
            inf_full[:, :, 1:n_slc] = interfstack

        interfstack = inf_full
        del inf_full

        # Get SLC amplitude
        interfstack = np.abs(self.slcstack["datastack"]) * np.exp(1j * np.angle(interfstack))

        # Initial mini stack indices (0-based)
        mini_ind = list(range(0, n_slc, self.miniStackSize))

        # Check if reference index is in mini_ind
        if reference_idx in mini_ind:
            self.reference_ComSAR_ind = mini_ind.index(reference_idx)
        else:
            # Add reference index and sort
            mini_ind.append(reference_idx)
            mini_ind = sorted(mini_ind)

            temp_diff = [self.miniStackSize] + [mini_ind[i] - mini_ind[i-1] for i in range(1, len(mini_ind))]

            one_image_ind = [i for i, diff in enumerate(temp_diff) if diff < 2]

            self.reference_ComSAR_ind = mini_ind.index(reference_idx)

            # Two images needed for interferometric phase
            if one_image_ind:
                for idx in one_image_ind:
                    if self.reference_ComSAR_ind != idx:
                        mini_ind[idx] = mini_ind[idx] + 1
                    elif idx > 1:
                        mini_ind[idx - 1] = mini_ind[idx - 1] - 1
                    else:
                        mini_ind[idx + 1] = mini_ind[idx + 1] + 1

        # Ensure last index is not equal to n_slc (need 2 images per mini-stack)
        if mini_ind[-1] == n_slc - 1:
            mini_ind[-1] = mini_ind[-1] - 1

        # Number of mini stacks
        numMiniStacks = len(mini_ind)

        # Compressed SLCs stack
        compressed_SLCs = np.zeros((nlines, nwidths, numMiniStacks), dtype=np.complex64)

        if self.Unified_flag:
            Unified_ind = np.arange(mini_ind[0], n_slc)
            try:
                reference_UnifiedSAR_ind = list(Unified_ind).index(reference_idx)
            except ValueError:
                reference_UnifiedSAR_ind = -1  # not found

            N_unified_SAR = len(Unified_ind)
            Unified_SAR = np.zeros((nlines, nwidths, N_unified_SAR), dtype=np.float32)

        for k in range(numMiniStacks):
            if k == numMiniStacks - 1:
                cal_ind = np.arange(mini_ind[k], n_slc)
            else:
                cal_ind = np.arange(mini_ind[k], mini_ind[k+1])

            # Coherence matrix from selected stack
            Coh_temp = self.slc_cov(interfstack[:, :, cal_ind], self.shp)

            # Phase linking
            phi_PL, _, v_PL = self.interf_linking(Coh_temp)

            # Compress SLCs via coherent summation
            compressed_SLCs[:, :, k] = np.sum(v_PL * interfstack[:, :, cal_ind], axis=2)

            if self.Unified_flag:
                Unified_SAR[:, :, cal_ind - mini_ind[0]] = phi_PL

            print(f"-> Compressed SAR progress: {k+1}/{len(mini_ind)} is finished.")

        # Optional SHP recomputation
        # if compressed_SLCs.shape[2] > 15:
        #     self.shp = SHP_SelPoint(np.abs(compressed_SLCs), CalWin, Alpha)

        # Phase linking on compressed SLCs
        cov_compressed_slc = self.slc_cov(compressed_SLCs, self.shp)
        phi_PL_compressed, Coh_cal, _ = self.interf_linking(cov_compressed_slc, self.reference_ComSAR_ind)

        if self.Unified_flag:
            # Update full time series
            for k in range(numMiniStacks):
                if k == numMiniStacks - 1:
                    cal_ind = np.arange(mini_ind[k], n_slc)
                else:
                    cal_ind = np.arange(mini_ind[k], mini_ind[k+1])
                
                # Equation 3 in [1]
                Unified_SAR[:, :, cal_ind - mini_ind[0]] += np.repeat(
                    phi_PL_compressed[:, :, k:k+1], 
                    len(cal_ind), 
                    axis=2
                )

            # Remove reference from unified SAR
            Unified_SAR = np.delete(Unified_SAR, reference_UnifiedSAR_ind, axis=2)

            # Phase filtering
            mask_coh = Coh_cal > self.Cohthre
            mask_PS = self.shp["BroNum"] > self.BroNumthre
            mask = np.logical_and(mask_PS, mask_coh)  # PS keep
            mask = np.repeat(mask[:, :, np.newaxis], N_unified_SAR - 1, axis=2)

            # Get indices without reference
            Unified_ind_no_ref = np.delete(Unified_ind, reference_UnifiedSAR_ind)
            interfstack_ComSAR = self.interfstack[:, :, Unified_ind_no_ref]
            interfstack_ComSAR[mask] = np.abs(interfstack_ComSAR[mask]) * np.exp(1j * Unified_SAR[mask])

            # DeSpeckle for unified SLCs
            slcstack_ComSAR = self.slcstack["datastack"][:, :, Unified_ind]
            mli_despeckle = self.slc_despeckle()  # Assuming this returns the despeckled image
            mask_coh = Coh_cal > self.Cohthre_slc_filt
            mask = np.logical_and(mask_PS, mask_coh)  # PS keep
            mask = np.repeat(mask[:, :, np.newaxis], N_unified_SAR, axis=2)
            slcstack_ComSAR[mask] = np.abs(mli_despeckle[mask]) * np.exp(1j * np.angle(slcstack_ComSAR[mask]))

            # Update filenames for unified ComSAR
            self.slcstack_ComSAR_filename = [self.slcstack["filename"][i] for i in Unified_ind]
            self.interfstack_ComSAR_filename = self.interfstack["filename"][
                np.any(np.isin(self.interfstack["filename"], self.slcstack_ComSAR_filename), axis=1)
            ]

        else:
            # Work only with compressed data
            phi_PL_compressed = np.delete(phi_PL_compressed, self.reference_ComSAR_ind, axis=2)

            # Phase filtering
            mask_coh = Coh_cal > self.Cohthre
            mask_PS = self.shp["BroNum"] > self.BroNumthre
            mask = np.logical_and(mask_PS, mask_coh)  # PS keep
            mask = np.repeat(mask[:, :, np.newaxis], numMiniStacks - 1, axis=2)

            # Get indices without reference
            mini_ind_no_ref = np.delete(mini_ind, self.reference_ComSAR_ind)
            interfstack_ComSAR = interfstack[:, :, mini_ind_no_ref]
            interfstack_ComSAR[mask] = np.abs(interfstack_ComSAR[mask]) * np.exp(1j * phi_PL_compressed[mask])

            # DeSpeckle for Compressed SLCs
            slcstack_ComSAR = self.slcstack["datastack"][:, :, mini_ind]
            mli_despeckle = self.slc_despeckle()  # Assuming this returns the despeckled image
            mask_coh = Coh_cal > self.Cohthre_slc_filt
            mask = np.logical_and(mask_PS, mask_coh)
            mask = np.repeat(mask[:, :, np.newaxis], numMiniStacks, axis=2)
            slcstack_ComSAR[mask] = np.abs(mli_despeckle[mask]) * np.exp(1j * np.angle(compressed_SLCs[mask]))

            # Update filenames for compressed ComSAR
            self.slcstack_ComSAR_filename = [self.slcstack["filename"][i] for i in mini_ind]
            self.interfstack_ComSAR_filename = self.interfstack["filename"][
                np.any(np.isin(self.interfstack["filename"], self.slcstack_ComSAR_filename), axis=1)
            ]

        # Export results
        self.interf_export(self.InSAR_path + '/diff0', '.comp')
        self.slc_export(self.InSAR_path +'/rslc', '.csar')