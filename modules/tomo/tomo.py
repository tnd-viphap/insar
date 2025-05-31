# type: ignore
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.tomo.comsar_estimator import ComSAR
from modules.tomo.input_parm import Input
from modules.tomo.psds_estimator import PSDS
from modules.tomo.shp import SHP


class TomoSARControl:
    def __init__(self, patch_info=None):
        print("###### TOMOSAR Processing ######")
        self.inputfile = os.path.join(os.path.split(os.path.abspath(__file__))[0].split("modules")[0], "modules/snap2stamps/bin/project.conf")
        self._load_config()
        
        # For SHP analysis
        self.CalWin = [7, 25]  # [row, col]
        self.Alpha = 0.05
        self.BroNumthre = 20
        self.Cohthre = 0.25
        self.Cohthre_slc_filt = 0.05
        self.patch_info = patch_info
        if not self.patch_info:
            self.patch_info = ["1", "1", "50", "50"]

        print("Step 1: Preparing inputs and SHP analysis...\n")
        self.input = Input(self.CalWin, False)
        self.slcstack, self.interfstack = self.input.run()
        
        # Process SHP in chunks
        self.shp = self._process_shp_chunks()
        print("\n")

    @staticmethod
    def process_chunk(chunk_info, calwin, alpha):
        """Process a single chunk of data for SHP analysis."""
        _shp = SHP(chunk_info['datastack'], calwin, alpha)
        chunk_result = _shp.run()
        return {
            'result': chunk_result,
            'start_line': chunk_info['start_line'],
            'end_line': chunk_info['end_line'],
            'start_col': chunk_info['start_col'],
            'end_col': chunk_info['end_col']
        }

    def _process_shp_chunks(self):
        """Process SHP for each chunk and combine results."""
        n_chunks = len(self.slcstack["datastack"])
        nlines = self.slcstack["nlines"]
        ncols = self.slcstack["ncols"]
        
        # Initialize combined results
        combined_pixelind = np.zeros(
            (self.CalWin[0] * self.CalWin[1], nlines, ncols),
            dtype=bool
        )
        combined_bronum = np.zeros((nlines, ncols), dtype=np.float32)
        
        # Process chunks in parallel with batch size of 50
        batch_size = 50
        n_batches = (n_chunks + batch_size - 1) // batch_size
        
        print(f"-> Processing {n_chunks} chunks for SHP in {n_batches} batches of {batch_size}")
        
        with ProcessPoolExecutor(max_workers=int(self.CPU)) as executor:
            for batch_idx in tqdm(range(n_batches), desc="   -> SHP Computation", unit="batch"):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_chunks)
                batch_chunks = self.slcstack["datastack"][start_idx:end_idx]
                
                # Process batch in parallel
                futures = [
                    executor.submit(
                        self.process_chunk,
                        chunk_info,
                        self.CalWin,
                        self.Alpha
                    ) for chunk_info in batch_chunks
                ]
                
                for future in as_completed(futures):
                    chunk_data = future.result()
                    chunk_result = chunk_data['result']
                    
                    # Get chunk boundaries
                    start_line = chunk_data['start_line']
                    end_line = chunk_data['end_line']
                    start_col = chunk_data['start_col']
                    end_col = chunk_data['end_col']
                    
                    # Reshape PixelInd to match chunk dimensions
                    chunk_pixelind = chunk_result['PixelInd'].reshape(
                        (self.CalWin[0] * self.CalWin[1], end_line - start_line, end_col - start_col)
                    )
                    
                    # Combine PixelInd
                    combined_pixelind[:, start_line:end_line, start_col:end_col] = chunk_pixelind
                    
                    # Combine BroNum
                    combined_bronum[start_line:end_line, start_col:end_col] = chunk_result['BroNum']
        
        return {
            'PixelInd': combined_pixelind,
            'BroNum': combined_bronum,
            'CalWin': self.CalWin
        }

    def _load_config(self):
        with open(self.inputfile, 'r') as file:
            for line in file.readlines():
                key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                if key:
                    setattr(self, key, value)  # Dynamically set variables
        
    @staticmethod
    def compute_coherence_chunk(args):
        """
        Process a chunk of pixels for a single image pair with appropriate window operations
        """
        ii, ss, chunk_start, chunk_end, nlines, nwidths, RadiusRow, RadiusCol, slc_ii, slc_ss, inf_ii, inf_ss, shp_pixelind = args
        
        # Extract the specific slice needed for this chunk
        chunk_width = chunk_end - chunk_start
        
        # Calculate coherence for the specific pair (ii, ss) for this chunk of columns
        Dphi = np.exp(1j * np.angle(inf_ii[:, chunk_start:chunk_end] * np.conj(inf_ss[:, chunk_start:chunk_end])))
        Interf = np.sqrt(slc_ii[:, chunk_start:chunk_end] * slc_ss[:, chunk_start:chunk_end]) * Dphi
        
        # Pad the chunk data for window operations
        m1 = np.pad(slc_ii[:, chunk_start:chunk_end], ((RadiusRow, RadiusRow), (RadiusCol, RadiusCol)), mode='symmetric')
        m2 = np.pad(slc_ss[:, chunk_start:chunk_end], ((RadiusRow, RadiusRow), (RadiusCol, RadiusCol)), mode='symmetric')
        Interf = np.pad(Interf, ((RadiusRow, RadiusRow), (RadiusCol, RadiusCol)), mode='symmetric')
        
        # Initialize result array for this chunk
        result = np.zeros((nlines, chunk_width), dtype=np.complex64)
        
        # Process each pixel in the chunk
        for jj in range(chunk_width):
            for kk in range(nlines):
                # Local coordinates within the padded arrays
                x_local = jj + RadiusCol
                y_local = kk + RadiusRow
                
                # Extract windows around the current pixel
                master_window = m1[y_local - RadiusRow:y_local + RadiusRow + 1,
                               x_local - RadiusCol:x_local + RadiusCol + 1].flatten()
                slave_window = m2[y_local - RadiusRow:y_local + RadiusRow + 1,
                              x_local - RadiusCol:x_local + RadiusCol + 1].flatten()
                interf_window = Interf[y_local - RadiusRow:y_local + RadiusRow + 1,
                                   x_local - RadiusCol:x_local + RadiusCol + 1].flatten()
                
                # Calculate global pixel index for mask lookup
                global_pixel_idx = (chunk_start + jj) * nlines + kk
                mask = shp_pixelind[:, global_pixel_idx]
                
                MasterValue = master_window[mask]
                SlaveValue = slave_window[mask]
                InterfValue = interf_window[mask]
                
                nu = np.sum(InterfValue)
                de1 = np.sum(MasterValue)
                de2 = np.sum(SlaveValue)
                
                result[kk, jj] = nu / np.sqrt(de1 * de2)
        
        return ii, ss, chunk_start, chunk_end, result

    def intf_cov(self):
        """
        Estimate coherence matrix for a stack of interferograms and SLC images.
        """
        start_time = time.time()
        nlines, nwidths, npages = self.interfstack["datastack"].shape
        
        # Get SLC amplitude
        slcstack = np.abs(self.slcstack["datastack"])

        # Normalize interferograms
        nonzero_mask = self.interfstack["datastack"] != 0
        interfstack = self.interfstack["datastack"].copy()
        interfstack[nonzero_mask] = self.interfstack["datastack"][nonzero_mask]/np.abs(self.interfstack["datastack"][nonzero_mask])

        # Determine reference index
        reference_ind = list(set(self.slcstack["filename"]) - set(self.interfstack["filename"]))[0]
        reference_ind = self.slcstack["filename"].index(reference_ind)

        # Build full interferogram stack including reference image
        inf_full = np.zeros((nlines, nwidths, npages + 1), dtype=np.complex64)
        if reference_ind > 0:
            inf_full[:, :, :reference_ind] = interfstack[:, :, :reference_ind]
            inf_full[:, :, reference_ind+1:] = interfstack[:, :, reference_ind:]
            inf_full[:, :, reference_ind] = np.abs(slcstack[:, :, reference_ind])
        else:
            inf_full[:, :, 0] = np.abs(slcstack[:, :, 0])
            inf_full[:, :, 1:npages+1] = interfstack
        # Coherence estimation
        RadiusRow = (self.CalWin[0] - 1) // 2
        RadiusCol = (self.CalWin[1] - 1) // 2
        
        Coh = np.zeros((npages + 1, npages + 1, nlines, nwidths), dtype=np.complex64)
        
        # Define chunk size as a multiple of the window size for better efficiency
        chunk_width_multiplier = max(1, 4)  # Adjust based on memory considerations
        chunk_width = chunk_width_multiplier * (2 * RadiusCol + 1)
        
        # Calculate number of chunks based on the width and chunk_width
        num_chunks = (nwidths + chunk_width - 1) // chunk_width
        print(f"   -> {npages * (npages+1)} epochs detected. Progressing...")
        
        # Set diagonal elements to 1 (coherence of an image with itself is 1)
        for ii in range(npages + 1):
            Coh[ii, ii, :, :] = 1.0
        
        # Process one pair (ii, ss) at a time
        pair_counter = 0
        
        for ii in range(npages + 1):
            slc_ii = slcstack[:, :, ii]
            
            for ss in range(ii + 1, npages + 1):
                pair_counter += 1
                slc_ss = slcstack[:, :, ss]
                inf_ii = inf_full[:, :, ii]
                inf_ss = inf_full[:, :, ss]
                
                # Prepare arguments for parallel processing of chunks
                args_list = []
                for c in range(num_chunks):
                    chunk_start = c * chunk_width
                    chunk_end = min((c + 1) * chunk_width, nwidths)
                    
                    args_list.append((
                        ii, ss, chunk_start, chunk_end, nlines, nwidths,
                        RadiusRow, RadiusCol, slc_ii, slc_ss, inf_ii, inf_ss, 
                        self.shp["PixelInd"]
                    ))
                
                # Process chunks sequentially with progress bar or use ProcessPoolExecutor with progress tracking
                if num_chunks <= 10:  # For small number of chunks, process sequentially with progress bar
                    for args in tqdm(
                        args_list, 
                        desc="   ", 
                        ncols=120,
                        bar_format='   {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} chunks [{elapsed}<{remaining}]'
                    ):
                        ii_res, ss_res, start, end, result = TomoSARControl.compute_coherence_chunk(args)
                        Coh[ii_res, ss_res, :, start:end] = result
                else:
                    # For many chunks, use parallel processing with progress tracking
                    print(f"   -> Parallel processing for pair ({ii}, {ss}) with {num_chunks} chunks...")
                    
                    with ProcessPoolExecutor(max_workers=int(self.CPU)) as executor:
                        futures = [executor.submit(TomoSARControl.compute_coherence_chunk, args) for args in args_list]
                        
                        for future in tqdm(
                            as_completed(futures), 
                            total=len(futures), 
                            desc="   ", 
                            ncols=120,
                            bar_format='   {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} chunks [{elapsed}<{remaining}]'
                        ):
                            ii_res, ss_res, start, end, result = future.result()
                            Coh[ii_res, ss_res, :, start:end] = result
        # Make mirror operator
        print("-> Applying Hermitian conjugation...")
        temp = np.ones(npages + 1)
        for jj in range(nwidths):
            for kk in range(nlines):
                W = Coh[:, :, kk, jj]
                Coh[:, :, kk, jj] = W + (W - np.diag(temp)).T

        end_time = time.time()
        print(f"-> Coherence estimation operation completed in {(end_time - start_time)/60.0:.2f} minutes\n")
        return Coh, reference_ind

    def run(self):
        if self.input.ComSAR_flag:
            print("Step 2: COMSAR estimation\n")
            ComSAR(self.slcstack,
                   self.interfstack,
                   self.shp,
                   self.input.InSAR_path,
                   self.BroNumthre,
                   self.Cohthre,
                   self.input.miniStackSize,
                   self.Cohthre_slc_filt,
                   self.input.Unified_flag,
                   self.input.InSAR_processor).run()
        else:
            print("Step 2: PSDS estimation\n")
            print("-> Computing SHP-based coherence started...")
            Coh_matrix, reference_idx = self.intf_cov()
            print("-> Refining pixels...")
            PSDS(Coh_matrix, self.slcstack,
                 self.interfstack, self.shp, reference_idx,
                 self.input.InSAR_path,
                 self.BroNumthre, self.Cohthre,
                 self.Cohthre_slc_filt,
                 self.input.InSAR_processor).run()

if __name__ == "__main__":
    TomoSARControl()