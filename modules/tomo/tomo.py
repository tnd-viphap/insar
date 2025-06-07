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
        if not os.path.exists(os.path.join(self.CURRENT_RESULT, "shp.npz")):
            self.shp = self._process_shp_chunks()['patches']
            np.savez(os.path.join(self.CURRENT_RESULT, "shp.npz"), shp=self.shp)
            time.sleep(2)
        else:
            self.shp = np.load(os.path.join(self.CURRENT_RESULT, "shp.npz"), allow_pickle=True)['shp'].item()['patches']
        print("\n")

    @staticmethod
    def process_chunk(chunk_info, calwin, alpha):
        """Process a single chunk of data for SHP analysis."""
        _shp = SHP(chunk_info['datastack'], calwin, alpha)
        chunk_result = _shp.run()
        return {
            'PixelInd': chunk_result['PixelInd'],
            'BroNum': chunk_result['BroNum'],
            'CalWin': calwin,
            'start_line': chunk_info['start_line'],
            'end_line': chunk_info['end_line'],
            'start_col': chunk_info['start_col'],
            'end_col': chunk_info['end_col']
        }

    def _process_shp_chunks(self):
        """Process SHP for each chunk and combine results."""
        n_chunks = len(self.slcstack["datastack"])
        
        # Process chunks in parallel with batch size of 50
        batch_size = 50
        n_batches = (n_chunks + batch_size - 1) // batch_size
        
        print(f"-> Processing {n_chunks} chunks for SHP in {n_batches} batches of {batch_size}")
        
        # Store results for each patch
        shp_patches = []
        
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
                    shp_patches.append(chunk_data)
        
        return {
            'patches': shp_patches
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
        
        # Calculate coherence for the specific pair (ii, ss) for this chunk of columns
        Dphi = np.exp(1j * np.angle(inf_ii[:, chunk_start:chunk_end] * np.conj(inf_ss[:, chunk_start:chunk_end])))
        Interf = np.sqrt(slc_ii[:, chunk_start:chunk_end] * slc_ss[:, chunk_start:chunk_end]) * Dphi
        
        # Pad the chunk data for window operations
        m1 = np.pad(slc_ii[:, chunk_start:chunk_end], ((RadiusRow, RadiusRow), (RadiusCol, RadiusCol)), mode='symmetric')
        m2 = np.pad(slc_ss[:, chunk_start:chunk_end], ((RadiusRow, RadiusRow), (RadiusCol, RadiusCol)), mode='symmetric')
        Interf = np.pad(Interf, ((RadiusRow, RadiusRow), (RadiusCol, RadiusCol)), mode='symmetric')
        
        # Initialize result array for this chunk
        chunk_width = chunk_end - chunk_start
        result = np.zeros((nlines, chunk_width), dtype=np.complex64)
        
        # Get the actual dimensions of the SHP data
        _, total_pixels = shp_pixelind.shape
        shp_width = total_pixels // nlines
        
        # Process each pixel in the chunk
        for jj in range(chunk_width):
            for kk in range(nlines):
                # Local coordinates within the padded arrays
                x_local = jj + RadiusCol
                y_local = kk + RadiusRow
                
                # Extract windows around the current pixel
                master_window = m1[y_local - RadiusRow:y_local + RadiusRow + 1,
                                x_local - RadiusCol:x_local + RadiusCol + 1]
                slave_window = m2[y_local - RadiusRow:y_local + RadiusRow + 1,
                               x_local - RadiusCol:x_local + RadiusCol + 1]
                interf_window = Interf[y_local - RadiusRow:y_local + RadiusRow + 1,
                                   x_local - RadiusCol:x_local + RadiusCol + 1]
                
                # Calculate the pixel index in the SHP data
                # Map the current pixel position to the SHP data space
                shp_col = int((chunk_start + jj) * shp_width / nwidths)
                pixel_idx = kk * shp_width + shp_col
                
                # Get the mask for this pixel
                mask = shp_pixelind[:, pixel_idx]
                
                # Apply mask to windows
                MasterValue = master_window.flatten()[mask]
                SlaveValue = slave_window.flatten()[mask]
                InterfValue = interf_window.flatten()[mask]
                
                nu = np.sum(InterfValue)
                de1 = np.sum(MasterValue)
                de2 = np.sum(SlaveValue)
                
                result[kk, jj] = nu / np.sqrt(de1 * de2)
        
        return ii, ss, chunk_start, chunk_end, result

    def _combine_patches_to_full_stack(self):
        """Combine patched data back into full stacks."""
        print("-> Combining patches into full stacks...")
        
        # Get dimensions from the stack info
        nlines = self.slcstack["nlines"]
        ncols = self.slcstack["ncols"]
        
        # Get number of pages from the first chunk
        first_chunk = self.slcstack["datastack"][0]
        npages = first_chunk["datastack"].shape[2]
        
        # Initialize full stacks
        full_slc = np.zeros((nlines, ncols, npages), dtype=np.complex64)
        full_interf = np.zeros((nlines, ncols, npages-1), dtype=np.complex64)
        
        # Combine SLC patches
        for chunk in self.slcstack["datastack"]:
            start_line = chunk["start_line"]
            end_line = chunk["end_line"]
            start_col = chunk["start_col"]
            end_col = chunk["end_col"]
            full_slc[start_line:end_line, start_col:end_col, :] = chunk["datastack"]
        
        # Combine interferogram patches
        for chunk in self.interfstack["datastack"]:
            start_line = chunk["start_line"]
            end_line = chunk["end_line"]
            start_col = chunk["start_col"]
            end_col = chunk["end_col"]
            full_interf[start_line:end_line, start_col:end_col, :] = chunk["datastack"]
        
        # Create new stack dictionaries
        full_slcstack = {
            "datastack": full_slc,
            "nlines": nlines,
            "ncols": ncols,
            "filename": self.slcstack["filename"]
        }
        
        full_interfstack = {
            "datastack": full_interf,
            "nlines": nlines,
            "ncols": ncols,
            "filename": self.interfstack["filename"]
        }
        
        return full_slcstack, full_interfstack

    def intf_cov(self, slcstack=None, interfstack=None):
        """
        Estimate coherence matrix for a stack of interferograms and SLC images.
        """
        start_time = time.time()
        
        # Process each patch separately
        all_coherence_matrices = []
        all_reference_indices = []
        
        for patch_idx, (slc_patch, interf_patch) in tqdm(enumerate(zip(slcstack["datastack"], interfstack["datastack"])), total=len(slcstack["datastack"]), desc="   -> Coherence estimation", unit="patch"):
            # Get patch dimensions
            nlines = slc_patch["end_line"] - slc_patch["start_line"]
            nwidths = slc_patch["end_col"] - slc_patch["start_col"]
            npages = interf_patch["datastack"].shape[2]
            
            # Get SHP data for this patch
            shp_data = self.shp[patch_idx]
            shp_pixelind = shp_data['PixelInd']
            shp_bronum = shp_data['BroNum']
            
            
            # Get SLC amplitude
            slcstack_data = np.abs(slc_patch["datastack"])

            # Normalize interferograms
            nonzero_mask = interf_patch["datastack"] != 0
            interfstack_data = interf_patch["datastack"].copy()
            interfstack_data[nonzero_mask] = interf_patch["datastack"][nonzero_mask]/np.abs(interf_patch["datastack"][nonzero_mask])

            # Determine reference index
            reference_ind = list(set(slcstack["filename"]) - set(interfstack["filename"]))[0]
            reference_ind = slcstack["filename"].index(reference_ind)

            # Build full interferogram stack including reference image
            inf_full = np.zeros((nlines, nwidths, npages+1), dtype=np.complex64)
            if reference_ind > 0:
                inf_full[:, :, :reference_ind] = interfstack_data[:, :, :reference_ind]
                inf_full[:, :, reference_ind+1:] = interfstack_data[:, :, reference_ind:]
                inf_full[:, :, reference_ind] = np.abs(slcstack_data[:, :, reference_ind])
            else:
                inf_full[:, :, 0] = np.abs(slcstack_data[:, :, 0])
                inf_full[:, :, 1:] = interfstack_data

            # Coherence estimation
            RadiusRow = (self.CalWin[0] - 1) // 2
            RadiusCol = (self.CalWin[1] - 1) // 2
            
            Coh = np.zeros((npages + 1, npages + 1, nlines, nwidths), dtype=np.complex64)
            
            # Set diagonal elements to 1 (coherence of an image with itself is 1)
            for ii in range(npages + 1):
                Coh[ii, ii, :, :] = 1.0
            
            # Prepare tasks for parallel processing
            tasks = []
            batch_size = 50  # Process 50 image pairs at a time
            
            # Create all image pairs
            image_pairs = []
            for ii in range(npages + 1):
                for ss in range(ii + 1, npages + 1):
                    image_pairs.append((ii, ss))
            
            # Split image pairs into batches
            for batch_start in range(0, len(image_pairs), batch_size):
                batch_end = min(batch_start + batch_size, len(image_pairs))
                batch_pairs = image_pairs[batch_start:batch_end]
                
                # For each pair in the batch, create tasks for each chunk
                for ii, ss in batch_pairs:
                    slc_ii = slcstack_data[:, :, ii]
                    slc_ss = slcstack_data[:, :, ss]
                    inf_ii = inf_full[:, :, ii]
                    inf_ss = inf_full[:, :, ss]
                    
                    # Process the entire patch width at once
                    tasks.append((ii, ss, 0, nwidths, nlines, nwidths, 
                                RadiusRow, RadiusCol, slc_ii, slc_ss, inf_ii, inf_ss, 
                                shp_pixelind))
            
            # Process batches in parallel
            with ProcessPoolExecutor(max_workers=int(self.CPU)) as executor:
                futures = [executor.submit(self.compute_coherence_chunk, task) for task in tasks]
                
                # Process results as they complete
                for future in as_completed(futures):
                    try:
                        ii, ss, chunk_start, chunk_end, result = future.result()
                        Coh[ii, ss, :, chunk_start:chunk_end] = result
                        Coh[ss, ii, :, chunk_start:chunk_end] = result.conj()
                    except Exception as e:
                        print(f"Error processing chunk: {str(e)}")
                        raise

            all_coherence_matrices.append({
                'coherence': Coh,
                'start_line': slc_patch["start_line"],
                'end_line': slc_patch["end_line"],
                'start_col': slc_patch["start_col"],
                'end_col': slc_patch["end_col"]
            })
            all_reference_indices.append(reference_ind)

        end_time = time.time()
        print(f"   -> Coherence estimation operation completed in {(end_time - start_time)/60.0:.2f} minutes\n")
        return all_coherence_matrices, all_reference_indices
    
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

        master_id = [self.slcstack["filename"].index(f) for f in self.slcstack["filename"] if f not in self.interfstack["filename"]][0]
        for i in range(n_interf):
            slave_id = self.interfstack["filename"][i]
            if self.input.InSAR_processor == 'snap':
                filename = os.path.join(path, f"{self.slcstack['filename'][master_id]}_{slave_id}{extension}")
                fid = open(filename, 'wb')
            elif self.input.InSAR_processor == 'isce':
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
            if self.input.InSAR_processor == 'snap':
                filename = os.path.join(path, f"{self.slcstack['filename'][i]}{extension}")
                fid = open(filename, 'wb')
            elif self.input.InSAR_processor == 'isce':
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
        if self.input.ComSAR_flag:
            print("Step 2: COMSAR estimation\n")
            # Process each patch separately
            for patch_idx, (slc_patch, interf_patch) in enumerate(zip(self.slcstack["datastack"], self.interfstack["datastack"])):
                print(f"-> Processing patch {patch_idx + 1}/{len(self.slcstack['datastack'])}")
                ComSAR(slc_patch,
                       interf_patch,
                       self.shp['patches'][patch_idx],  # Use patch-specific SHP
                       self.input.InSAR_path,
                       self.BroNumthre,
                       self.Cohthre,
                       self.input.miniStackSize,
                       self.Cohthre_slc_filt,
                       self.input.Unified_flag,
                       self.input.InSAR_processor).run()
        else:
            print("Step 2: PSDS estimation\n")
            if not os.path.exists(os.path.join(self.CURRENT_RESULT, "coherence.npz")):
                print("-> Computing SHP-based coherence started...")
                all_coherence_matrices, all_reference_indices = self.intf_cov(self.slcstack, self.interfstack)
                np.savez(os.path.join(self.CURRENT_RESULT, "coherence.npz"), coherence=all_coherence_matrices, reference_indices=all_reference_indices)
            else:
                print("-> Loading existing coherence matrix...")
                all_coherence_matrices = np.load(os.path.join(self.CURRENT_RESULT, "coherence.npz"), allow_pickle=True)['coherence']
                all_reference_indices = np.load(os.path.join(self.CURRENT_RESULT, "coherence.npz"), allow_pickle=True)['reference_indices']
            
            # Check if PSDS results already exist
            psds_result_file = os.path.join(self.CURRENT_RESULT, "psds.npz")
            if os.path.exists(psds_result_file):
                print("-> Loading existing PSDS results...")
                psds_data = np.load(psds_result_file, allow_pickle=True)
                full_slc_despeckle = psds_data['slc_despeckle']
                full_interf_filtered = psds_data['interf_filtered']
            else:
                # Initialize full stacks
                print("-> Refining pixels...")
                nlines = self.slcstack["nlines"]
                ncols = self.slcstack["ncols"]
                npages = self.slcstack["datastack"][0]["datastack"].shape[2]
                npages_interf = self.interfstack["datastack"][0]["datastack"].shape[2]
                
                full_slc_despeckle = np.zeros((nlines, ncols, npages), dtype=np.complex64)
                full_interf_filtered = np.zeros((nlines, ncols, npages_interf), dtype=np.complex64)
                
                for patch_idx, (slc_patch, interf_patch, coh_matrix, ref_idx) in tqdm(enumerate(
                    zip(self.slcstack["datastack"], self.interfstack["datastack"], 
                        all_coherence_matrices, all_reference_indices)), total=len(self.slcstack['datastack']), desc="   -> PSDS estimation", unit="patch"):
                    
                    # Get patch dimensions and positions
                    start_line = slc_patch["start_line"]
                    end_line = slc_patch["end_line"]
                    start_col = slc_patch["start_col"]
                    end_col = slc_patch["end_col"]
                    
                    # Process the patch
                    psds = PSDS(coh_matrix['coherence'], slc_patch,
                             interf_patch,
                             self.slcstack["filename"],
                             self.interfstack["filename"],
                             self.shp[patch_idx],
                             ref_idx,
                             self.input.InSAR_path,
                             self.BroNumthre,
                             self.Cohthre,
                             self.Cohthre_slc_filt,
                             self.input.InSAR_processor)
                    
                    # Get processed results
                    slc_despeckle, interf_filtered = psds.run()
                    
                    # Place results in full stacks
                    full_slc_despeckle[start_line:end_line, start_col:end_col, :] = slc_despeckle
                    full_interf_filtered[start_line:end_line, start_col:end_col, :] = interf_filtered
                
                # Save PSDS results
                print("-> Saving PSDS results...")
                np.savez(psds_result_file,
                        slc_despeckle=full_slc_despeckle,
                        interf_filtered=full_interf_filtered)
            
            # Create output directories if they don't exist
            slc_output_dir = os.path.join(self.input.InSAR_path, 'rslc')
            interf_output_dir = os.path.join(self.input.InSAR_path, 'diff0')
            os.makedirs(slc_output_dir, exist_ok=True)
            os.makedirs(interf_output_dir, exist_ok=True)
            
            # Export SLC files
            print("-> Exporting processed SLC files...")
            self.slcstack["datastack"] = full_slc_despeckle
            self.slc_export(slc_output_dir, '.psar')
            
            # Export interferogram files
            print("-> Exporting processed interferogram files...")
            self.interfstack["datastack"] = full_interf_filtered
            self.interf_export(interf_output_dir, '.psds')

if __name__ == "__main__":
    TomoSARControl().run()