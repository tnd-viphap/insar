# type: ignore
import glob
import os
import sys
import warnings

import numpy as np
from config.parser import ConfigParser

project_path = os.path.abspath(os.path.join(__file__, '../../..')).replace("/config", "")
sys.path.append(project_path)
warnings.filterwarnings("ignore")

class Input:
    def __init__(self, calwin, output=True, project_name="default"):
        # Initialize config parser
        config_path = os.path.join(project_path, "config", "config.json")
        self.config_parser = ConfigParser(config_path)
        self.config = self.config_parser.get_project_config(project_name)

        self.COMSAR_fetch = str(self.config['api_flags']['comsar'])
        self.ComSAR_flag = self.COMSAR_fetch.lower() in ('true', '1')
        print("-> ComSAR Enabled" if self.ComSAR_flag else "-> PSDS Enabled")

        self.miniStackSize = int(self.config['processing_parameters']['ministack'])

        self.Unified_fetch = str(self.config['processing_parameters']['unified'])
        self.Unified_flag = self.Unified_fetch.lower() in ('true', '1')
        print("-> Unified ComSAR Enabled" if self.Unified_flag else "-> Unified ComSAR Disabled")

        self.InSAR_processor = 'snap'
        self.InSAR_path = self.config['processing_parameters']['current_result']

        self.slcstack = {}
        self.interfstack = {}

        self.CalWin = calwin
        self.print_flag = output

    def imgread(self, directory, prefix, nlines, dtype_str='cpxfloat32', extension='', verbose=False):
        """
        Read a stack of binary files with consistent dimensions and type.

        Args:
            directory (str): Directory containing the files.
            prefix (str): Prefix of the filenames to match.
            nlines (int): Number of lines (rows) per image.
            dtype_str (str): Data type (e.g., 'cpxfloat32', 'float32', 'int16').
            extension (str): Optional extension to filter files (e.g., '.rslc', '.diff').
            verbose (bool): If True, print file loading info.

        Returns:
            list: List of 3D numpy arrays (chunk_lines, chunk_cols, num_images)
        """
        # Data type mapping
        dtype_map = {
            'cpxfloat32': np.complex64,
            'float32': np.float32,
            'int16': np.int16,
            'uint8': np.uint8
        }

        if dtype_str not in dtype_map:
            raise ValueError(f"Unsupported dtype_str: {dtype_str}")

        dtype = dtype_map[dtype_str]

        pattern = f"*.{prefix}.{extension}" if extension else f"*.{prefix}"
        file_pattern = os.path.join(directory, pattern)
        files = [f.replace("\\", "/") for f in sorted(glob.glob(file_pattern))]

        if not files:
            raise FileNotFoundError(f"No files found for pattern: {file_pattern}")

        stack = []
        for file in files:
            file = file.replace("\\", "/")
            data = np.fromfile(file, dtype=dtype)
            if len(data) % nlines != 0:
                raise ValueError(f"File {file} does not match expected line size.")

            ncols = len(data) // nlines
            img = data.reshape((nlines, ncols))
            stack.append(img)
            if verbose:
                if self.print_flag:
                    print(f"-> Loaded {file} -> shape ({nlines}, {ncols})")
                    
        if "rslc" == prefix:
            files = [os.path.split(f.replace("\\", "/"))[-1].split(".")[0] for f in sorted(glob.glob(file_pattern))]
            self.slcstack["filename"] = files
        elif "diff" == prefix:
            files = [os.path.split(f.replace("\\", "/"))[-1].split(".")[0].split("_")[-1] for f in sorted(glob.glob(file_pattern))]
            self.interfstack["filename"] = files

        # Calculate chunk dimensions
        chunk_lines = self.CalWin[0] * 3
        chunk_cols = self.CalWin[1] * 3
        
        # Calculate number of chunks in each dimension
        n_chunks_lines = (nlines + chunk_lines - 1) // chunk_lines
        n_chunks_cols = (ncols + chunk_cols - 1) // chunk_cols
        
        # Create chunks
        chunked_stack = []
        for i in range(n_chunks_lines):
            for j in range(n_chunks_cols):
                start_line = i * chunk_lines
                end_line = min((i + 1) * chunk_lines, nlines)
                start_col = j * chunk_cols
                end_col = min((j + 1) * chunk_cols, ncols)
                
                # Extract chunk from each image
                chunk = np.stack([img[start_line:end_line, start_col:end_col] for img in stack], axis=-1)
                chunked_stack.append({
                    'datastack': chunk,
                    'start_line': start_line,
                    'end_line': end_line,
                    'start_col': start_col,
                    'end_col': end_col
                })
                
        return chunked_stack, nlines, ncols

    def run(self):
        if self.InSAR_processor == 'snap':
            reference_date = self.InSAR_path.strip().split('/')[-1].split('_')[1]
            
            file_par = f"{self.InSAR_path}/rslc/{reference_date}.rslc.par"
            
            with open(file_par, 'r') as f:
                lines = f.readlines()
            
            # Find line containing 'zimuth_lines'
            nlines = None
            for line in lines:
                if 'zimuth_lines' in line:
                    nlines = int(line.strip().split()[-1])
                    break

            # Load image stacks in chunks
            _slcstack_chunks, nlines, ncols = self.imgread(f"{self.InSAR_path}/rslc", 'rslc', nlines, 'cpxfloat32', verbose=True)
            self.slcstack["datastack"] = _slcstack_chunks
            self.slcstack["nlines"] = nlines
            self.slcstack["ncols"] = ncols
            
            _interfstack_chunks, _, _ = self.imgread(f"{self.InSAR_path}/diff0", 'diff', nlines, 'cpxfloat32', verbose=True)
            self.interfstack["datastack"] = _interfstack_chunks
            self.interfstack["nlines"] = nlines
            self.interfstack["ncols"] = ncols

        else:
            print("not yet supported")

        if self.slcstack["datastack"] is not None and self.interfstack["datastack"] is not None:
            return self.slcstack, self.interfstack
        else:
            raise AssertionError('Can not broadcast enough data')

if __name__ == "__main__":
    slcstack, interfstack = Input([7, 25], project_name="default").run()
    print(slcstack["datastack"][0]["data"].shape)

