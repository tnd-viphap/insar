import glob
import os
import sys
import warnings

import numpy as np

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(project_path)
warnings.filterwarnings("ignore")

class Input:
    def __init__(self, output=True):

        self.project_conf = os.path.join(project_path, "modules/snap2stamps/bin/project.conf").replace("\\", "/")

        self.COMSAR_fetch = self.read_conf_value('COMSAR')
        self.ComSAR_flag = self.COMSAR_fetch.lower() in ('true', '1')
        print("-> ComSAR Enabled" if self.ComSAR_flag else "-> PSDS Enabled")

        self.miniStackSize = int(self.read_conf_value('MINISTACK'))

        self.Unified_fetch = self.read_conf_value('UNIFIED')
        self.Unified_flag = self.Unified_fetch.lower() in ('true', '1')
        print("-> Unified ComSAR Enabled" if self.Unified_flag else "-> Unified ComSAR Disabled")

        self.InSAR_processor = 'snap'
        self.InSAR_path = self.read_conf_value('CURRENT_RESULT')

        self.slcstack = {}
        self.interfstack = {}

        self.print_flag = output

    def read_conf_value(self, key):
        """Reads the value of a key from a configuration file."""
        try:
            with open(self.project_conf, 'r') as file:
                for line in file:
                    if line.strip().startswith(key):
                        parts = line.split('=')
                        if len(parts) > 1:
                            return parts[1].strip()
            print(f"WARNING: {key} not found in the file.")
            return ''
        except FileNotFoundError:
            raise FileNotFoundError("Cannot open the file.")
        
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
            np.ndarray: 3D numpy array (nlines, ncols, num_images)
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

        return np.stack(stack, axis=-1)

    
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

            # Load image stacks
            _slcstack = self.imgread(f"{self.InSAR_path}/rslc", 'rslc', nlines, 'cpxfloat32', verbose=True)
            self.slcstack["datastack"] = _slcstack
            _interfstack = self.imgread(f"{self.InSAR_path}/diff0", 'diff', nlines, 'cpxfloat32', verbose=True)
            self.interfstack["datastack"] = _interfstack

        else:
            print("not yet supported")

        if self.slcstack["datastack"] is not None and self.interfstack["datastack"] is not None:
            return self.slcstack, self.interfstack
        else:
            raise AssertionError('Can not broadcast enough data')

if __name__ == "__main__":
    # test_ImgRead()
    # None
    slcstack, interfstack = Input().run()

