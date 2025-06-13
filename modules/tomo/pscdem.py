#!/usr/bin/env python3
"""
pscdem.py: Extract height for PS Candidates
Author: Based on original C++ code by Andy Hooper
"""

import numpy as np
import os
import sys
from pathlib import Path
from typing import Tuple, Optional, Union

class PSCDEM:
    def __init__(self, parmfile: str, ij_file: str = "pscands.1.ij", 
                 out_file: str = "pscands.1.hgt", precision: str = "f", log_file: str = None):
        """
        Initialize PscDEM class
        
        Args:
            parmfile: Path to parameter file containing width and DEM filename
            ij_file: Path to PS candidates ij file
            out_file: Path to output height file
            precision: 'f' for float32 or 'd' for float64
        """
        self.parmfile = parmfile
        self.ij_file = ij_file
        self.out_file = out_file
        self.precision = precision
        self.width = None
        self.dem_file = None
        self.pscands = None
        self.dem_data = None
        self.heights = None
        self.log_file = log_file
        # Initialize parameters
        self._read_parameters()
        
    def _read_parameters(self) -> None:
        """Read parameters from parmfile"""
        try:
            with open(self.parmfile, 'r') as f:
                self.width = int(f.readline().strip())
                self.dem_file = f.readline().strip()
            if self.log_file:
                with open(self.log_file, 'a') as log:
                    log.write(f"Width: {self.width}\n")
                    log.write(f"DEM file: {self.dem_file}\n")
        except Exception as e:
            print(f"Error reading parameter file {self.parmfile}: {e}")
            sys.exit(1)

    def read_pscands_ij(self) -> np.ndarray:
        """
        Read PS candidate locations from ij file
        Returns: numpy array of [pscid, y, x] coordinates
        """
        try:
            self.pscands = np.loadtxt(self.ij_file, dtype=int)
            if self.log_file:
                with open(self.log_file, 'a') as log:
                    log.write(f"Read {len(self.pscands)} PS candidates from {self.ij_file}\n")
            return self.pscands
        except Exception as e:
            print(f"Error reading ij file {self.ij_file}: {e}")
            sys.exit(1)

    def read_dem(self) -> np.ndarray:
        """
        Read DEM file
        Returns: 2D numpy array of DEM data
        """
        try:
            dtype = np.float32 if self.precision == 'f' else np.float64
            # Check if file exists
            if not os.path.exists(self.dem_file):
                if self.log_file:
                    with open(self.log_file, 'a') as log:
                        log.write(f"Warning: DEM file {self.dem_file} not found - using zeros\n")
                self.dem_data = np.zeros((1, self.width), dtype=dtype)
                return self.dem_data
                
            # Read DEM file as binary
            with open(self.dem_file, 'rb') as f:
                # Check for sun raster header
                header = f.read(32)
                if int.from_bytes(header[:4], byteorder='little') == 0x59a66a95:
                    if self.log_file:
                        with open(self.log_file, 'a') as log:
                            log.write("pscdem: sun raster file - skipping header\n")
                else:
                    f.seek(0)  # Reset to start if not sun raster
                
                # Read binary data
                dem_data = np.frombuffer(f.read(), dtype=dtype)
                
            # Reshape to 2D array
            height = dem_data.size // self.width
            self.dem_data = dem_data.reshape(height, self.width)
            return self.dem_data
        except Exception as e:
            if self.log_file:
                with open(self.log_file, 'a') as log:
                    log.write(f"Error reading DEM file {self.dem_file}: {e}\n")
            sys.exit(1)

    def extract_heights(self) -> np.ndarray:
        """
        Extract heights for PS candidates from DEM
        Returns: Array of heights corresponding to PS candidates
        """
        if self.pscands is None:
            self.read_pscands_ij()
        if self.dem_data is None:
            self.read_dem()
            
        self.heights = np.zeros(len(self.pscands), dtype=self.dem_data.dtype)
        for i, (_, y, x) in enumerate(self.pscands):
            if 0 <= y < self.dem_data.shape[0] and 0 <= x < self.dem_data.shape[1]:
                self.heights[i] = self.dem_data[y, x]
        return self.heights

    def save_heights(self) -> None:
        """Save extracted heights to output file as binary"""
        if self.heights is None:
            self.extract_heights()
            
        try:
            # Save as binary data
            with open(self.out_file, 'wb') as f:
                f.write(self.heights.tobytes())
            if self.log_file:
                with open(self.log_file, 'a') as log:
                    log.write(f"Saved heights to {self.out_file}\n")
        except Exception as e:
            if self.log_file:
                with open(self.log_file, 'a') as log:
                    log.write(f"Error saving heights to {self.out_file}: {e}\n")
            sys.exit(1)

    def run(self) -> None:
        """Process the entire workflow"""
        self.read_pscands_ij()
        self.read_dem()
        self.extract_heights()
        self.save_heights()

def main():
    if len(sys.argv) < 2:
        print("Usage: python pscdem.py parmfile [pscands.1.ij] [pscands.1.hgt] [precision]")
        print("\nInput parameters:")
        print("  parmfile   (input)  width of dem files (range bins)")
        print("                      name of dem file (radar coords, float)")
        print("  pscands.1.ij (input)  location of PS candidates")
        print("  pscands.1.hgt (output) height of PS candidates")
        print("  precision(input) d or f (default)")
        sys.exit(1)

    # Parse arguments
    parmfile = sys.argv[1]
    ij_file = sys.argv[2] if len(sys.argv) > 2 else "pscands.1.ij"
    out_file = sys.argv[3] if len(sys.argv) > 3 else "pscands.1.hgt"
    precision = sys.argv[4] if len(sys.argv) > 4 else "f"

    # Create PscDEM instance and process
    pscdem = PscDEM(parmfile, ij_file, out_file, precision)
    pscdem.process()

if __name__ == "__main__":
    main() 