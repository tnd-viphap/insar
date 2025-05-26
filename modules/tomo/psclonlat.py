import os
import sys
import numpy as np
import struct

class PSLonLat:
    def __init__(self, parm_file, ij_file=None, ll_file=None, log_file=None):
        """
        Initialize PS LonLat extraction
        
        Args:
            parm_file (str): Path to parameter file containing width and lon/lat file paths
            ij_file (str, optional): Path to PS candidate locations file. Defaults to 'pscands.1.ij'
            ll_file (str, optional): Path to output lon/lat file. Defaults to 'pscands.1.ll'
            log_file (str, optional): Path to log file. If None, no logging will be performed.
        """
        self.parm_file = parm_file
        self.ij_file = ij_file or 'pscands.1.ij'
        self.ll_file = ll_file or 'pscands.1.ll'
        self.log_file = log_file

        # Initialize variables
        self.width = None
        self.lon_file = None
        self.lat_file = None
        
        # Sun raster file magic number
        self.SUN_RASTER_MAGIC = 0x59a66a95

    def _log(self, message, level="INFO"):
        """Write message to log file with timestamp"""
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{message}\n")
        print(message)  # Also print to console
            
    def read_parm_file(self):
        """Read parameter file to get width and lon/lat file paths"""
        try:
            with open(self.parm_file, 'r') as f:
                # Read width
                self.width = int(f.readline().strip())
                
                # Read lon file path
                self.lon_file = f.readline().strip()
                
                # Read lat file path
                self.lat_file = f.readline().strip()
                
        except Exception as e:
            self._log(f"Error reading parameter file: {str(e)}", "ERROR")
            sys.exit(1)
            
    def read_ps_candidates(self):
        """Read PS candidate locations from input file"""
        try:
            # Read PS candidate locations
            ps_data = np.loadtxt(self.ij_file, dtype=int)
            return ps_data
        except Exception as e:
            self._log(f"Error reading PS candidate file: {str(e)}", "ERROR")
            sys.exit(1)
            
    def check_sun_raster_header(self, file_path):
        """Check if file has Sun raster header and skip if present"""
        try:
            with open(file_path, 'rb') as f:
                # Read first 32 bytes
                header = f.read(32)
                if len(header) >= 4:
                    # Check magic number (first 4 bytes)
                    magic = struct.unpack('>I', header[:4])[0]
                    if magic == self.SUN_RASTER_MAGIC:
                        self._log("Sun raster file detected - skipping header")
                        return True
            return False
        except Exception as e:
            self._log(f"Error checking Sun raster header: {str(e)}", "ERROR")
            return False
            
    def read_lonlat_files(self):
        """Read lon/lat binary files"""
        try:
            # Check for Sun raster header in lon file
            has_header_lon = self.check_sun_raster_header(self.lon_file)
            has_header_lat = self.check_sun_raster_header(self.lat_file)
            
            # Read lon/lat files as binary
            lon_data = np.fromfile(self.lon_file, dtype=np.float32)
            lat_data = np.fromfile(self.lat_file, dtype=np.float32)
            
            # Skip header if present
            if has_header_lon:
                lon_data = lon_data[8:]  # Skip 32 bytes (8 float32 values)
            if has_header_lat:
                lat_data = lat_data[8:]  # Skip 32 bytes (8 float32 values)
            
            # Reshape to 2D arrays based on width
            lon_data = lon_data.reshape(-1, self.width)
            lat_data = lat_data.reshape(-1, self.width)
            
            return lon_data, lat_data
        except Exception as e:
            self._log(f"Error reading lon/lat files: {str(e)}", "ERROR")
            sys.exit(1)
            
    def extract_lonlat(self):
        """Extract lon/lat values for PS candidates"""
        # Read parameter file
        self.read_parm_file()
        self._log(f"Width: {self.width}")
        self._log(f"Lon file: {self.lon_file}")
        self._log(f"Lat file: {self.lat_file}")
        
        # Read PS candidates
        ps_data = self.read_ps_candidates()
        
        # Read lon/lat files
        lon_data, lat_data = self.read_lonlat_files()
        
        # Extract lon/lat values for each PS candidate
        lonlat_values = []
        for pscid, y, x in ps_data:
            # Convert to 0-based indexing
            y_idx = y - 1
            x_idx = x - 1
            
            # Get lon/lat values
            lon = lon_data[y_idx, x_idx]
            lat = lat_data[y_idx, x_idx]
            
            lonlat_values.append([lon, lat])
            
            # Print progress every 100,000 candidates
            if pscid % 100000 == 0:
                self._log(f"{pscid} PS candidates processed")
                
        return np.array(lonlat_values)
    
    def write_output(self, lonlat_values):
        """Write lon/lat values to output file"""
        try:
            # Write only lon/lat values in binary format
            with open(self.ll_file, 'wb') as f:
                # Write each lon/lat pair as two float32 values
                for lon, lat in lonlat_values:
                    f.write(struct.pack('ff', lon, lat))
            self._log(f"Successfully wrote lon/lat values to {self.ll_file}")
        except Exception as e:
            self._log(f"Error writing output file: {str(e)}", "ERROR")
            sys.exit(1)
            
    def run(self):
        """Main execution method"""
        self._log(f"Processing PS candidates from {self.ij_file}")
        
        # Extract lon/lat values
        lonlat_values = self.extract_lonlat()
        
        # Write output
        self.write_output(lonlat_values)

def main():
    if len(sys.argv) < 2:
        print("Usage: psclonlat.py parmfile [pscands.1.ij] [pscands.1.ll]")
        print("\nInput parameters:")
        print("  parmfile   (input)  width of lon/lat files (range bins)")
        print("                      name of lon file (float)")
        print("                      name of lat file (float)")
        print("  pscands.1.ij (input)  location of permanent scatterer candidates")
        print("  pscands.1.ll (output) lon/lat of permanent scatterer candidates")
        sys.exit(1)
        
    parm_file = sys.argv[1]
    ij_file = sys.argv[2] if len(sys.argv) > 2 else None
    ll_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    ps_lonlat = PSLonLat(parm_file, ij_file, ll_file)
    ps_lonlat.run()

if __name__ == "__main__":
    main() 