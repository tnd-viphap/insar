import glob
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_path)

from modules.tomo.extract_pixels import MTExtractCands
from modules.tomo.ps_parms import Parms


class PSDS_Prep:
    def __init__(self, master_date, data_dir, da_thresh=None, rg_patches=1, az_patches=1, 
                 rg_overlap=50, az_overlap=50, maskfile=None):
        """
        Initialize PSDS preparation class
        
        Args:
            master_date (str): Master date in YYYYMMDD format
            data_dir (str): Data directory path
            da_thresh (float): Amplitude dispersion threshold (default: 0.4 for PS, 0.6 for SB)
            rg_patches (int): Number of patches in range (default: 1)
            az_patches (int): Number of patches in azimuth (default: 1)
            rg_overlap (int): Overlapping pixels between patches in range (default: 50)
            az_overlap (int): Overlapping pixels between patches in azimuth (default: 50)
            maskfile (str): Optional mask file path
        """
        self.conf_path = Path(os.path.join(project_path, "modules/snap2stamps/bin/project.conf"))
        self._load_config()
        if platform.system() == "Linux":
            self.calamp_path = Path(os.path.join(project_path, "modules/StaMPS/bin/calamp"))
        else:
            self.calamp_path = Path(os.path.join(project_path, "modules/StaMPS/src/calamp.exe"))

        self.master_date = master_date
        self.data_dir = Path(data_dir.replace('\\', '/'))
        self.rg_patches = int(rg_patches)
        self.az_patches = int(az_patches)
        self.rg_overlap = int(rg_overlap)
        self.az_overlap = int(az_overlap)
        self.maskfile = maskfile
        self.work_dir = self.data_dir
        
        # Set up log file
        log_dir = self.work_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_dir / f"psds_prep_{timestamp}.log"
        
        # Check if small baseline processing
        self.sb_flag = (self.data_dir / "SMALL_BASELINES").exists()
        
        # Set default da_thresh based on processing type
        if da_thresh is None:
            self.da_thresh = 0.6 if self.sb_flag else 0.4
        else:
            self.da_thresh = da_thresh
            
        # Initialize other attributes
        self.width = None
        self.length = None
        self.rsc_file = None

    def _load_config(self):
        with open(self.conf_path, 'r') as file:
            for line in file.readlines():
                key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                if key:
                    setattr(self, key, value)  # Dynamically set variables
        
    def find_rsc_file(self):
        """Find the RSC file based on processing type"""
        if self.sb_flag:
            rsc_files = list(self.data_dir.glob(f"SMALL_BASELINES/*/{self.master_date}.*slc.par"))
            if rsc_files:
                self.rsc_file = str(rsc_files[-1]).replace('\\', '/')
        else:
            rsc_files = list(self.data_dir.glob(f"*slc/{self.master_date}.*slc.par"))
            if rsc_files:
                self.rsc_file = str(rsc_files[0]).replace('\\', '/')
                
        if not self.rsc_file:
            raise FileNotFoundError("Could not find RSC file")
            
    def get_dimensions(self):
        """Get width and length from RSC file"""
        with open(self.rsc_file, 'r') as f:
            for line in f:
                if 'range_samples' in line:
                    self.width = int(line.split()[1])
                elif 'azimuth_lines' in line:
                    self.length = int(line.split()[1])
                    
        if not self.width or not self.length:
            raise ValueError("Could not determine image dimensions from RSC file")
            
    def create_patches(self):
        """Create patch directories and configuration files"""
        width_p = self.width // self.rg_patches
        length_p = self.length // self.az_patches
        
        # Remove existing patch directories
        for patch_dir in self.work_dir.glob("PATCH_*"):
            if patch_dir.is_dir():
                import shutil
                shutil.rmtree(patch_dir)
                
        # Create patch list file
        with open(self.work_dir / "patch.list", 'w') as f:
            pass
            
        patch_num = 0
        for irg in range(self.rg_patches):
            for iaz in range(self.az_patches):
                patch_num += 1
                
                # Calculate patch boundaries
                start_rg1 = width_p * irg + 1
                start_rg = max(1, start_rg1 - self.rg_overlap)
                end_rg1 = width_p * (irg + 1)
                end_rg = min(self.width, end_rg1 + self.rg_overlap)
                
                start_az1 = length_p * iaz + 1
                start_az = max(1, start_az1 - self.az_overlap)
                end_az1 = length_p * (iaz + 1)
                end_az = min(self.length, end_az1 + self.az_overlap)
                
                # Create patch directory
                patch_dir = self.work_dir / f"PATCH_{patch_num}"
                patch_dir.mkdir(exist_ok=True)
                
                # Write patch configuration files
                with open(patch_dir / "patch.in", 'w') as f:
                    f.write(f"{start_rg}\n{end_rg}\n{start_az}\n{end_az}\n")
                    
                with open(patch_dir / "patch_noover.in", 'w') as f:
                    f.write(f"{start_rg1}\n{end_rg1}\n{start_az1}\n{end_az1}\n")
                    
                with open(self.work_dir / "patch.list", 'a') as f:
                    f.write(f"PATCH_{patch_num}\n")
                    
    def prepare_psds_files(self):
        """Prepare PSDs files for processing"""
        # Write width to pscphase.in
        with open(self.work_dir / "pscphase.in", 'w') as f:
            f.write(f"{self.width}\n")
            if self.sb_flag:
                psds_files = list(self.data_dir.glob("SMALL_BASELINES/*/*.psds"))
            else:
                psds_files = list(self.data_dir.glob("diff0/*.psds"))
            for psds_file in psds_files:
                psds_file = str(psds_file).replace('\\', '/')
                if platform.system() == "Linux":
                    f.write(f"{psds_file}\n")
                else:
                    f.write("    "+psds_file+"\n")
                
        # Write width to pscdem.in
        with open(self.work_dir / "pscdem.in", 'w') as f:
            f.write(f"{self.width}\n")
            dem_files = list(self.data_dir.glob("geo/*dem.rdc"))
            for dem_file in dem_files:
                dem_file = str(dem_file).replace('\\', '/')
                f.write(f"{dem_file}\n")

        # Write width to psclonlat.in
        with open(self.work_dir / "psclonlat.in", 'w') as f:
            f.write(f"{self.width}\n")
            lon_files = list(self.data_dir.glob("geo/*.lon"))
            lat_files = list(self.data_dir.glob("geo/*.lat"))
            for lon_file in lon_files:
                lon_file = str(lon_file).replace('\\', '/')
                f.write(f"{lon_file}\n")
            for lat_file in lat_files:
                lat_file = str(lat_file).replace('\\', '/')
                f.write(f"{lat_file}\n")
        
                
    def run(self):
        """Main execution method"""
        # try:
        start_time = time.time()
        self.find_rsc_file()
        self.get_dimensions()
        
        # Write processor type
        with open(self.work_dir / "processor.txt", 'w') as f:
            f.write("snap\n")

        # Write initial parameters
        parms = Parms(self.conf_path)
        parms.initialize()

        # Calibrate amplitudes
        selfile = self.work_dir / "selsbc.in" if self.sb_flag else self.work_dir / "selpsc.in"
        if os.path.exists(selfile):
            os.remove(selfile)
        with open(self.work_dir / "calamp.in", 'w') as f:
            for file in glob.glob(f"{self.data_dir}/*slc/*.psar"):
                file = file.replace('\\', '/')
                f.write(f"{file}\n")
        command = f"{self.calamp_path} {self.work_dir}/calamp.in {self.width} {self.work_dir}/calamp.out f 1 {self.maskfile}"
        
        # Run calamp and log its output
        with open(self.log_file, 'w') as log:
            log.write(f"=== Running calamp command ===\n")
            log.write(f"Command: {command}\n\n")
            log.write("Output:\n")
            try:
                # Use subprocess.run instead of Popen for better control
                process = subprocess.run(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    check=False,  # Don't raise exception on non-zero return code
                    creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
                )
                
                if process.stdout:
                    log.write(process.stdout)
                if process.stderr:
                    log.write(f"\nErrors/Warnings:\n{process.stderr}")
                log.write(f"\nReturn code: {process.returncode}\n")
                
                if process.returncode != 0:
                    print(f"Error: calamp failed with return code {process.returncode}")
                    print(f"Check log file for details: {self.log_file}")
                    sys.exit(1)
                    
            except Exception as e:
                error_msg = f"Error executing calamp command: {str(e)}"
                print(error_msg)
                log.write(f"\n{error_msg}\n")
                sys.exit(1)
            
        # Write dimensions to files
        with open(self.work_dir / "width.txt", 'w') as f:
            f.write(f"{self.width}\n")
        with open(self.work_dir / "len.txt", 'w') as f:
            f.write(f"{self.length}\n")
        with open(self.work_dir / "rsc.txt", 'w') as f:
            f.write(f"{self.rsc_file}\n")
        with open(self.work_dir / selfile, 'w') as f:
            f.write(f"{self.da_thresh}\n")
            f.write(f"{self.width}\n")
            with open(self.work_dir / "calamp.out", 'r') as f2:
                for line in f2.readlines():
                    if platform.system() == "Linux":
                        f.write(line)
                    else:
                        f.write("     "+line)
            
        self.create_patches()
        self.prepare_psds_files()
        
        print(f"-> Processing {self.rg_patches} patch(es) in range and {self.az_patches} in azimuth")
        print(f"-> Amplitude Dispersion Threshold: {self.da_thresh}")

        # Run mt_extract_cands
        print("-> Extracting candidate PS pixels...")
        mt_extract_cands = MTExtractCands()
        mt_extract_cands.run()

        end_time = time.time()
        print(f"-> Total preparation time: {(end_time - start_time)/60:.2f} minutes")
            
        # except Exception as e:
        #     print(f"Error: {str(e)}")
        #     sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: psds_prep.py yyyymmdd datadir da_thresh [rg_patches az_patches rg_overlap az_overlap maskfile]")
        print("    yyyymmdd                 = master date")
        print("    datadir                  = data directory (with expected structure)")
        print("    da_thresh                = (delta) amplitude dispersion threshold")
        print("                                typical values: 0.4 for PS, 0.6 for SB")
        print("    rg_patches (default 1)   = number of patches in range")
        print("    az_patches (default 1)   = number of patches in azimuth")
        print("    rg_overlap (default 50)  = overlapping pixels between patches in range")
        print("    az_overlap (default 50) = overlapping pixels between patches in azimuth")
        print("    maskfile (optional) char file, same dimensions as slcs, 0 to include, 1 otherwise")
        sys.exit(4)
        
    master_date = sys.argv[1]
    data_dir = sys.argv[2]
    da_thresh = float(sys.argv[3]) if len(sys.argv) > 3 else None
    rg_patches = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    az_patches = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    rg_overlap = int(sys.argv[6]) if len(sys.argv) > 6 else 50
    az_overlap = int(sys.argv[7]) if len(sys.argv) > 7 else 50
    maskfile = sys.argv[8] if len(sys.argv) > 8 else None
    
    prep = PSDS_Prep(master_date, data_dir, da_thresh, rg_patches, az_patches, 
                     rg_overlap, az_overlap, maskfile)
    prep.run()