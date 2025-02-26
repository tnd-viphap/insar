import glob
import os
import platform
import subprocess
import sys
import time

import numpy as np

from _9_2_mt_extract_cands import MTExtractCands


class MTPrepSNAP:
    def __init__(self, da_threshold, patch_info, maskfile=None):
        super().__init__()
        
        self.plf = platform.system()
        
        self.inputfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], "project.conf")
        self._load_config()
        
        os.chdir(self.CURRENT_RESULT)
        
        self.master = os.path.split(self.MASTER)[1].split("_")[0]
        
        self.patch_info = patch_info
        
        self.prg = int(self.patch_info[0]) if len(sys.argv) > 4 else 1
        self.paz = int(self.patch_info[1]) if len(sys.argv) > 4 else 1
        self.overlap_rg = int(self.patch_info[2]) if len(sys.argv) > 4 else 50
        self.overlap_az = int(self.patch_info[-1]) if len(sys.argv) > 4 else 50
        
        self.maskfile = maskfile if len(sys.argv) > 5 else ""
        if maskfile and not os.path.exists(maskfile):
            print(f"{maskfile} does not exist, exiting")
            sys.exit(2)
        
        self.sb_flag = 1 if os.path.exists(f"{self.CURRENT_RESULT}/SMALL_BASELINES") else 0
        rsc_files = glob.glob(f"{self.CURRENT_RESULT}/SMALL_BASELINES/*/{self.master}.*slc.par" if self.sb_flag else f"{self.CURRENT_RESULT}/*slc/{self.master}.*slc.par")
        
        if not rsc_files:
            print("No RSC file found, exiting.")
            sys.exit(3)
        
        self.rsc_file = rsc_files[-1]
        self.da_thresh = float(da_threshold) if len(sys.argv) > 3 else (0.6 if self.sb_flag else 0.4)
        
        print(f"Amplitude Dispersion Threshold: {self.da_thresh}")
        print(f"Processing {self.prg} patch(es) in range and {self.paz} in azimuth\n")
        print(f"Platform: {self.plf}\n")
        
        if self.plf == "Windows":
            disk, project = self.PROJECTFOLDER.split(":")
            
            # Replace file path
            self.replace_in_wsl_file(f'/mnt/{disk.lower()}{project}modules/StaMPS/StaMPS_CONFIG.bash', 'STAMPS', f'"/mnt/{disk.lower()}{project}modules/StaMPS"')
            self.replace_in_wsl_file(f'/mnt/{disk.lower()}{project}modules/StaMPS/StaMPS_CONFIG.bash', 'TRIANGLE_BIN', f'/mnt/{disk.lower()}{project}modules/triangle/bin')
            self.replace_in_wsl_file(f'/mnt/{disk.lower()}{project}modules/StaMPS/StaMPS_CONFIG.bash', 'SNAPHU_BIN', f'/mnt/{disk.lower()}{project}modules/snaphu-1.4.2/bin')
            self.replace_in_wsl_file(f'/mnt/{disk.lower()}{project}modules/TRAIN/APS_CONFIG.sh', 'APS_toolbox', f'/mnt/{disk.lower()}{project}modules/TRAIN/')
            print("\n")
            time.sleep(1)
            
            # Add modules config files to Linux
            print("Adding StaMPS CONFIG and APS_CONFIG to .bashrc...")
            lines = [
                f'source /mnt/{disk.lower()}{project}modules/StaMPS/StaMPS_CONFIG.bash',
                f'source /mnt/{disk.lower()}{project}modules/TRAIN/APS_CONFIG.sh',
            ]
            self.add_to_wsl_bashrc(lines)
            print("\n")
        
    def _load_config(self):
        with open(self.inputfile, 'r') as file:
            for line in file.readlines():
                key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                if key:
                    setattr(self, key, value)  # Dynamically set variables
                    
    def replace_in_wsl_file(self, file_path, key, new_value):
        """
        Replaces a line in a WSL file where a specific key exists, keeping the format 'key=value'.

        :param file_path: Path to the file inside WSL (e.g., ~/.bashrc)
        :param key: The key (variable name) to search for
        :param new_value: The new value to assign to the key
        """
        # Construct the sed command to replace the line dynamically
        replace_command = f"wsl bash -c \"sed -i 's|^export {key}=.*|export {key}={new_value}|' {file_path}\""

        # Run the command in WSL
        result = subprocess.run(replace_command, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"-> Successfully configured '{key}' in {file_path} to '{new_value}'")
        else:
            print("Error:", result.stderr)
                    
    def add_to_wsl_bashrc(self, lines):
        """
        Appends lines to the ~/.bashrc file inside WSL.

        :param lines: List of strings to be added to .bashrc
        """
        wsl_bashrc_path = "/home/$USER/.bashrc"
        
        # Read the existing .bashrc file
        check_command = f"wsl bash -c \"grep -Fxq '{lines[0]}' {wsl_bashrc_path} && echo 'FOUND' || echo 'NOT_FOUND'\""
        result = subprocess.run(check_command, shell=True, capture_output=True, text=True)

        if not "NOT_FOUND" in result.stdout:
            print("Entries already exist in .bashrc. Skipping addition...")
            return

        # Escape special characters to prevent syntax issues
        escaped_lines = [line.replace('"', '\\"').replace("'", "'\\''") for line in lines]

        # Convert list of lines into a formatted string with newline handling
        formatted_lines = "\\n".join(escaped_lines)

        # WSL command to append lines to .bashrc
        append_command = f"wsl bash -c \"echo -e '{formatted_lines}' >> {wsl_bashrc_path}\""

        # Run the command
        result = subprocess.run(append_command, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Successfully added lines to {wsl_bashrc_path}")
        else:
            print("Error:", result.stderr)

    def _run_command(self, command):
        """Runs a shell command and returns the output."""
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error executing command: {command}\n{result.stderr}")
            sys.exit(result.returncode)
        return result.stdout.strip()
    
    def read_rslc_and_compute_mean(self, rslc_files, width, output_file):
        with open(output_file, "w") as out_f:
            for idx, rslc_file in enumerate(rslc_files):
                if not os.path.exists(rslc_file):
                    print(f"Warning: {rslc_file} not found, skipping.")
                    continue
                try:
                    # Read binary RSLC file as complex floats
                    data = np.fromfile(rslc_file, dtype=np.complex64)
                    swapped_data = data.byteswap()
                    swapped_data = swapped_data.reshape(-1, width)
                    amplitudes = np.abs(swapped_data)
                    
                    # Filter out near-zero values (threshold 0.001 as in C code)
                    valid_pixels = amplitudes[amplitudes > 0.001]
                    
                    # Compute mean amplitude
                    mean_amplitude = np.mean(valid_pixels) if valid_pixels.size > 0 else 0
                    
                    # Write to output file
                    out_f.write(f"{rslc_file} {mean_amplitude:.6f}\n")
                    
                    print(f"-> [{idx+1}] {rslc_file}, Mean Amplitude: {mean_amplitude:.6f}")
                except Exception as e:
                    print(f"Error processing {rslc_file}: {e}")
    
    def calibrate_amplitudes(self):
        """Creates calamp.in, runs calamp command, and generates calamp.out."""
        calamp_in_path = os.path.join(self.CURRENT_RESULT, "calamp.in").replace("\\", "/")
        calamp_out_path = os.path.join(self.CURRENT_RESULT, "calamp.out").replace("\\", "/")
        
        with open(calamp_in_path, "w") as f:
            if self.sb_flag:
                slc_files = glob.glob(f"{self.CURRENT_RESULT}/SMALL_BASELINES/*/*.*slc")
            else:
                slc_files = glob.glob(f"{self.CURRENT_RESULT}/*slc/*.*slc")
            f.write("\n".join([f.replace("\\", "/") for f in slc_files]))
        
        if self.plf == "Windows":
            self.read_rslc_and_compute_mean([f.replace("\\", "/") for f in slc_files], self.width, calamp_out_path)
        else:
            calamp_cmd = f"calamp {calamp_in_path} {self.width} {calamp_out_path} f 1 {self.maskfile}"
            subprocess.run(calamp_cmd, shell=True)
        
        time.sleep(3)

    def process(self):
        if len(sys.argv) < 1:
            print("usage: mt_prep_snap yyyymmdd datadir da_thresh [rg_patches az_patches rg_overlap az_overlap maskfile]")
            sys.exit(4)
        
        if self.plf == "Windows":
            self.width = int(self._run_command(f"wsl gawk '/range_samples/ {{print $2}}' < {self.rsc_file}"))
            self.length = int(self._run_command(f"wsl gawk '/azimuth_lines/ {{print $2}}' < {self.rsc_file}"))
        else:
            self.width = int(self._run_command(f"gawk '/range_samples/ {{print $2}}' < {self.rsc_file}"))
            self.length = int(self._run_command(f"gawk '/azimuth_lines/ {{print $2}}' < {self.rsc_file}"))
        
        with open(f"{self.CURRENT_RESULT}/processor.txt", "w") as f:
            f.write("snap\n")
        if self.plf == "Windows":
            matlab_script = f"{self.PROJECTFOLDER}modules/StaMPS/matlab/sb_parms_initial.m" if self.sb_flag else f"{self.PROJECTFOLDER}modules/StaMPS/matlab/ps_parms_initial_windows.m"
        else:
            matlab_script = f"{self.PROJECTFOLDER}modules/StaMPS/matlab/sb_parms_initial.m" if self.sb_flag else f"{self.PROJECTFOLDER}modules/StaMPS/matlab/ps_parms_initial.m"
        os.system(f"matlab -nojvm -nosplash -nodisp -r \"run('{matlab_script}'); exit;\" > {self.CURRENT_RESULT}/ps_parms_initial.log")
        
        time.sleep(3)
        
        with open(f"{self.CURRENT_RESULT}/width.txt", "w") as file:
            file.write(str(self.width))
            file.close()
        with open(f"{self.CURRENT_RESULT}/len.txt", "w") as file:
            file.write(str(self.length))
            file.close()
        with open(f"{self.CURRENT_RESULT}/rsc.txt", "w") as file:
            file.write(self.rsc_file.replace("\\", "/"))
            file.close()
            
        # Calibrate
        print("Calibrating amplitude...")
        self.calibrate_amplitudes()
        
        selfile_path = os.path.join(self.CURRENT_RESULT, "selsbc.in" if bool(self.sb_flag) else "selpsc.in").replace("\\", "/")

        # Write da_thresh to selfile
        with open(selfile_path, "w") as f:
            f.write(f"{self.da_thresh}\n")
            f.close()

        # Append width to selfile
        with open(selfile_path, "a") as f:
            f.write(f"{self.width}\n")
            f.close()

        # Append contents of calamp.out to selfile
        calamp_out_path = os.path.join(self.CURRENT_RESULT, "calamp.out")
        if os.path.exists(calamp_out_path):
            with open(calamp_out_path, "r") as calamp_file, open(selfile_path, "a") as selfile:
                selfile.write(calamp_file.read())
            calamp_file.close()
            selfile.close()
        
        width_p = self.width // self.prg
        length_p = self.length // self.paz
        
        irg = 0
        iaz = 0
        ip = 0
        
        os.system(f"rm -rf {self.CURRENT_RESULT}/PATCH_* {self.CURRENT_RESULT}/patch.list")

        with open(f"{self.CURRENT_RESULT}/patch.list", "w") as patch_list_file:
            while irg < self.prg:
                irg += 1
                while iaz < self.paz:
                    iaz += 1
                    ip += 1

                    start_rg1 = width_p * (irg - 1) + 1
                    start_rg = max(1, start_rg1 - self.overlap_rg)
                    end_rg1 = width_p * irg
                    end_rg = min(width_p, end_rg1 + self.overlap_rg)

                    start_az1 = length_p * (iaz - 1) + 1
                    start_az = max(1, start_az1 - self.overlap_az)
                    end_az1 = length_p * iaz
                    end_az = min(length_p, end_az1 + self.overlap_az)

                    patch_dir = f"{self.CURRENT_RESULT}/PATCH_{ip}"
                    os.makedirs(patch_dir, exist_ok=True)

                    patch_list_file.write(f"{patch_dir}\n")

                    with open(os.path.join(patch_dir, "patch.in"), "w") as patch_in:
                        patch_in.write(f"{start_rg}\n{end_rg}\n{start_az}\n{end_az}\n")

                    with open(os.path.join(patch_dir, "patch_noover.in"), "w") as patch_noover:
                        patch_noover.write(f"{start_rg1}\n{end_rg1}\n{start_az1}\n{end_az1}\n")

                iaz = 0  # Reset azimuth index for next range loop
            patch_in.close()
            patch_noover.close()
            patch_list_file.close()
                
        # Create pscphase.in
        with open(f"{self.CURRENT_RESULT}/pscphase.in", "w") as pscphase:
            pscphase.write(f"{self.width}\n")
            if self.sb_flag == 1:
                files = glob.glob(f"{self.CURRENT_RESULT}/SMALL_BASELINES/*/*.diff")
            else:
                files = glob.glob(f"{self.CURRENT_RESULT}/diff0/*.diff")
            
            for file in files:
                file = file.replace('\\', '/')
                pscphase.write(f"{file}\n")
            pscphase.close()

        # Create pscdem.in
        with open(f"{self.CURRENT_RESULT}/pscdem.in", "w", encoding="utf-8") as pscdem:
            pscdem.write(f"{self.width}\n")
            dem_files = glob.glob(f"{self.CURRENT_RESULT}/geo/*dem.rdc")
            for file in dem_files:
                file = file.replace('\\', '/')
                pscdem.write(f"{file}\n")
            pscdem.close()

        # Check for longitude files
        lon_files = sorted(glob.glob(f"{self.CURRENT_RESULT}/geo/*.lon"))
        lat_files = sorted(glob.glob(f"{self.CURRENT_RESULT}/geo/*.lat"))

        if lon_files and lat_files:
            with open(f"{self.CURRENT_RESULT}/psclonlat.in", "w", encoding="utf-8") as psclonlat:
                psclonlat.write(f"{self.width}\n")
                lon_files[0] = lon_files[0].replace('\\', '/')
                psclonlat.write(f"{lon_files[0]}\n")  # First longitude file
                lat_files[0] = lat_files[0].replace('\\', '/')
                psclonlat.write(f"{lat_files[0]}\n")  # First latitude file
                psclonlat.close()
                
        # Extract PS info
        extractor = MTExtractCands(f"{self.CURRENT_RESULT}/patch.list", selfile_path)
        extractor.process()
        
        lon_files = glob.glob(f"{self.CURRENT_RESULT}/geo/*.lon")
        
        if not lon_files:
            print("-> Lon file does not exist")
            sys.exit(3)
    
if __name__ == "__main__":
    #try:
    processor = MTPrepSNAP(0.4, None, None)
    processor.process()
    #except Exception as e:
    #    print(f"MTPreSnap fails to execute due to\n{e}")