#type:ignore
import os
import shutil
import subprocess
import sys
import time
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
import rasterio.errors
from shapely.io import from_wkt

warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)

class CoregIFG:
    ######################################################################################
    ## TOPSAR Coregistration and Interferogram formation ##
    ######################################################################################
    
    def __init__(self, max_perp):
        super().__init__()
        inputfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], "project.conf")
        self.bar_message = '\n#####################################################################\n'

        # Getting configuration variables from inputfile
        with open(inputfile, 'r') as file:
            for line in file.readlines():
                key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                if key:
                    setattr(self, key, value)  # Dynamically set variables

        self.polygon = f"POLYGON (({self.LONMIN} {self.LATMIN},{self.LONMAX} {self.LATMIN},{self.LONMAX} {self.LATMAX},{self.LONMIN} {self.LATMAX},{self.LONMIN} {self.LATMIN}))"

        self.outlog = self.LOGFOLDER + 'coreg_ifg_proc_stdout.log'
        self.graphxml = self.GRAPHSFOLDER + 'coreg_ifg_computation_subset.xml'
        self.graph2run = self.GRAPHSFOLDER + 'coreg_ifg2run.xml'

        self.out_file = open(self.outlog, 'a')
        self.err_file = self.out_file

        message = '## Coregistration and Interferogram computation started:\n'
        print(message)
        self.out_file.write(message)
        
        self.prepare_folder()
        self.load_cache_files()
        
        self.max_perp = max_perp
        
    def prepare_folder(self):
        for folder in [self.COREGFOLDER, self.IFGFOLDER, self.LOGFOLDER]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
    def load_cache_files(self):
        """Loads cache files for broken and baseline checks."""
        self.cache_broken_path = self.BROKEN_CACHE
        self.baseline_cache_path = self.BASELINE_CACHE
        self.broken_entries = self.load_cache(self.cache_broken_path)
        self.bs_entries = self.load_cache(self.baseline_cache_path)

    def load_cache(self, cache_path):
        """Loads a cache file into a set for fast lookup."""
        if os.path.exists(cache_path):
            with open(cache_path, "r") as cb_file:
                return set(line.strip() for line in cb_file.readlines())
        return set()
    
    def parse_bbox(self, dim_file):
        """Parses a .dim XML file and extracts necessary metadata"""
        tree = ET.parse(dim_file)
        root = tree.getroot()
        
        # Extract bounding box coordinates
        bbox_params = {
            'first_near_lat': float(root.find(".//MDATTR[@name='first_near_lat']").text),
            'first_near_long': float(root.find(".//MDATTR[@name='first_near_long']").text),
            'first_far_lat': float(root.find(".//MDATTR[@name='first_far_lat']").text),
            'first_far_long': float(root.find(".//MDATTR[@name='first_far_long']").text),
            'last_near_lat': float(root.find(".//MDATTR[@name='last_near_lat']").text),
            'last_near_long': float(root.find(".//MDATTR[@name='last_near_long']").text),
            'last_far_lat': float(root.find(".//MDATTR[@name='last_far_lat']").text),
            'last_far_long': float(root.find(".//MDATTR[@name='last_far_long']").text)
        }
        
        return bbox_params

    def parse_baseline(self, dim_file):
        print("Checking valid baseline...")
        """Parses a .dim XML file and extracts necessary metadata"""
        tree = ET.parse(dim_file)
        root = tree.getroot()
        
        # Extract Bperp info from Master to Slave
        # Positive value corresponds to a higher orbit in Master compared to Slave
        # Vice-versa, negative
        baseline_params = {
            'perp_bs': [float(f.text) for f in root.findall(".//MDATTR[@name='Perp Baseline']")][1]
        }
        
        return baseline_params

    def remove_poor_coreg(self, coreg_dim):
        shutil.rmtree(coreg_dim.replace(".dim", ".data"))
        shutil.rmtree(coreg_dim.replace("coreg", "ifg").replace(".dim", ".data"))
        os.remove(coreg_dim)
        os.remove(coreg_dim.replace("coreg", "ifg"))

    def check_coregistration(self, dim_file, window_size=128, stride=64, threshold=85):
        """
        Checks coregistration of SLV images against the MST image using a scanning window approach.
        
        - The function slides a window across the images and compares the non-zero pixel ratio.
        - If the SLV image has less than `threshold`% valid pixels compared to the MST image, 
        it is flagged as "poor coregistration detected".
        
        :param data_folder: Path to the directory containing the .img files.
        :param window_size: Size of the scanning window (default: 128x128).
        :param stride: Step size for the window movement (default: 64).
        :param threshold: Coregistration threshold percentage (default: 85%).
        """

        # Identify SLV and MST images
        slv_files = []
        mst_file = None

        data_folder = dim_file.replace(".dim", ".data")
        for file in os.listdir(data_folder):
            if file.endswith(".img"):
                if file.startswith("i") and "slv" in file:
                    slv_files.append(os.path.join(data_folder, file))
                elif file.startswith("i"):
                    mst_file = os.path.join(data_folder, file)  # Assume only one MST file

        if not mst_file or not slv_files:
            print("-> Missing required files.")
            return

        # Read MST image
        with rasterio.open(mst_file) as mst_src:
            mst_data = mst_src.read(1)  # Read first band
            height, width = mst_data.shape

        # Compare each SLV file
        for slv_file in slv_files:
            with rasterio.open(slv_file) as slv_src:
                slv_data = slv_src.read(1)  # Read first band

            # Sliding window computation
            total_valid_mst = 0
            total_valid_slv = 0

            for i in range(0, height - window_size + 1, stride):
                for j in range(0, width - window_size + 1, stride):
                    # Extract window
                    slv_window = slv_data[i:i + window_size, j:j + window_size]

                    # Count non-zero pixels in each window
                    mst_valid_pixels = 128*128
                    slv_valid_pixels = np.count_nonzero(slv_window)

                    # Accumulate total valid pixels
                    total_valid_mst += mst_valid_pixels
                    if (slv_valid_pixels / mst_valid_pixels * 100) >= 85:
                        total_valid_slv += slv_valid_pixels

            # Compute overall overlap percentage (fraction of pixels presents in slv image compared to mst image)
            overlap_percentage = (total_valid_slv / total_valid_mst) * 100 if total_valid_mst > 0 else 0

            # Check coregistration condition
            if total_valid_slv == 0:
                print(f"-> {os.path.basename(slv_file)}: Poor coregistration detected (Empty image)\n")
                if os.path.exists(self.cache_broken_path):
                    with open(self.cache_broken_path, "a") as cb_file:
                        cb_file.write(f"{os.path.split(dim_file)[1]}\n")
                        cb_file.close()
                self.remove_poor_coreg(dim_file)
            elif overlap_percentage < threshold:
                print(f"-> {os.path.basename(slv_file)}: Poor coregistration detected (Low overlap: {overlap_percentage:.2f}%)\n")
                if os.path.exists(self.cache_broken_path):
                    with open(self.cache_broken_path, "a") as cb_file:
                        cb_file.write(f"{os.path.split(dim_file)[1]}\n")
                        cb_file.close()
                self.remove_poor_coreg(dim_file)
            else:
                print(f"-> {os.path.basename(slv_file)}: Good coregistration detected (Overlap: {overlap_percentage:.2f}% in pixels)\n")

    def check_overlapping(self, master_dim, coreg_dim):
        bbox_master = self.parse_bbox(master_dim)
        bbox_coreg = self.parse_bbox(coreg_dim)
        
        # Master BBOX
        LATMIN = min(bbox_master['first_near_lat'], bbox_master['first_far_lat'], bbox_master['last_near_lat'], bbox_master['last_far_lat'])
        LATMAX = max(bbox_master['first_near_lat'], bbox_master['first_far_lat'], bbox_master['last_near_lat'], bbox_master['last_far_lat'])
        LONMIN = min(bbox_master['first_near_long'], bbox_master['first_far_long'], bbox_master['last_near_long'], bbox_master['last_far_long'])
        LONMAX = max(bbox_master['first_near_long'], bbox_master['first_far_long'], bbox_master['last_near_long'], bbox_master['last_far_long'])
        master_polygon = from_wkt(f"POLYGON (({LONMIN} {LATMIN},{LONMAX} {LATMIN},{LONMAX} {LATMAX},{LONMIN} {LATMAX},{LONMIN} {LATMIN}))")
        
        # Coreg BBOX
        LATMIN = min(bbox_coreg['first_near_lat'], bbox_coreg['first_far_lat'], bbox_coreg['last_near_lat'], bbox_coreg['last_far_lat'])
        LATMAX = max(bbox_coreg['first_near_lat'], bbox_coreg['first_far_lat'], bbox_coreg['last_near_lat'], bbox_coreg['last_far_lat'])
        LONMIN = min(bbox_coreg['first_near_long'], bbox_coreg['first_far_long'], bbox_coreg['last_near_long'], bbox_coreg['last_far_long'])
        LONMAX = max(bbox_coreg['first_near_long'], bbox_coreg['first_far_long'], bbox_coreg['last_near_long'], bbox_coreg['last_far_long'])
        coreg_polygon = from_wkt(f"POLYGON (({LONMIN} {LATMIN},{LONMAX} {LATMIN},{LONMAX} {LATMAX},{LONMIN} {LATMAX},{LONMIN} {LATMIN}))")
        
        if master_polygon.intersects(coreg_polygon):
            print("-> Overlap detected. Checking coreg file...")
            intersection = master_polygon.intersection(coreg_polygon).area
            ref_area = coreg_polygon.area
            overlap_percentage = (intersection / ref_area) * 100
            if overlap_percentage >= 85:
                print(f"-> Spatial footprint has a best-fit of {overlap_percentage}\n\nChecking for digital dimension...")
                self.check_coregistration(coreg_dim)
            else:
                if os.path.exists(self.cache_broken_path):
                    with open(self.cache_broken_path, "a") as cb_file:
                        cb_file.write(f"{master_dim[0:8]}_{coreg_dim}\n")
                        cb_file.close()
                self.remove_poor_coreg(coreg_dim)
                
    def process(self):
        k = 0
        sorted_slavesplittedfolder = []
        for folder in os.listdir(self.SLAVESFOLDER):
            for file in os.listdir(os.path.join(self.SLAVESFOLDER, folder)):
                if file.endswith('.dim'):
                    sorted_slavesplittedfolder.append(os.path.join(self.SLAVESFOLDER, folder, file))
        sorted_slavesplittedfolder = sorted(sorted_slavesplittedfolder)
        tailm = self.MASTER.split('/')[-1].split('_')[0]

        for dimfile in sorted_slavesplittedfolder:
            k += 1
            _, tail = os.path.split(os.path.join(self.SLAVESFOLDER, dimfile))
            if tail[0:8] != tailm:
                message = f"[{k}] Processing slave file : {tail}\n"
                print(message)
                self.out_file.write(message)
                outputname = tailm + '_' + tail[0:8] + '_' + self.IW1
                if outputname + '.dim' in self.bs_entries:
                    print(f"Slave {outputname}: Poor Interferogram for PS processing. Skipping...")
                    continue
                check_outputname = [outputname + '.dim', outputname + '.data']

                if any(f in os.listdir(self.COREGFOLDER) for f in check_outputname) or any(f in os.listdir(self.IFGFOLDER) for f in check_outputname):
                    print(f"Slave {tail[0:8]} is coregistered and does have interferogram. Validating spatial coverage...\n")
                    self.check_overlapping(self.MASTER, os.path.join(self.COREGFOLDER, outputname+'.dim'))
                    # Check baseline and remove weak interferogram in terms of baseline
                    baseline = self.parse_baseline(os.path.join(self.IFGFOLDER, outputname+'.dim'))["perp_bs"]
                    if abs(baseline) >= self.max_perp:
                        if os.path.exists(self.baseline_cache_path):
                            with open(self.baseline_cache_path, "a") as cb_file:
                                cb_file.write(f"{outputname+'.dim'}\n")
                                cb_file.close()
                        self.remove_poor_coreg(os.path.join(self.COREGFOLDER, outputname+'.dim'))
                        print(f"Slave {outputname}: Poor Interferogram (Bperp = {baseline} m) for PS processing. Skipping...")
                    else:
                        print(f"-> {outputname}: Valid baseline = {baseline}\n")
                    continue

                # **Check if outputname is in broken_cache.txt**
                if outputname+'.dim' in self.broken_entries:
                    print(f"Skipping {outputname}: Broken coreg\n")
                    self.out_file.write(f"Skipping {outputname}: Broken coreg\n")
                    continue
                
                with open(self.graphxml, 'r') as file:
                    filedata = file.read()

                # Replace the target string
                filedata = filedata.replace('MASTER', self.MASTER)
                filedata = filedata.replace('SLAVE', dimfile)
                filedata = filedata.replace('OUTPUTCOREGFOLDER', self.COREGFOLDER)
                filedata = filedata.replace('OUTPUTIFGFOLDER', self.IFGFOLDER)
                filedata = filedata.replace('OUTPUTFILE', outputname)
                filedata = filedata.replace('POLYGON', self.polygon)

                # Write the file out again
                with open(self.graph2run, 'w') as file:
                    file.write(filedata)

                args = [self.GPTBIN_PATH, self.graph2run, '-c', self.CACHE, '-q', self.CPU]

                # **Launch the processing**
                process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                timeStarted = time.time()
                stdout = process.communicate()[0]
                print('SNAP STDOUT:{}'.format(stdout))

                timeDelta = time.time() - timeStarted  # Get execution time.
                print('[' + str(k) + '] Finished process in ' + str(timeDelta) + ' seconds.')
                self.out_file.write('[' + str(k) + '] Finished process in ' + str(timeDelta) + ' seconds.\n')

                if process.returncode != 0:
                    message = 'Error computing coregistration and interferogram generation of splitted slave ' + str(dimfile)
                    self.err_file.write(message + '\n')
                else:
                    message = 'Coregistration and Interferogram computation for data ' + str(tail) + ' successfully completed.\n'
                    print(message)
                    self.out_file.write(message)

                # Check overlapping then perform pixel check to ensure we have valid coreg data
                self.check_overlapping(self.MASTER, os.path.join(self.COREGFOLDER, outputname+'.dim'))
                
                # Check baseline and remove weak interferogram in terms of baseline
                baseline = self.parse_baseline(os.path.join(self.IFGFOLDER, outputname+'.dim'))["perp_bs"]
                if abs(baseline) >= self.max_perp:
                    if os.path.exists(self.baseline_cache_path):
                        with open(self.baseline_cache_path, "r") as cb_file:
                            lines = cb_file.readlines()
                            cb_file.close()
                        lines.append(f"{outputname+'.dim'}\n")
                        lines = list(set(lines))
                        with open(self.baseline_cache_path, "w") as cb:
                            cb.writelines(lines)
                            cb.close()
                    self.remove_poor_coreg(os.path.join(self.COREGFOLDER, outputname+'.dim'))
                    print(f"Slave {outputname}: Poor Interferogram (Bperp = {baseline} m) for PS processing. Skipping...")
                else:
                    print(f"-> {outputname}: Valid baseline = {baseline}\n")
                print(self.bar_message)
                self.out_file.write(self.bar_message)

        self.out_file.close()

if __name__ == "__main__":
    CoregIFG(150.0).process()
    '''
    try:
        start_time = time.time()
        CoregIFG(150.0).process()
        end_time = time.time()
        print(f"Coregistration and Interferogram executes in {(end_time - start_time)/60} minutes.")
    except Exception as e:
        print(f"Coregistration and Interferogram fails due to\n{e}")
    '''


