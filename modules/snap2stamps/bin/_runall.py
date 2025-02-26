import os
import platform
import subprocess
import time
from _0_engage import Initialize
from _1_download import SLC_Search, Download
from _2_master_sel import MasterSelect
from _3_find_bursts import Burst
from _4_splitting_master import MasterSplitter
from _5_splitting_slaves import SlavesSplitter
from _6_coreg_ifg_topsar import CoregIFG
from _7_stamps_export import StaMPSExporter

class Manager:
    def __init__(self):
        super().__init__()

        # List of Python files to execute
        self.python_files = ["_0_engage.py", "_1_download.py", "_2_master_sel.py", "_3_find_bursts.py", "_4_splitting_master.py", "_5_splitting_slaves.py", "_6_coreg_ifg_topsar.py"]#, "7_stamps_export.py"]
        self.python_files = [os.path.join(os.path.abspath(__file__).replace("_runall.py", ""), f) for f in self.python_files]
        self.conf_file = os.path.abspath(__file__).replace("_runall.py", "project.conf")
        
        with open(self.conf_file, 'r') as file:
            for line in file.readlines():
                key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                if key:
                    setattr(self, key, value)  # Dynamically set variables
        
        print("Running VIPHAP Adaptive InSAR Solution:\n")
        for idx, file in enumerate(self.python_files):
            print(f"------ Step {file.split('_')[1]}: {self.python_files[idx]}\n")
        print(f"Found configuration file: {self.conf_file}")

        self.failed_scripts = {}
    
    def run_cmd(self):
        for idx, script in enumerate(self.python_files):
            if platform.system() == "Linux":
                command = ["python3", script, self.conf_file]
            else:
                command = ["C:/Users/Admin/.conda/envs/insar/python.exe", script, self.conf_file]
                if idx == 2:
                    command = ["C:/Users/Admin/.conda/envs/insar/python.exe", script, self.conf_file, str(0)]
                
            print(f"############## Running: Step {script.split('_')[1]} ##############")
            
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            if stdout:
                print("[STDOUT] " + script + ":\n" + stdout.decode())
            if stderr:
                print("[STDERR] " + script + ":\n" + stderr.decode())

            if process.returncode != 0:
                self.failed_scripts[script] = stderr.decode()
                print("[ERROR] " + script + " failed with return code " + str(process.returncode))

        if self.failed_scripts:
            print("\n====== Execution Summary ======")
            print("[ERROR] The following scripts failed:")
            for script, error in self.failed_scripts.items():
                print(" - " + script + ": " + error.strip())
        else:
            print("\n[INFO] All scripts executed successfully.")
            
    def run_stages(self):
        # Initialze the project configuration
        print(f"############## Running: Step 1: Gather project structure ##############")
        bbox = [106.6969, 10.7615, 106.7275, 10.7945] ############### NEED REPLACING TO YOUR AOI ###############
        Initialize(bbox)
        print("\n")
        
        # Do searching for data
        print(f"############## Running: Step 2: Download SLC Images ##############")
        results = SLC_Search("Descending", 553).search()
        downloader = Download(results)
        if results:
            downloader.download(self.RAWDATAFOLDER)
            time.sleep(2)
            
        # Select master
        print(f"############## Running: Step 3: Select MASTER ##############")
        MasterSelect(1).select_master()
        print("\n")
        
        # Find master busrt
        print(f"############## Running: Step 4: Find MASTER burst ##############")
        Burst().find_burst()
        print("\n")
        
        # Master split and slaves split
        print(f"############## Running: Step 5: Split MASTER ##############")
        MasterSplitter().process()
        print("\n")
        print(f"############## Running: Step 6: Split SLAVES ##############")
        SlavesSplitter().process()
        print("\n")
        
        # Run coregistration and make interferogram
        print(f"############## Running: Step 7: Coregistration and Interferogram ##############")
        CoregIFG(150.0).process()
        
        # StaMPS export
        print(f"############## Running: Step 8: StaMPS Export ##############")
        StaMPSExporter("DEMOBaSon", 0).process()
        
if __name__ == "__main__":
    Manager().run_stages()
    '''
    try:
        Manager().run_stages()
    except Exception as e:
        print(f"Solution Execution Fails Due to\n{e}")
    '''