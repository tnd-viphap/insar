import os
import platform
import shutil
import sys
import time

sys.path.append(os.path.join(os.path.abspath(__file__), "../../../.."))

from modules.tomo.tomo import TomoSARControl
from modules.tomo.psds_prep import PSDS_Prep
from modules.tomo.stamps_prep import StampsPrep
from modules.tomo.comsar_prep import ComSAR_Prep

class StaMPSPrep:
    def __init__(self, stamps_flag, threshold, patch_info=None):
        super().__init__()
        
        self.inputfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], "project.conf")
        self._load_config()
        
        self.plf = platform.system()
        
        self.master_date = os.path.split(self.CURRENT_RESULT)[1].split("_")[1]
        self.threshold = threshold
        self.patch_info = patch_info
        if self.patch_info is None:
            self.patch_info = ["1", "1", "50", "50"]
        self.stamps_flag = stamps_flag
        
    def _load_config(self):
        with open(self.inputfile, 'r') as file:
            for line in file.readlines():
                key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                if key:
                    setattr(self, key, value)  # Dynamically set variables
                    
    def process(self):
        timeStarted = time.time()
        os.chdir(self.CURRENT_RESULT)
        if self.stamps_flag == 'NORMAL':
            StampsPrep(self.master_date, self.CURRENT_RESULT, self.threshold, self.patch_info[0], self.patch_info[1], self.patch_info[2], self.patch_info[-1]).run()
        else:
            flag = 'comsar' if self.COMSAR == "1" else 'psds'
            # Python-based PSDS_main.m
            TomoSARControl().run()
            # Python-based mt_prep_snap_psds.m
            print(f"-> Preparing {flag} patches...")
            if flag == 'comsar':
                ComSAR_Prep(self.master_date, self.CURRENT_RESULT, self.threshold, self.patch_info[0], self.patch_info[1], self.patch_info[2], self.patch_info[-1]).run()
            else:
                PSDS_Prep(self.master_date, self.CURRENT_RESULT, self.threshold, self.patch_info[0], self.patch_info[1], self.patch_info[2], self.patch_info[-1]).run()
        timeDelta = time.time() - timeStarted
        print(f'-> Finished process in {timeDelta} seconds.')
            
if __name__ == "__main__":
    StaMPSPrep('TOMO', 0.4, None).process()
    
