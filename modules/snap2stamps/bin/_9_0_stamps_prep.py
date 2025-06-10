# type: ignore
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

project_path = os.path.abspath(os.path.join(__file__, '../../../..')).replace("/config", "")
sys.path.append(project_path)
from config.parser import ConfigParser

class StaMPSPrep:
    def __init__(self, stamps_flag, threshold, patch_info=None, project_name="default"):
        super().__init__()
        self.project_name = project_name
        self.config_parser = ConfigParser(os.path.join(project_path, "config", "config.json"))
        self.config = self.config_parser.get_project_config(self.project_name)
        
        self.plf = platform.system()
        
        self.master_date = os.path.split(self.config["processing_parameters"]["current_result"])[1].split("_")[1]
        self.threshold = threshold
        self.patch_info = patch_info
        if self.patch_info is None:
            self.patch_info = ["1", "1", "50", "50"]
        self.stamps_flag = stamps_flag
        
    def process(self):
        timeStarted = time.time()
        # os.chdir(self.CURRENT_RESULT)
        if self.stamps_flag == 'NORMAL':
            StampsPrep(self.master_date, self.config["processing_parameters"]["current_result"], self.threshold, self.patch_info[0], self.patch_info[1], self.patch_info[2], self.patch_info[-1], self.project_name).run()
        else:
            flag = 'comsar' if self.config["api_flags"]["comsar"] == "1" else 'psds'
            # Python-based PSDS_main.m
            TomoSARControl(project_name=self.project_name).run()
            # Python-based mt_prep_snap_psds.m
            print(f"-> Preparing {flag} patches...")
            if flag == 'comsar':
                ComSAR_Prep(self.master_date, self.config["processing_parameters"]["current_result"], self.threshold, self.patch_info[0], self.patch_info[1], self.patch_info[2], self.patch_info[-1], self.project_name).run()
            else:
                PSDS_Prep(self.master_date, self.config["processing_parameters"]["current_result"], self.threshold, self.patch_info[0], self.patch_info[1], self.patch_info[2], self.patch_info[-1], self.project_name).run()
        timeDelta = time.time() - timeStarted
        print(f'-> Finished process in {timeDelta} seconds.')
            
if __name__ == "__main__":
    StaMPSPrep('TOMO', 0.4, None).process()
    
