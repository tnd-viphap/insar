# type: ignore
import os
import platform
import sys
import time

sys.path.append(os.path.join(os.path.abspath(__file__), "../../../.."))
project_path = os.path.abspath(os.path.join(__file__, '../../../..')).replace("/config", "")
sys.path.append(project_path)
from config.parser import ConfigParser

class StaMPSPrep:
    def __init__(self, stamps_flag, threshold, patch_info=None, project_name="default"):
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
        os.chdir(self.config["processing_parameters"]["current_result"])
        with open(os.path.join(project_path, "config", "project.conf"), "w") as f:
            f.write(f"COMSAR={self.config['api_flags']['comsar']}" + "\n")
            f.write(f"UNIFIED={self.config['processing_parameters']['unified']}" + "\n")
            f.write(f"MINISTACK={self.config['processing_parameters']['ministack']}" + "\n")
            f.write(f"CURRENT_RESULT={self.config['processing_parameters']['current_result']}" + "\n")
            f.write(f"MAX_PERP={self.config['processing_parameters']['max_perp']}")
            f.close()
        if self.stamps_flag == 'NORMAL':
            os.system(f"mt_prep_snap {self.master_date} {self.config['processing_parameters']['current_result']} {self.threshold} {self.patch_info[0]} {self.patch_info[1]} {self.patch_info[2]} {self.patch_info[-1]}")
        else:
            flag = 'comsar' if self.config["api_flags"]["comsar"] == "1" else 'psds'
            os.system(f"matlab -nojvm -nosplash -nodisplay -r \"run('{self.config['project_definition']['project_folder']}/modules/TomoSAR/Tomography/scripts/PSDS_main.m'); exit;\" > {self.config['processing_parameters']['current_result']}/{flag.upper()}.log")
            print(f"-> Preparing {flag} patches...")
            os.chdir(self.config["processing_parameters"]["current_result"])
            if flag == 'comsar':
                os.system(f"mt_prep_snap_comsar {self.master_date} {self.config['processing_parameters']['current_result']} {self.threshold} {self.patch_info[0]} {self.patch_info[1]} {self.patch_info[2]} {self.patch_info[-1]}")
            else:
                os.system(f"mt_prep_snap_psds {self.master_date} {self.config['processing_parameters']['current_result']} {self.threshold} {self.patch_info[0]} {self.patch_info[1]} {self.patch_info[2]} {self.patch_info[-1]}")
        timeDelta = time.time() - timeStarted
        print(f'-> Finished process in {timeDelta} seconds.')
            
if __name__ == "__main__":
    StaMPSPrep('TOMO', 0.3, None, "maychai").process()