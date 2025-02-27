import os
import platform
import time

from _9_1_mt_prep_snap import MTPrepSNAP


class StaMPSPrep:
    def __init__(self, threshold, patch_info=None):
        super().__init__()
        
        self.inputfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], "project.conf")
        self._load_config()
        
        self.plf = platform.system()
        
        self.master_date = os.path.split(self.MASTER)[1].split("_")[0]
        self.threshold = threshold
        self.patch_info = patch_info
        
    def _load_config(self):
        with open(self.inputfile, 'r') as file:
            for line in file.readlines():
                key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                if key:
                    setattr(self, key, value)  # Dynamically set variables
                    
    def process(self):
        os.chdir(self.CURRENT_RESULT)
        # Default for Linux
        if self.plf == "Windows":
            processor = MTPrepSNAP(self.threshold, self.patch_info, None)
            processor.process()
        elif self.plf == "Linux":
            command = ["mt_prep_snap", self.master_date, self.CURRENT_RESULT, str(self.threshold), "1", "1", "50", "50"]
            if self.patch_info:
                command = ["mt_prep_snap", self.master_date, self.CURRENT_RESULT,
                        str(self.threshold), str(self.patch_info[0]), str(self.patch_info[1]), str(self.patch_info[2]), str(self.patch_info[-1])]
            timeStarted = time.time()
            os.system(" ".join(command))
            timeDelta = time.time() - timeStarted
            print(f'Finished process in {timeDelta} seconds.')
            
if __name__ == "__main__":
    try:
        StaMPSPrep(0.4, None).process()
    except Exception as e:
        print(f"StaMPS Preparation fails to execute due to\n{e}")
    