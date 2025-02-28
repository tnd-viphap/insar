import os
import sys

class StaMPSEXE:
    def __init__(self, process_type=None, display=' -nodisplay'):
        super().__init__()
        
        self.process_type = process_type
        self.display = display
        
        self.inputfile = os.path.join(os.path.split(os.path.abspath(__file__))[0].replace("\\", "/")+'/modules/snap2stamps/bin', "project.conf")
        self._load_config()
        print(f"############## Running: Step 10: StaMPS ##############")
        
    def _load_config(self):
        with open(self.inputfile, 'r') as file:
            for line in file.readlines():
                key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                if key:
                    setattr(self, key, value)  # Dynamically set variables
        
    def run(self):
        if self.process_type == 'NORMAL':
            try:
                os.system(f"matlab -nojvm -nosplash{self.display} -r \"run('{os.path.split(os.path.abspath(__file__))[0]}/modules/StaMPS/autorun_normal.m'); exit;\" > {self.CURRENT_RESULT}/STAMPS.log")
            except:
                sys.exit(0)    
        elif self.process_type == 'TOMO':
            None
        elif self.process_type == "COM":
            None
            
if __name__ == "__main__":
    StaMPSEXE("NORMAL", '').run()