# type: ignore

import os
import time
import sys
import shutil

project_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_path)

from config.parser import ConfigParser, Initialize
from modules.snap2stamps.bin._1_download import SLC_Search, Download
from modules.snap2stamps.bin._2_master_sel import MasterSelect
from modules.snap2stamps.bin._6_coreg_ifg_topsar import CoregIFG
from modules.snap2stamps.bin._7_stamps_export import StaMPSExporter
from modules.snap2stamps.bin._9_0_stamps_prep import StaMPSPrep


class Manager:
    def __init__(self, bbox, direction, frame, download_on: list = [None, None], max_date: int = None, reest_flag=1, identity_master=None, max_perp=150.0, da_threshold=0.45,
                 result_folder="", renew_flag=0, process_range=None,
                 stamps_flag='NORMAL', ptype=None,
                 stack_size=5, uni=0, project_name: str = "default"):
        super().__init__()
        self.bbox = bbox
        self.direction = direction
        self.frame = frame
        self.max_date = max_date
        self.download_on = download_on
        self.reest_flag = reest_flag
        self.identity_master = identity_master
        self.max_perp = max_perp
        self.da_threshold = da_threshold
        self.result_folder = result_folder
        self.renew_flag = renew_flag
        self.process_range = process_range
        self.stamps_flag = stamps_flag
        self.project_name = project_name
        
        if self.stamps_flag != 'NORMAL' and ptype == None:
            print("TomoSAR requires processing type flag (comsar):\n-> 0: PSDS\n->1: ComSAR")
            sys.exit(0)
        self.ptype = ptype
        self.stack_size = stack_size
        self.uni = uni

        # List of Python files to execute
        self.python_files = [
            "_1_download.py",
            "_2_master_sel.py",
            "_6_coreg_ifg_topsar.py",
            "_7_stamps_export.py",
            "_9_0_stamps_prep.py",
        ]
        self.python_files = [
            os.path.join("modules/snap2stamps/bin", f)
            for f in self.python_files
        ]
        self.python_files = [f.replace("\\", "/") for f in self.python_files]

        # Initialize config parser
        config_path = os.path.join(project_path, 'config', 'config.json')
        self.config_parser = ConfigParser(config_path)
        self.config = self.config_parser.get_project_config(self.project_name)

        # Set attributes from config
        for section, values in self.config.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    setattr(self, key.upper(), value)

        print("Running VINSAR:\n")
        for idx, file in enumerate(self.python_files):
            step_num = idx + 1
            if step_num == 1:
                step_name = "Download SLC Images"
            elif step_num == 2:
                step_name = "Select MASTER"
            elif step_num == 3:
                step_name = "Coregistration and Interferogram"
            elif step_num == 4:
                step_name = "StaMPS Export"
            elif step_num == 5:
                step_name = "StaMPS Preparation"
            else:
                step_name = file.split('_')[1]
            print(f"------ Step {step_num}: {step_name} ------\n")
        print(f"Using configuration for project: {self.project_name}")

        self.failed_scripts = {}

    def run_stages(self):
        # Initialize the project configuration
        print(
            f"############## Running: Step 1: Gather project structure ##############"
        )
        Initialize(self.bbox, self.direction, self.frame, self.max_perp, self.ptype, 
                  self.stack_size, self.uni, self.project_name)
        print("\n")
        
        # Do searching for data
        print(f"############## Running: Step 2: Download SLC Images ##############")
        print("-> Searching for new products...")
        results = SLC_Search(self.max_date, self.download_on, self.project_name).search()
        time.sleep(2)
        downloader = Download(results, self.download_on, self.project_name)
        if results:
            print(f"-> Found {len(results)} products. Downloading...")
            print("-> Note: This step also includes burst finding and splitting for all products")
            download_success = downloader.download(self.config['project_definition']['raw_data_folder'])
            if not download_success:
                print("ERROR: Download and processing stage failed! Stopping workflow.")
                return False
            time.sleep(2)
        else:
            print("-> No new products. Skip downloading and processing!")
            
        # Select master
        print(f"############## Running: Step 3: Select MASTER ##############")
        selected_master = MasterSelect(self.reest_flag, self.identity_master, None, self.project_name).select_master()
        print("\n")

        # Note: Burst finding, master splitting, and slave splitting are now handled 
        # during the download and processing phase in Step 2
        print(f"############## Running: Step 4: Coregistration and Interferogram ##############")
        # self.config_parser._load_config()
        # self.config = self.config_parser.get_project_config(self.project_name)
        # if self.identity_master:
        #     if self.config['processing_parameters']['old_master'] == self.identity_master:
        #         self.renew_flag = 0
        # else:
        #     if not self.config['processing_parameters']['old_master'] in selected_master:
        #         self.renew_flag = 1
        #     else:
        #         self.renew_flag = 0
        if bool(self.renew_flag):
            with open(self.config['cache_files']['baseline_cache'], 'w') as f:
                f.close()
        if bool(self.reest_flag) and bool(self.renew_flag):
            shutil.rmtree(self.config['project_definition']['coreg_folder'])
            shutil.rmtree(self.config['project_definition']['ifg_folder'])
            
        time.sleep(2)
        CoregIFG(self.max_perp, self.process_range, self.project_name).process()
        print('\n')

        # StaMPS export
        print(f"############## Running: Step 5: StaMPS Export ##############")
        StaMPSExporter(self.stamps_flag, self.result_folder, self.renew_flag, None, self.project_name).process()
        print('\n')

        # StaMPS preparation
        print(f"############## Running: Step 6: StaMPS Preparation ##############")
        StaMPSPrep(self.stamps_flag, self.da_threshold, None, self.project_name).process()
        print('\n')
        
        return True

if __name__ == "__main__":
    # Parameters
    ## Phase 1: SNAP2STAMPS
    bbox = [106.691059, 20.837039, 106.776203, 20.899435]
    direction = 'DESCENDING'
    frame_no = 522
    max_date = 2
    download_range = ["20240101", None] # ["20220901", None] means downloading from 01/09/2022 until now
    reest_flag = 1
    process_range = ["20241001", None]
    identity_master = "20250323"
    max_perp = 150.0
    da_threshold = 0.4
    renew_flag = 0
    unified_flag = 0
    ministack_size = 5 
    ## Phase 2: STAMPS
    oobj = "normal"

    # Running phases
    session = Manager(bbox, direction, frame_no, download_range, max_date,
                      reest_flag, identity_master, max_perp, da_threshold,
                      renew_flag=renew_flag,
                      process_range=process_range,
                      stamps_flag='TOMO', ptype=0,
                      stack_size=ministack_size, uni=unified_flag, project_name="maychai")
    
    success = session.run_stages()
    if success:
        print("Workflow completed successfully!")
    else:
        print("Workflow failed! Please check the logs for details.")
        sys.exit(1)
    """
    try:
        Manager().run_stages()
    except Exception as e:
        print(f"Solution Execution Fails Due to\n{e}")
    """