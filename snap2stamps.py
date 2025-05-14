import os
import time
import sys
import shutil
sys.path.append(os.path.join(os.path.abspath(__file__), "../../../../"))

from modules.snap2stamps.bin._0_engage import Initialize
from modules.snap2stamps.bin._1_download import SLC_Search, Download
from modules.snap2stamps.bin._2_master_sel import MasterSelect
from modules.snap2stamps.bin._3_find_bursts import Burst
from modules.snap2stamps.bin._4_splitting_master import MasterSplitter
from modules.snap2stamps.bin._5_splitting_slaves import SlavesSplitter
from modules.snap2stamps.bin._6_coreg_ifg_topsar import CoregIFG
from modules.snap2stamps.bin._7_stamps_export import StaMPSExporter
from modules.snap2stamps.bin._9_0_stamps_prep import StaMPSPrep


class Manager:
    def __init__(self, bbox, direction, frame, download_on, max_date, reest_flag=1, identity_master=None, max_perp=150.0, da_threshold=0.45,
                 result_folder="", renew_flag=0, process_range=None,
                 stamps_flag='NORMAL', ptype=None,
                 stack_size=5, uni=0):
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
        self.result_folder=result_folder
        self.renew_flag=renew_flag
        self.process_range = process_range
        self.stamps_flag = stamps_flag
        if self.stamps_flag != 'NORMAL' and ptype == None:
            print("TomoSAR requires processing type flag (comsar):\n-> 0: PSDS\n->1: ComSAR")
            sys.exit(0)
        self.ptype = ptype
        self.stack_size = stack_size
        self.uni = uni

        # List of Python files to execute
        self.python_files = [
            "_0_engage.py",
            "_1_download.py",
            "_2_master_sel.py",
            "_3_find_bursts.py",
            "_4_splitting_master.py",
            "_5_splitting_slaves.py",
            "_6_coreg_ifg_topsar.py",
            "_7_stamps_export.py",
            "_9_0_stamps_prep.py",
        ]
        self.python_files = [
            os.path.join("modules/snap2stamps/bin", f)
            for f in self.python_files
        ]
        self.python_files = [f.replace("\\", "/") for f in self.python_files]
        self.conf_file = "modules/snap2stamps/bin/project.conf"

        with open(self.conf_file, "r") as file:
            for line in file.readlines():
                key, value = (
                    (line.split("=")[0].strip(), line.split("=")[1].strip())
                    if "=" in line
                    else (None, None)
                )
                if key:
                    setattr(self, key, value)  # Dynamically set variables

        print("Running VINSAR:\n")
        for idx, file in enumerate(self.python_files):
            print(f"------ Step {file.split('_')[1]}: {self.python_files[idx]}\n")
        print(f"Found configuration file: {self.conf_file}")

        self.failed_scripts = {}

    def run_stages(self):
        # Initialze the project configuration
        print(
            f"############## Running: Step 1: Gather project structure ##############"
        )
        Initialize(self.bbox, self.direction, self.frame, self.max_perp, self.ptype, self.stack_size, self.uni)
        print("\n")
        
        # Do searching for data
        print(f"############## Running: Step 2: Download SLC Images ##############")
        print("-> Searching for new products...")
        results = SLC_Search(self.max_date).search()
        time.sleep(2)
        downloader = Download(results, self.download_on)
        if results:
            print(f"-> Found {len(results)} products. Downloading...")
            downloader.download(self.RAWDATAFOLDER)
            time.sleep(2)
        else:
            print("-> No new products. Skip downloading and processing!")
            
        # Select master
        print(f"############## Running: Step 3: Select MASTER ##############")
        MasterSelect(self.reest_flag, self.identity_master).select_master()
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
        print(
            f"############## Running: Step 7: Coregistration and Interferogram ##############"
        )
        if bool(self.reest_flag):
            shutil.rmtree("process/coreg")
            shutil.rmtree("process/ifg")
        time.sleep(2)
        CoregIFG(self.max_perp).process()
        print('\n')

        # StaMPS export
        print(f"############## Running: Step 8: StaMPS Export ##############")
        StaMPSExporter(self.process_range, self.stamps_flag, self.result_folder, self.renew_flag).process()
        print('\n')

        # StaMPS preparation
        print(f"############## Running: Step 9: StaMPS Preparation ##############")
        StaMPSPrep(self.stamps_flag, self.da_threshold).process()
        print('\n')
        

if __name__ == "__main__":
    bbox = [
        106.6969,
        10.7615,
        106.7275,
        10.7945,
    ]  ############### NEED REPLACING TO YOUR AOI ###############
    bbox = [106.6783, 10.7236, 106.7746, 10.8136]
    Manager(bbox).run_stages()
    """
    try:
        Manager().run_stages()
    except Exception as e:
        print(f"Solution Execution Fails Due to\n{e}")
    """