import os
import platform
import subprocess
import time
import sys
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
    def __init__(self, bbox, direction, frame, reest_flag=1, max_perp=150.0, da_threshold=0.45,
                 result_folder="", renew_flag=0, stamps_flag='NORMAL', ptype=None):
        super().__init__()
        self.bbox = bbox
        self.direction = direction
        self.frame = frame
        self.reest_flag = reest_flag
        self.max_perp = max_perp
        self.da_threshold = da_threshold
        self.result_folder=result_folder
        self.renew_flag=renew_flag
        self.stamps_flag = stamps_flag
        if self.stamps_flag != 'NORMAL' and ptype == None:
            print("TomoSAR requires processing type flag (comsar):\n-> 0: PSDS\n->1: ComSAR")
            sys.exit(0)
        self.ptype = ptype

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

    def run_cmd(self):
        for idx, script in enumerate(self.python_files):
            if platform.system() == "Linux":
                command = ["python3", script, self.conf_file]
            else:
                command = [
                    "C:/Users/Admin/.conda/envs/insar/python.exe",
                    script,
                    self.conf_file,
                ]
                if idx == 2:
                    command = [
                        "C:/Users/Admin/.conda/envs/insar/python.exe",
                        script,
                        self.conf_file,
                        str(0),
                    ]

            print(f"############## Running: Step {script.split('_')[1]} ##############")

            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()

            if stdout:
                print("[STDOUT] " + script + ":\n" + stdout.decode())
            if stderr:
                print("[STDERR] " + script + ":\n" + stderr.decode())

            if process.returncode != 0:
                self.failed_scripts[script] = stderr.decode()
                print(
                    "[ERROR] "
                    + script
                    + " failed with return code "
                    + str(process.returncode)
                )

        if self.failed_scripts:
            print("\n====== Execution Summary ======")
            print("[ERROR] The following scripts failed:")
            for script, error in self.failed_scripts.items():
                print(" - " + script + ": " + error.strip())
        else:
            print("\n[INFO] All scripts executed successfully.")

    def run_stages(self):
        # Initialze the project configuration
        print(
            f"############## Running: Step 1: Gather project structure ##############"
        )
        Initialize(self.bbox, self.direction, self.frame, self.ptype)
        print("\n")

        # Do searching for data
        print(f"############## Running: Step 2: Download SLC Images ##############")
        print("-> Searching for new products...")
        results = SLC_Search().search()
        time.sleep(2)
        downloader = Download(results)
        if results:
            print(f"-> Found {len(results)} products. Downloading...")
            downloader.download(self.RAWDATAFOLDER)
            time.sleep(2)

            # Select master
            print(f"############## Running: Step 3: Select MASTER ##############")
            MasterSelect(self.reest_flag).select_master()
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
            CoregIFG(150.0).process()
            print('\n')
        else:
            print("-> No new products. Skip downloading and processing!")

        # StaMPS export
        print(f"############## Running: Step 8: StaMPS Export ##############")
        StaMPSExporter(self.result_folder, self.renew_flag).process()
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
