#type:ignore
from snap2stamps import Manager
from stamps import StaMPSEXE
import time
import os

if __name__ == "__main__":
    bbox = [106.6969, 10.7615, 106.7275, 10.7945] ############### NEED REPLACING TO YOUR AOI ###############
    bbox = [106.6783, 10.7236, 106.7746, 10.8136]
    inputfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], "modules/snap2stamps/bin/project.conf").replace("\\", "/")
    with open(inputfile, 'r') as file:
            for line in file.readlines():
                key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                if key:
                    locals()[key] = value             
    session = Manager(bbox, 'DESCENING', 553, REEST_FLAG, MAX_PERP, DA_THRESHOLD, result_folder="DEMOBaSon", renew_flag=0, stamps_flag='TOMO', ptype=1).run_stages()
    time.sleep(1)
    StaMPSEXE('').run()

