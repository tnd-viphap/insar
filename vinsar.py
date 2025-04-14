#type:ignore
from snap2stamps import Manager
from stamps import StaMPSEXE
import time
import os

if __name__ == "__main__":
    
    inputfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], "modules/snap2stamps/bin/project.conf").replace("\\", "/")
    
    # Parameters
    ## Phase 1: SNAP2STAMPS
    bbox = [106.6783, 10.7236, 106.7746, 10.8136]
    direction = 'DESCENDING'
    frame_no = 540
    download_range = ["", None] # ["20220901", None] means downloading from 01/09/2022 until now
    reest_flag = 1
    identity_master = None # "20221019"
    max_perp = 150.0
    da_threshold = 0.4 
    renew_flag = 1
    process_range = None # ["20220901", "20221019"]
    unified_flag = 0
    ministack_size = 5 
    ## Phase 2: STAMPS
    oobj = "normal"

    ## Running phases
    session = Manager(bbox, direction, frame_no, download_range, reest_flag, max_perp, da_threshold, renew_flag=renew_flag,
                      process_range=process_range,
                      stamps_flag='TOMO', ptype=1,
                      stack_size=ministack_size, uni=unified_flag).run_stages()
    time.sleep(1)
    StaMPSEXE(oobj, '').run()

