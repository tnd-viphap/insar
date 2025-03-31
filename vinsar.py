#type:ignore
from snap2stamps import Manager
from stamps import StaMPSEXE
import time
import os

if __name__ == "__main__":
    
    inputfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], "modules/snap2stamps/bin/project.conf").replace("\\", "/")
    
    # Parameters
    bbox = [106.6783, 10.7236, 106.7746, 10.8136]
    oobj = "normal"
    reest_flag = 1
    max_perp = 150.0
    da_threshold = 0.4 
    unified_flag = 0
    ministack_size = 5            
    session = Manager(bbox, 'DESCENDING', 540, reest_flag, max_perp, da_threshold, renew_flag=1,
                      stamps_flag='TOMO', ptype=1,
                      stack_size=ministack_size, uni=unified_flag).run_stages()
    time.sleep(1)
    StaMPSEXE(oobj, '').run()

