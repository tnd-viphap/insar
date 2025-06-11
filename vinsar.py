#type:ignore
from snap2stamps import Manager
from stamps import StaMPSEXE
from crlink import CRLink
import time
import os

if __name__ == "__main__":
    
    project_name = "default"
    
    # Parameters
    ## Phase 1: SNAP2STAMPS
    bbox = 
    direction =
    frame_no = 
    max_date = 
    download_range = 
    reest_flag = 
    identity_master = 
    max_perp = 
    da_threshold = 
    renew_flag = 
    process_range = 
    unified_flag = 
    ministack_size =  
    ## Phase 2: STAMPS
    oobj = 

    ## Running phases
    session = Manager(bbox, direction, frame_no, download_range, max_date,
                      reest_flag,
                      identity_master,
                      max_perp, da_threshold, renew_flag=renew_flag,
                      process_range=process_range,
                      stamps_flag='TOMO', ptype=1,
                      stack_size=ministack_size, uni=unified_flag, project_name=project_name).run_stages()
    time.sleep(1)
    ps_results = StaMPSEXE(oobj, project_name).run()

    ## Phase 3: CRLink
    n_rovers =
    crlink = CRLink(ps_results, n_rovers).run()


