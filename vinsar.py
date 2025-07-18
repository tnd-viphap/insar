#type:ignore
from snap2stamps import Manager
from stamps import StaMPSEXE
from crlink import CRLink
import time
import json

if __name__ == "__main__":
    with open("in.json", "r") as config_file:
        config = json.load(config_file)

    # Now assign variables from the validated config
    project_name = str(config["PROJECT_NAME"])
    bbox = list(config["BBOX"])
    direction = str(config["DIRECTION"])
    frame_no = int(config["FRAME"])
    max_date = int(config["MAX_DATE"])
    download_range = list(config["DOWNLOAD_RANGE"])
    reest_flag = int(config["REEST"])
    identity_master = str(config["MASTER"])
    max_perp = float(config["MAX_PERP"])
    da_threshold = float(config["DA"])
    renew_flag = int(config["RENEW"])
    process_range = list(config["PROCESS_RANGE"])
    stamps_flag = str(config["STAMPS"])
    ptype = int(config["TOMO_TYPE"])
    unified_flag = int(config["UNIFIED"])
    ministack_size = int(config["MSIZE"])
    oobj = str(config["OOBJ"])
    n_rovers = int(config["NROVERS"])

    #########################################################
    # Running phases
    #########################################################
    session = Manager(bbox, direction, frame_no, download_range, max_date, reest_flag, identity_master,
                      max_perp, da_threshold, renew_flag=renew_flag, process_range=process_range,
                      stamps_flag=stamps_flag, ptype=ptype,
                      stack_size=ministack_size, uni=unified_flag, project_name=project_name).run_stages()
    time.sleep(1)
    ps_results = StaMPSEXE(oobj, project_name).run()

    time.sleep(1)
    crlink = CRLink(ps_results, n_rovers).run()


