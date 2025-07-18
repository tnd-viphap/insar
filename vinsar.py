#type:ignore
from snap2stamps import Manager
from stamps import StaMPSEXE
from crlink import CRLink
import time

if __name__ == "__main__":

    # Load inputs
    inputs = {}
    with open(".in", "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # skip empty lines and comments
            if "=" not in line:
                raise ValueError(f"Malformed line: {line}")
            key, value = line.split("=", 1)
            inputs[key.strip()] = value.strip()

    # Now assign variables from the dictionary
    project_name = inputs.get("PROJECT_NAME")
    bbox = inputs.get("BBOX")
    direction = inputs.get("DIRECTION")
    frame_no = int(inputs.get("FRAME"))
    max_date = int(inputs.get("MAX_DATE"))
    download_range = inputs.get("DOWNLOAD_RANGE")
    reest_flag = int(inputs.get("REEST"))
    identity_master = inputs.get("MASTER")
    max_perp = float(inputs.get("MAX_PERP", 150.0))
    da_threshold = float(inputs.get("DA", 0.4))
    renew_flag = int(inputs.get("RENEW"))
    process_range = inputs.get("PROCESS_RANGE")
    stamps_flag = inputs.get("STAMPS")
    ptype = int(inputs.get("TOMO_TYPE"))
    unified_flag = int(inputs.get("UNIFIED"))
    ministack_size = int(inputs.get("MSIZE"))
    oobj = inputs.get("OOBJ")
    n_rovers = int(inputs.get("NROVERS"))

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


