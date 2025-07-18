#type:ignore
from snap2stamps import Manager
from stamps import StaMPSEXE
from crlink import CRLink
import time
import json

if __name__ == "__main__":
    with open("in.json", "r") as config_file:
        config = json.load(config_file)

    # Validate required fields and types
    for key, spec in schema.items():
        if spec.get("required", False) and key not in config:
            raise ValueError(f"Missing required parameter: {key}")
        
        value = config.get(key, spec.get("default"))
        if value is None:
            raise ValueError(f"No value or default provided for: {key}")

        # Type validation
        if spec["type"] == "integer":
            config[key] = int(value)
        elif spec["type"] == "number":
            config[key] = float(value)
        elif spec["type"] == "string":
            config[key] = str(value)

    # Now assign variables from the validated config
    project_name = config["PROJECT_NAME"]
    bbox = config["BBOX"]
    direction = config["DIRECTION"]
    frame_no = config["FRAME"]
    max_date = config["MAX_DATE"]
    download_range = config["DOWNLOAD_RANGE"]
    reest_flag = config["REEST"]
    identity_master = config["MASTER"]
    max_perp = config["MAX_PERP"]
    da_threshold = config["DA"]
    renew_flag = config["RENEW"]
    process_range = config["PROCESS_RANGE"]
    stamps_flag = config["STAMPS"]
    ptype = config["TOMO_TYPE"]
    unified_flag = config["UNIFIED"]
    ministack_size = config["MSIZE"]
    oobj = config["OOBJ"]
    n_rovers = config["NROVERS"]

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


