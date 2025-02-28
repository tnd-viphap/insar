#type:ignore
from snap2stamps import Manager
from stamps import StaMPSEXE
import time
import os

if __name__ == "__main__":
    bbox = []
    
    inputfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], "modules/snap2stamps/bin/project.conf").replace("\\", "/")
    with open(inputfile, 'r') as file:
            for line in file.readlines():
                key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                if key:
                    locals()[key] = value
                    
    session = Manager(bbox, DIRECTION, FRAME, REEST_FLAG, MAX_PERP, DA_THRESHOLD, "", 0).run_stages()
    time.sleep(1)
    StaMPSEXE("NORMAL", '').run()