from snap2stamps import Manager
from stamps import StaMPSEXE
import time

if __name__ == "__main__":
    bbox = [106.6969, 10.7615, 106.7275, 10.7945] ############### NEED REPLACING TO YOUR AOI ###############
    bbox = [106.6783, 10.7236, 106.7746, 10.8136]
    session = Manager(bbox).run_stages()
    time.sleep(1)
    StaMPSEXE("NORMAL", '').run()