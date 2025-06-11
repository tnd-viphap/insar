import os
import sys

class PSCPhase:
    def __init__(self, parm_file, ij_file=None, ph_file=None, log_file=None):
        """
        Initialize PSCPhase extraction

        Args:
            parm_file (str): Path to parameter file containing width and ifg file paths
            ij_file (str, optional): Path to PS candidate locations file. Defaults to 'pscands.1.ij'
            ph_file (str, optional): Path to output phase file. Defaults to 'pscands.1.ph'
            log_file (str, optional): Path to log file. If None, no logging will be performed.
        """
        self.parm_file = parm_file
        self.ij_file = ij_file or 'pscands.1.ij'
        self.ph_file = ph_file or 'pscands.1.ph'
        self.log_file = log_file

        self.width = None
        self.ifg_filenames = []
        self.SUN_RASTER_MAGIC = 0x59a66a95

    def _log(self, message, level="INFO"):
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{message}\n")

    def read_parm_file(self):
        try:
            with open(self.parm_file, 'r') as f:
                self.width = int(f.readline().strip())
                savepos = f.tell()
                num_files = 0
                while True:
                    line = f.readline()
                    if not line:
                        break
                    num_files += 1
                f.seek(savepos)
                self.ifg_filenames = [f.readline().strip() for _ in range(num_files)]
        except Exception as e:
            self._log(f"Error reading parameter file: {str(e)}", "ERROR")
            sys.exit(1)

    def read_ps_candidates(self):
        ij_list = []
        try:
            with open(self.ij_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        pscid, y, x = map(int, parts[:3])
                        ij_list.append((pscid, y, x))
            return ij_list
        except Exception as e:
            self._log(f"Error reading PS candidate file: {str(e)}", "ERROR")
            sys.exit(1)

    def run(self):
        self.read_parm_file()
        ij_list = self.read_ps_candidates()
        with open(self.ph_file, 'wb') as out_f:
            for i, ifg_filename in enumerate(self.ifg_filenames):
                if not os.path.exists(ifg_filename):
                    print(f"Error opening file {ifg_filename}")
                    sys.exit(1)
                with open(ifg_filename, 'rb') as ifg_f:
                    header = ifg_f.read(32)
                    if int.from_bytes(header[:4], 'little') == self.SUN_RASTER_MAGIC:
                        print('sun raster file - skipping header')
                    else:
                        ifg_f.seek(0)
                    for pscid, y, x in ij_list:
                        xyaddr = (y * self.width + x) * 8  # 8 bytes per complex float (2x4)
                        ifg_f.seek(xyaddr)
                        data = ifg_f.read(8)
                        if len(data) != 8:
                            print(f"Warning: could not read 8 bytes at {(y, x)} in {ifg_filename}")
                            data = b'\x00' * 8
                        out_f.write(data)
                # print(f"{i+1} of {len(self.ifg_filenames)} interferograms processed")

def main():
    if len(sys.argv) < 2:
        print("Usage: pscphase.py parmfile [pscands.1.ij] [pscands.1.ph]")
        sys.exit(1)
    parm_file = sys.argv[1]
    ij_file = sys.argv[2] if len(sys.argv) > 2 else None
    ph_file = sys.argv[3] if len(sys.argv) > 3 else None
    psc_phase = PSCPhase(parm_file, ij_file, ph_file)
    psc_phase.run()

if __name__ == "__main__":
    main() 