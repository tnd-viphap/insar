import os
import sys

project_path = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(project_path)

from modules.tomo.ps_parms import Parms
from dev.stamps.flow import StaMPSStep

class AutoRun:
    def __init__(self, oobj="normal"):
        self.input_file = os.path.join(project_path, 'modules/snap2stamps/bin/project.conf')
        self._load_config()

        self.oobj = oobj
        if self.oobj == None:
            self.oobj = 'normal'

        self.patches = [self.CURRENT_RESULT + '/' + f for f in os.listdir(self.CURRENT_RESULT) if f.startswith('PATCH_')]
        self.parms = Parms(self.input_file)
        self.parms.load()
        self._load_rslcpar()

        self.stamps = StaMPSStep(self.parms)

    def _load_config(self):
        with open(self.input_file, 'r') as file:
            for line in file.readlines():
                key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                if key:
                    setattr(self, key, value)  # Dynamically set variables

    def _load_rslcpar(self):
        master_date = self.CURRENT_RESULT.split('_')[1]
        with open(os.path.join(self.CURRENT_RESULT, f'rslc/{master_date}.rslc.par'), 'r') as file:
            for line in file.readlines():
                line = line.strip().split('\t')
                if line[0].startswith('range_pixel_spacing'):
                    self.parms.set('range_pixel_spacing', float(line[1]))
                elif line[0].startswith('near_range_slc'):
                    self.parms.set('near_range_slc', float(line[1]))
                elif line[0].startswith('sar_to_earth_center'):
                    self.parms.set('sar_to_earth_center', float(line[1]))
                elif line[0].startswith('earth_radius_below_sensor'):
                    self.parms.set('earth_radius_below_sensor', float(line[1]))
                elif line[0].startswith('center_range_slc'):
                    self.parms.set('center_range_slc', float(line[1]))
                elif line[0].startswith('azimuth_lines'):
                    self.parms.set('azimuth_lines', float(line[1]))
                elif line[0].startswith('prf'):
                    self.parms.set('prf', float(line[1]))
                elif line[0].startswith('heading'):
                    self.parms.set('heading', float(line[1]))
                elif line[0].startswith('radar_frequency'):
                    self.parms.set('lambda', 299792458 / float(line[1]))
                elif line[0].startswith('sensor'):
                    if 'ASAR' in line[1]:
                        self.parms.set('platform', 'ENVISAT')
                    else:
                        self.parms.set('platform', line[1])
        self.parms.save()

    def aps_linear(self):
        None
    
    def run(self):
        if self.oobj == 'normal':
            for patch in self.patches:
                print(f"-> Processing patch: {patch}")
                print("   -> Step 1: Load initial gamma")
                self.parms.set('n_cores', 30)
                self.parms.set('plot_scatterer_size', 30)
                self.parms.save()
                self.stamps.run(1, 1)
                print("   -> Step 2: Calculate coherence")
                self.parms.set('max_topo_err', 10)
                self.parms.set('gamma_change_convergence', 0.005)
                self.parms.set('filter_grid_size', 50)
                self.parms.save()
                self.stamps.run(2, 2)
                print("   -> Step 3: Calculate deformation")
                self.parms.set('select_method', 'PERCENT')
                self.parms.set('percent_rand', 80)
                self.parms.save()
                self.stamps.run(3, 3, True)
                print("   -> Step 4: Weed PS")
                self.parms.set('weed_zero_elevation', 'n')
                self.parms.set('weed_neighbours', 'n')
                self.parms.save()
                self.stamps.run(4, 4)
                print("   -> Step 5: Phase correction")
                self.parms.set('merge_resample_size', 10)
                self.parms.save()
                self.stamps.run(5, 5, True)
                print("   -> Step 6: Phase unwrapping")
                self.parms.set('unwrap_time_win', 24)
                self.parms.set('unwrap_gold_n_win', 8)
                self.parms.set('unwrap_grid_size', 10)
                self.parms.save()
                self.stamps.run(6, 6, 'y')
                self.aps_linear()
                print("   -> Step 7: Phase unwrapping correction")
                self.parms.set('scla_deramp', 'y')
                self.parms.save()
                self.stamps.run(7, 7, 'y')
                self.stamps.run(6, 6, 'y')
                self.stamps.run(7, 7, 'y')
                print("   -> Step 8: Atmospheric correction")
                self.parms.set('scn_time_win', 30)
                self.parms.save()
                self.stamps.run(8, 8, 'y')
                print("\n")
            
if __name__ == "__main__":
    autorun = AutoRun('normal')
    autorun.run()
