import os
import sys
import time
import geopandas as gpd
from shapely.geometry import Point

class StaMPSEXE:
    def __init__(self, process_type=None, display=' -nodisplay'):
        super().__init__()
        
        self.process_type = process_type
        self.display = display
        
        self.inputfile = os.path.join(os.path.split(os.path.abspath(__file__))[0].replace("\\", "/")+'/modules/snap2stamps/bin', "project.conf")
        self._load_config()
        print(f"############## Running: Step 10: StaMPS ##############")
        
    def _load_config(self):
        with open(self.inputfile, 'r') as file:
            for line in file.readlines():
                key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                if key:
                    setattr(self, key, value)  # Dynamically set variables
        
    def run(self):
        if self.process_type == 'NORMAL':
            try:
                #os.system(f"matlab -nojvm -nosplash{self.display} -r \"run('{os.path.split(os.path.abspath(__file__))[0]}/modules/StaMPS/autorun_normal.m'); exit;\" > {self.CURRENT_RESULT}/STAMPS.log")
                time.sleep(1)
                for patch in [os.path.join(self.CURRENT_RESULT, f) for f in os.listdir(self.CURRENT_RESULT) if f.startswith('PATCH_')]:
                    csv_file = [os.path.join(patch, f) for f in os.listdir(patch) if f.endswith('.csv')]
                    folder = patch.split('/')[-1]
                    if csv_file:
                        data = gpd.read_file(csv_file[0])
                        geom = [Point(f) for f in zip(data.LON, data.LAT)]
                        data = gpd.GeoDataFrame(data, geometry=geom, crs="EPSG:4326")
                        outfile = os.path.join(self.DATAFOLDER, f'geom/{self.CURRENT_RESULT.split("/")[-1]}_{folder}.shp')
                        data.to_file(outfile, driver="ESRI Shapefile")
            except:
                sys.exit(0)    
        elif self.process_type == 'TOMO':
            None
        elif self.process_type == "COM":
            None
            
if __name__ == "__main__":
    StaMPSEXE("NORMAL", '').run()