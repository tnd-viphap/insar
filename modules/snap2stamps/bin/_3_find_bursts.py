#type:ignore
import glob
import os

import geopandas as gpd
import stsa
from shapely.geometry import Point, Polygon
import time


class Burst:
    def __init__(self):
        super().__init__()
        with open(os.path.join(os.path.split(os.path.abspath(__file__))[0], "project.conf"), 'r') as file:
            for line in file.readlines():
                key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                if key:
                    setattr(self, key, value)
                    
        self.polygon = Polygon([Point(self.LONMIN, self.LATMIN), Point(self.LONMAX, self.LATMIN), Point(self.LONMAX, self.LATMAX), Point(self.LONMIN, self.LATMAX), Point(self.LONMIN, self.LATMIN)])
    
    # Update config
    def modify_master(self, output: list):
        lines = ''''''
        with open(self.CONFIG_PATH, 'r') as file:
            lines = file.readlines()
            for idx, line in enumerate(lines):
                if output[0] and line.startswith("IW1"):
                    lines[idx] = "IW1=" + output[0] + '\n'
                if output[1] and line.startswith("FIRST_BURST"):
                    lines[idx] = "FIRST_BURST=" + output[1] + '\n'
                if output[1] and line.startswith("LAST_BURST"):
                    lines[idx] = "LAST_BURST=" + output[2] + '\n'
            
        with open(self.CONFIG_PATH, "w") as file:
            file.writelines(lines)
            file.close()
            
    def find_burst(self, folder=None):
        # Find burst footprints
        s1 = stsa.TopsSplitAnalyzer(target_subswaths=['iw1', 'iw2', 'iw3'], polarization="vv")
        zip = [f for f in glob.iglob(f"{folder}/*.zip")] if folder else [f for f in glob.iglob(f"{self.MASTERFOLDER}/*/*.zip")]
        if zip:
            s1.load_zip(os.path.join(self.PROJECTFOLDER, zip[0].replace("./", "")))
            s1.to_shapefile(f"{self.DATAFOLDER}geom/master_bursts.shp")
            
            # Find bursts
            gdf = gpd.read_file(f"{self.DATAFOLDER}geom/master_bursts.shp")
            if any(gdf.geometry.intersects(self.polygon)):
                gdf = gdf[gdf.geometry.intersects(self.polygon)]
                subswath = gdf.subswath.values[0]
                first_burst = gdf.burst.values[0]
                last_burst = gdf.burst.values[-1]
            
                # Update configuration
                self.modify_master([subswath, first_burst, last_burst])
                print(f"-> Found AOI in bursts {first_burst} - {last_burst}")
                return True
            else:
                print("-> Found no overlapping bursts")
                return False
        else:
            print("NO RAW IMAGE FOUND: Ensure at least 1 master image is in master folder")
            print("NO RAW IMAGE FOUND: Keep current burst index\n")
            return False

if __name__ == "__main__":
    try:
        Burst().find_burst()
    except Exception as e:
        print(f"Find bursts fails due to\n{e}")