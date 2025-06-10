#type:ignore
import glob
import os

import geopandas as gpd
import stsa
from shapely.geometry import Point, Polygon
import time
import sys

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_path)
from config.parser import ConfigParser


class Burst:
    def __init__(self, project_name="default"):
        super().__init__()
        self.project_name = project_name
        self.config_parser = ConfigParser(os.path.join(project_path, "config", "config.json"))
        self.config = self.config_parser.get_project_config(self.project_name)
                    
        self.polygon = Polygon([Point(self.config["aoi_bbox"]["lon_min"], self.config["aoi_bbox"]["lat_min"]), Point(self.config["aoi_bbox"]["lon_max"], self.config["aoi_bbox"]["lat_min"]), Point(self.config["aoi_bbox"]["lon_max"], self.config["aoi_bbox"]["lat_max"]), Point(self.config["aoi_bbox"]["lon_min"], self.config["aoi_bbox"]["lat_max"]), Point(self.config["aoi_bbox"]["lon_min"], self.config["aoi_bbox"]["lat_min"])])
    
    # Update config
    def modify_master(self, output: list):
        self.config["processing_parameters"]["iw1"] = output[0]
        self.config["processing_parameters"]["first_burst"] = output[1]
        self.config["processing_parameters"]["last_burst"] = output[2]
        self.config_parser.update_project_config(self.project_name, self.config)
            
    def find_burst(self, folder=None):
        # Find burst footprints
        s1 = stsa.TopsSplitAnalyzer(target_subswaths=['iw1', 'iw2', 'iw3'], polarization="vv")
        zip = [f for f in glob.iglob(f"{folder}/*.zip")] if folder else [f for f in glob.iglob(f"{self.config["project_definition"]["master_folder"]}/*/*.zip")]
        if zip:
            s1.load_zip(os.path.join(self.config["project_definition"]["project_folder"], zip[0].replace("./", "")))
            if not os.path.exists(os.path.join(self.config["project_definition"]["data_folder"], "geom")):
                os.makedirs(os.path.join(self.config["project_definition"]["data_folder"], "geom"))
            s1.to_shapefile(f"{self.config["project_definition"]["data_folder"]}geom/master_bursts.shp")
            
            # Find bursts
            gdf = gpd.read_file(f"{self.config["project_definition"]["data_folder"]}geom/master_bursts.shp")
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