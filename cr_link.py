import os
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from modules.utils.compute_cr_ea import gps2ecef_pyproj

class CRLink:
    def __init__(self, psi_result_file):
        self.psi_file = psi_result_file

    def _get_total_loc_err(self):
        pta_folder = "process/pta/"
        pta_result_folders = os.listdir(pta_folder)
        pta_result_files = []

        # Get all pta_results.csv in each pta folder
        for folder in os.listdir(pta_result_folders):
            for file in os.listdir(pta_folder+folder):
                if "results" in file:
                    pta_result_files.append(pta_folder+folder+"/"+file)
        
        ground_range_loc_errs = []
        azimuth_loc_errs = []

        lats = []
        lons = []

        for file in pta_result_files:
            data = pd.read_csv(file)
            # Get errors on dimensions
            ground_range_loc_errs.append(data["ground_range_localization_error_[m]"])
            azimuth_loc_errs.append(data["azimuth_localization_error_[m]"])
            # Get corrected lat, lon information
            lats.append(data["latitude_corrected_[deg]"])
            lons.append(data["longitude_corrected_[deg]"])
        alt = data["altitude_m"]
        # Compute average error
        avg_gr_err = max(ground_range_loc_errs)
        avg_az_err = max(azimuth_loc_errs)
        avg_lat = min(lats)
        avg_lon = min(lons)

        return avg_lat, avg_lon, alt, avg_gr_err, avg_az_err    

    def run(self):
        # Retrieve PTA results
        avg_lat, avg_lon, alt, avg_gr_err, avg_az_err = self._get_total_loc_err()

        # Get search radius for searching nearby points among PS ones
        search_radius = np.sqrt([avg_gr_err, avg_az_err])

        # Create a GeoDataFrame with WGS84 coordinates
        gdf = gpd.GeoDataFrame(geometry=[Point(avg_lon, avg_lat)], crs="EPSG:4326")
        utm_crs = gdf.estimate_utm_crs()
        gdf_utm = gdf.to_crs(utm_crs)
        gdf_buffered = gdf_utm.buffer(search_radius)
        gdf_geom = gpd.GeoSeries(gdf_buffered, crs=utm_crs).to_crs("EPSG:4326").geometry

        # Load psi data
        psi = gpd.read_file(self.psi_file)

        # Capture nearby points
        nearby = psi.loc[gdf_geom.covers(psi.geometry)]
        nearby.to_file("data/geom/CRLink.shp", driver="ESRI Shapefiles")



