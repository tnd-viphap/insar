# type: ignore
import os
import re
import shutil
import sys
import time
import platform
import subprocess

import geopandas as gpd
import jenkspy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import scipy.io as sio
from shapely.geometry import Point
from affine import Affine

project_folder = os.path.split(os.path.abspath(__file__))[0].replace("\\", "/")
sys.path.append(project_folder)

from config.parser import ConfigParser
from modules.tomo.ps_parms import Parms


class StaMPSEXE:
    def __init__(self, oobj="normal", project_name="default"):
        super().__init__()
        
        if platform.system() == 'Windows':
            self.display = '-nodesktop'
        else:
            self.display = '-nodisplay'
        self.oobj = oobj
        self.project_name = project_name
        
        # Initialize config parser
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.json')
        self.config_parser = ConfigParser(config_path)
        self.config = self.config_parser.get_project_config(self.project_name)
        
        # Set attributes from config
        for section, values in self.config.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    setattr(self, key.upper(), value)
        
        self._load_rslcpar()
        print(f"############## Running: Step 10: StaMPS ##############")
        
    def _csvtoshp(self):
        for patch in [os.path.join(self.config['processing_parameters']['current_result'], f) for f in os.listdir(self.config['processing_parameters']['current_result']) if f.startswith('PATCH_')]:
            csv_file = [f for f in os.listdir(patch) if f.endswith('.csv')]
            folder = patch.split('/')[-2]
            if csv_file:
                data = gpd.read_file(csv_file[0])
                data.geometry = [Point(x, y) for x, y in zip(data.LON, data.LAT)]
                data.to_file(os.path.join(self.config['project_definition']['data_folder'], f"geom/{self.config['processing_parameters']['current_result'].split('/')[-1]}_{folder}.shp"))

    def _load_rslcpar(self):
        parms = Parms(self.project_name)
        parms.load()
        master_date = self.config['processing_parameters']['current_result'].split('/')[-1].split('_')[1]
        with open(os.path.join(self.config['processing_parameters']['current_result'], f'rslc/{master_date}.rslc.par'), 'r') as file:
            for line in file.readlines():
                line = line.strip().split('\t')
                if line[0].startswith('range_pixel_spacing'):
                    parms.set('range_pixel_spacing', float(line[1]))
                elif line[0].startswith('near_range_slc'):
                    parms.set('near_range_slc', float(line[1]))
                elif line[0].startswith('sar_to_earth_center'):
                    parms.set('sar_to_earth_center', float(line[1]))
                elif line[0].startswith('earth_radius_below_sensor'):
                    parms.set('earth_radius_below_sensor', float(line[1]))
                elif line[0].startswith('center_range_slc'):
                    parms.set('center_range_slc', float(line[1]))
                elif line[0].startswith('azimuth_lines'):
                    parms.set('azimuth_lines', float(line[1]))
                elif line[0].startswith('prf'):
                    parms.set('prf', float(line[1]))
                elif line[0].startswith('heading'):
                    parms.set('heading', float(line[1]))
                elif line[0].startswith('radar_frequency'):
                    parms.set('lambda', 299792458 / float(line[1]))
                elif line[0].startswith('sensor'):
                    if 'ASAR' in line[1]:
                        parms.set('platform', 'ENVISAT')
                    else:
                        parms.set('platform', line[1])
        parms.save()

    def rasterize_preserve_and_stack(self, points_gdf, 
                                    columns, 
                                    pixel_size=10, 
                                    window_radius=15, 
                                    bounds=None):
        """
        Rasterize and stack multiple columns from point GeoDataFrame:
        - Preserve original values at point locations.
        - Only assign values to pixels containing points; do not interpolate or fill other pixels.
        Returns:
            stacked_raster: (bands, height, width) array
            transform: affine transform
        """
        # Ensure projected CRS
        if points_gdf.crs.is_geographic:
            raise ValueError("GeoDataFrame must use a projected CRS in meters.")
        
        # Compute bounds
        if bounds is None:
            minx, miny, maxx, maxy = points_gdf.total_bounds
            buffer = window_radius
            minx = np.floor((minx - buffer) / pixel_size) * pixel_size
            miny = np.floor((miny - buffer) / pixel_size) * pixel_size
            maxx = np.ceil((maxx + buffer) / pixel_size) * pixel_size
            maxy = np.ceil((maxy + buffer) / pixel_size) * pixel_size
            bounds = (minx, miny, maxx, maxy)
        else:
            minx, miny, maxx, maxy = bounds

        width = int(np.ceil((maxx - minx) / pixel_size))
        height = int(np.ceil((maxy - miny) / pixel_size))
        
        transform = Affine(pixel_size, 0, minx, 0, -pixel_size, maxy)

        stacked_bands = []

        point_coords = np.array([[geom.x, geom.y] for geom in points_gdf.geometry])

        for col in columns:
            values = points_gdf[col].values
            band = np.full((height, width), np.nan)

            # Assign known point values only
            for i, (x, y) in enumerate(point_coords):
                col_idx = int((x - minx) // pixel_size)
                row_idx = int((maxy - y) // pixel_size)
                if 0 <= row_idx < height and 0 <= col_idx < width:
                    band[row_idx, col_idx] = values[i]
            stacked_bands.append(band)

        stacked_array = np.stack(stacked_bands, axis=0)  # shape: (bands, height, width)
        return stacked_array, transform
    
    def raster_to_points(self, stacked_array, transform, columns):
        """
        Convert raster array back to point GeoDataFrame with center coords.
        
        Parameters:
            stacked_array: (bands, height, width) numpy array
            transform: Affine transform used for the raster
            columns: list of band names corresponding to array layers

        Returns:
            GeoDataFrame with columns from stacked_array and geometry as point centers
        """
        bands, height, width = stacked_array.shape
        assert bands == len(columns), "Mismatch between number of bands and columns"

        # Compute coordinates of pixel centers
        xs = np.arange(width)
        ys = np.arange(height)
        xv, yv = np.meshgrid(xs, ys)

        x_coords, y_coords = transform * (xv + 0.5, yv + 0.5)  # center of pixel
        x_coords = x_coords.ravel()
        y_coords = y_coords.ravel()

        # Extract values from each band
        data = {}
        for i, col in enumerate(columns):
            band_values = stacked_array[i].ravel()
            data[col] = band_values

        # Only keep pixels with at least one non-NaN value
        valid_mask = ~np.all(np.isnan(stacked_array), axis=0).ravel()
        
        # Build GeoDataFrame
        geometries = [Point(x, y) for x, y in zip(x_coords[valid_mask], y_coords[valid_mask])]
        data = {col: np.array(vals)[valid_mask] for col, vals in data.items()}
        gdf = gpd.GeoDataFrame(data, geometry=geometries, crs="EPSG:32648")  # Change to match original CRS
        
        return gdf
    
    def ps_dem_err(self):
        parms = Parms(self.project_name)
        parms.load()
        slant_range = parms.get('range_pixel_spacing')
        near_range = parms.get('near_range_slc')
        re = parms.get('earth_radius_below_sensor')
        rs = parms.get('sar_to_earth_center')
        lambda_val = parms.get('lambda')
        
        ij = sio.loadmat("ps2.mat")['ij']
        K_ps_uw = sio.loadmat("scla2.mat")['K_ps_uw']
        
        
        range_pixel = ij[:, 2].reshape(-1, 1)
        del ij
        rg = near_range + range_pixel * slant_range
        alpha = np.pi - np.arccos((rg**2 + re**2 - rs**2) / (2 * rg * re))
        dem_err = -K_ps_uw * lambda_val * rg * np.sin(alpha) / (4 * np.pi)
        dem_err = dem_err.reshape(-1, 1)
        
        # Save result
        sio.savemat("dem_err.mat", {"dem_err": dem_err})
        return dem_err
        
    def ps_lonlat_err(self, dem_err):
        parms = Parms(self.project_name)
        parms.load()
        la = sio.loadmat("la2.mat")['la']
        heading = parms.get('heading')
        re = parms.get('earth_radius_below_sensor')
        theta = (180 - heading) * np.pi / 180
        Dx = dem_err * (1 / np.tan(la)) * np.cos(theta)
        Dy = dem_err * (1 / np.tan(la)) * np.sin(theta)
        Dlon = np.arccos(1 - (Dx**2) / (2 * re**2)) * 180 / np.pi
        Dlat = np.arccos(1 - (Dy**2) / (2 * re**2)) * 180 / np.pi
        self.lonlat_err = np.array([Dlon, Dlat])
        sio.savemat("lonlat_err.mat", {"lonlat_err": self.lonlat_err})

    def ps_export_gis(self, csv_filename, shapefile_name, lon_rg=None, lat_rg=None, ortho=None):
        if not os.path.exists("ps_plot_ts_v-dao.mat"):
            if platform.system() == 'Windows':
                matlab_cmd = (
                f"\"C:/Program Files/MATLAB/R2024a/bin/matlab.exe\" -wait -nosplash {self.display} "
                f"-r \"ps_plot('v-dao', 'a_linear', 'ts'); exit;\""
            )
                subprocess.run(matlab_cmd, shell=True)
            else:
                os.system(f"matlab -nojvm -nosplash {self.display} -r \"ps_plot('v-dao', 'a_linear', 'ts'); exit;\"")
        print("   -> TS V-dao done")
        
        if not os.path.exists("ps_plot_v-dao.mat"):
            if platform.system() == 'Windows':
                matlab_cmd = (
                f"\"C:/Program Files/MATLAB/R2024a/bin/matlab.exe\" -wait -nosplash {self.display} "
                f"-r \"ps_plot('v-dao', 'a_linear', -1); exit;\""
            )
                subprocess.run(matlab_cmd, shell=True)
            else:
                os.system(f"matlab -nojvm -nosplash {self.display} -r \"ps_plot('v-dao', 'a_linear', -1); exit;\"")
        print("   -> V-dao done")
        
        # Load necessary data
        ph_disp = sio.loadmat("ps_plot_v-dao.mat")['ph_disp'].flatten()
        ts_data = sio.loadmat("ps_plot_ts_v-dao.mat")
        ph_mm = ts_data['ph_mm']
        day = ts_data['day'].flatten()
        ps2_data = sio.loadmat("ps2.mat")
        lonlat = ps2_data['lonlat']
        xy = ps2_data['xy']
        lonlat_err = sio.loadmat("lonlat_err.mat")['lonlat_err']
        dem_err = sio.loadmat("dem_err.mat")['dem_err'].flatten()
        hgt = sio.loadmat("hgt2.mat")['hgt'].flatten()
        coh_ps = sio.loadmat("pm2.mat")['coh_ps'].flatten()
        
        # Filter data based on region of interest
        if lon_rg and lat_rg:
            mask = (lonlat[:, 0] >= min(lon_rg)) & (lonlat[:, 0] <= max(lon_rg)) & \
                (lonlat[:, 1] >= min(lat_rg)) & (lonlat[:, 1] <= max(lat_rg))
        else:
            mask = np.ones(lonlat.shape[0], dtype=bool)
        
        xy = xy[mask]
        lonlat = lonlat[mask]
        lonlat_err = lonlat_err[:, mask, 0]
        hgt = hgt[mask]
        dem_err = dem_err[mask]
        coh_ps = coh_ps[mask]
        ph_disp = ph_disp[mask]
        ph_mm = ph_mm[mask]
        
        # DEM ORTHO CORRECTION
        if ortho == 'ortho':
            lonlat_err = lonlat_err.T
            lonlat -= lonlat_err
            hgt += dem_err
        
        # Format data
        ids = [f'PS_{int(x)}' for x in xy[:, 0]]
        lon = [round(float(x), 8) for x in lonlat[:, 0]]
        lat = [round(float(x), 8) for x in lonlat[:, 1]]
        hgt = [round(float(x), 8) for x in hgt]
        coh = [round(float(x), 8) for x in coh_ps]
        vlos = [round(float(x), 8) for x in ph_disp]
        days = [f'D{pd.to_datetime(d - 693960, origin="1899-12-30", unit="D").strftime("%Y%m%d")}' for d in day]
        
        d_mm0 = ph_mm - ph_mm[:, [0]]  # First measurement as reference
        d_mm = d_mm0.tolist()
        
        # Create DataFrame
        header = ['CODE', 'LON', 'LAT', 'HEIGHT', 'COHERENCE', 'VLOS'] + days
        data = list(zip(ids, lon, lat, hgt, coh, vlos, *zip(*d_mm)))
        gis_data = pd.DataFrame(data, columns=header)
        gis_data = gis_data[['CODE', 'LON', 'LAT', 'HEIGHT', 'COHERENCE', 'VLOS'] + list(sorted(days, key=lambda x: int(x[1:])))]
        gis_data.to_csv(csv_filename, index=False) # need to save file for CRLink
        coordinates = gpd.GeoSeries([Point(float(lon[i]), float(lat[i])) for i in range(len(lon))])
        gdf = gpd.GeoDataFrame(gis_data, geometry=coordinates, crs="EPSG:4326")
        gdf.to_file(f'{self.config["project_definition"]["data_folder"]}geom/temp_cr.shp', driver='ESRI Shapefile')

        # Rasterize the data
        print("   -> Interpolating data...")
        gdf = gpd.read_file(f'{self.config["project_definition"]["data_folder"]}geom/temp_cr.shp')  # or create it manually
        gdf = gdf.set_crs(4326)
        if gdf.crs.is_geographic:
            gdf = gdf.to_crs(epsg=32648)
        
        # Use smaller buffer for more precise pixel centers
        buffer = 8  # reduced from 16 for better precision
        columns = ['HEIGHT', 'VLOS', 'COHERENCE'] + [c for c in gdf.columns if re.match(r'^D', c)]
        stacked_array, transform = self.rasterize_preserve_and_stack(gdf, columns, pixel_size=10, window_radius=buffer)
        
        print("   -> Converting raster to points...")
        interpolated_data = self.raster_to_points(stacked_array, transform, columns)
        
        # Ensure proper coordinate conversion back to WGS84
        interpolated_data = interpolated_data.to_crs(epsg=4326)
        interpolated_data = interpolated_data.dropna(subset=['VLOS'])
        
        # Update LON and LAT columns with the new coordinates
        interpolated_data['LON'] = None
        interpolated_data['LAT'] = None
        interpolated_data['LON'] = interpolated_data.geometry.x
        interpolated_data['LAT'] = interpolated_data.geometry.y
        
        # Generate unique PS codes
        interpolated_data["CODE"] = [f"PS_{i}" for i in range(len(interpolated_data))]
        
        del gis_data
        del gdf
        print("   -> Adding color schema...")
        # Choose the number of classes
        n_classes = 10  # adjust based on how much detail you want

        # Compute Jenks natural breaks
        vlos_values = interpolated_data['VLOS'].values
        
        breaks = jenkspy.jenks_breaks(vlos_values, n_classes=n_classes)

        # Create colormap (Spectral has max 11 distinct steps)
        cmap = plt.get_cmap('Spectral', n_classes)
        # norm = mcolors.BoundaryNorm(breaks, cmap.N)

        # Assign class index and corresponding color
        interpolated_data['BINS'] = np.digitize(vlos_values, breaks, right=True) - 1
        interpolated_data['COLOR'] = interpolated_data['BINS'].apply(lambda i: mcolors.to_hex(cmap(i)) if 0 <= i < cmap.N else "#000000")

        group_df = interpolated_data.groupby('COLOR')
        legend_values = {"legend_settings": []}
        color_dict = []
        for idx, row in group_df["VLOS"].describe().iterrows():
            color_dict.append({
                "upper_threshold": row["max"],
                "color": idx
            })
        legend_values["legend_settings"] = color_dict
        legend_values["legend_settings"] = sorted(legend_values["legend_settings"], key=lambda x: x['upper_threshold'])
        interpolated_data = interpolated_data.drop(columns=["BINS", "COLOR"])
        interpolated_data["COLOR"] = None
        interpolated_data.at[0, "COLOR"] = legend_values
        
        # Save to CSV
        gis_data = pd.DataFrame(interpolated_data.drop(columns=["geometry"]))
        gis_data.to_csv(csv_filename.replace('_cr', ''), index=False)
        print(f"   -> CSV data saved to {csv_filename.replace('_cr', '')}")
        
        # Convert to Shapefile
        interpolated_data = interpolated_data.to_crs(epsg=4326)
        interpolated_data = interpolated_data.drop(columns=["COLOR"])
        interpolated_data.to_file(shapefile_name, driver='ESRI Shapefile')
        print(f"   -> Shapefile saved to {shapefile_name}")
        
        # Clean up temporary file
        if os.path.exists(f'{self.config["project_definition"]["data_folder"]}geom/temp_cr.shp'):
            os.remove(f'{self.config["project_definition"]["data_folder"]}geom/temp_cr.shp')
            os.remove(f'{self.config["project_definition"]["data_folder"]}geom/temp_cr.shx')
            os.remove(f'{self.config["project_definition"]["data_folder"]}geom/temp_cr.dbf')
            os.remove(f'{self.config["project_definition"]["data_folder"]}geom/temp_cr.prj')
            os.remove(f'{self.config["project_definition"]["data_folder"]}geom/temp_cr.cpg')
    
    def run(self):
        self.csv_files = []
        if platform.system() == 'Windows':
            matlab_cmd = (
                f"\"C:/Program Files/MATLAB/R2024a/bin/matlab.exe\" -wait -nosplash {self.display} "
                f"-r \"run('{os.path.split(os.path.abspath(__file__))[0].replace(os.sep, '/')}/modules/StaMPS/autorun_{self.oobj.lower()}.m'); exit;\" "
                f"> \"{self.config['processing_parameters']['current_result'].replace(os.sep, '/')}/STAMPS.log\""
            )
            subprocess.run(matlab_cmd, shell=True)
        else:
            os.system(f"matlab -nojvm -nosplash -nodisplay -r \"run('{os.path.split(os.path.abspath(__file__))[0]}/modules/StaMPS/autorun_{self.oobj.lower()}.m'); exit;\" > {self.config['processing_parameters']['current_result']}/STAMPS.log")
        time.sleep(1)
        print('-> Exporting CSV data and Shapefiles...')
        patch_paths = [os.path.join(self.config["processing_parameters"]["current_result"], f) for f in os.listdir(self.config["processing_parameters"]["current_result"]) if f.startswith('PATCH_')]
        patch_identifier = [f for f in os.listdir(self.config["processing_parameters"]["current_result"]) if f.startswith('PATCH_')]
        for path, identity in zip(patch_paths, patch_identifier):
            csv_filename = os.path.join(self.config["project_definition"]["data_folder"], f"geom/{self.config['processing_parameters']['current_result'].split('/')[-1]}_{identity}_cr.csv")
            self.csv_files.append(csv_filename)
            shutil.copy(os.path.join(self.config["processing_parameters"]["current_result"], 'parms.json'), path)
            os.chdir(path)
            dem_err = self.ps_dem_err()
            self.ps_lonlat_err(dem_err)
            self.ps_export_gis(csv_filename, os.path.join(self.config["project_definition"]["data_folder"], f"geom/{self.config['processing_parameters']['current_result'].split('/')[-1]}_{identity}.shp"), [], [], 'ortho')
            os.chdir(self.config["processing_parameters"]["current_result"]) 
        os.chdir(self.config["project_definition"]["project_folder"])
        return self.csv_files

if __name__ == "__main__":
    StaMPSEXE(project_name="default").run()