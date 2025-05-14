import os
import re
import sys
import time

import geopandas as gpd
import jenkspy
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import scipy.io as sio
from shapely.geometry import Point
from affine import Affine


class StaMPSEXE:
    def __init__(self, oobj="normal", display=' -nodisplay'):
        super().__init__()
        
        self.display = display
        self.oobj = oobj
        self.inputfile = os.path.join(os.path.split(os.path.abspath(__file__))[0].replace("\\", "/")+'/modules/snap2stamps/bin', "project.conf")
        self._load_config()
        print(f"############## Running: Step 10: StaMPS ##############")
        
    def _load_config(self):
        with open(self.inputfile, 'r') as file:
            for line in file.readlines():
                key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                if key:
                    setattr(self, key, value)  # Dynamically set variables
                    
    def _csvtoshp(self):
        for patch in [os.path.join(self.CURRENT_RESULT, f) for f in os.listdir(self.CURRENT_RESULT) if f.startswith('PATCH_')]:
            csv_file = [f for f in os.listdir(patch) if f.endswith('.csv')]
            folder = patch.split('/')[-2]
            if csv_file:
                data = gpd.read_file(csv_file[0])
                data.geometry = [Point(x, y) for x, y in zip(data.LON, data.LAT)]
                data.to_file(os.path.join(self.DATAFOLDER, f'geom/{self.CURRENT_RESULT.split("/")[-1]}_{folder}.shp'))

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

    def ps_export_gis(self, csv_filename, shapefile_name, lon_rg=None, lat_rg=None, ortho=None):
        if not os.path.exists("ps_plot_ts_v-dao.mat"):
            command = "matlab -nosplash -nodisplay -r \"ps_plot('v-dao', 'a_linear', 'ts'); exit;\""
            os.system(command)
        print("-> TS V-dao done")
        
        if not os.path.exists("ps_plot_v-dao.mat"):
            command = "matlab -nosplash -nodisplay -r \"ps_plot('v-dao', 'a_linear', -1); exit;\""
            os.system(command)
        print("-> V-dao done")
        
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
        lonlat_err = lonlat_err[mask]
        hgt = hgt[mask]
        dem_err = dem_err[mask]
        coh_ps = coh_ps[mask]
        ph_disp = ph_disp[mask]
        ph_mm = ph_mm[mask]
        
        # DEM ORTHO CORRECTION
        if ortho == 'ortho':
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

        # Rasterize the data
        print("-> Rasterizing data...")
        coordinates = gpd.GeoSeries([Point(float(lon[i]), float(lat[i])) for i in range(len(lon))])
        gdf = gpd.GeoDataFrame(gis_data, geometry=coordinates, crs="EPSG:4326")
        if gdf.crs.is_geographic:
            gdf = gdf.to_crs(epsg=32648)
        buffer = 16  # normal resampling spacing
        columns = ['LON', 'LAT', 'HEIGHT', 'VLOS', 'COHERENCE'] + [c for c in gdf.columns if re.match(r'^D', c)]
        print("Interpolating data...")
        stacked_array, transform = self.rasterize_preserve_and_stack(gdf, columns, pixel_size=10.0, window_radius=buffer)
        print("Converting raster to points...")
        interpolated_data = self.raster_to_points(stacked_array, transform, columns)
        interpolated_data["CODE"] = ""
        for idx, row in interpolated_data.iterrows():
            row["CODE"] = "PS_" + str(idx)
        del gis_data
        del gdf
        print("-> Adding color schema...")
        # Choose the number of classes
        n_classes = 10  # adjust based on how much detail you want

        # Compute Jenks natural breaks
        vlos_values = interpolated_data['VLOS'].values
        breaks = jenkspy.jenks_breaks(vlos_values, n_classes=n_classes)

        # Create colormap (Spectral has max 11 distinct steps)
        cmap = cm.get_cmap('Spectral', n_classes)
        #norm = mcolors.BoundaryNorm(breaks, cmap.N)

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
        interpolated_data.to_file(csv_filename, driver='CSV', index=False, encoding='utf-8')
        print(f"-> CSV data saved to {csv_filename}")
        
        # Convert to Shapefile
        interpolated_data.set_crs("EPSG:4326", inplace=True)
        interpolated_data.to_file(shapefile_name, driver='ESRI Shapefile')
        print(f"-> Shapefile saved to {shapefile_name}")
    
    def run(self):
        self.csv_files = []
        os.system(f"matlab -nojvm -nosplash{self.display} -r \"run('{os.path.split(os.path.abspath(__file__))[0]}/modules/StaMPS/autorun_{self.oobj.lower()}.m'); exit;\" > {self.CURRENT_RESULT}/STAMPS.log")
        time.sleep(1)
        print('-> Exporting CSV data and Shapefiles...')
        patch_paths = [os.path.join(self.CURRENT_RESULT, f) for f in os.listdir(self.CURRENT_RESULT) if f.startswith('PATCH_')]
        patch_identifier = [f for f in os.listdir(self.CURRENT_RESULT) if f.startswith('PATCH_')]
        for path, identity in zip(patch_paths, patch_identifier):
            os.chdir(path)
            self.csv_files.append(os.path.join(self.DATAFOLDER, f'geom/{self.CURRENT_RESULT.split("/")[-1]}_{identity}.csv'))
            self.ps_export_gis(os.path.join(self.DATAFOLDER, f'geom/{self.CURRENT_RESULT.split("/")[-1]}_{identity}.csv'), os.path.join(self.DATAFOLDER, f'geom/{self.CURRENT_RESULT.split("/")[-1]}_{identity}.shp'), [], [], 'ortho')
            os.chdir(self.CURRENT_RESULT) 
        os.chdir(self.PROJECTFOLDER)
        return self.csv_files
if __name__ == "__main__":
    StaMPSEXE('').run()
