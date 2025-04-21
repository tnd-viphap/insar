import os
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

    def ps_export_gis(self, filename, shapefile_name, lon_rg=None, lat_rg=None, ortho=None):
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
        # Choose the number of classes
        n_classes = 10  # adjust based on how much detail you want

        # Compute Jenks natural breaks
        vlos_values = gis_data['VLOS'].values
        breaks = jenkspy.jenks_breaks(vlos_values, n_classes=n_classes)

        # Create colormap (Spectral has max 11 distinct steps)
        cmap = cm.get_cmap('Spectral', n_classes)
        norm = mcolors.BoundaryNorm(breaks, cmap.N)

        # Assign class index and corresponding color
        gis_data['BINS'] = np.digitize(vlos_values, breaks, right=True) - 1
        gis_data['COLOR'] = gis_data['BINS'].apply(lambda i: mcolors.to_hex(cmap(i)) if 0 <= i < cmap.N else "#000000")

        group_df = gis_data.groupby('COLOR')
        legend_values = {"legend_settings": []}
        color_dict = []
        for idx, row in group_df["VLOS"].describe().iterrows():
            color_dict.append({
                "upper_threshold": row["max"],
                "color": idx
            })
        legend_values["legend_settings"] = color_dict
        legend_values["legend_settings"] = sorted(legend_values["legend_settings"], key=lambda x: x['upper_threshold'])
        gis_data = gis_data.drop(columns=["BINS", "COLOR"])
        gis_data["COLOR"] = None
        gis_data.at[0, "COLOR"] = legend_values
        
        # Save to CSV
        gis_data.to_csv(filename, index=False)
        print(f"-> Data saved to {filename}")
        
        # Convert to Shapefile
        geometry = [Point(float(lon[i]), float(lat[i])) for i in range(len(lon))]
        gdf = gpd.GeoDataFrame(gis_data, geometry=geometry, crs="EPSG:4326")
        gdf.to_file(shapefile_name, driver='ESRI Shapefile')
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
            self.ps_export_gis(os.path.join(self.DATAFOLDER, f'geom/{self.CURRENT_RESULT.split("/")[-1]}_{identity}.shp'), os.path.join(self.DATAFOLDER, 'geom') + f"/{self.CURRENT_RESULT.split('/')[-1]}_{identity}.shp", [], [], 'ortho')
            os.chdir(self.CURRENT_RESULT) 
        os.chdir(self.PROJECTFOLDER)
        return self.csv_files
if __name__ == "__main__":
    StaMPSEXE('').run()
