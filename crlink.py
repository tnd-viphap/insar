# type: ignore
import os
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import pandas as pd
import time
import paramiko

class CRLink:
    def __init__(self, psi_result_file, n_rovers):
        # Read input file
        try:
            inputfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], "modules/snap2stamps/bin/project.conf")
            self._update_n_rovers(inputfile, n_rovers)
            with open(inputfile, 'r') as file:
                for line in file.readlines():
                    key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                    if key:
                        setattr(self, key, value)
        except Exception as e:
            print(f"Error reading input file: {e}")

        self.psi_file = psi_result_file
        self.n_rovers = n_rovers
        self.local_gnss_file = f"{self.DATAFOLDER}gnss/output.csv"

    def _update_n_rovers(self, config_file, n_rovers):
        lines = ''''''
        with open(config_file, 'r') as file:
            lines = file.readlines()
            for idx, line in enumerate(lines):
                if line.startswith("NROVERS"):
                    lines[idx] = "NROVERS=" + str(n_rovers) + '\n'
        with open(config_file, 'w') as file:
            file.writelines(lines)
            file.close()
        time.sleep(1)

    def _fetch_gnss_data(self):
        if not os.path.exists(f"{self.DATAFOLDER}gnss/"):
            os.makedirs(f"{self.DATAFOLDER}gnss/", exist_ok=True)
        # SSH Connection
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.SERVERIP, username=self.SERVERNAME, password="Viphap@2023")
        # SFTP Connection
        sftp = ssh.open_sftp()
        sftp.get(self.REMOTE_GNSSFILE, self.local_gnss_file)
        sftp.close()
        ssh.close()
        time.sleep(1)

    def _get_total_loc_err(self):
        pta_folder = f"{self.PROJECTFOLDER}process/pta/"
        if not os.path.exists(pta_folder):
            os.makedirs(pta_folder)
        pta_result_folders = os.listdir(pta_folder)
        pta_result_files = []

        # Get all pta_results.csv in each pta folder
        if pta_result_folders:
            for folder in pta_result_folders:
                for file in os.listdir(pta_folder+folder):
                    if "pta_results.csv" in file:
                        pta_result_files.append(pta_folder+folder+"/"+file)

            data = []
            if pta_result_files:
                for file in pta_result_files:
                    datum = pd.read_csv(file)
                    data.append(datum)
                data = pd.concat(data, ignore_index=True)
            else:
                print("-> Missing PTA on data")
        else:
            data = None
            
        # Compute average error
        if len(data) > 0:
            target_names = data["target_name"].unique().tolist()
            
            self.target_data = {}
            grouped = data.groupby("target_name")
            del data
            for group in target_names:
                self.target_data[group] = {
                    "target_name": grouped.get_group(group)["target_name"],
                    "incidence_angle_[deg]": grouped.get_group(group)["incidence_angle_[deg]"].mean(),
                    "azimuth_angle_[deg]": 90 - ((np.pi)**-1)*180*grouped.get_group(group)["squint_angle_[rad]"].mean(),
                    "ground_range_localization_error_[m]": grouped.get_group(group)["ground_range_localization_error_[m]"].max(),
                    "azimuth_localization_error_[m]": grouped.get_group(group)["azimuth_localization_error_[m]"].max(),
                    "latitude_corrected_[deg]": grouped.get_group(group)["latitude_corrected_[deg]"].mean(),
                    "longitude_corrected_[deg]": grouped.get_group(group)["longitude_corrected_[deg]"].mean()
                }
        else:
            print("-> Missing PTA on data")

    def _cr_with_insar(self):
        # Load psi data
        psi = gpd.read_file(self.psi_file)
        
        # Get estimated errors
        self._get_total_loc_err()

        # Search for PS points in error radius range
        found_points = []
        for _, value in self.target_data.items():
            # Get search radius for searching nearby points among PS ones
            search_radius = np.sqrt((value["ground_range_localization_error_[m]"])**2 + (value["azimuth_localization_error_[m]"])**2)
            
            # Create a GeoDataFrame with WGS84 coordinates
            gdf = gpd.GeoDataFrame(geometry=[Point(value["longitude_corrected_[deg]"], value["latitude_corrected_[deg]"])], crs="EPSG:4326")
            utm_crs = gdf.estimate_utm_crs()
            gdf_utm = gdf.to_crs(utm_crs)
            gdf_buffered = gdf_utm.buffer(search_radius, 1)
            gdf_geom = gpd.GeoSeries(gdf_buffered, crs=utm_crs).to_crs("EPSG:4326").geometry[0]

            # Capture nearby points
            nearby = psi.loc[gdf_geom.covers(psi.geometry)]
            nearby = nearby.reset_index()
            nearby["target_id"] = value["target_name"]
            
            # Compute the nearest point
            cr_point = np.array([float(value["longitude_corrected_[deg]"]), float(value["latitude_corrected_[deg]"])])
            if len(nearby) > 0:
                distances = []
                for idx, row in nearby.iterrows():
                    ps_point = np.array([float(row["LON"]), float(row["LAT"])])
                    distance = np.linalg.norm(ps_point - cr_point)
                    distances.append(distance)
                # Get the point
                try:
                    min_dist = np.min(distances)
                    if min_dist != 'nan':
                        ps_idx = distances.index(min_dist)
                        found_points.append(nearby.loc[ps_idx])
                    else:
                        found_points.append(nearby.loc[0])
                except:
                    print(f"-> No {value['target_name'].values[0].upper()} point found on PS data")

        if found_points:
            self.insar_points = gpd.GeoDataFrame(found_points)
            del found_points
            self.insar_points.to_file(self.psi_file.replace("INSAR", "CRLink"), index=False)

    def _cr_with_gnss(self):
        # Load GNSS data
        gnss_data = pd.read_csv(self.local_gnss_file)
        
        # Convert TIMESTAMP to datetime
        gnss_data['TIMESTAMP'] = pd.to_datetime(gnss_data['TIMESTAMP'])
        
        # Get slave dates
        if len(os.listdir(self.SLAVESFOLDER)) > 0:
            slave_dates = [d for d in os.listdir(self.SLAVESFOLDER) if os.path.isdir(os.path.join(self.SLAVESFOLDER, d))]
            slave_dates = [pd.to_datetime(d, format="%Y%m%d") for d in slave_dates]
        
            # Process each target
            self.gnss_results = []
            for target_name, target_info in self.target_data.items():
                # Find GNSS data for each slave date
                for slave_date in slave_dates:
                    # Get GNSS data for this date
                    date_data = gnss_data[gnss_data['TIMESTAMP'].dt.date == slave_date.date()]
                    
                    if len(date_data) == 0:
                        continue
                    
                    # Process each rover
                    for rover in range(1, self.n_rovers + 1 if self.n_rovers else len([col for col in gnss_data.columns if 'Delta_E' in col]) + 1):
                        # Get displacement columns for this rover
                        e_col = f'Delta_E{rover}'
                        n_col = f'Delta_N{rover}'
                        u_col = f'Delta_U{rover}'
                        
                        if not all(col in date_data.columns for col in [e_col, n_col, u_col]):
                            continue
                        
                        # Calculate average displacement for this rover on this date
                        avg_e = date_data[e_col].mean()
                        avg_n = date_data[n_col].mean()
                        avg_u = date_data[u_col].mean()

                        # Convert to VLOS of GNSS metrics
                        incidence_angle = target_info["incidence_angle_[deg]"]
                        azimuth_angle = target_info["azimuth_angle_[deg]"]
                        gnss_vlos = -(avg_e*np.cos(azimuth_angle*np.pi/180) + avg_n*np.sin(azimuth_angle*np.pi/180))*np.cos(incidence_angle*np.pi/180) - avg_u*np.cos(incidence_angle*np.pi/180)
                        
                        # Calculate total horizontal displacement
                        total_horizontal = np.sqrt(avg_e**2 + avg_n**2)
                        
                        # Store results
                        self.gnss_results.append({
                            'target_name': target_name,
                            'date': slave_date.strftime('%Y%m%d'),
                            'rover': rover,
                            'longitude': target_info["longitude_corrected_[deg]"],
                            'latitude': target_info["latitude_corrected_[deg]"],
                            'total_h_error_m': total_horizontal,
                            'ground_range_error_m': target_info["ground_range_localization_error_[m]"],
                            'azimuth_error_m': target_info["azimuth_localization_error_[m]"],
                            'gnss_vlos_m': gnss_vlos
                        })
                        
            # Convert results to DataFrame and save
            if self.gnss_results:
                result_dfs = []
                for result in self.gnss_results:
                    result_dfs.append(pd.DataFrame(result))
                self.gnss_results = pd.concat(result_dfs, ignore_index=True)
                output_file = self.local_gnss_file.replace('output.csv', f'GNSS_CRLink.csv')
                self.gnss_results.to_csv(output_file, index=False)
                print(f"-> GNSS-CR link results saved to {output_file}")
            else:
                print("-> No matching GNSS data found for the CR points")
            del result_dfs
        else:
            print("-> Empty SLAVES data")

    def _combined_insar_gnss(self):
        # Combined INSAR and GNSS results
        if not os.path.exists(f"{self.DATAFOLDER}crlink/"):
            os.makedirs(f"{self.DATAFOLDER}crlink/", exist_ok=True)
        self.combined_results = pd.merge(self.insar_points, self.gnss_results, on=['target_name'], how='inner')
        self.combined_results.to_csv(f"{self.DATAFOLDER}crlink/CRLink_results.csv", index=False)

    def run(self):
        self._fetch_gnss_data()
        self._cr_with_insar()
        self._cr_with_gnss(self.n_rovers)
        self._combined_insar_gnss()

if __name__ == "__main__":
    from modules.snap2stamps.bin._0_engage import Initialize
    Initialize([108.8721, 15.1294, 108.8996, 15.1569], "DESCENDING", 540, 150.0, 0, 5, 0)
    time.sleep(2)
    cr_link = CRLink("dev/pta/ps_data/INSAR_20200408_PSDS_v2_PATCH_1.shp", 2)
    #cr_link._fetch_gnss_data()
    cr_link._get_total_loc_err()
    cr_link._cr_with_insar()
    #cr_link._cr_with_gnss()
    None