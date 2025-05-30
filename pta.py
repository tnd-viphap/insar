# type: ignore
import os
import shutil
import time
import zipfile

import numpy as np
import pandas as pd
from sct.analyses.graphical_output import sct_pta_graphs
from sct.analyses.point_target_analysis import point_target_analysis_with_corrections
from sct.configuration.sct_configuration import SCTConfiguration

from modules.utils.compute_cr_ea import gps2ecef_pyproj, shift_target_point


class PTA:
    def __init__(self, prod, eof=None):

        # Read input file
        inputfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], "modules/snap2stamps/bin/project.conf")
        with open(inputfile, 'r') as file:
            for line in file.readlines():
                key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                if key:
                    setattr(self, key, value)
        
        self.config = SCTConfiguration.from_toml("modules/pta/pta.toml")
        self.prod = prod
        self.eof = eof
        
        prod_date = os.path.split(self.prod)[-1][17:25]
        self.pta_output_folder = f"{self.PROJECTFOLDER}/process/pta/{prod_date}"
        self.pta_target_file = f"{self.PROJECTFOLDER}/process/pta/{prod_date}/pta_target.csv"
        self.pta_output_result_file = f"{self.PROJECTFOLDER}/process/pta/{prod_date}/pta_results.csv"
        self.pta_output_graphs = f"{self.PROJECTFOLDER}/process/pta/{prod_date}/pta_graphs"
        if not os.path.exists(f"{self.PROJECTFOLDER}/process/pta"):
            os.makedirs(f"{self.PROJECTFOLDER}/process/pta")
        if not os.path.exists(self.pta_output_folder):
            os.makedirs(self.pta_output_folder)
        if not os.path.exists(self.pta_output_graphs):
            os.makedirs(self.pta_output_graphs)
            
        self.extract_folder_name = os.path.join(self.pta_output_folder, os.path.split(self.prod)[-1].replace(".zip", ".SAFE"))
        if not os.path.exists(self.pta_target_file):
            shutil.copy(f"modules/pta/pta_target.csv", self.pta_output_folder)
        
    def pta(self):
        # Unzip raw file
        if self.prod.endswith(".zip"):
            with zipfile.ZipFile(self.prod,"r") as zip_ref:
                if not os.path.exists(self.extract_folder_name):
                    zip_ref.extractall(self.pta_output_folder)
                zip_ref.close()
        
        # Separate target points
        target_point_filenames = []
        target_points = pd.read_csv(self.pta_target_file)
        
        # Create a directory for individual target files
        target_points_dir = os.path.join(self.pta_output_folder, "target_points")
        if not os.path.exists(target_points_dir):
            os.makedirs(target_points_dir)
        target_results_dir = os.path.join(self.pta_output_folder, "target_results")
        if not os.path.exists(target_results_dir):
            os.makedirs(target_results_dir)
            
        # Save each target point to a separate file
        for idx, row in target_points.iterrows():
            # Create a new DataFrame with the same columns and single row
            sep_df = pd.DataFrame([row], columns=target_points.columns)
            target_filename = os.path.join(target_points_dir, f"pta_target_{row['target_name']}.csv")
            sep_df.to_csv(target_filename, index=False)
            target_point_filenames.append(target_filename)

        # PTA for each target point
        for target_file in target_point_filenames:
            count = 0
            self.shift_df = None
            target_data = pd.read_csv(target_file)
            while True:
                # if count == 1:
                #     break
                results_df, data_for_graphs = point_target_analysis_with_corrections(
                    product_path=self.extract_folder_name,
                    external_target_source=target_file,
                    external_orbit_path=self.eof,
                    config=self.config.point_target_analysis,
                )
                results_df = results_df.loc[results_df["polarization"] == "V/V"]
                result_columns = results_df.columns
                results_df = results_df.merge(target_data, on="target_name", how="left")
                results_df = results_df[result_columns.append(pd.Index(["latitude_deg", "longitude_deg", "altitude_m"]))]

                
                # Save results with target name in filename
                target_name = target_data['target_name'][0]
                results_filename = os.path.join(target_results_dir, f"pta_results_{target_name}.csv")
                if self.shift_df is not None and len(self.shift_df) > 0:
                    results_df["latitude_corrected_[deg]"] = self.shift_df["latitude_corrected_[deg]"]
                    results_df["longitude_corrected_[deg]"] = self.shift_df["longitude_corrected_[deg]"]
                results_df.to_csv(results_filename, index=False)

                # optional, if graphical output is needed
                for file in os.listdir(self.pta_output_graphs):
                    if target_name in file:
                        os.remove(os.path.join(self.pta_output_graphs, file))
                sct_pta_graphs(graphs_data=data_for_graphs, results_df=results_df, output_dir=self.pta_output_graphs)

                # Get the error values for the first (and only) row
                gr_error = float(results_df["ground_range_localization_error_[m]"].iloc[0])
                az_error = float(results_df["azimuth_localization_error_[m]"].iloc[0])

                # Track which coordinates have been optimized
                gr_optimized = abs(gr_error) <= 2.3
                az_optimized = abs(az_error) <= 14.1

                # Check if both criteria are met
                if gr_optimized and az_optimized:
                    print(f"-> Target {target_name} meets all allowed errors after {count} iterations")
                    break

                # Track if we're making progress
                if count > 0:
                    prev_gr_error = float(self.shift_df["ground_range_localization_error_[m]"].iloc[0])
                    prev_az_error = float(self.shift_df["azimuth_localization_error_[m]"].iloc[0])
                    
                    # Check if errors are getting worse
                    if (not gr_optimized and abs(gr_error) > abs(prev_gr_error)) or \
                       (not az_optimized and abs(az_error) > abs(prev_az_error)):
                        print(f"-> Target {target_name} errors are increasing, stopping at iteration {count}")
                        break

                # Maximum iterations based on optimization status
                max_iterations = 20 if not (gr_optimized or az_optimized) else 10
                if count >= max_iterations:
                    print(f"-> Target {target_name} reached maximum iterations ({count})")
                    print(f"   Ground range error: {gr_error:.2f}m (optimized: {gr_optimized})")
                    print(f"   Azimuth error: {az_error:.2f}m (optimized: {az_optimized})")
                    break

                # Update coordinates based on which criteria are not met
                self.shift_df = shift_target_point(results_filename)
                lat_cors, lon_cors = self.shift_df["latitude_corrected_[deg]"], self.shift_df["longitude_corrected_[deg]"]
                if lat_cors[0] and lon_cors[0]:
                    alts = target_data["altitude_m"]
                    
                    # Only update coordinates that haven't been optimized
                    if not gr_optimized:
                        target_data.loc[0, "longitude_deg"] = lon_cors[0]
                    if not az_optimized:
                        target_data.loc[0, "latitude_deg"] = lat_cors[0]
                    
                    # Always update ECEF coordinates
                    x, y, z = gps2ecef_pyproj(target_data.loc[0, "latitude_deg"], 
                                            target_data.loc[0, "longitude_deg"], 
                                            alts[0])
                    target_data.loc[0, "x_coord_m"] = x
                    target_data.loc[0, "y_coord_m"] = y
                    target_data.loc[0, "z_coord_m"] = z

                    target_data.to_csv(target_file, index=False)
                    count += 1
                    time.sleep(1)
        
        # Gather results
        shutil.rmtree(target_points_dir)
        dfs = []
        for file in os.listdir(target_results_dir):
            dfs.append(pd.read_csv(os.path.join(target_results_dir, file)))
        results_df = pd.concat(dfs, ignore_index=True)
        results_df.to_csv(self.pta_output_result_file, index=False)
        shutil.rmtree(target_results_dir)
        shutil.rmtree(self.extract_folder_name)


if __name__ == "__main__":
    prod = "dev/S1A_IW_SLC__1SDV_20250301T223610_20250301T223640_058117_072D83_9435.zip"
    #eof = "dev/S1A_OPER_AUX_POEORB_OPOD_20250322T070533_V20250301T225942_20250303T005942.EOF"
    pta = PTA(prod, None)
    pta.pta()