import os
import shutil
import time
import zipfile

import pandas as pd
from sct.analyses.graphical_output import sct_pta_graphs
from sct.analyses.point_target_analysis import point_target_analysis_with_corrections
from sct.configuration.sct_configuration import SCTConfiguration

from modules.utils.compute_cr_ea import gps2ecef_pyproj, shift_target_point


class PTA:
    def __init__(self, prod):

        # Read input file
        inputfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], "modules/snap2stamps/bin/project.conf")
        with open(inputfile, 'r') as file:
            for line in file.readlines():
                key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                if key:
                    setattr(self, key, value)
        
        self.config = SCTConfiguration.from_toml(self.PROJECTFOLDER+"modules/pta/pta.toml")
        self.prod = prod

    def pta(self):
        count = 0
        prod_date = os.path.split(prod)[-1][17:25]
        pta_output_folder = f"{self.PROJECTFOLDER}process/{prod_date}"
        pta_target_file = f"{self.PROJECTFOLDER}process/{prod_date}/pta_target.csv"
        pta_output_result_file = f"{self.PROJECTFOLDER}process/{prod_date}/pta_results.csv"
        pta_output_graphs = f"{self.PROJECTFOLDER}process/{prod_date}/pta_graphs"
        if not os.path.exists(self.PROJECTFOLDER+"process/pta"):
            os.makedirs(self.PROJECTFOLDER+"process/pta")
        if not os.path.exists(pta_output_folder):
            os.makedirs(pta_output_folder)
        if not os.path.exists(pta_output_graphs):
            os.makedirs(pta_output_graphs)

        if self.prod.endswith(".zip"):
            with zipfile.ZipFile(self.prod,"r") as zip_ref:
                zip_ref.extractall(pta_output_folder)
                parent_folder = os.listdir(pta_output_folder)
                main_folder = f"{pta_output_folder}/{parent_folder}"
                shutil.move(f"{main_folder}/{os.listdir(main_folder)[0]}", pta_output_folder)
                os.remove(parent_folder)
                zip_ref.close()

        if not os.path.exists(pta_target_file):
            shutil.copy(f"{self.PROJECTFOLDER}modules/pta/pta_target.csv", pta_output_folder)

        while True:
            results_df, data_for_graphs = point_target_analysis_with_corrections(
                product_path=self.prod,
                external_target_source=pta_target_file,
                external_orbit_path=None,
                config=self.config.point_target_analysis,
            )
            target_data = pd.read_csv(pta_target_file)
            results_df = pd.merge(results_df, target_data, on=['target_name'], how="inner")
            results_df.to_csv(pta_output_result_file, index=False)

            # optional, if graphical output is needed
            sct_pta_graphs(graphs_data=data_for_graphs, results_df=results_df, output_dir=pta_output_graphs)
        
            # check for valid ground range and azimuth localization error
            if abs(float(results_df["ground_range_localization_error_[m]"])) <= 2.3 and abs(float(results_df["azimuth_range_localization_error"])) <= 14.1:
                print(f"-> Update CR localization {count} times for {prod_date}")
                os.remove(main_folder)
                time.sleep(1)
                break

            # Update coordinates
            lat_cors, lon_cors = shift_target_point(pta_target_file)
            alts = target_data.loc[(target_data['target_name'].str.contains('on')) & (target_data['polarization'].str.contains('V/V'))]["altitude_m"]
            for lat, lon, alt in zip(lat_cors, lon_cors, alts):
                x, y, z, = gps2ecef_pyproj(lat, lon, alt)
                for axis, key in zip([x, y, z], ["longitude_deg", "latitude_deg", "altitude_m"]):
                    target_data[key] = axis
            new_target_data = target_data.copy()
            target_data = None
            new_target_data.to_csv(pta_target_file)
            count += 1
            time.sleep(1)


if __name__ == "__main__":
    prod = "data/S1A_IW_SLC__1SDV_20250301T223610_20250301T223640_058117_072D83_9435.SAFE"
    pta = PTA(prod)
    pta.pta()

