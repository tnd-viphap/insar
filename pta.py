import os

from sct.analyses.point_target_analysis import point_target_analysis_with_corrections
from sct.analyses.graphical_output import sct_pta_graphs
from sct.configuration.sct_configuration import SCTConfiguration

class PTA:
    def __init__(self, config, prod):
        self.config = SCTConfiguration.from_toml(config)
        self.prod = prod

    def process_pta(self):
        path_to_external_orbit = None
        targets_csv_file_path = "modules/pta/pta_target.csv"
        output_results_csv_file = "process/pta/pta_results.csv"
        graphs_output_directory = "process/pta/pta_graphs"
        if not os.path.exists(graphs_output_directory):
            os.makedirs(graphs_output_directory, exist_ok=True)

        results_df, data_for_graphs = point_target_analysis_with_corrections(
            product_path=self.prod,
            external_target_source=targets_csv_file_path,
            external_orbit_path=path_to_external_orbit,
            config=self.config.point_target_analysis,
        )
        results_df.to_csv(output_results_csv_file, index=False)

        # optional, if graphical output is needed
        sct_pta_graphs(graphs_data=data_for_graphs, results_df=results_df, output_dir=graphs_output_directory)

if __name__ == "__main__":
    config = "modules/pta/pta.toml"
    prod = "data/S1A_IW_SLC__1SDV_20250301T223610_20250301T223640_058117_072D83_9435.SAFE"
    pta = PTA(config, prod)
    pta.process_pta()

