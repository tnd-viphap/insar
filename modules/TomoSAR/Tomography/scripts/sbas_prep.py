import os
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import json
import logging

class SBAS_Prep:
    def __init__(self, project_conf_path=None):
        """
        Initialize SBAS preparation class
        
        Args:
            project_conf_path (str): Path to project configuration file
        """
        self.project_conf_path = project_conf_path or "../../snap2stamps/bin/project.conf"
        self.parms = {
            'Created': datetime.today(),
            'small_baseline_flag': 'y'  # small baseline ifgs with multiple masters
        }
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging configuration"""
        logger = logging.getLogger('SBAS_Prep')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
        
    def _read_project_conf(self, key):
        """Read value from project configuration file"""
        try:
            with open(self.project_conf_path, 'r') as f:
                for line in f:
                    if key in line:
                        return line.split('=')[1].strip()
        except FileNotFoundError:
            self.logger.error(f"Could not open {self.project_conf_path}")
        return ''
        
    def _get_processor(self):
        """Get processor type from processor.txt file"""
        processor_file = Path('processor.txt')
        
        # Try to find processor.txt in current and parent directories
        for _ in range(3):
            if processor_file.exists():
                with open(processor_file, 'r') as f:
                    processor = f.read().strip()
                    if processor.lower() not in ['gamma', 'doris']:
                        self.logger.warning("This processor is not supported (doris and gamma)")
                    return processor
            processor_file = Path('..') / processor_file
            
        return 'doris'  # default processor
        
    def _set_default_parameters(self):
        """Set default parameter values for SBAS processing"""
        defaults = {
            'max_topo_err': 20,
            'quick_est_gamma_flag': 'y',
            'select_reest_gamma_flag': 'y',
            'filter_grid_size': 50,
            'filter_weighting': 'P-square',
            'gamma_change_convergence': 0.005,
            'gamma_max_iterations': 3,
            'slc_osf': 1,
            'clap_win': 32,
            'clap_low_pass_wavelength': 800,
            'clap_alpha': 1,
            'clap_beta': 0.3,
            'select_method': 'DENSITY',
            'density_rand': 2,  # Random-phase pixels per km2 (before weeding)
            'percent_rand': 1,  # Percent random-phase pixels (before weeding)
            'gamma_stdev_reject': 0,
            'weed_time_win': 730,
            'weed_max_noise': float('inf'),
            'weed_standard_dev': float('inf'),
            'weed_zero_elevation': 'n',
            'weed_neighbours': 'n',
            'unwrap_method': '3D_QUICK',
            'unwrap_patch_phase': 'n',
            'drop_ifg_index': [],
            'unwrap_la_error_flag': 'y',
            'unwrap_spatial_cost_func_flag': 'n',
            'unwrap_prefilter_flag': 'y',
            'unwrap_grid_size': 200,
            'unwrap_gold_n_win': 32,
            'unwrap_alpha': 8,
            'unwrap_time_win': 730,
            'unwrap_gold_alpha': 0.8,
            'unwrap_hold_good_values': 'n',
            'scla_drop_index': [],
            'scn_wavelength': 100,
            'scn_time_win': 365,
            'scn_deramp_ifg': [],
            'scn_kriging_flag': 'n',
            'ref_lon': [-float('inf'), float('inf')],
            'ref_lat': [-float('inf'), float('inf')],
            'ref_centre_lonlat': [0, 0],
            'ref_radius': float('inf'),
            'ref_velocity': 0,
            'n_cores': 1,
            'plot_dem_posting': 90,
            'plot_scatterer_size': 120,
            'plot_pixels_scatterer': 3,
            'plot_color_scheme': 'inflation',
            'shade_rel_angle': [90, 45],
            'lonlat_offset': [0, 0],
            'merge_resample_size': 100,  # grid size (in m) to resample to during merge of patches
            'merge_standard_dev': float('inf'),
            'scla_method': 'L2',
            'scla_deramp': 'n',
            'sb_scla_drop_index': [],
            'subtr_tropo': 'n',
            'tropo_method': 'a_l'
        }
        
        # Apply defaults for missing parameters
        for key, value in defaults.items():
            if key not in self.parms:
                self.parms[key] = value
                
    def _load_lambda_heading(self):
        """Load lambda and heading from files"""
        for param, filename in [('lambda', 'lambda.1.in'), ('heading', 'heading.1.in')]:
            if param not in self.parms:
                filepath = Path(filename)
                for _ in range(3):
                    if filepath.exists():
                        try:
                            with open(filepath, 'r') as f:
                                self.parms[param] = float(f.read().strip())
                            break
                        except (ValueError, IOError):
                            pass
                    filepath = Path('..') / filepath
                else:
                    self.parms[param] = float('nan')
                    
    def initialize(self):
        """Initialize parameters with default values"""
        # Get processor type
        self.parms['insar_processor'] = self._get_processor()
        
        # Set default parameters
        self._set_default_parameters()
        
        # Load lambda and heading
        self._load_lambda_heading()
        
        # Save parameters
        self.save()
        
    def save(self):
        """Save parameters to file"""
        result_folder = self._read_project_conf("CURRENT_RESULT")
        parmfile = Path(result_folder) / 'parms.json'
        
        try:
            # Convert datetime to string for JSON serialization
            parms_dict = self.parms.copy()
            parms_dict['Created'] = parms_dict['Created'].strftime('%Y-%m-%d')
            
            with open(parmfile, 'w') as f:
                json.dump(parms_dict, f, indent=4)
                
            # Log new parameters
            for key, value in self.parms.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"{key} = {value}")
                else:
                    self.logger.info(f"{key} = {str(value)}")
                    
        except (IOError, PermissionError) as e:
            self.logger.warning(f"Could not save parameters: {str(e)}")
            
    def load(self):
        """Load parameters from file"""
        result_folder = self._read_project_conf("CURRENT_RESULT")
        parmfile = Path(result_folder) / 'parms.json'
        
        if parmfile.exists():
            try:
                with open(parmfile, 'r') as f:
                    loaded_parms = json.load(f)
                    # Convert Created string back to datetime
                    loaded_parms['Created'] = datetime.strptime(
                        loaded_parms['Created'], '%Y-%m-%d')
                    self.parms.update(loaded_parms)
            except (IOError, json.JSONDecodeError) as e:
                self.logger.error(f"Error loading parameters: {str(e)}")
                
    def get(self, key, default=None):
        """Get parameter value"""
        return self.parms.get(key, default)
        
    def set(self, key, value):
        """Set parameter value"""
        self.parms[key] = value
        
    def __getitem__(self, key):
        """Allow dictionary-like access to parameters"""
        return self.parms[key]
        
    def __setitem__(self, key, value):
        """Allow dictionary-like setting of parameters"""
        self.parms[key] = value

if __name__ == "__main__":
    # Example usage
    sbas = SBAS_Prep()
    sbas.initialize() 