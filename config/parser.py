# type: ignore
import json
import os
import psutil
import platform
from typing import Optional, Dict, Any, List, Union

class ConfigParser:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in configuration file {self.config_path}")
    
    def get_project_config(self, project_name: str) -> Dict[str, Any]:
        """Get configuration for a specific project"""
        if project_name not in self.config['projects']:
            raise KeyError(f"Project '{project_name}' not found in configuration")
        return self.config['projects'][project_name]
    
    def update_project_config(self, project_name: str, config_data: Dict[str, Any]) -> None:
        """Update configuration for a specific project"""
        if project_name not in self.config['projects']:
            raise KeyError(f"Project '{project_name}' not found in configuration")
        self.config['projects'][project_name] = config_data
        self._save_config()
    
    def _save_config(self) -> None:
        """Save configuration to JSON file"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

class Initialize:
    def __init__(self, bbox: List[float], direction: str, frame: int, 
                 max_perp: Optional[float] = None, ptype: Optional[str] = None, 
                 stack_size: Optional[int] = None, unified_flag: Optional[int] = None,
                 project_name: str = "default", config_path: Optional[str] = None):
        super().__init__()
        plf = platform.system()
        
        self.bbox = bbox
        self.direction = direction
        self.frame = frame
        self.max_perp = max_perp
        self.ptype = ptype
        self.stack_size = stack_size
        self.unified_flag = unified_flag
        self.project_name = project_name
        
        # Initialize config parser
        if not config_path:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     'config', 'config.json')
        self.config_parser = ConfigParser(config_path)
        
        # Get project folder
        project_folder = os.path.abspath(os.path.dirname(os.path.dirname(__file__))).replace('\\', '/')
        if plf == "Windows":
            project_in_disk = project_folder[0].upper()
            project_folder = f"{project_in_disk}:{project_folder.split(':')[1]}"
        
        # Define folder structure
        folders = {
            'log_folder': project_folder + '/logs/',
            'data_folder': project_folder + '/data/',
            'master_folder': project_folder + '/data/master/',
            'slaves_folder': project_folder + '/data/slaves/',
            'rawdata_folder': project_folder + '/data/raw/',
            'coreg_folder': project_folder + '/process/coreg/',
            'ifg_folder': project_folder + '/process/ifg/',
            'stamp_folder': project_folder + '/results/',
            'process_folder': project_folder + '/process/'
        }
        
        # Create folders
        for folder in folders.values():
            os.makedirs(folder, exist_ok=True)
        
        # Set up file paths
        graphs_folder = project_folder + '/modules/snap2stamps/graphs/'
        
        # Set up cache files
        cache_files = {
            'download_cache': project_folder + '/data/download_cache.txt',
            'broken_cache': project_folder + '/data/broken_cache.txt',
            'baseline_cache': project_folder + '/data/baseline_cache.txt',
            'datalake': project_folder + '/data/lake.json'
        }
        
        # Create cache files if they don't exist
        for file_path in cache_files.values():
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    if file_path.endswith('.json'):
                        json.dump([], f, indent=4)
                    f.close()
        
        # Set up system resources
        n_cores = round(os.cpu_count() * 0.8)
        total_ram = round(psutil.virtual_memory().total / (1024 ** 3) * 0.8)
        
        # Set up GPT path based on platform
        if plf == "Windows":
            gpt = "C:/Program Files/snap/bin/gpt.exe"
        elif plf == "Linux":
            gpt = os.path.split(os.path.abspath(__file__))[0].split('insar')[0] + 'snap/bin/gpt'
        
        # Update configuration
        config_data = {
            'project_definition': {
                'project_folder': project_folder,
                'graphs_folder': graphs_folder,
                'log_folder': folders['log_folder'],
                'data_folder': folders['data_folder'],
                'master_folder': folders['master_folder'],
                'master': folders['master_folder'] + os.listdir(folders['master_folder'])[0] + f"/{os.listdir(folders['master_folder'])[0]}_M.dim" if os.listdir(folders['master_folder']) else "",
                'slaves_folder': folders['slaves_folder'],
                'raw_data_folder': folders['rawdata_folder'],
                'coreg_folder': folders['coreg_folder'],
                'ifg_folder': folders['ifg_folder'],
                'stamp_folder': folders['stamp_folder'],
                'config_path': project_folder + '/modules/snap2stamps/bin/project.conf'
            },
            'cache_files': {
                'download_cache': cache_files['download_cache'],
                'broken_cache': cache_files['broken_cache'],
                'baseline_cache': cache_files['baseline_cache']
            },
            'search_parameters': {
                'datalake': cache_files['datalake'],
                'direction': self.direction,
                'frame': self.frame
            },
            'processing_parameters': {
                'iw1': 'IW1',  # Default value from conf files
                'first_burst': 5,  # Default value from conf files
                'last_burst': 5,  # Default value from conf files
                'old_master': os.listdir(folders['master_folder'])[0],  # Will be set when master is selected
                'max_perp': self.max_perp or 150.0,  # Default from conf files
                'da_threshold': 0.4,  # Default from conf files
                'reest_flag': 1,  # Default from conf files
                'current_result': folders['stamp_folder'] + list(sorted(os.listdir(folders['stamp_folder']), key=lambda x: int(x.split('_')[1])))[-1],  # Will be set during processing
                'ministack': self.stack_size or 5,  # Default from conf files
                'unified': self.unified_flag or 0,  # Default from conf files
                'ptype': self.ptype
            },
            'aoi_bbox': {
                'lon_min': self.bbox[0],
                'lat_min': self.bbox[1],
                'lon_max': self.bbox[2],
                'lat_max': self.bbox[3]
            },
            'snap_gpt': {
                'gptbin_path': gpt
            },
            'api_flags': {
                'comsar': 0  # Default from conf files
            },
            'gnss': {
                'remote_gnssfile': 'C:/Users/ViphapCom2/Documents/GNSS_Viphap/gnss/data/output.csv',  # Default from conf files
                'server_name': 'ViphapCom2',  # Default from conf files
                'server_ip': '192.168.101.100',  # Default from conf files
                'nrovers': 2  # Default from conf files
            },
            'computing_resources': {
                'cpu': n_cores,
                'cache': f"{total_ram}G"
            }
        }
        
        # Update JSON config
        self.config_parser.update_project_config(self.project_name, config_data)

if __name__ == "__main__":
    try:
        bbox = [106.6969, 10.7615, 106.7275, 10.7945]
        Initialize(bbox, "DESCENDING", 540, project_name="default")
    except Exception as e:
        print(f"Engage project structure fails due to\n{e}\n")