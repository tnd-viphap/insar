# type: ignore
import json
import os
import psutil
import platform
import sys

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')).replace("\\", "/")
sys.path.append(project_path)

from config.parser import ConfigParser

class Initialize:
    def __init__(self, bbox, direction, frame, max_perp=None, ptype=None, stack_size=None, unified_flag=None, project_name="default"):
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
        config_path = os.path.join(project_path, "config", "config.json")
        self.config_parser = ConfigParser(config_path)
        self.config = self.config_parser.get_project_config(self.project_name)
        
        # Set up project folders
        project_folder = os.path.join(project_path).replace("\\", "/")
        if plf == "Windows":
            project_in_disk = project_folder[0].upper()
            project_folder = f"{project_in_disk}:{project_folder.split(':')[1]}"
            
        # Create folder structure
        folders = {
            'project_folder': project_folder,
            'graphs_folder': os.path.join(project_folder, 'modules/snap2stamps/graphs/'),
            'log_folder': os.path.join(project_folder, 'logs/'),
            'data_folder': os.path.join(project_folder, 'data/'),
            'master_folder': os.path.join(project_folder, 'data/master/'),
            'master': os.path.join(project_folder, 'data/master/20200101/20200101_M.dim'),
            'slaves_folder': os.path.join(project_folder, 'data/slaves/'),
            'raw_data_folder': os.path.join(project_folder, 'data/raw/'),
            'coreg_folder': os.path.join(project_folder, 'process/coreg/'),
            'ifg_folder': os.path.join(project_folder, 'process/ifg/'),
            'stamp_folder': os.path.join(project_folder, 'results/'),
            'config_path': os.path.join(project_folder, 'modules/snap2stamps/bin/project.conf')
        }

        # Create directories
        for folder in folders.values():
            os.makedirs(folder, exist_ok=True)
            
        # Set up cache files
        cache_files = {
            'download_cache': os.path.join(project_folder, 'data/download_cache.txt'),
            'broken_cache': os.path.join(project_folder, 'data/broken_cache.txt'), 
            'baseline_cache': os.path.join(project_folder, 'data/baseline_cache.txt')
        }

        # Create empty cache files if they don't exist
        for cache_file in cache_files.values():
            if not os.path.exists(cache_file):
                with open(cache_file, 'w') as f:
                    pass

        # Set up datalake
        datalake = os.path.join(project_folder, 'data/lake.json')
        if not os.path.exists(datalake):
            with open(datalake, 'w') as f:
                json.dump([], f, indent=4)

        # Set up computing resources
        n_cores = round(os.cpu_count() * 0.8)
        total_ram = round(psutil.virtual_memory().total / (1024 ** 3) * 0.8)
        
        # Set up GPT path
        if plf == "Windows":
            gpt = "C:/Program Files/snap/bin/gpt.exe"
        else:
            gpt = "/home/viphap/snap9/bin/gpt"

        # Update config dictionary
        config_update = {
            "project_definition": folders,
            "cache_files": cache_files,
            "search_parameters": {
                "datalake": datalake,
                "direction": self.direction,
                "frame": self.frame
            },
            "processing_parameters": {
                "current_result": folders['stamp_folder']+'/INSAR_20200101_PSDS_v0/',
                "max_perp": self.max_perp,
                "ptype": self.ptype,
                "ministack": self.stack_size,
                "unified": self.unified_flag
            },
            "aoi_bbox": {
                "lon_min": self.bbox[0],
                "lat_min": self.bbox[1], 
                "lon_max": self.bbox[2],
                "lat_max": self.bbox[3]
            },
            "snap_gpt": {
                "gptbin_path": gpt
            },
            "computing_resources": {
                "cpu": n_cores,
                "cache": f"{total_ram}G"
            }
        }

        # Update config file
        self.config.update(config_update)
        self.config_parser.update_project_config(self.project_name, self.config)

if __name__ == "__main__":
    try:
        bbox = [106.6969, 10.7615, 106.7275, 10.7945]
        Initialize(bbox, "DESCENDING", 522, project_name="default")
    except Exception as e:
        print(f"Engage project structure fails due to\n{e}\n")