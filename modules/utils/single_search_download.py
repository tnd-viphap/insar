#type: ignore
import asf_search as asf
import os
import sys
import logging
import threading
import time
import concurrent.futures

project_path = os.path.join(os.path.dirname(__file__), "../..")
sys.path.append(project_path)
from config.parser import ConfigParser

class Downloader:
    """ This is designed for redownloading when entering splitting phase but finding no data """
    def __init__(self, search_result, project_name):
        super().__init__()
        self.project_name = project_name
        self.config_parser = ConfigParser(os.path.join(project_path, "config", "config.json"))
        self.config = self.config_parser.get_project_config(self.project_name)
        self.logger = logging.getLogger()
        self.session = asf.ASFSession()
        self.session.auth_with_creds("tnd2000", "Nick0327#@!!")  # Replace with real credentials
        self.search_result = search_result
            
        self.print_lock = threading.Lock()  # Ensure thread-safe printing
        
        self.processed_files = os.listdir(self.config['project_definition']['master_folder']) + os.listdir(self.config['project_definition']['slaves_folder'])

    def _resume_download(self, result, savepath):
        """Resume an interrupted download using HTTP Range requests, showing progress."""
        file_id = str(result.properties['fileID'])
        file_name = file_id.split("-")[0] + ".zip"
        file_path = os.path.join(savepath, file_name)
        expected_size = result.properties["bytes"]

        if expected_size is None:
            with self.print_lock:
                self.logger.info(f"Skipping {file_name}: Not found in datalake")
            return None

        current_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        if current_size == expected_size:
            with self.print_lock:
                self.logger.info(f"{file_name} already downloaded.")
            return file_name

        with self.print_lock:
            print(f"-> Starting download: {file_name} ({current_size}/{expected_size} bytes)...")

        url = result.properties["url"]
        headers = {"Range": f"bytes={current_size}-"} if current_size > 0 else {}

        with self.session.get(url, headers=headers, stream=True) as response:
            response.raise_for_status()
            mode = "ab" if "Range" in headers else "wb"
            with open(file_path, mode) as file:
                downloaded = current_size
                start_time = time.time()

                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        downloaded += len(chunk)

                        percent = (downloaded / expected_size) * 100
                        elapsed_time = time.time() - start_time
                        speed = downloaded / (elapsed_time + 1e-6)  # Bytes per second

                        with self.print_lock:
                            #sys.stdout.write(f"\r[{file_name}] {percent:.2f}% ({downloaded}/{expected_size} bytes) | {speed / 1e6:.2f} MB/s")
                            sys.stdout.flush()
        print("\n")

        with self.print_lock:
            self.logger.info(f"\nDownloaded: {file_name}")

        # **Save fileID to download_cache.txt**
        if os.path.exists(self.config['cache_files']['download_cache']):
            with open(self.config['cache_files']['download_cache'], "r") as cache:
                lines = cache.readlines()
                file_id = file_id+'\n'
                lines.append(file_id)
                lines = list(sorted(set(lines)))
                with open(self.config['cache_files']['download_cache'], "w") as cache_file:  # Open in append mode
                    cache_file.writelines(lines)
                    cache_file.close()
                cache.close()
        else:
            with open(self.config['cache_files']['download_cache'], "a") as cache:
                cache.write(file_id+"\n")
                cache.close()

        return file_name

    def download(self, savepath):
        """Download files in parallel, resuming if needed."""
        os.makedirs(savepath, exist_ok=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self._resume_download, result, savepath): result for result in self.search_result}
            for future in concurrent.futures.as_completed(futures):
                file_name = future.result()
                if file_name:
                    self.logger.info(f"Downloaded: {file_name}") 

class Search_Download:
    def __init__(self, project_name):
        self.project_name = project_name
        self.session = asf.ASFSession()
        self.session.auth_with_creds("tnd2000", "Nick0327#@!!")

        self.config_parser = ConfigParser(os.path.join(project_path, "config", "config.json"))
        self.config = self.config_parser.get_project_config(self.project_name)

        self.AOI = f"POLYGON (({self.config['aoi_bbox']['lon_min']} {self.config['aoi_bbox']['lat_min']},{self.config['aoi_bbox']['lon_max']} {self.config['aoi_bbox']['lat_min']},{self.config['aoi_bbox']['lon_max']} {self.config['aoi_bbox']['lat_max']},{self.config['aoi_bbox']['lon_min']} {self.config['aoi_bbox']['lat_max']},{self.config['aoi_bbox']['lon_min']} {self.config['aoi_bbox']['lat_min']}))"
    
    def search(self, start=None, end=None):
        # Search around the incomplete file's date
        results = asf.search(
            platform=["Sentinel-1A", "Sentinel-1C"],
            processingLevel="SLC",
            intersectsWith=self.AOI,
            flightDirection=self.config['search_parameters']['direction'],
            frame=int(self.config['search_parameters']['frame']),
            start=start,
            end=end
        )
        return results
    def download(self, results, savepath):
        if results:
            if savepath:
                results[0].download(savepath, session=self.session)
            else:
                results[0].download(self.config['project_definition']['raw_data_folder'], session=self.session)
            with open(self.config['cache_files']['download_cache'], "r") as cache:
                lines = cache.readlines()
                lines.append(results[0].properties["fileID"]+"\n")
                lines = list(sorted(set(lines)))
            with open(self.config['cache_files']['download_cache'], "w") as write_cache:
                write_cache.writelines(lines)
            cache.close()
            write_cache.close()
