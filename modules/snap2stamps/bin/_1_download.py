#type:ignore
import concurrent.futures
import json
import logging
import os
import sys
import threading
import time
import shutil
from datetime import datetime, timedelta
from shapely.geometry import Point, Polygon
from shapely import from_wkt
import asf_search as asf
from asf_search.exceptions import ASFSearch5xxError
from tenacity import retry, stop_after_attempt, wait_exponential
import platform

from modules.snap2stamps.bin._2_master_sel import MasterSelect
from modules.snap2stamps.bin._3_find_bursts import Burst
from modules.snap2stamps.bin._4_splitting_master import MasterSplitter
from modules.snap2stamps.bin._5_splitting_slaves import SlavesSplitter

project_path = os.path.abspath(os.path.join(__file__, '../../../..')).replace("/config", "")
sys.path.append(project_path)

from config.parser import ConfigParser

class Download:
    def __init__(self, search_result, download_on: list = [None, None], project_name="default"):
        super().__init__()
        
        self.logger = logging.getLogger()
        self.session = asf.ASFSession()
        self.session.auth_with_creds("tnd2000", "Nick0327#@!!")  # Replace with real credentials
        self.search_result = search_result
        self.download_on = download_on
        
        # Handle date filtering based on download_on parameter
        if self.download_on[0] is not None or self.download_on[1] is not None:
            start_date = self.download_on[0]
            end_date = self.download_on[1]
            
            # Filter by start date if provided
            if start_date is not None:
                start_idx = [self.search_result.index(f) for f in self.search_result if int(f.properties["fileID"][17:25]) >= int(start_date)]
                if start_idx:
                    start_idx = start_idx[0]
                    self.search_result = self.search_result[start_idx:]
            
            # Filter by end date if provided
            if end_date is not None:
                end_idx = [self.search_result.index(f) for f in self.search_result if int(f.properties["fileID"][17:25]) <= int(end_date)]
                if end_idx:
                    end_idx = end_idx[-1] + 1
                    self.search_result = self.search_result[:end_idx]
        
        # Read input file
        self.project_name = project_name
        self.config_parser = ConfigParser(os.path.join(project_path, "config", "config.json"))
        self.config = self.config_parser.get_project_config(self.project_name)
            
        self.print_lock = threading.Lock()  # Ensure thread-safe printing
        
        # Create necessary directories if they don't exist
        os.makedirs(self.config["project_definition"]["raw_data_folder"], exist_ok=True)
        os.makedirs(self.config["project_definition"]["master_folder"], exist_ok=True)
        os.makedirs(self.config["project_definition"]["slaves_folder"], exist_ok=True)
        os.makedirs(self.config["project_definition"]["data_folder"], exist_ok=True)
        
        self.processed_files = os.listdir(self.config["project_definition"]["master_folder"]) + os.listdir(self.config["project_definition"]["slaves_folder"])

    def _get_expected_size(self, file_id):
        """Retrieve expected file size from lake.json."""
        try:
            with open(self.config["search_parameters"]["datalake"], "r") as file:
                self.data = json.load(file)
                for entry in self.data:
                    if isinstance(entry, dict) and "properties" in entry:
                        properties = entry["properties"]
                        if isinstance(properties, dict) and properties.get("fileID") == file_id:
                            return properties.get("bytes")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Error reading datalake: {str(e)}")
        return None

    def _resume_download(self, result, savepath):
        """Resume an interrupted download using HTTP Range requests, showing progress."""
        file_id = str(result.properties['fileID'])
        file_name = file_id.split("-")[0] + ".zip"
        file_path = os.path.join(savepath, file_name)
        expected_size = self._get_expected_size(file_id)

        if expected_size is None:
            with self.print_lock:
                self.logger.info(f"Skipping {file_name}: Not found in datalake")
            return None

        current_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        if current_size == expected_size:
            with self.print_lock:
                self.logger.info(f"{file_name} already downloaded.")
            return file_name
        
        if file_id[17:25] in self.processed_files:
            print("-> Processed data detected. Skipping this product...")
            return None

        with self.print_lock:
            print(f"Starting download: {file_name} ({current_size}/{expected_size} bytes)...")

        url = result.properties["url"]
        headers = {"Range": f"bytes={current_size}-"} if current_size > 0 else {}

        try:
            with self.session.get(url, headers=headers, stream=True) as response:
                response.raise_for_status()
                mode = "ab" if "Range" in headers else "wb"
                
                # Create a temporary file for the download
                temp_file_path = file_path + ".tmp"
                with open(temp_file_path, mode) as file:
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
                                sys.stdout.write(f"\r[{file_name}] {percent:.2f}% ({downloaded}/{expected_size} bytes) | {speed / 1e6:.2f} MB/s")
                                sys.stdout.flush()
                
                # Only rename the file if download was successful
                if downloaded == expected_size:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    os.rename(temp_file_path, file_path)
                    
                    # Update download cache
                    self._update_download_cache(file_id)
                    
                    with self.print_lock:
                        self.logger.info(f"\nDownloaded: {file_name}")
                    return file_name
                else:
                    # If download was incomplete, keep the temporary file
                    with self.print_lock:
                        self.logger.warning(f"\nIncomplete download: {file_name}")
                    return None
                    
        except Exception as e:
            with self.print_lock:
                self.logger.error(f"Error downloading {file_name}: {str(e)}")
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return None

    def _update_download_cache(self, file_id):
        """Update the download cache with the new file ID."""
        try:
            if os.path.exists(self.config["cache_files"]["download_cache"]):
                with open(self.config["cache_file"]["download_cache"], "r") as cache:
                    lines = cache.readlines()
                    file_id = file_id + '\n'
                    if file_id not in lines:
                        lines.append(file_id)
                        lines = list(sorted(set(lines)))
                        with open(self.config["cache_files"]["download_cache"], "w") as cache_file:
                            cache_file.writelines(lines)
            else:
                with open(self.config["cache_files"]["download_cache"], "w") as cache:
                    cache.write(file_id + "\n")
        except Exception as e:
            self.logger.error(f"Error updating download cache: {str(e)}")

    def _get_disk_space(self, path):
        """Get disk space information for Windows."""
        if platform.system() == "Windows":
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            total_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(path), None, ctypes.pointer(total_bytes), ctypes.pointer(free_bytes)
            )
            return (free_bytes.value / total_bytes.value) * 100
        else:
            stat = os.statvfs("/")
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_bfree * stat.f_frsize
            percentage = float(free) / float(total) * 100
            return percentage

    def download(self, savepath):
        """
        Download files in parallel, resuming if needed, 
        with sequence trigger after 10 downloads and continuous disk space monitoring.
        """
        os.makedirs(savepath, exist_ok=True)
        
        def check_and_clean_disk_space():
            """Internal function to check disk space and perform cleaning if needed."""
            percentage = self._get_disk_space(self.config["project_definition"]["raw_data_folder"])
            if percentage <= 30.0:
                print("-> Disk space is about full. Performing data cleaning...")
                incomplete_download = []
                
                # Identify and move incomplete downloads
                for product in os.listdir(self.config["project_definition"]["raw_data_folder"]):
                    product_path = os.path.join(self.config["project_definition"]["raw_data_folder"], product)
                    if float(os.path.getsize(product_path)) / 1024**3 <= 3.8:
                        incomplete_path = os.path.join(self.config["project_definition"]["data_folder"], product).replace("\\", "/")
                        incomplete_download.append(incomplete_path)
                        shutil.move(product_path, self.config["project_definition"]["data_folder"])
                
                # Cleaning sequence
                time.sleep(2)
                MasterSelect(self.config["processing_parameters"]["reeest_flag"], None, True).select_master()
                time.sleep(2)
                
                # Move incomplete files back to RAWDATAFOLDER
                for file in incomplete_download:
                    shutil.move(file, self.config["project_definition"]["raw_data_folder"])
                
                time.sleep(2)
                Burst().find_burst()
                time.sleep(2)
                MasterSplitter().process()
                time.sleep(2)
                SlavesSplitter().process()
                time.sleep(2)
        
        # Download tracking
        download_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self._resume_download, result, savepath): result for result in self.search_result}
            
            for future in concurrent.futures.as_completed(futures):
                file_name = future.result()
                if file_name:
                    self.logger.info(f"Downloaded: {file_name}")
                    download_count += 1
                    
                    # Check disk space after each download
                    check_and_clean_disk_space()
                    
                    # Trigger sequence after every 10 downloads
                    if download_count % 10 == 0:
                        print(f"-> Triggered processing sequence after {download_count} downloads")
                        
                        # Temporary move of current downloads
                        incomplete_download = []
                        for product in os.listdir(self.config["project_definition"]["raw_data_folder"]):
                            product_path = os.path.join(self.config["project_definition"]["raw_data_folder"], product)
                            if float(os.path.getsize(product_path)) / 1024**3 <= 3.8:
                                incomplete_path = os.path.join(self.config["project_definition"]["data_folder"], product).replace("\\", "/")
                                incomplete_download.append(incomplete_path)
                                shutil.move(product_path, self.config["project_definition"]["data_folder"])
                        
                        # Run cleaning sequence
                        time.sleep(2)
                        MasterSelect(self.config["processing_parameters"]["reeest_flag"], None, True, self.project_name).select_master()
                        
                        # Move incomplete files back to RAWDATAFOLDER
                        for file in incomplete_download:
                            shutil.move(file, self.config["project_definition"]["raw_data_folder"])
                        
                        time.sleep(2)
                        Burst(self.project_name).find_burst()
                        time.sleep(2)
                        MasterSplitter(self.project_name).process()
                        time.sleep(2)
                        SlavesSplitter(self.project_name).process()
                        time.sleep(2)
                        
                        print("-> Continue downloading...")

class SLC_Search:
    def __init__(self, max_date=None, download_on: list = [None, None], project_name="default"):
        super().__init__()
        
        # Read input file
        self.project_name = project_name
        self.config_parser = ConfigParser(os.path.join(project_path, "config", "config.json"))
        self.config = self.config_parser.get_project_config(self.project_name)

        # Define AOI
        self.AOI = f"POLYGON (({self.config['aoi_bbox']['lon_min']} {self.config['aoi_bbox']['lat_min']},{self.config['aoi_bbox']['lon_max']} {self.config['aoi_bbox']['lat_min']},{self.config['aoi_bbox']['lon_max']} {self.config['aoi_bbox']['lat_max']},{self.config['aoi_bbox']['lon_min']} {self.config['aoi_bbox']['lat_max']},{self.config['aoi_bbox']['lon_min']} {self.config['aoi_bbox']['lat_min']}))"
        
        self.logger = self._setup_logger()
        self.download_on = download_on
        self.start_date, self.end_date = self._determine_date_range()
        self.current_date = self.start_date
        self.final_results = []
        self.session = asf.ASFSession()
        self.session.auth_with_creds("tnd2000", "Nick0327#@!!")
        self.resume = False
        self.max_date = max_date
        
        # Track existing data by month
        self.monthly_data = {}
        
        # Get list of already processed images
        if os.path.exists(self.config["project_definition"]["master_folder"]):
            for file in os.listdir(self.config["project_definition"]["master_folder"]):
                month_key = file[0:6]  # YYYYMM
                if month_key not in self.monthly_data:
                    self.monthly_data[month_key] = {'processed': [], 'incomplete': []}
                self.monthly_data[month_key]['processed'].append(file)
                
        if os.path.exists(self.config["project_definition"]["slaves_folder"]):
            for file in os.listdir(self.config["project_definition"]["slaves_folder"]):
                month_key = file[0:6]  # YYYYMM
                if month_key not in self.monthly_data:
                    self.monthly_data[month_key] = {'processed': [], 'incomplete': []}
                self.monthly_data[month_key]['processed'].append(file)
        
        # Track incomplete downloads
        if os.path.exists(self.config["project_definition"]["raw_data_folder"]):
            for file in os.listdir(self.config["project_definition"]["raw_data_folder"]):
                if file.endswith('.zip'):
                    month_key = file[17:23]  # YYYYMM
                    if month_key not in self.monthly_data:
                        self.monthly_data[month_key] = {'processed': [], 'incomplete': []}
                    self.monthly_data[month_key]['incomplete'].append(file)

    def _setup_logger(self):
        """Set up logging."""
        log_file = f"./logs/log_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.txt"
        os.makedirs("./logs", exist_ok=True)
        logging.basicConfig(filename=log_file, format="%(asctime)s %(message)s", filemode="w")
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        return logger

    def _determine_date_range(self):
        """Determine the search date range based on existing data and download_on parameter."""
        time.sleep(1)
        
        # If download_on is specified, use those dates
        if self.download_on[0] is not None or self.download_on[1] is not None:
            start_date = None
            end_date = None
            
            if self.download_on[0] is not None:
                start_date = datetime.strptime(self.download_on[0], "%Y%m%d")
            if self.download_on[1] is not None:
                end_date = datetime.strptime(self.download_on[1], "%Y%m%d")
                
            # If only one date is provided, use appropriate default for the other
            if start_date is None:
                start_date = datetime(2014, 10, 1)  # Default start date
            if end_date is None:
                end_date = datetime.now()  # Default end date
                
            self.logger.info(f"-> Using specified date range: {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}")
            return start_date, end_date
            
        # If no download_on specified, use existing logic
        if os.listdir(self.config["project_definition"]["master_folder"]) or os.listdir(self.config["project_definition"]["slaves_folder"]):
            if os.path.exists(self.config["cache_files"]["download_cache"]):
                with open(self.config["cache_files"]["download_cache"], "r") as file:
                    lines = file.readlines()
                    if lines:
                        latest_product = lines[-1].strip()
                        latest_date = datetime.strptime(latest_product[17:25], "%Y%m%d") + timedelta(1)
                    else:
                        latest_date = datetime.strptime("20141001", "%Y%m%d")
                self.logger.info(f"-> Resuming from latest available data: {latest_date}")
                return latest_date, datetime.now()
            else:
                images = os.listdir(self.config["project_definition"]["master_folder"]) + os.listdir(self.config["project_definition"]["slaves_folder"])
                latest_date = datetime.strptime(list(sorted(images))[-1], "%Y%m%d")
                return latest_date, datetime.now()
        else:
            images = [f[17:25] for f in os.listdir(self.config["project_definition"]["raw_data_folder"])]
            if images:
                latest_date = datetime.strptime(list(sorted(images))[-1], "%Y%m%d")
                return latest_date, datetime.now()
                
        self.logger.info("-> No data found in master/slaves, downloading from beginning.")
        return datetime(2014, 10, 1), datetime.now()
    
    def _determine_best_overlap(self, results):
        best = []
        for result in results:
            coordinates = result.geometry["coordinates"][0]
            points = [Point(f[0], f[1]) for f in coordinates]
            polygon = Polygon(points)
            if polygon.contains_properly(from_wkt(self.AOI)):
                best.append(result)
        self.logger.info(f"-> Found {len(best)} useable products")
        return best

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
    def _safe_search(self, **kwargs):
        """Wrapper for asf.search with retry logic"""
        try:
            return asf.search(**kwargs)
        except ASFSearch5xxError as e:
            self.logger.warning(f"ASF Search API error: {str(e)}. Retrying...")
            raise  # This will trigger the retry
        except Exception as e:
            self.logger.error(f"Unexpected error during ASF search: {str(e)}")
            raise

    def _get_monthly_count(self, month_key):
        """Get total count of images for a month (processed + incomplete)."""
        if month_key not in self.monthly_data:
            return 0
        return len(self.monthly_data[month_key]['processed']) + len(self.monthly_data[month_key]['incomplete'])

    def _is_file_incomplete(self, file_path):
        """Check if a file is incomplete based on size."""
        return float(os.path.getsize(file_path)) / 1024**3 <= 3.8

    def _filter_monthly_results(self, results):
        """Filter results based on existing data and max_date per month."""
        filtered_results = []
        monthly_counts = {month: self._get_monthly_count(month) for month in self.monthly_data.keys()}
        
        # First, handle incomplete downloads
        for month_key, data in self.monthly_data.items():
            for file in data['incomplete']:
                file_path = os.path.join(self.config["project_definition"]["raw_data_folder"], file)
                if self._is_file_incomplete(file_path):
                    # Find matching result in results
                    for result in results:
                        if result.properties['fileID'].split("-")[0] == file.split("-")[0]:
                            filtered_results.append(result)
                            monthly_counts[month_key] += 1
                            break
        
        # Then handle new downloads, respecting max_date
        for result in results:
            month_key = result.properties['fileID'][17:23]
            if month_key not in monthly_counts:
                monthly_counts[month_key] = 0
            
            # Skip if we already have max_date images for this month
            if monthly_counts[month_key] >= self.max_date:
                continue
                
            # Skip if already processed
            if month_key in self.monthly_data and result.properties['fileID'][17:25] in [f[0:8] for f in self.monthly_data[month_key]['processed']]:
                continue
                
            # Skip if already in filtered results
            if result in filtered_results:
                continue
                
            filtered_results.append(result)
            monthly_counts[month_key] += 1
        
        return filtered_results

    def search(self):
        """Perform a full search for Sentinel-1 data."""
        # Load existing lake data or initialize an empty list
        try:
            with open(self.config["search_parameters"]["datalake"], "r") as file:
                lake_data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            lake_data = []

        # First, search for incomplete downloads
        print("-> Checking for incomplete downloads...")
        incomplete_results = []
        for month_key, data in self.monthly_data.items():
            for file in data['incomplete']:
                file_path = os.path.join(self.config["project_definition"]["raw_data_folder"], file)
                if self._is_file_incomplete(file_path):
                    file_date = datetime.strptime(file[17:25], "%Y%m%d")
                    
                    try:
                        results = self._safe_search(
                            platform=["Sentinel-1A", "Sentinel-1C"],
                            processingLevel="SLC",
                            intersectsWith=self.AOI,
                            flightDirection=self.config["search_parameters"]["direction"],
                            frame=int(self.config["search_parameters"]["frame_no"]),
                            start=file_date - timedelta(days=1),
                            end=file_date + timedelta(days=1)
                        )
                        
                        # Find matching result for the incomplete file
                        for result in results:
                            if result.properties['fileID'].split("-")[0] == file.split("-")[0]:
                                incomplete_results.append(result)
                                if not result in lake_data:
                                    lake_data.append(result.geojson())
                                break
                                
                    except Exception as e:
                        self.logger.error(f"Error searching for incomplete file {file}: {str(e)}")
                        continue

        print(f"-> Found {len(incomplete_results)} incomplete downloads")

        # Proceed with regular search from latest date
        print(f"-> Search for products from {self.start_date.strftime('%d/%m/%Y')} to {self.end_date.strftime('%d/%m/%Y')}")
        
        while self.current_date <= self.end_date:
            start = datetime(self.current_date.year, self.current_date.month, 1)
            next_month = self.current_date.month + 1 if self.current_date.month < 12 else 1
            next_year = self.current_date.year if self.current_date.month < 12 else self.current_date.year + 1
            end = datetime(next_year, next_month, 1) - timedelta(days=1)

            self.logger.info(f"Searching data from {start.strftime('%d/%m/%Y')} to {end.strftime('%d/%m/%Y')}")

            try:
                results = self._safe_search(
                    platform=["Sentinel-1A", "Sentinel-1C"],
                    processingLevel="SLC",
                    intersectsWith=self.AOI,
                    flightDirection=self.config["search_parameters"]["direction"],
                    frame=int(self.config["search_parameters"]["frame_no"]),
                    start=start,
                    end=end
                )
                # Find the best overlapping footprint on AOI
                results = self._determine_best_overlap(results)
                
                for result in results:
                    if not result in lake_data:
                        lake_data.append(result.geojson())
                
                if results:
                    # Filter results based on existing data and max_date
                    filtered_results = self._filter_monthly_results(results)
                    self.final_results.extend(filtered_results)

            except Exception as e:
                self.logger.error(f"Error during monthly search: {str(e)}")
                # Continue to next month even if current month fails
                self.current_date += timedelta(days=30)
                continue

            # Move to the next month
            self.current_date += timedelta(days=30)
        
        # Add incomplete results to final results
        for result in incomplete_results:
            if result not in self.final_results:
                self.final_results.append(result)
        
        # Save updated lake data
        with open(self.config["search_parameters"]["datalake"], 'w') as file:
            json.dump(lake_data, file, indent=4) 
        
        self.logger.info(f"Found {len(self.final_results)} images for download.")
        return list(sorted(self.final_results, key=lambda x: int(x.geojson()["properties"]["fileID"][17:25])))

if __name__ == "__main__":
    search = SLC_Search()
    results = search.search()
    #if results[0:1]:
    #    downloader = Download(results[0:1])
    #    downloader.download(search.RAWDATAFOLDER)