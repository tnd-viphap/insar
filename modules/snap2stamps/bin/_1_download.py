#type:ignore
import concurrent.futures
import json
import logging
import os
import random
import sys
import threading
import time
from datetime import datetime, timedelta

import asf_search as asf

class Download:
    def __init__(self, search_result):
        super().__init__()
        
        self.logger = logging.getLogger()
        self.session = asf.ASFSession()
        self.session.auth_with_creds("tnd2000", "Nick0327#@!!")  # Replace with real credentials
        self.search_result = search_result
        
        # Read input file
        inputfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], "project.conf")
        with open(inputfile, 'r') as file:
            for line in file.readlines():
                key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                if key:
                    setattr(self, key, value)
                    
        with open(self.DATALAKE, "r") as file:
            self.data = json.load(file)
            
        self.print_lock = threading.Lock()  # Ensure thread-safe printing
        
        self.processed_files = os.listdir(self.MASTERFOLDER) + os.listdir(self.SLAVESFOLDER)

    def _get_expected_size(self, file_id):
        """Retrieve expected file size from lake.json."""
        for entry in self.data:
            if entry["properties"]["fileID"] == file_id:
                return entry["properties"]["bytes"]
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
            print("Processed data detected. Skipping this product...")
            return

        with self.print_lock:
            print(f"Starting download: {file_name} ({current_size}/{expected_size} bytes)...")

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
                            sys.stdout.write(f"\r[{file_name}] {percent:.2f}% ({downloaded}/{expected_size} bytes) | {speed / 1e6:.2f} MB/s")
                            sys.stdout.flush()
        print("\n")

        with self.print_lock:
            self.logger.info(f"\nDownloaded: {file_name}")

        # **Save fileID to download_cache.txt**
        if os.path.exists(self.DOWNLOAD_CACHE):
            with open(self.DOWNLOAD_CACHE, "r") as cache:
                lines = cache.readlines()
                file_id = file_id+'\n'
                lines.append(file_id)
        
        lines = list(sorted(set(lines)))
        with open(self.DOWNLOAD_CACHE, "w") as cache_file:  # Open in append mode
            cache_file.writelines(lines)
            cache_file.close()
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

class SLC_Search:
    def __init__(self, flightDirection, frame):
        super().__init__()
        
        # Read input file
        inputfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], "project.conf")
        with open(inputfile, 'r') as file:
            for line in file.readlines():
                key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                if key:
                    setattr(self, key, value)

        # Define AOI
        self.AOI = f"POLYGON (({self.LONMIN} {self.LATMIN},{self.LONMAX} {self.LATMIN},{self.LONMAX} {self.LATMAX},{self.LONMIN} {self.LATMAX},{self.LONMIN} {self.LATMIN}))"
        
        self.logger = self._setup_logger()
        self.start_date, self.end_date = self._determine_date_range()
        self.current_date = self.start_date
        self.final_results = []
        self.session = asf.ASFSession()
        self.session.auth_with_creds("tnd2000", "Nick0327#@!!")

        # Define paths
        self.lake_json_path = self.DATALAKE
        
        self.resume = False
        self.flightDirection = flightDirection
        self.frame = frame

    def _setup_logger(self):
        """Set up logging."""
        log_file = f"./logs/log_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.txt"
        os.makedirs("./logs", exist_ok=True)
        logging.basicConfig(filename=log_file, format="%(asctime)s %(message)s", filemode="w")
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        return logger

    def _determine_date_range(self):
        """Determine the search date range based on existing data."""
        if os.listdir(self.MASTERFOLDER) or os.listdir(self.SLAVESFOLDER):
            if os.path.exists(self.DOWNLOAD_CACHE) and os.path.getsize(self.DOWNLOAD_CACHE) > 0:
                with open("./data/download_cache.txt", "r") as file:
                    latest_product = file.readlines()[-1].strip()
                latest_date = datetime.strptime(latest_product[17:25], "%Y%m%d") + timedelta(1)
                self.logger.info(f"Resuming from latest available data: {latest_date}")
                return latest_date, datetime.now()
        self.logger.info("No data found in master/slaves, downloading from beginning.")
        return datetime(2014, 10, 1), datetime.now()

    def search(self):
        """Perform a full search for Sentinel-1 data."""
        # Load lake.json if it exists
        if os.path.exists(self.lake_json_path):
            try:
                with open(self.lake_json_path, "r") as file:
                    lake_data = json.load(file)
                    if not isinstance(lake_data, list):
                        lake_data = []
            except json.JSONDecodeError:
                lake_data = []
        else:
            lake_data = []

        while self.current_date <= self.end_date:
            start = datetime(self.current_date.year, self.current_date.month, 1)
            next_month = self.current_date.month + 1 if self.current_date.month < 12 else 1
            next_year = self.current_date.year if self.current_date.month < 12 else self.current_date.year + 1
            end = datetime(next_year, next_month, 1) - timedelta(days=1)

            self.logger.info(f"Searching data from {start.strftime('%d/%m/%Y')} to {end.strftime('%d/%m/%Y')}")

            results = asf.search(
                platform=["Sentinel-1A", "Sentinel-1B"],
                processingLevel="SLC",
                intersectsWith=self.AOI,
                flightDirection="Descending",
                frame=553,
                start=start,
                end=end
            )
            if results:
                # Select one random result from the available images
                selected_result = random.choice(results)
                
                # Append new product to the download queue
                self.final_results.append(selected_result)
                # Save the new product to lake.json
                if not selected_result.geojson() in lake_data:
                    lake_data.append(selected_result.properties)
                    with open(self.lake_json_path, "w") as file:
                        json.dump(lake_data, file, indent=4)
                
                if os.listdir(self.RAWDATAFOLDER):
                    for file in os.listdir(self.RAWDATAFOLDER):
                        if file[17:23] == selected_result.properties['fileName'][17:23]:
                            print("Raw files detected. Checking for resuming or reloading...")
                            result = asf.search(
                                platform=["Sentinel-1A", "Sentinel-1B"],
                                processingLevel="SLC",
                                intersectsWith=self.AOI,
                                flightDirection=self.flightDirection,
                                frame=self.frame,
                                start=datetime.strptime(file[17:25], "%Y%m%d")-timedelta(1),
                                end=datetime.strptime(file[17:25], "%Y%m%d")+timedelta(1)
                            )
                            if result:
                                self.final_results.remove(selected_result)
                                self.final_results.append(result[0])
                                self.resume = True
            # Move to the next month
            self.current_date += timedelta(days=30)

        self.logger.info(f"Found {len(self.final_results)} images for download.")
        return self.final_results

if __name__ == "__main__":
    search = SLC_Search("Descending", 553)
    results = search.search()
    if results[0:1]:
        downloader = Download(results[0:1])
        downloader.download(search.RAWDATAFOLDER)