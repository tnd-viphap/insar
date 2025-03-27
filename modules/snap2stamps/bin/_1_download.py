#type:ignore
import concurrent.futures
import json
import logging
import os
import random
import sys
import threading
import time
import shutil
from datetime import datetime, timedelta
from shapely.geometry import Point, Polygon
from shapely import from_wkt

from modules.snap2stamps.bin._2_master_sel import MasterSelect
from modules.snap2stamps.bin._3_find_bursts import Burst
from modules.snap2stamps.bin._4_splitting_master import MasterSplitter
from modules.snap2stamps.bin._5_splitting_slaves import SlavesSplitter

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
            
        self.print_lock = threading.Lock()  # Ensure thread-safe printing
        
        self.processed_files = os.listdir(self.MASTERFOLDER) + os.listdir(self.SLAVESFOLDER)

    def _get_expected_size(self, file_id):
        """Retrieve expected file size from lake.json."""
        with open(self.DATALAKE, "r") as file:
            self.data = json.load(file)
        """Retrieve expected file size from lake.json safely."""
        for entry in self.data:
            if isinstance(entry, dict) and "properties" in entry:
                properties = entry["properties"]
                if isinstance(properties, dict) and properties.get("fileID") == file_id:
                    file.close()
                    return properties.get("bytes")
        file.close()
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
                            #sys.stdout.write(f"\r[{file_name}] {percent:.2f}% ({downloaded}/{expected_size} bytes) | {speed / 1e6:.2f} MB/s")
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
        else:
            with open(self.DOWNLOAD_CACHE, "a") as cache:
                cache.write(file_id+"\n")
                cache.close()

        return file_name

    def download(self, savepath):
        """
        Download files in parallel, resuming if needed, 
        with sequence trigger after 10 downloads and continuous disk space monitoring.
        """
        os.makedirs(savepath, exist_ok=True)
        
        def check_and_clean_disk_space():
            """Internal function to check disk space and perform cleaning if needed."""
            stat = os.statvfs("/")
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_bfree * stat.f_frsize
            percentage = float(free) / float(total) * 100
            
            if percentage <= 30.0:
                print("-> Disk space is about full. Performing data cleaning...")
                incomplete_download = []
                
                # Identify and move incomplete downloads
                for product in os.listdir(self.RAWDATAFOLDER):
                    product_path = os.path.join(self.RAWDATAFOLDER, product)
                    if float(os.path.getsize(product_path)) / 1024**3 <= 3.8:
                        incomplete_path = os.path.join(self.DATAFOLDER, product).replace("\\", "/")
                        incomplete_download.append(incomplete_path)
                        shutil.move(product_path, self.DATAFOLDER)
                
                # Cleaning sequence
                time.sleep(2)
                MasterSelect(self.REEST_FLAG, True).select_master()
                time.sleep(2)
                
                # Move incomplete files back to RAWDATAFOLDER
                for file in incomplete_download:
                    shutil.move(file, self.RAWDATAFOLDER)
                
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
                        for product in os.listdir(self.RAWDATAFOLDER):
                            product_path = os.path.join(self.RAWDATAFOLDER, product)
                            if float(os.path.getsize(product_path)) / 1024**3 <= 3.8:
                                incomplete_path = os.path.join(self.DATAFOLDER, product).replace("\\", "/")
                                incomplete_download.append(incomplete_path)
                                shutil.move(product_path, self.DATAFOLDER)
                        
                        # Run cleaning sequence
                        time.sleep(2)
                        MasterSelect(self.REEST_FLAG, True).select_master()
                        
                        # Move incomplete files back to RAWDATAFOLDER
                        for file in incomplete_download:
                            shutil.move(file, self.RAWDATAFOLDER)
                        
                        time.sleep(2)
                        Burst().find_burst()
                        time.sleep(2)
                        MasterSplitter().process()
                        time.sleep(2)
                        SlavesSplitter().process()
                        time.sleep(2)
                        
                        print("-> Continue downloading...")

class SLC_Search:
    def __init__(self):
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
        self.resume = False

    def _setup_logger(self):
        """Set up logging."""
        log_file = f"./logs/log_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.txt"
        os.makedirs("./logs", exist_ok=True)
        logging.basicConfig(filename=log_file, format="%(asctime)s %(message)s", filemode="w")
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        return logger

    def _determine_date_range(self):
        time.sleep(1)
        """Determine the search date range based on existing data."""
        if os.listdir(self.MASTERFOLDER) or os.listdir(self.SLAVESFOLDER):
            if os.path.exists(self.DOWNLOAD_CACHE):
                with open(self.DOWNLOAD_CACHE, "r") as file:
                    lines = file.readlines()
                    if lines:
                        latest_product = lines[-1].strip()
                        latest_date = datetime.strptime(latest_product[17:25], "%Y%m%d") + timedelta(1)
                    else:
                        latest_date = datetime.strptime("20141001", "%Y%m%d")
                self.logger.info(f"-> Resuming from latest available data: {latest_date}")
                return latest_date, datetime.now()
            else:
                images = os.listdir(self.MASTERFOLDER) + os.listdir(self.SLAVESFOLDER)
                latest_date = datetime.strptime(list(sorted(images))[-1], "%Y%m%d")
                return latest_date, datetime.now()
        else:
            images = [f[17:25] for f in os.listdir(self.RAWDATAFOLDER)]
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

    def search(self):
        """Perform a full search for Sentinel-1 data."""
        # Load existing lake data or initialize an empty list
        try:
            with open(self.DATALAKE, "r") as file:
                lake_data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            lake_data = []

        # Check for incomplete raw files
        print("-> Finding incomplete products that are not in the searching date range...")
        incomplete_files = []
        for file in os.listdir(self.RAWDATAFOLDER):
            # Identify potentially incomplete files (you might need to adjust this logic)
            file_date = datetime.strptime(file[17:25], "%Y%m%d")
            
            # Check if file is not in processed folders
            if not any(file in processed_dir for processed_dir in [self.MASTERFOLDER, self.SLAVESFOLDER]):
                incomplete_files.append(file)

        # Search for incomplete files first
        incomplete_results = []
        for incomplete_file in incomplete_files:
            file_date = datetime.strptime(incomplete_file[17:25], "%Y%m%d")
            
            # Search around the incomplete file's date
            start = file_date - timedelta(days=1)
            end = file_date + timedelta(days=1)

            results = asf.search(
                platform=["Sentinel-1A", "Sentinel-1B"],
                processingLevel="SLC",
                intersectsWith=self.AOI,
                flightDirection=self.DIRECTION,
                frame=int(self.FRAME),
                start=start,
                end=end
            )
            for result in results:
                if not result in lake_data:
                    lake_data.append(result.geojson())
            incomplete_results.extend(results)
        print(f"-> Found {len(incomplete_results)} incomplete products")

        # Proceed with regular search from latest date
        print(f"-> Search for products from {self.start_date.strftime('%d/%m/%Y')} to {self.end_date.strftime('%d/%m/%Y')}")
        
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
                flightDirection=self.DIRECTION,
                frame=int(self.FRAME),
                start=start,
                end=end
            )
            # Find the best overlapping footprint on AOI
            results = self._determine_best_overlap(results)
            
            for result in results:
                if not result in lake_data:
                    lake_data.append(result.geojson())
                    
            if results:
                
                # Select one random result from the new available images
                selected_result = random.choice(results)
                
                # Append new product to the download queue
                self.final_results.append(selected_result)
                
                # Check for existing raw files and potential resume
                if os.listdir(self.RAWDATAFOLDER):
                    for file in os.listdir(self.RAWDATAFOLDER):
                        if file[17:23] == selected_result.properties['fileName'][17:23]:
                            print(f"-> Raw file on {file[21:23]}/{file[17:21]} detected. Checking for resuming or reloading...")
                            result = asf.search(
                                platform=["Sentinel-1A", "Sentinel-1B"],
                                processingLevel="SLC",
                                intersectsWith=self.AOI,
                                flightDirection=self.DIRECTION,
                                frame=int(self.FRAME),
                                start=datetime.strptime(file[17:25], "%Y%m%d")-timedelta(1),
                                end=datetime.strptime(file[17:25], "%Y%m%d")+timedelta(1)
                            )
                            if result:
                                self.final_results.remove(selected_result)
                                self.final_results.append(result[0])
                                self.resume = True
                elif os.listdir(self.MASTERFOLDER) and os.listdir(self.SLAVESFOLDER):
                    master_month = str(os.listdir(self.MASTERFOLDER)[0])[0:6]
                    slaves_month = [str(f)[0:6] for f in os.listdir(self.SLAVESFOLDER)]
                    months = list(sorted(slaves_month + [master_month]))
                    if selected_result and selected_result.properties['fileName'][17:23] in months:
                        self.final_results.remove(selected_result)

            # Move to the next month
            self.current_date += timedelta(days=30)
        
        # Combine incomplete results with regular search results
        for result in incomplete_results:
            if not result in self.final_results:
                self.final_results.append(result)
        
        # Save updated lake data
        with open(self.DATALAKE, 'w') as file:
            json.dump(lake_data, file, indent=4) 
        
        # Final filtering of results
        processed_data = os.listdir(self.MASTERFOLDER) + os.listdir(self.SLAVESFOLDER)
        if processed_data:
            processed_month = [f[0:6] for f in processed_data]
            self.final_results = [f for f in self.final_results if not f.geojson()["properties"]["fileID"][17:23] in processed_month]
        self.logger.info(f"Found {len(self.final_results)} images for download.")
        return list(sorted(self.final_results, key=lambda x: int(x.geojson()["properties"]["fileID"][17:25])))

if __name__ == "__main__":
    search = SLC_Search()
    results = search.search()
    #if results[0:1]:
    #    downloader = Download(results[0:1])
    #    downloader.download(search.RAWDATAFOLDER)
