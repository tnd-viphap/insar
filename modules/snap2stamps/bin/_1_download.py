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
from tqdm import tqdm

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
        self.processing_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)  # Increased workers for processing
        self.processing_futures = []  # Track processing tasks
        self.progress_lock = threading.Lock()  # Lock for progress bar updates
        self.total_bytes = 0
        self.downloaded_bytes = 0
        self.progress_bar = None
        self.download_status = {}  # Track download status for each file
        self.successful_downloads = 0  # Counter for successful downloads
        self.processing_complete = threading.Event()  # Event to signal processing completion
        self.successful_dates = []  # Track dates of successful downloads
        
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

    def _get_total_download_size(self):
        """Calculate total download size for all files."""
        total_size = 0
        for result in self.search_result:
            file_id = str(result.properties['fileID'])
            expected_size = self._get_expected_size(file_id)
            if expected_size:
                total_size += expected_size
        return total_size

    def _get_master_burst_range(self):
        """Get the burst range for the current master image."""
        master_folder = self.config["project_definition"]["master_folder"]
        if not os.listdir(master_folder):
            return None
        master_date = os.listdir(master_folder)[0]
        master_path = os.path.join(master_folder, master_date)
        # Use Burst to get burst info
        burst_info = Burst(self.project_name)
        burst_found = burst_info.find_burst(master_path)
        if not burst_found:
            return None
        config = self.config_parser.get_project_config(self.project_name)
        return (config["processing_parameters"].get("first_burst"), config["processing_parameters"].get("last_burst"))

    def _get_slave_burst_range(self, slave_folder):
        """Get the burst range for a slave image."""
        burst_info = Burst(self.project_name)
        burst_found = burst_info.find_burst(slave_folder)
        if not burst_found:
            return None
        config = self.config_parser.get_project_config(self.project_name)
        return (config["processing_parameters"].get("first_burst"), config["processing_parameters"].get("last_burst"))

    def _bursts_overlap(self, master_range, slave_range):
        if not master_range or not slave_range:
            return False
        m_first, m_last = master_range
        s_first, s_last = slave_range
        # Check if there is any overlap
        return not (s_last < m_first or s_first > m_last)

    def _trigger_master_and_splitters(self):
        self.logger.info("Triggering master selection and splitting after 10 successful downloads...")
        # 1. Master selection
        master_selector = MasterSelect(1, None, False, self.project_name)
        master_selector.select_master()
        # 2. Master splitting
        master_splitter = MasterSplitter(self.project_name)
        master_splitter.process()
        # 3. Slaves splitting
        slaves_splitter = SlavesSplitter(self.project_name)
        slaves_splitter.process()
        self.logger.info("Master selection and splitting completed.")

    def _process_single_product(self, file_name):
        """Process a single downloaded product with complete checking and processing logic."""
        self.logger.info(f"Starting processing for {file_name}")
        try:
            # Get the date from the file name
            date_str = file_name[17:25]
            slave_folder = os.path.join(self.config["project_definition"]["slaves_folder"], date_str)
            os.makedirs(slave_folder, exist_ok=True)
            
            # Move the file to the slave folder
            src_path = os.path.join(self.config["project_definition"]["raw_data_folder"], file_name)
            dst_path = os.path.join(slave_folder, file_name)
            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)
            
            # Check if already processed
            output_name = os.path.join(slave_folder, f"{date_str}_{self.config['processing_parameters']['iw1']}.dim")
            if os.path.exists(output_name):
                self.logger.info(f"Slave image {date_str} is already processed.")
                if os.path.exists(dst_path):
                    os.remove(dst_path)
                return True
                
            # Check for orbit data renewal
            try:
                folder_date = datetime.strptime(date_str, "%Y%m%d")
                last_modified_date = datetime.fromtimestamp(os.path.getmtime(slave_folder))
                days_diff = int(abs((last_modified_date - folder_date).days))
                
                if days_diff <= 21 and datetime.now() >= folder_date + timedelta(days=21):
                    self.logger.info(f"-> Image {date_str} has new orbit data. Renewing...")
                    if os.path.exists(slave_folder):
                        shutil.rmtree(slave_folder)
                        os.makedirs(slave_folder)
                        # Re-copy the file since we just deleted the folder
                        if os.path.exists(src_path):
                            shutil.copy2(src_path, dst_path)
                        # Clean up coreg folder
                        for file in os.listdir(self.config["project_definition"]["coreg_folder"]):
                            if date_str in file:
                                file_path = os.path.join(self.config["project_definition"]["coreg_folder"], file)
                                if os.path.isfile(file_path):
                                    os.remove(file_path)
                                elif os.path.isdir(file_path):
                                    shutil.rmtree(file_path)
            except Exception as e:
                self.logger.error(f"Error checking orbit data for {date_str}: {str(e)}")
            
            # If folder is empty (could happen after orbit renewal), try to re-download
            if not os.path.exists(dst_path):
                self.logger.info(f"-> Found no data. Re-downloading data for {date_str}...")
                try:
                    product_date = datetime.strptime(date_str, "%Y%m%d")
                    start = product_date - timedelta(days=1)
                    end = product_date + timedelta(days=1)
                    instance = Search_Download(self.project_name)
                    results = instance.search(start, end)
                    if results:
                        instance.download(results, slave_folder)
                    time.sleep(1)
                except Exception as e:
                    self.logger.error(f"Error re-downloading {date_str}: {str(e)}")
                    return False
            
            # Verify file exists after all checks
            if not os.path.exists(dst_path):
                self.logger.error(f"No valid file found for {date_str} after all checks")
                return False
            
            # Find bursts
            try:
                self.logger.info(f"-> Raw data detected. Checking products & Finding bursts for {date_str}...")
                
                # First try to verify/repair the download
                try:
                    product_date = datetime.strptime(date_str, "%Y%m%d")
                    start = product_date - timedelta(days=1)
                    end = product_date + timedelta(days=1)
                    
                    time.sleep(1)
                    results = Search_Download(self.project_name).search(start, end)
                    if results:
                        downloader = Downloader(results, self.project_name)
                        downloader._resume_download(results[0], slave_folder)
                except Exception as e:
                    self.logger.warning(f"Error verifying download for {date_str}: {str(e)}")
                
                # Now try to find bursts
                time.sleep(1)
                found_burst = Burst(self.project_name).find_burst(slave_folder)
                if not found_burst:
                    self.logger.error(f"No valid bursts found for {date_str}")
                    shutil.rmtree(slave_folder)
                    self._remove_from_download_cache(date_str)
                    return False
                # Burst overlap check
                master_burst_range = self._get_master_burst_range()
                slave_burst_range = self._get_slave_burst_range(slave_folder)
                if not self._bursts_overlap(master_burst_range, slave_burst_range):
                    self.logger.warning(f"Burst mismatch: Slave {date_str} bursts {slave_burst_range} do not overlap with master bursts {master_burst_range}. Skipping.")
                    shutil.rmtree(slave_folder)
                    self._remove_from_download_cache(date_str)
                    return False
            except Exception as e:
                # If burst finding fails, try one more time with fresh download
                self.logger.warning(f"First burst finding attempt failed for {date_str}: {str(e)}")
                try:
                    self.logger.info(f"-> Broken {date_str}. Re-downloading...")
                    if os.path.exists(dst_path):
                        os.remove(dst_path)
                    time.sleep(1)
                    product_date = datetime.strptime(date_str, "%Y%m%d")
                    start = product_date - timedelta(days=1)
                    end = product_date + timedelta(days=1)
                    instance = Search_Download(self.project_name)
                    results = instance.search(start, end)
                    if results:
                        instance.download(results, slave_folder)
                    time.sleep(1)
                    found_burst = Burst(self.project_name).find_burst(slave_folder)
                    if not found_burst:
                        raise Exception("No valid bursts found after re-download")
                except Exception as e2:
                    self.logger.error(f"Error in second burst finding attempt for {date_str}: {str(e2)}")
                    if os.path.exists(slave_folder):
                        shutil.rmtree(slave_folder)
                    return False
            
            # Run splitting process
            try:
                splitter = SlavesSplitter(self.project_name)
                splitter.split_single_slave(slave_folder, dst_path, output_name)
                self.logger.info(f"Successfully processed {date_str}")
                # Track successful downloads
                self.successful_dates.append(date_str)
                # Trigger master/slave splitting after every 10
                if len(self.successful_dates) % 10 == 0:
                    self._trigger_master_and_splitters()
                return True
            except Exception as e:
                self.logger.error(f"Error splitting {date_str}: {str(e)}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing {file_name}: {str(e)}")
            return False

    def _remove_from_download_cache(self, date_str):
        """Remove a date from the download cache."""
        try:
            if os.path.exists(self.config["cache_files"]["download_cache"]):
                with open(self.config["cache_files"]["download_cache"], "r") as cache:
                    lines = cache.readlines()
                    line_to_rm = None
                    for line in lines:
                        if date_str in line:
                            line_to_rm = lines.index(line)
                            break
                    if line_to_rm is not None:
                        del lines[line_to_rm]
                with open(self.config["cache_files"]["download_cache"], "w") as write_cache:
                    write_cache.writelines(lines)
        except Exception as e:
            self.logger.error(f"Error removing {date_str} from download cache: {str(e)}")

    def _log_status(self, message):
        """Log status without interfering with progress bar."""
        if self.progress_bar:
            with self.progress_lock:
                self.progress_bar.clear()
                print(message)
                self.progress_bar.refresh()
        else:
            self.logger.info(message)

    def _resume_download(self, result, savepath):
        """Resume an interrupted download using HTTP Range requests, showing progress with tqdm."""
        file_id = str(result.properties['fileID'])
        file_name = file_id.split("-")[0] + ".zip"
        file_path = os.path.join(savepath, file_name)
        temp_file_path = file_path + ".tmp"
        expected_size = self._get_expected_size(file_id)

        if expected_size is None:
            self._log_status(f"Skipping {file_name}: Not found in datalake")
            with self.progress_lock:
                self.download_status[file_name] = "Skipped (Not in Datalake)"
                if self.progress_bar:
                    self.progress_bar.set_description(f"Downloaded: {len([s for s in self.download_status.values() if s == 'Complete'])}/{len(self.search_result)}")
            return None

        # Check both the main file and temporary file
        current_size = 0
        download_path = None
        
        if os.path.exists(file_path):
            current_size = os.path.getsize(file_path)
            download_path = file_path
        elif os.path.exists(temp_file_path):
            current_size = os.path.getsize(temp_file_path)
            download_path = temp_file_path
            
        if current_size == expected_size:
            # If temp file is complete, rename it
            if download_path == temp_file_path:
                os.rename(temp_file_path, file_path)
            with self.progress_lock:
                self.downloaded_bytes += expected_size
                if self.progress_bar:
                    self.progress_bar.update(expected_size)
                    self.download_status[file_name] = "Complete"
                    self.progress_bar.set_description(f"Downloaded: {len([s for s in self.download_status.values() if s == 'Complete'])}/{len(self.search_result)}")
            return file_name
        
        if file_id[17:25] in self.processed_files:
            self._log_status("-> Processed data detected. Skipping this product...")
            with self.progress_lock:
                self.downloaded_bytes += expected_size
                if self.progress_bar:
                    self.progress_bar.update(expected_size)
                    self.download_status[file_name] = "Skipped (Already Processed)"
                    self.progress_bar.set_description(f"Downloaded: {len([s for s in self.download_status.values() if s == 'Complete'])}/{len(self.search_result)}")
            return file_name

        url = result.properties["url"]
        headers = {"Range": f"bytes={current_size}-"} if current_size > 0 else {}

        try:
            with self.session.get(url, headers=headers, stream=True) as response:
                response.raise_for_status()
                mode = "ab" if current_size > 0 else "wb"
                
                # Use the existing temp file if it exists, otherwise create new
                target_path = temp_file_path if not os.path.exists(file_path) else file_path
                
                with open(target_path, mode) as file:
                    downloaded = current_size
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            downloaded += len(chunk)
                            with self.progress_lock:
                                if self.progress_bar:
                                    self.progress_bar.update(len(chunk))
                
                # Verify download completion
                if downloaded == expected_size:
                    # If we were downloading to temp file, rename it
                    if target_path == temp_file_path:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        os.rename(temp_file_path, file_path)
                    
                    # Update download cache
                    self._update_download_cache(file_id)
                    
                    with self.progress_lock:
                        self.download_status[file_name] = "Complete"
                        if self.progress_bar:
                            self.progress_bar.set_description(f"Downloaded: {len([s for s in self.download_status.values() if s == 'Complete'])}/{len(self.search_result)}")
                    
                    # Submit processing task to thread pool
                    future = self.processing_pool.submit(self._process_single_product, file_name)
                    self.processing_futures.append(future)
                    
                    return file_name
                else:
                    self._log_status(f"Incomplete download: {file_name} ({downloaded}/{expected_size} bytes)")
                    return None
                    
        except Exception as e:
            self._log_status(f"Error downloading {file_name}: {str(e)}")
            return None

    def _update_download_cache(self, file_id):
        """Update the download cache with the new file ID."""
        try:
            if os.path.exists(self.config["cache_files"]["download_cache"]):
                with open(self.config["cache_files"]["download_cache"], "r") as cache:
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
        """Download files in parallel, properly resuming incomplete downloads."""
        os.makedirs(savepath, exist_ok=True)
        
        # Calculate total download size
        self.total_bytes = self._get_total_download_size()
        
        # Group files by their completion status
        incomplete_downloads = []
        new_downloads = []
        
        for result in self.search_result:
            file_id = str(result.properties['fileID'])
            file_name = file_id.split("-")[0] + ".zip"
            file_path = os.path.join(savepath, file_name)
            temp_path = file_path + ".tmp"
            self.download_status[file_name] = "Pending"
            
            # Check if file exists in any form
            if os.path.exists(file_path) or os.path.exists(temp_path):
                incomplete_downloads.append(result)
            else:
                new_downloads.append(result)
        
        download_success = True
        processing_success = True
        
        try:
            # Initialize single progress bar for all downloads
            with tqdm(total=self.total_bytes, unit='B', unit_scale=True, ncols=80,
                     bar_format='{desc}: {percentage:3.1f}%|{bar:20}| {n_fmt}/{total_fmt}',
                     desc=f"Downloaded: 0/{len(self.search_result)}", position=0, leave=True) as self.progress_bar:
                
                # First handle incomplete downloads
                if incomplete_downloads:
                    self._log_status(f"Resuming {len(incomplete_downloads)} incomplete downloads...")
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                        futures = {executor.submit(self._resume_download, result, savepath): result 
                                 for result in incomplete_downloads}
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                file_name = future.result()
                                if file_name:
                                    self._log_status(f"Successfully downloaded {file_name}")
                                else:
                                    download_success = False
                            except Exception as e:
                                self._log_status(f"Error in download: {str(e)}")
                                download_success = False
                
                # Then handle new downloads
                if new_downloads:
                    self._log_status(f"Starting {len(new_downloads)} new downloads...")
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                        futures = {executor.submit(self._resume_download, result, savepath): result 
                                 for result in new_downloads}
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                file_name = future.result()
                                if file_name:
                                    self._log_status(f"Successfully downloaded {file_name}")
                                else:
                                    download_success = False
                            except Exception as e:
                                self._log_status(f"Error in download: {str(e)}")
                                download_success = False
            
            print("\n")  # Add a newline after progress bar completes
            
            # Wait for processing tasks and show their status
            if self.processing_futures:
                processing_success = self._wait_for_processing_completion()
                if not processing_success:
                    self._log_status("WARNING: Some processing tasks failed or timed out!")
            
        finally:
            # Clean up processing pool
            self.processing_pool.shutdown(wait=True)
        
        # Final status report
        if download_success and processing_success:
            self._log_status("All downloads and processing completed successfully!")
        else:
            if not download_success:
                self._log_status("ERROR: Some downloads failed!")
            if not processing_success:
                self._log_status("ERROR: Some processing tasks failed!")
        
        return download_success and processing_success

    def _is_file_incomplete(self, file_path):
        """Check if a file is incomplete based on size."""
        return float(os.path.getsize(file_path)) / 1024**3 <= 3.8

    def _wait_for_processing_completion(self, timeout=3600):
        """Wait for all processing tasks to complete with timeout."""
        if not self.processing_futures:
            return True
            
        self._log_status(f"Waiting for {len(self.processing_futures)} processing tasks to complete...")
        
        try:
            # Wait for all futures to complete with timeout
            done, not_done = concurrent.futures.wait(
                self.processing_futures, 
                timeout=timeout,
                return_when=concurrent.futures.ALL_COMPLETED
            )
            
            if not_done:
                self._log_status(f"WARNING: {len(not_done)} processing tasks did not complete within timeout")
                # Cancel remaining tasks
                for future in not_done:
                    future.cancel()
                return False
            
            # Check results
            processing_results = []
            for future in done:
                try:
                    result = future.result()
                    processing_results.append(result)
                    if result:
                        self._log_status("Processing task completed successfully")
                    else:
                        self._log_status("Processing task failed")
                except Exception as e:
                    self._log_status(f"Processing task error: {str(e)}")
                    processing_results.append(False)
            
            # Summary of processing results
            successful = len([r for r in processing_results if r])
            failed = len(processing_results) - successful
            self._log_status(f"Processing complete: {successful} successful, {failed} failed")
            
            return failed == 0
            
        except Exception as e:
            self._log_status(f"Error waiting for processing completion: {str(e)}")
            return False

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
        # Ensure max_date is at least 1
        if max_date is None or max_date < 1:
            self.max_date = 1
        else:
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
        """Determine the search date range."""
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
            
        # If no download_on specified, always search from beginning
        start_date = datetime(2014, 10, 1)
        end_date = datetime.now()
        self.logger.info(f"-> Searching full date range: {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}")
        return start_date, end_date

    def _determine_best_overlap(self, results):
        best = []
        for result in results:
            coordinates = result.geometry["coordinates"][0]
            points = [Point(f[0], f[1]) for f in coordinates]
            polygon = Polygon(points)
            if polygon.intersects(from_wkt(self.AOI)):
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

    def _handle_existing_raw_data(self):
        """Handle existing data in raw folder, process complete ones and return incomplete ones."""
        if not os.path.exists(self.config["project_definition"]["raw_data_folder"]):
            return []

        complete_downloads = []
        incomplete_downloads = []
        processed_dates = set()  # Track processed dates to avoid duplicates

        # First, get all processed dates from master and slaves folders
        if os.path.exists(self.config["project_definition"]["master_folder"]):
            processed_dates.update(f[0:8] for f in os.listdir(self.config["project_definition"]["master_folder"]))
        if os.path.exists(self.config["project_definition"]["slaves_folder"]):
            processed_dates.update(f[0:8] for f in os.listdir(self.config["project_definition"]["slaves_folder"]))

        # Categorize existing raw data
        for file in os.listdir(self.config["project_definition"]["raw_data_folder"]):
            if file.endswith('.zip'):
                file_path = os.path.join(self.config["project_definition"]["raw_data_folder"], file)
                file_date = file[17:25] if len(file) > 25 else None

                # Skip if already processed
                if file_date in processed_dates:
                    continue

                if not self._is_file_incomplete(file_path):
                    complete_downloads.append(file)
                else:
                    incomplete_downloads.append(file)

        # Process complete downloads if any
        if complete_downloads:
            self.logger.info(f"Found {len(complete_downloads)} complete downloads to process")
            self._process_downloaded_products()

        # Return file IDs of incomplete downloads for resuming
        return [f[0:32] for f in incomplete_downloads]  # Assuming standard Sentinel-1 file ID length

    def _filter_monthly_results(self, results):
        """Filter results based on existing data and max_date per month."""
        filtered_results = []
        monthly_counts = {month: self._get_monthly_count(month) for month in self.monthly_data.keys()}
        max_date = self.max_date if self.max_date and self.max_date > 0 and self.max_date != None else 1
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
            if monthly_counts[month_key] >= max_date:
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

    def _get_incomplete_products(self):
        """Get list of incomplete products from raw data folder."""
        incomplete_products = []
        if not os.path.exists(self.config["project_definition"]["raw_data_folder"]):
            return incomplete_products

        for file in os.listdir(self.config["project_definition"]["raw_data_folder"]):
            if not file.endswith('.zip'):
                continue
                
            file_path = os.path.join(self.config["project_definition"]["raw_data_folder"], file)
            if self._is_file_incomplete(file_path):
                # Get the full file ID from filename
                file_id = file.split('.')[0]  # Remove .zip extension
                incomplete_products.append({
                    'file_id': file_id,
                    'date': file_id[17:25] if len(file_id) >= 25 else None,
                    'path': file_path
                })
        
        self.logger.info(f"Found {len(incomplete_products)} incomplete products in raw data folder")
        return incomplete_products

    def _get_processed_products(self):
        """Get list of already processed products from master and slaves folders."""
        processed_products = set()
        
        # Check master folder
        if os.path.exists(self.config["project_definition"]["master_folder"]):
            processed_products.update(f[0:8] for f in os.listdir(self.config["project_definition"]["master_folder"]))
            
        # Check slaves folder
        if os.path.exists(self.config["project_definition"]["slaves_folder"]):
            processed_products.update(f[0:8] for f in os.listdir(self.config["project_definition"]["slaves_folder"]))
            
        self.logger.info(f"Found {len(processed_products)} processed products")
        return processed_products

    def _search_for_product(self, date):
        """Search for a specific product by date."""
        try:
            search_date = datetime.strptime(date, "%Y%m%d")
            results = self._safe_search(
                platform=["Sentinel-1A", "Sentinel-1C"],
                processingLevel="SLC",
                intersectsWith=self.AOI,
                flightDirection=self.config["search_parameters"]["direction"],
                frame=int(self.config["search_parameters"]["frame"]),
                start=search_date - timedelta(days=1),
                end=search_date + timedelta(days=1)
            )
            return self._determine_best_overlap(results)
        except Exception as e:
            self.logger.error(f"Error searching for date {date}: {str(e)}")
            return []

    def search(self):
        """
        Perform search in steps:
        1. Find incomplete products
        2. Search for all products in time range
        3. Filter results
        4. Return final download list
        """
        # Step 1: Get incomplete products from raw folder
        incomplete_products = self._get_incomplete_products()
        incomplete_dates = set(p['date'] for p in incomplete_products if p['date'])
        
        # Step 2: Get already processed products
        processed_dates = self._get_processed_products()
        
        # Step 3: Search for incomplete products first
        incomplete_results = []
        for product in incomplete_products:
            if not product['date']:
                continue
            # Only include incomplete products within the download_on range if specified
            if self.download_on[0] is not None and product['date'] < self.download_on[0]:
                continue
            if self.download_on[1] is not None and product['date'] > self.download_on[1]:
                continue
            results = self._search_for_product(product['date'])
            for result in results:
                if result.properties['fileID'].startswith(product['file_id']):
                    incomplete_results.append(result)
                    break
        
        self.logger.info(f"Found {len(incomplete_results)} matching products for incomplete downloads")
        
        # Step 4: Search for all products in time range
        all_results = []
        print(f"-> Searching for products from {self.start_date.strftime('%d/%m/%Y')} to {self.end_date.strftime('%d/%m/%Y')}")
        
        current_date = self.start_date
        while current_date <= self.end_date:
            start = datetime(current_date.year, current_date.month, 1)
            next_month = current_date.month + 1 if current_date.month < 12 else 1
            next_year = current_date.year if current_date.month < 12 else current_date.year + 1
            end = datetime(next_year, next_month, 1) - timedelta(days=1)
            
            self.logger.info(f"Searching data from {start.strftime('%d/%m/%Y')} to {end.strftime('%d/%m/%Y')}")
            
            try:
                results = self._safe_search(
                    platform=["Sentinel-1A", "Sentinel-1C"],
                    processingLevel="SLC",
                    intersectsWith=self.AOI,
                    flightDirection=self.config["search_parameters"]["direction"],
                    frame=int(self.config["search_parameters"]["frame"]),
                    start=start,
                    end=end
                )
                results = self._determine_best_overlap(results)
                all_results.extend(results)
            except Exception as e:
                self.logger.error(f"Error during monthly search: {str(e)}")
            
            current_date += timedelta(days=30)
            
        self.logger.info(f"Found total {len(all_results)} products in date range")
        
        # Step 5: Filter results
        scheduled_fileids = set()
        final_results = []
        
        # First add incomplete products that need to be resumed
        for result in incomplete_results:
            fileid = result.properties['fileID']
            if fileid not in scheduled_fileids:
                final_results.append(result)
                scheduled_fileids.add(fileid)
        
        # Then add all products that haven't been processed yet
        for result in all_results:
            fileid = result.properties['fileID']
            date = fileid[17:25]
            
            # Skip if already scheduled for download
            if fileid in scheduled_fileids:
                continue
                
            # Skip if already processed successfully
            if date in processed_dates:
                continue
                
            # Apply monthly limit filter only for dates that we already have some data for
            month_key = date[0:6]
            if month_key in self.monthly_data:
                if len(self.monthly_data[month_key]['processed']) + len(self.monthly_data[month_key]['incomplete']) >= self.max_date:
                    continue
            
            final_results.append(result)
            scheduled_fileids.add(fileid)
            if month_key not in self.monthly_data:
                self.monthly_data[month_key] = {'processed': [], 'incomplete': []}
            self.monthly_data[month_key]['incomplete'].append(date)
        
        # Step 6: Update lake data
        try:
            with open(self.config["search_parameters"]["datalake"], "r") as file:
                lake_data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            lake_data = []
            
        for result in final_results:
            if not result in lake_data:
                lake_data.append(result.geojson())
                
        with open(self.config["search_parameters"]["datalake"], 'w') as file:
            json.dump(lake_data, file, indent=4)
        
        # Final summary
        missing_products = len([r for r in final_results if r not in incomplete_results])
        search_range_str = "full date range" if self.download_on[0] is None and self.download_on[1] is None else f"specified range {self.start_date.strftime('%Y%m%d')} to {self.end_date.strftime('%Y%m%d')}"
        summary = f"""
Search Results Summary ({search_range_str}):
- Incomplete products found: {len(incomplete_products)}
- Matching products for incomplete: {len(incomplete_results)}
- Total products in date range: {len(all_results)}
- Missing products found: {missing_products}
- Final products to download: {len(final_results)}
  * Resume incomplete: {len(incomplete_results)}
  * New downloads: {len(final_results) - len(incomplete_results)}
"""
        print(summary)
        self.logger.info(summary)
        print(summary)
        
        return list(sorted(final_results, key=lambda x: int(x.geojson()["properties"]["fileID"][17:25])))

if __name__ == "__main__":
    search = SLC_Search(2, ["20240101", None], "maychai")
    results = search.search()
    print(results)
    #if results[0:1]:
    #    downloader = Download(results[0:1])
    #    downloader.download(search.RAWDATAFOLDER)