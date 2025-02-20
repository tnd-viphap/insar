import time
import asf_search as asf
from datetime import datetime, timedelta
import os
import logging
import sys
from random import Random
import concurrent

inputfile = sys.argv[1]

with open(inputfile, 'r') as file:
    for line in file.readlines():
        if "PROJECTFOLDER" in line:
            PROJECT = line.split('=')[1].strip()
            print(PROJECT)
        elif "IW1" in line:
            IW = line.split('=')[1].strip()
            print(IW)
        elif "MASTER" in line:
            MASTER = line.split('=')[1].strip()
            print(MASTER)
        elif "GRAPHSFOLDER" in line:
            GRAPH = line.split('=')[1].strip()
            print(GRAPH)
        elif "GPTBIN_PATH" in line:
            GPT = line.split('=')[1].strip()
            print(GPT)
        elif "LONMIN" in line:
            LONMIN = line.split('=')[1].strip()
        elif "LATMIN" in line:
            LATMIN = line.split('=')[1].strip()
        elif "LONMAX" in line:
            LONMAX = line.split('=')[1].strip()
        elif "LATMAX" in line:
            LATMAX = line.split('=')[1].strip()
        elif "CACHE" in line:
            CACHE = line.split('=')[1].strip()
        elif "CPU" in line:
            CPU = line.split('=')[1].strip()

# Define area of interest (AOI) using WKT string
area_of_interest = polygon = 'POLYGON ((' + LONMIN + ' ' + LATMIN + ',' + LONMAX + ' ' + LATMIN + ',' + LONMAX + ' ' + LATMAX + ',' + LONMIN + ' ' + LATMAX + ',' + LONMIN + ' ' + LATMIN + '))'

class Download:
    def __init__(self):
        super().__init__()
        self.start_time = time.time()
        self.logger = logging.getLogger()
        self.session = asf.ASFSession()
        self.session.auth_with_creds("tnd2000", "Nick0327@")

    def _download_single(self, result, savepath):
        """Helper method to download a single product."""
        file_name = result.properties['fileID'].split("-")[0] + ".zip"
        if file_name not in os.listdir("./data"):
            print(f"Downloading: {result.properties['fileID']}")
            result.download(path=savepath, session=self.session)
        return file_name

    def download(self, search_result, savepath: str):
        """Download products in parallel (4 at a time)."""
        if not os.path.exist(savepath):
            os.mkdir(savepath)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self._download_single, result[0], savepath): result[0] for result in search_result}
            
            for future in concurrent.futures.as_completed(futures):
                file_name = future.result()
                self.logger.info(f"Downloaded: {file_name}")

        self.logger.info(f"Done downloading in {(time.time() - self.start_time) / 60:.2f} mins")


class SLC_Search:
    def __init__(self):
        super().__init__()
        
        # Set initial date
        if os.path.exists("./data/cache.txt"):
            latest_product = open("./data/cache.txt", "r").readlines()
            if latest_product:
                latest_product = latest_product[-1]
                latest_product_date = latest_product[17:25]
                # Check if the product exists
                self.start_date = datetime(int(latest_product_date[0:4]), int(latest_product_date[4:6]), int(latest_product_date[6:]))
            else:
                self.start_date = datetime(2014, 10, 1)
        else:
            self.start_date = datetime(2014, 10, 1)  # Sentinel-1A launched in 2014
        self.end_date = datetime.now()
        
        print(self.start_date)
        print(self.end_date)

        # Loop through every 1 months
        self.current_date = self.start_date
        self.total_imgs = 0

        self.final_results = []

        log_file = "./logs/log_{}.txt".format(datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
        logging.basicConfig(
            filename=log_file,
            format="%(asctime)s %(message)s",
            filemode="w"
        )
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

    def search(self):
        start_time = time.time()
        while self.current_date < self.end_date:
            next_date = self.current_date + timedelta(days=30)  # Approx. 6 months

            # Get first day of next month
            start = datetime(next_date.year, next_date.month, 1)

            # Get last day of the month
            if next_date.month == 12:
                end = datetime(next_date.year + 1, 1, 1) - timedelta(days=1)
            else:
                end = datetime(next_date.year, next_date.month + 1, 1) - timedelta(days=1)

            self.logger.info(f"Searching for data from {start.strftime('%d/%m/%Y')} to {end.strftime('%d/%m/%Y')}")
            # Query for Sentinel-1 SLC data using ASF API

            results = asf.search(
                platform=["Sentinel-1A", "Sentinel-1B"],
                processingLevel="SLC",
                intersectsWith=area_of_interest,
                flightDirection="Descending",
                frame=553,
                start=start,
                end=end
            )
            numbers = len(results)
            if numbers > 1:
                results = [results[Random().randint(a=0, b=int(numbers-1))]]
                self.final_results.append(results)
            elif numbers == 1:
                results = [results[0]]
                self.final_results.append(results)
            elif numbers < 1:
                pass
            
            for product in results:
                self.logger.info(f"In store: {product.properties['fileID']}")

            # Print number of products found
            self.total_imgs += len(results)

            # Move to the next date range
            self.current_date = next_date
        
        # Save all product IDs to cache.txt
        self.save_to_cache()

        self.logger.info(f"Done searching on {time.time() - start_time:.2f} seconds")
        self.logger.info(f"Found {self.total_imgs} images")
        return self.final_results

    def save_to_cache(self):
        """Save all product IDs to cache.txt"""
        cache_file = "./data/cache.txt"
        with open(cache_file, "a") as f:
            for result in self.final_results:
                for product in result:
                    f.write(f"{product.properties['fileID']}\n")
        self.logger.info(f"Saved {len(self.final_results)} product IDs to {cache_file}")


if __name__ == "__main__":

    search = SLC_Search()
    results = search.search()

    download = Download()
    download.download(results, "./data/raw/")