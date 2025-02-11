import time
import asf_search as asf
from datetime import datetime, timedelta
import os
import logging
from random import Random

# Define area of interest (AOI) using WKT string
area_of_interest = "POLYGON((106.6969 10.7615,106.7275 10.7615,106.7275 10.7945,106.6969 10.7945,106.6969 10.7615))"

class Download:
    def __init__(self):
        super().__init__()
        self.start_time = time.time()
        self.logger = logging.getLogger()
        self.session = asf.ASFSession()
        self.session.auth_with_creds("tnd2000", "Nick0327@")

    def download(self, search_result, savepath: str):
        # Download the products
        for result in search_result:
            if not result[0].properties['fileID'].split("-")[0]+".zip" in os.listdir("./data"):
                print(f"Downloading: {result[0].properties['fileID']}")
                result[0].download(path=savepath, session=self.session)
        self.logger.info(f"Done downloading in {(time.time()-self.start_time)/60} mins")

class SLC_Search:
    def __init__(self):
        super().__init__()

        # Set initial date
        self.start_date = datetime(2014, 10, 1)  # Sentinel-1A launched in 2014
        self.end_date = datetime.now()

        # Loop through every 6 months
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
                platform=["Sentinel-1A", "Sentinel-1B", "Sentinel-1C"],
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
        self.logger.info(f"Done searching on {time.time()-start_time}")
        self.logger.info(f"Found {self.total_imgs} images")
        return self.final_results

if __name__ == "__main__":

    search = SLC_Search()
    results = search.search()

    download = Download()
    download.download(results, "./data/")