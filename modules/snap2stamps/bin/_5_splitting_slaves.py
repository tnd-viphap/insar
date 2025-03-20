#type:ignore
import os
import glob
import subprocess
import time
import shutil
import sys
from datetime import datetime, timedelta
sys.path.append(os.path.join(os.path.abspath(__file__), "../../../../"))
from modules.utils.single_search_download import Search_Download, Downloader
from modules.snap2stamps.bin._3_find_bursts import Burst

class SlavesSplitter:
    def __init__(self):
        self.inputfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], "project.conf")
        if not os.path.exists(self.inputfile):
            print("Configuration file is missing")
        self.bar_message = '\n#####################################################################\n'
        self._load_config()
        self._create_folders()
        self.graph2run = os.path.join(self.GRAPHSFOLDER, 'splitgraph2run.xml')
        self.outlog = os.path.join(self.LOGFOLDER, 'split_proc_stdout.log')
    
    def _load_config(self):
        with open(self.inputfile, 'r') as file:
            for line in file.readlines():
                key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                if key:
                    setattr(self, key, value)
    
    def _create_folders(self):
        for folder in [self.LOGFOLDER]:
            if not os.path.exists(folder):
                os.makedirs(folder)
    
    def process(self):
        with open(self.outlog, 'a') as out_file:
            out_file.write(self.bar_message)
            out_file.write('## TOPSAR Splitting and Apply Orbit\n')
            out_file.write(self.bar_message)
            
            k = 0
            for acdatefolder in sorted(os.listdir(self.SLAVESFOLDER)):
                k += 1
                print(f'[{k}] Folder: {acdatefolder}')
                out_file.write(f'[{k}] Folder: {acdatefolder}\n')

                folder_path = os.path.join(self.SLAVESFOLDER, acdatefolder)
                files = glob.glob(folder_path + '/*.zip')
                out_file.write(str(files) + '\n')
                
                if not os.listdir(folder_path):
                    print(f"-> Found no data. Re-downloading data for {acdatefolder}...")
                    try:
                        product_date = datetime.strptime(acdatefolder, "%Y%m%d")
                        start = product_date - timedelta(days=1)
                        end = product_date + timedelta(days=1)
                        instance = Search_Download()
                        results = instance.search(start, end)
                        if results:
                            instance.download(results, folder_path)
                        time.sleep(1)
                    except:
                        print(f"-> Please check {acdatefolder}")
                files = glob.glob(folder_path + '/*.zip')
                if files:
                    head, tail = os.path.split(files[0])
                    outputname = head + '/' + tail[17:25] + '_' + self.IW1 + '.dim'
                
                    if os.path.exists(outputname):
                        print(f"-> Slave image {tail[17:25]} is already processed.")
                        out_file.write(f"Slave image {tail[17:25]} is already processed.\n")
                        if files and files[0].endswith(".zip"):
                            os.remove(files[0])
                        continue  
                
                    if files[0].endswith(".zip"):
                        print(f"-> Raw data detected. Checking products & Finding {tail} bursts...")
                        try:
                            product_date = datetime.strptime(acdatefolder, "%Y%m%d")
                            start = product_date - timedelta(days=1)
                            end = product_date + timedelta(days=1)
                            
                            time.sleep(1)
                            results = Search_Download().search(start, end)
                            if results:
                                downloader = Downloader(results)
                                downloader._resume_download(results[0], folder_path)
                                    
                            time.sleep(1)
                            found_burst = Burst().find_burst(folder_path)
                            if found_burst == False:
                                shutil.rmtree(folder_path)
                                with open(self.DOWNLOAD_CACHE, "r") as cache:
                                    lines = cache.readlines()
                                    line_to_rm = None
                                    for line in lines:
                                        if acdatefolder in line:
                                            line_to_rm = lines.index(line)
                                    if line_to_rm:
                                        del lines[line_to_rm]
                                with open(self.DOWNLOAD_CACHE, "w") as write_cache:
                                    write_cache.writelines(lines)
                                cache.close()
                                write_cache.close()
                                continue
                        except:
                            try:
                                print(f"-> Broken {tail[17:25]}. Re-downloading...")
                                os.remove(files[0])
                                time.sleep(1)
                                product_date = datetime.strptime(acdatefolder, "%Y%m%d")
                                start = product_date - timedelta(days=1)
                                end = product_date + timedelta(days=1)
                                instance = Search_Download()
                                results = instance.search(start, end)
                                if results:
                                    instance.download(results, folder_path)
                                time.sleep(1)
                                Burst().find_burst(folder_path)
                            except:
                                time.sleep(1)
                                shutil.rmtree(folder_path)
                                continue
                    time.sleep(1)

                    graphxml = os.path.join(self.GRAPHSFOLDER, 'slave_split_applyorbit.xml') if len(files) == 1 else os.path.join(self.GRAPHSFOLDER, 'slaves_assemble_split_applyorbit.xml')
                    self._load_config()
                    with open(graphxml, 'r') as file:
                        filedata = file.read()
                        filedata = filedata.replace('INPUTFILE', files[0])
                        filedata = filedata.replace('IWs', self.IW1)
                        filedata = filedata.replace('FIRST_BURST', self.FIRST_BURST)
                        filedata = filedata.replace('LAST_BURST', self.LAST_BURST)
                        filedata = filedata.replace('OUTPUTFILE', outputname)

                    with open(self.graph2run, 'w') as file:
                        file.write(filedata)

                    args = [self.GPTBIN_PATH, self.graph2run, '-c', self.CACHE, '-q', self.CPU]
                    print(args)
                    out_file.write(str(args) + '\n')

                    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    timeStarted = time.time()
                    stdout, stderr = process.communicate()
                    process.wait()

                    print('SNAP STDOUT: ', stdout.decode())
                    out_file.write(stdout.decode() + '\n')

                    timeDelta = time.time() - timeStarted
                    print(f'[{k}] Finished process in {timeDelta} seconds.')
                    out_file.write(f'[{k}] Finished process in {timeDelta} seconds.\n')

                    if process.returncode != 0:
                        message = f'Error splitting slaves {files}\n{stderr.decode()}'
                        print(message)
                        out_file.write(message + '\n')
                    else:
                        message = f'Split slave {files} successfully completed.\n'
                        print(message)
                        out_file.write(message)
                    
                    # Delete raw data
                    if files[0].endswith(".zip"):
                        os.remove(files[0])
                        
                    out_file.write(self.bar_message)
                else:
                    continue

if __name__ == "__main__":
    try:
        splitter = SlavesSplitter()
        splitter.process()
    except Exception as e:
        print(f"Splitting slaves fails due to\n{e}")