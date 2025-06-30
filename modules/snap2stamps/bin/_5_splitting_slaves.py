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
from pta import PTA
project_path = os.path.abspath(os.path.join(__file__, '../../../..')).replace("/config", "")
sys.path.append(project_path)
from config.parser import ConfigParser

class SlavesSplitter:
    def __init__(self, project_name="default"):
        self.project_name = project_name
        self.config_parser = ConfigParser(os.path.join(project_path, "config", "config.json"))
        self.config = self.config_parser.get_project_config(self.project_name)
        self.bar_message = '\n#####################################################################\n'
        self._create_folders()
        self.graph2run = os.path.join(self.config["project_definition"]["graphs_folder"], 'splitgraph2run.xml')
        self.outlog = os.path.join(self.config["project_definition"]["log_folder"], 'split_proc_stdout.log')
    
    def _create_folders(self):
        for folder in [self.config["project_definition"]["log_folder"]]:
            if not os.path.exists(folder):
                os.makedirs(folder)
    
    def split_single_slave(self, slave_folder, input_file, output_name):
        """Split a single slave file."""
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        # Run PTA first
        try:
            # PTA(input_file, None, self.project_name).pta()
            time.sleep(1)
        except Exception as e:
            print(f"Warning: PTA processing failed: {str(e)}")
            
        graphxml = os.path.join(self.config["project_definition"]["graphs_folder"], 'slave_split_applyorbit.xml')
        with open(graphxml, 'r') as file:
            filedata = file.read()
            filedata = filedata.replace('INPUTFILE', input_file)
            filedata = filedata.replace('IWs', self.config['processing_parameters']['iw1'])
            filedata = filedata.replace('FIRST_BURST', str(self.config['processing_parameters']['first_burst']))
            filedata = filedata.replace('LAST_BURST', str(self.config['processing_parameters']['last_burst']))
            filedata = filedata.replace('OUTPUTFILE', output_name)

        with open(self.graph2run, 'w') as file:
            file.write(filedata)

        args = [self.config["snap_gpt"]["gptbin_path"], self.graph2run, '-c', str(self.config["computing_resources"]["cache"]), '-q', str(self.config["computing_resources"]["cpu"])]
        
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        timeStarted = time.time()
        stdout, stderr = process.communicate()
        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f'Error splitting slave {input_file}\n{stderr.decode()}')
            
        # Delete raw data after successful processing
        if os.path.exists(input_file):
            os.remove(input_file)
            
        return True
    
    def process(self):
        with open(self.outlog, 'a') as out_file:
            out_file.write(self.bar_message)
            out_file.write('## TOPSAR Splitting and Apply Orbit\n')
            out_file.write(self.bar_message)
            
            k = 0
            for acdatefolder in sorted(os.listdir(self.config["project_definition"]["slaves_folder"])):
                k += 1
                print(f'[{k}] Folder: {acdatefolder}')
                out_file.write(f'[{k}] Folder: {acdatefolder}\n')

                folder_path = os.path.join(self.config["project_definition"]["slaves_folder"], acdatefolder)
                files = glob.glob(folder_path + '/*.zip')
                out_file.write(str(files) + '\n')

                if len(files) >= 1:
                    pass
                else:
                    try:
                        folder_date = datetime.strptime(acdatefolder, "%Y%m%d")
                        last_modified_date = datetime.fromtimestamp(os.path.getmtime(folder_path))
                        days_diff = int(abs((last_modified_date - folder_date).days))
                        
                        if days_diff <= 21 and datetime.now() >= folder_date + timedelta(days=21):
                            print(f"-> Image {acdatefolder} has new orbit data. Renewing...")
                            if os.path.exists(folder_path):
                                shutil.rmtree(folder_path)
                                os.makedirs(folder_path)
                                for file in os.listdir(self.config["project_definition"]["coreg_folder"]):
                                    if acdatefolder in file:
                                        if os.path.isfile(os.path.join(self.config["project_definition"]["coreg_folder"], file)):
                                            os.remove(os.path.join(self.config["project_definition"]["coreg_folder"], file))
                                        elif os.path.isdir(os.path.join(self.config["project_definition"]["coreg_folder"], file)):
                                            shutil.rmtree(os.path.join(self.config["project_definition"]["coreg_folder"], file))
                    except:
                        print(f"-> Error checking folder date {acdatefolder}")
                        pass
                
                if not os.listdir(folder_path):
                    print(f"-> Found no data. Re-downloading data for {acdatefolder}...")
                    try:
                        product_date = datetime.strptime(acdatefolder, "%Y%m%d")
                        start = product_date - timedelta(days=1)
                        end = product_date + timedelta(days=1)
                        instance = Search_Download(self.project_name)
                        results = instance.search(start, end)
                        if results:
                            instance.download(results, folder_path)
                        time.sleep(1)
                    except:
                        print(f"-> Please check {acdatefolder}")
                files = glob.glob(folder_path + '/*.zip')
                if files:
                    head, tail = os.path.split(files[0])
                    outputname = head + '/' + tail[17:25] + '_' + self.config['processing_parameters']['iw1'] + '.dim'
                
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
                            results = Search_Download(self.project_name).search(start, end)
                            if results:
                                downloader = Downloader(results, self.project_name)
                                downloader._resume_download(results[0], folder_path)
                                    
                            time.sleep(1)
                            found_burst = Burst(self.project_name).find_burst(folder_path)
                            if found_burst == False:
                                shutil.rmtree(folder_path)
                                with open(self.config["cache_files"]["download_cache"], "r") as cache:
                                    lines = cache.readlines()
                                    line_to_rm = None
                                    for line in lines:
                                        if acdatefolder in line:
                                            line_to_rm = lines.index(line)
                                    if line_to_rm:
                                        del lines[line_to_rm]
                                with open(self.config["cache_files"]["download_cache"], "w") as write_cache:
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
                                instance = Search_Download(self.project_name)
                                results = instance.search(start, end)
                                if results:
                                    instance.download(results, folder_path)
                                time.sleep(1)
                                Burst(self.project_name).find_burst(folder_path)
                            except:
                                time.sleep(1)
                                shutil.rmtree(folder_path)
                                continue
                    
                    try:
                        self.split_single_slave(folder_path, files[0], outputname)
                        message = f'Split slave {files[0]} successfully completed.\n'
                        print(message)
                        out_file.write(message)
                    except Exception as e:
                        message = f'Error splitting slaves {files[0]}\n{str(e)}'
                        print(message)
                        out_file.write(message + '\n')
                        
                    out_file.write(self.bar_message)
                else:
                    continue

if __name__ == "__main__":
    try:
        splitter = SlavesSplitter()
        splitter.process()
    except Exception as e:
        print(f"Splitting slaves fails due to\n{e}")