#type:ignore
import glob
import os
import subprocess
import sys
import time

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_path)
from config.parser import ConfigParser

class MasterSplitter:
    def __init__(self, project_name="default"):
        super().__init__()
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
    
    def process(self):
        with open(self.outlog, 'a') as out_file:
            out_file.write(self.bar_message)
            out_file.write('## TOPSAR Splitting and Apply Orbit\n')
            out_file.write(self.bar_message)
            
            k = 0
            for acdatefolder in sorted(os.listdir(self.config["project_definition"]["master_folder"])):
                k += 1
                print(f'[{k}] Folder: {acdatefolder}')
                out_file.write(f'[{k}] Folder: {acdatefolder}\n')
                
                folder_path = os.path.join(self.config["project_definition"]["master_folder"], acdatefolder)
                files = glob.glob(folder_path + '/*.zip')
                out_file.write(str(files) + '\n')
                
                if not files:
                    continue
                
                _, tail = os.path.split(files[0])
                outputname = tail[17:25] + '_M.dim'
                if tail[17:25] in os.listdir(self.config["project_definition"]["master_folder"]) and len(os.listdir(folder_path)) > 1:
                    print("Master image is already processed")
                    if tail in os.listdir(folder_path):
                        print("Raw data detected. Deleting...")
                        os.remove(files[0])
                    continue
                
                if len(files) == 1:
                    graphxml = os.path.join(self.config["project_definition"]["graphs_folder"], 'slave_split_applyorbit.xml')
                    
                    with open(graphxml, 'r') as file:
                        filedata = file.read()
                        filedata = filedata.replace('INPUTFILE', files[0])
                        filedata = filedata.replace('IWs', self.config["processing_parameters"]["iw1"])
                        filedata = filedata.replace('FIRST_BURST', self.config["processing_parameters"]["first_burst"])
                        filedata = filedata.replace('LAST_BURST', self.config["processing_parameters"]["last_burst"])
                        filedata = filedata.replace('OUTPUTFILE', os.path.join(folder_path, outputname))
                    
                    with open(self.graph2run, 'w') as file:
                        file.write(filedata)
                
                args = [self.config["snap_gpt"]["gptbin_path"], self.graph2run, '-c', self.config["computing_resources"]["cache"], '-q', self.config["computing_resources"]["cpu"]]
                print(args)
                out_file.write(str(args) + '\n')
                
                process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                timeStarted = time.time()
                stdout, stderr = process.communicate()
                
                print('SNAP STDOUT: ' + stdout.decode())
                out_file.write(stdout.decode() + '\n')
                
                timeDelta = time.time() - timeStarted
                print(f'[{k}] Finished process in {timeDelta} seconds.')
                out_file.write(f'[{k}] Finished process in {timeDelta} seconds.\n')
                
                if process.returncode != 0:
                    message = f'Error splitting master {files[0]}\n{stderr.decode()}'
                    print(message)
                    out_file.write(message + '\n')
                else:
                    message = f'Split master {files[0]} successfully completed.\n'
                    print(message)
                    out_file.write(message)
                    
                # Delete raw data
                if files[0]:
                    os.remove(files[0])
                
                out_file.write(self.bar_message)

if __name__ == "__main__":
    try:
        splitter = MasterSplitter()
        splitter.process()
    except Exception as e:
        print(f"Splitting master fails due to\n{e}")