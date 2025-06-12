# type: ignore
import os
import subprocess
import sys
import shutil
import platform
import time

import numpy as np

project_path = os.path.abspath(os.path.join(__file__, '../../../..')).replace("/config", "")
sys.path.append(project_path)
from config.parser import ConfigParser

class MasterSelect:
    def __init__(self, reest_flag, identity_master, subprocess=False, project_name="default"):
        super().__init__()
        
        self.reset_master = bool(int(reest_flag))  # 0 or 1 (reselect)
        self.subprocess = subprocess
        self.identity_master = identity_master

        self.project_name = project_name
        self.config_parser = ConfigParser(os.path.join(project_path, "config", "config.json"))
        self.config = self.config_parser.get_project_config(self.project_name)
                    
        self.graph2runms = os.path.join(self.config["project_definition"]["graphs_folder"], "toslaves2run.xml")
        self.graph2runsm = os.path.join(self.config["project_definition"]["graphs_folder"], "tomaster2run.xml")
        self.plf = platform.system()
        
        self.completed = open(self.config["cache_files"]["download_cache"], "r").readlines()

        self.raw_files = self.get_files_in_directory(self.config["project_definition"]["raw_data_folder"], ".zip")
        if subprocess:
            self.raw_files = [f for f in self.raw_files if f.replace(".zip", "-SLC")+"\n" in self.completed]
        self.master_files = self.get_files_in_directory(self.config["project_definition"]["master_folder"])
        self.slave_files = self.get_files_in_directory(self.config["project_definition"]["slaves_folder"])
        
        self.all_files = sorted(set(self.raw_files + self.master_files + self.slave_files))
        duplicates = []
        if self.raw_files:
            for file in self.raw_files:
                if (not file[17:25] in self.master_files or not file[17:25] in self.slave_files):
                    duplicates.append(file)
        self.raw_files = duplicates
        
        self.no_initial_master = False

    def modify_master(self, master_info):
        """Modify the project.conf file with the new MASTER and OLD_MASTER values."""
        self.config['project_definition']['master'] = str(master_info[0]).replace('\\', '/').replace('//', '/')
        self.config['processing_parameters']['old_master'] = master_info[1][0:8]
        self.config_parser.update_project_config(self.project_name, self.config)
        self.config_parser._load_config()
        self.config = self.config_parser.get_project_config(self.project_name)

    def get_files_in_directory(self, directory, extension=None):
        """Returns a sorted list of unique filenames (or specific extensions) from a directory."""
        if not os.path.exists(directory):
            return []
        return sorted(set(
            f for f in os.listdir(directory) if (not extension or f.endswith(extension))
        ))

    def move_slaves(self, exclude=None):
        if len(self.raw_files) > 0:
            self.raw_files = os.listdir(self.config["project_definition"]["raw_data_folder"])
            for item in self.raw_files:
                src_path = os.path.join(self.config["project_definition"]["raw_data_folder"], item)
                dest_path = os.path.join(self.config["project_definition"]["slaves_folder"], item[17:25])
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path, exist_ok=True)
                shutil.move(src_path, dest_path)
        if len(os.listdir(self.config["project_definition"]["master_folder"])) >= 2:
            exclude = os.path.split(exclude)[1][17:25] if ".zip" in exclude else exclude.replace("\\", "/").split("/")[-1]
            for item in os.listdir(self.config["project_definition"]["master_folder"]):
                if item != exclude:
                    src_path = os.path.join(self.config["project_definition"]["master_folder"], item)
                    try:
                        shutil.move(src_path, self.config["project_definition"]["slaves_folder"])
                    except:
                        pass

    def move_master(self, src_dir):
        if not os.path.isdir(src_dir):
            date = os.path.split(src_dir)[1][17:25]
            dest = os.path.join(self.config["project_definition"]["master_folder"], date)
            if not date in os.listdir(self.config["project_definition"]["master_folder"]):
                os.makedirs(dest, exist_ok=True)
            shutil.move(src_dir, dest)
        else:
            shutil.move(src_dir, self.config["project_definition"]["master_folder"])
        
    def master_to_slave(self, om, cm):
        print(f"Master to slave: {om}, {cm}")
        if not os.path.exists(os.path.join(self.config["project_definition"]["slaves_folder"], om)):
            return
        new_slave = os.path.join(self.config["project_definition"]["slaves_folder"], om, om+"_M").replace("_M", f"_{self.config['processing_parameters']['iw1']}.dim")
        old_master = os.path.join(self.config["project_definition"]["slaves_folder"], om, om+"_M.dim")
        graphxml = os.path.join(self.config["project_definition"]["graphs_folder"], 'toslaves.xml')
        
        with open(graphxml, 'r') as file:
            filedata = file.read()
            filedata = filedata.replace('SOURCE', old_master)
            filedata = filedata.replace('OUTPUT', new_slave)
        
        with open(self.graph2runms, 'w') as file:
            file.write(filedata)
        
        if (om in cm) or (new_slave in os.listdir(os.path.join(self.config["project_definition"]["slaves_folder"], om))):
            print("-> Converted M-S data detected. Skipping...")
        else:
            print("-> Converting M-S...")
            args = [self.config["snap_gpt"]["gptbin_path"], self.graph2runms, '-c', str(self.config["computing_resources"]["cache"]), '-q', str(self.config["computing_resources"]["cpu"])]
            process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process.communicate()
            time.sleep(2)
            try:
                os.remove(old_master)
                shutil.rmtree(old_master.replace(".dim", ".data"))
            except:
                pass
            print("-> Converting DONE")
        
    def slave_to_master(self, om, cm):
        current_master = os.path.join(self.config["project_definition"]["master_folder"], os.listdir(self.config["project_definition"]["master_folder"])[0])
        current_master = [os.path.join(current_master, f) for f in os.listdir(current_master) if '.dim' in f]
        if current_master:
            current_master = current_master[0]
        
            graphxml = os.path.join(self.config["project_definition"]["graphs_folder"], 'tomaster.xml')
            with open(graphxml, 'r') as file:
                filedata = file.read()
                filedata = filedata.replace('SOURCE', current_master)
                filedata = filedata.replace('OUTPUT', current_master.replace(f"_{self.config['processing_parameters']['iw1']}", "_M"))
            
            with open(self.graph2runsm, 'w') as file:
                file.write(filedata)
                
            if (om in cm) or (os.path.split(current_master)[1].replace(f"{self.config['processing_parameters']['iw1']}", "M") in os.listdir(os.path.split(current_master)[0])):
                print("-> Converted S-M data detected. Skipping...")
            else:
                print("-> Converting S-M...")
                args = [self.config["snap_gpt"]["gptbin_path"], self.graph2runsm, '-c', str(self.config["computing_resources"]["cache"]), '-q', str(self.config["computing_resources"]["cpu"])]
                process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                process.communicate()
                time.sleep(2)
                os.remove(current_master)
                shutil.rmtree(current_master.replace(".dim", ".data"))
                print("-> Converting DONE")
        else:
            if any(['.zip' in f for f in os.listdir(os.path.join(self.config["project_definition"]["master_folder"], os.listdir(self.config["project_definition"]["master_folder"])[0]))]):
                print(f"-> Raw data with ZIP detected. Skipping...\n")
            else:
                print(f"-> No data found. Skipping...\n")

    def select_master(self):
        if self.identity_master:
            selected_master = [f for f in self.all_files if str(self.identity_master) in f][0]
            print(f"Selected MASTER = {selected_master}\n")
            if self.reset_master:
                master_in = "r" if selected_master in self.raw_files else "s" if selected_master in self.slave_files else "m"
            else:
                master_in = "m"
            old_master = self.master_files[0] if self.master_files else self.config["project_definition"]["master"].split("/")[-1] if "/" in self.config["project_definition"]["master"] else self.config["project_definition"]["master"].split("\\")[-1]
                
            master_folder = os.path.join(self.config["project_definition"]["master_folder"], selected_master[17:25] if master_in=="r" else selected_master)
            output_name = f"{master_folder}/{selected_master}_M.dim" if self.plf == "Linux" else \
                        master_folder + f"/{selected_master[17:25] if master_in=='r' else selected_master}_M.dim"

            # Move master file
            if master_folder == os.path.split(self.config["project_definition"]["master"])[0].split("/")[-1]:
                print("-> No MASTER date changes. Skipping reselecting...\n")
                sys.exit(0)
            if master_in == "s":
                print("-> New MASTER found in /slaves/")
                self.move_master(os.path.join(self.config["project_definition"]["slaves_folder"], selected_master))
            elif master_in == "r":
                print("-> New MASTER found in /raw/")
                self.move_master(os.path.join(self.config["project_definition"]["raw_data_folder"], selected_master))
            time.sleep(1)
            
            # Move slaves
            self.move_slaves(selected_master)

            print(f"Moved {selected_master} to {master_folder}")
            self.modify_master([output_name, old_master])
            print("New MASTER updated to configuration\n")
            
            # Convert master-slave
            cm = self.config["project_definition"]["master"]
            om = self.config["processing_parameters"]["old_master"]
                        
            print("Converting old-new MASTER...")
            if '.zip' in selected_master:
                print("Raw data detected. Skipping reformatting MASTER file...") 
            else:
                self.slave_to_master(om, cm)
                time.sleep(1)
                self.master_to_slave(om, cm)
                time.sleep(1)
            print("\n")
        else:
            selected_master = self.all_files[int(len(self.all_files) // 2)] if len(self.all_files) > 2 else self.all_files[0]
            if self.reset_master:
                print(f"Selected MASTER = {selected_master}\n")
                master_in = "r" if selected_master in self.raw_files else "s" if selected_master in self.slave_files else "m"
            else:
                master_in = "m"
            old_master = self.master_files[0] if self.master_files else self.config["project_definition"]["master"].split("/")[-1] if "/" in self.config["project_definition"]["master"] else self.config["project_definition"]["master"].split("\\")[-1]
            if not self.master_files:
                print("No current master, forcing re-selection.")
                self.no_initial_master = True
                self.reset_master = True

            if self.reset_master:
                print("Reselecting MASTER...") 
                
                master_folder = os.path.join(self.config["project_definition"]["master_folder"], selected_master[17:25] if master_in=="r" else selected_master)
                output_name = f"{master_folder}/{selected_master}_M.dim" if self.plf == "Linux" else \
                            master_folder + f"/{selected_master[17:25] if master_in=='r' else selected_master}_M.dim"

                # Move master file
                if master_folder == os.path.split(self.config["project_definition"]["master"])[0].split("/")[-1]:
                    print("-> No MASTER date changes. Skipping reselecting...\n")
                    sys.exit(0)
                if master_in == "s":
                    print("-> New MASTER found in /slaves/")
                    self.move_master(os.path.join(self.config["project_definition"]["slaves_folder"], selected_master))
                elif master_in == "r":
                    print("-> New MASTER found in /raw/")
                    self.move_master(os.path.join(self.config["project_definition"]["raw_data_folder"], selected_master))
                time.sleep(1)
                
                # Move slaves
                self.move_slaves(selected_master)

                print(f"Moved {selected_master} to {master_folder}")
                self.modify_master([output_name, old_master])
                print("New MASTER updated to configuration\n")
                
                # Convert master-slave
                cm = self.config["project_definition"]["master"]
                om = self.config["processing_parameters"]["old_master"]
                            
                print("Converting old-new MASTER...")
                if '.zip' in selected_master:
                    print("Raw data detected. Skipping reformatting MASTER file...") 
                else:
                    self.slave_to_master(om, cm)
                    time.sleep(1)
                    self.master_to_slave(om, cm)
                    time.sleep(1)
                print("\n")

            else:
                master_subfolders = [folder for folder in os.listdir(self.config["project_definition"]["master_folder"]) if os.path.isdir(os.path.join(self.config["project_definition"]["master_folder"], folder))]
                empty_master_subfolder = next((sub for sub in master_subfolders if not os.listdir(os.path.join(self.config["project_definition"]["master_folder"], sub))), None)

                if empty_master_subfolder:
                    try:
                        print(f"Detected empty master folder: {empty_master_subfolder}, searching for {selected_master}...")
                        possible_locations = [self.config["project_definition"]["slaves_folder"], self.config["project_definition"]["raw_data_folder"]]
                        for loc in possible_locations:
                            if selected_master in os.listdir(loc):
                                print(f"Moving {selected_master} from {loc} to master folder {empty_master_subfolder}")
                                shutil.move(os.path.join(loc, selected_master), os.path.join(self.config["project_definition"]["master_folder"], empty_master_subfolder))
                                break
                    except:
                        print("Selected master data is missing. Check manually")

                self.move_slaves(selected_master)
                self.modify_master([None, old_master])
                print("Keeping current MASTER.")

        return selected_master

if __name__ == "__main__":
    try:
        MasterSelect(1).select_master()
    except Exception as e:
        print(f"Select master fails due to\n{e}")
        