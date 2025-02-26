import os
import sys
import shutil
import platform
import time

import numpy as np

class MasterSelect:
    def __init__(self, reest_flag):
        super().__init__()
        
        inputfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], "project.conf")
        self.reset_master = bool(int(reest_flag))  # 0 or 1 (reselect)

        with open(inputfile, 'r') as file:
            for line in file.readlines():
                key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                if key:
                    setattr(self, key, value)
                    
        self.plf = platform.system()

        self.raw_files = self.get_files_in_directory(self.RAWDATAFOLDER, ".zip")
        self.master_files = self.get_files_in_directory(self.MASTERFOLDER)
        self.slave_files = self.get_files_in_directory(self.SLAVESFOLDER)
        
        self.all_files = sorted(set(self.raw_files + self.master_files + self.slave_files))
        duplicates = []
        if self.raw_files:
            for file in self.raw_files:
                if (not file[17:25] in self.master_files or not file[17:25] in self.slave_files):
                    duplicates.append(file)
        self.raw_files = duplicates

    def modify_master(self, config_file, master_info):
        """Modify the project.conf file with the new MASTER and OLD_MASTER values."""
        config_path = os.path.join(config_file, "project.conf")
        with open(config_path, "r") as file:
            lines = file.readlines()

        for idx, line in enumerate(lines):
            if master_info[0] and line.startswith("MASTER="):
                strg = str(master_info[0]).replace('\\', '/').replace('//', '/')
                lines[idx] = f"MASTER={strg}\n"
            if master_info[1] and line.startswith("OLD_MASTER"):
                lines[idx] = f"OLD_MASTER={master_info[1]}\n"

        with open(config_path, "w") as file:
            file.writelines(lines)

    def get_files_in_directory(self, directory, extension=None):
        """Returns a sorted list of unique filenames (or specific extensions) from a directory."""
        if not os.path.exists(directory):
            return []
        return sorted(set(
            f for f in os.listdir(directory) if (not extension or f.endswith(extension))
        ))

    def move_slaves(self, exclude=None):
        if len(os.listdir(self.RAWDATAFOLDER)) > 0:
            for item in os.listdir(self.RAWDATAFOLDER):
                src_path = os.path.join(self.RAWDATAFOLDER, item)
                dest_path = os.path.join(self.SLAVESFOLDER, item[17:25])
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path, exist_ok=True)
                shutil.move(src_path, dest_path)
        if len(os.listdir(self.MASTERFOLDER)) >= 2:
            exclude = os.path.split(exclude)[1][17:25] if ".zip" in exclude else exclude.replace("\\", "/").split("/")[-1]
            for item in os.listdir(self.MASTERFOLDER):
                if item != exclude:
                    src_path = os.path.join(self.MASTERFOLDER, item)
                    shutil.move(src_path, self.SLAVESFOLDER)

    def move_master(self, src_dir):
        if not os.path.isdir(src_dir):
            date = os.path.split(src_dir)[1][17:25]
            dest = os.path.join(self.MASTERFOLDER, date)
            if not date in os.listdir(dest):
                os.makedirs(dest, exist_ok=True)
        else:
            dest = self.MASTERFOLDER
        shutil.move(src_dir, dest)     

    def select_master(self):
        
        selected_master = self.all_files[int(np.floor(len(self.all_files) // 2) - 1)] if len(self.all_files) > 2 else self.all_files[0]
        if self.reset_master:
            print(f"Selected MASTER = {selected_master}")
            master_in = "r" if selected_master in self.raw_files else "s" if selected_master in self.slave_files else "m"
        else:
            master_in = "m"
        old_master = self.master_files[0] if self.master_files else self.MASTER.split("/")[-1] if "/" in self.MASTER else self.MASTER.split("\\")[-1]
        if not self.master_files:
            print("No current master, forcing re-selection.")
            self.reset_master = True

        if self.reset_master:
            print("Reselecting MASTER...") 
            
            master_folder = os.path.join(self.MASTERFOLDER, selected_master[17:25] if master_in=="r" else selected_master)
            output_name = f"{master_folder}/{selected_master}_M.dim" if self.plf == "Linux" else \
                        master_folder + f"/{selected_master[17:25] if master_in=='r' else selected_master}_M.dim"

            # Move master file
            if master_folder == os.path.split(self.MASTER)[0].split("/")[-1]:
                print("No MASTER date changes. Skipping reselecting...\n")
                sys.exit(0)
            if master_in == "s":
                print("New MASTER found in /slaves/")
                self.move_master(os.path.join(self.SLAVESFOLDER, selected_master))
            elif master_in == "r":
                print("New MASTER found in /raw/")
                self.move_master(os.path.join(self.RAWDATAFOLDER, selected_master))
            time.sleep(1)
            
            # Move slaves
            self.move_slaves(selected_master)

            print(f"Moved {selected_master} to {master_folder}")
            self.modify_master(os.path.dirname(os.path.abspath(__file__)), [output_name, old_master])
            print("New MASTER updated in project.conf")

        else:
            master_subfolders = [folder for folder in os.listdir(self.MASTERFOLDER) if os.path.isdir(os.path.join(self.MASTERFOLDER, folder))]
            empty_master_subfolder = next((sub for sub in master_subfolders if not os.listdir(os.path.join(self.MASTERFOLDER, sub))), None)

            if empty_master_subfolder:
                try:
                    print(f"Detected empty master folder: {empty_master_subfolder}, searching for {selected_master}...")
                    possible_locations = [self.SLAVESFOLDER, self.RAWDATAFOLDER]
                    for loc in possible_locations:
                        if selected_master in os.listdir(loc):
                            print(f"Moving {selected_master} from {loc} to master folder {empty_master_subfolder}")
                            shutil.move(os.path.join(loc, selected_master), os.path.join(self.MASTERFOLDER, empty_master_subfolder))
                            break
                except:
                    print("Selected master data is missing. Check manually")

            self.move_slaves(selected_master)
            self.modify_master(os.path.dirname(os.path.abspath(__file__)), [None, old_master])
            print("Keeping current MASTER.")

if __name__ == "__main__":
    try:
        MasterSelect(0).select_master()
    except Exception as e:
        print(f"Select master fails due to\n{e}")
        