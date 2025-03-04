import json
import os
import psutil
import platform

class Initialize:
    def __init__(self, bbox, direction, frame, ptype=None):
        super().__init__()
        plf = platform.system()
        
        self.bbox = bbox
        self.direction = direction
        self.frame = frame
        self.ptype = ptype
        # To replace
        project_folder = os.path.split(os.path.abspath(__file__))[0].split('modules')[0]
        if plf == "Windows":
            project_in_disk = project_folder[0].upper()
            project_folder = f"{project_in_disk}:{project_folder.split(':')[1]}"
        log_folder = project_folder + '/logs/'
        datafolder = project_folder + '/data/'
        master_folder = project_folder + '/data/master/'
        slaves_folder = project_folder + '/data/slaves/'
        rawdata_folder = project_folder + '/data/raw/'
        coreg_folder = project_folder + '/process/coreg/'
        ifg_folder = project_folder + '/process/ifg/'
        stamp_folder = project_folder + '/results/'
        for folder in [log_folder, datafolder, master_folder, slaves_folder, rawdata_folder,
                       project_folder+'/process/', coreg_folder, ifg_folder, stamp_folder]:
            os.makedirs(folder, exist_ok=True)
        config_file = project_folder + '/modules/snap2stamps/bin/project.conf'
        graphs_folder = config_file.replace("bin/project.conf", "graphs/")
        lonmin = self.bbox[0]
        latmin = self.bbox[1]
        lonmax = self.bbox[2]
        latmax = self.bbox[3]
        dc = project_folder + '/data/download_cache.txt'
        bc = project_folder + '/data/broken_cache.txt'
        bsc = project_folder + '/data/baseline_cache.txt'
        datalake = project_folder + '/data/lake.json'
        for file in [dc, bc, bsc]:
            if not os.path.exists(file):
                with open(file, 'w') as f:
                    f.close()
        with open(datalake, 'w') as f:
            json.dump([], f, indent=4)
            f.close()
        n_cores = round(os.cpu_count() * 0.8)
        total_ram = round(psutil.virtual_memory().total / (1024 ** 3) * 0.8)
        if plf == "Windows":
            gpt = "C:/Program Files/snap9/bin/gpt.exe"
        elif plf == "Linux":
            gpt = os.path.split(os.path.abspath(__file__))[0].split('insar')[0] + 'snap9/bin/gpt'
        
        self.modify_master(config_file, [project_folder, graphs_folder, log_folder,
                                    master_folder, slaves_folder, rawdata_folder,
                                    coreg_folder, ifg_folder, stamp_folder,
                                    config_file,
                                    lonmin, latmin, lonmax, latmax,
                                    dc, bc, bsc, datalake, self.direction, self.frame, datafolder, self.ptype,
                                    gpt, n_cores, total_ram])
        
    # Update config
    def modify_master(self, config_file, path):
        lines = ''''''
        with open(config_file, 'r') as file:
            lines = file.readlines()
            for idx, line in enumerate(lines):
                if line.startswith("PROJECTFOLDER"):
                    lines[idx] = "PROJECTFOLDER=" + str(path[0]).replace('\\', '/').replace('//', '/') + '\n'
                elif line.startswith("GRAPHSFOLDER"):
                    lines[idx] = "GRAPHSFOLDER=" + str(path[1]).replace('\\', '/').replace('//', '/') + '\n'
                elif line.startswith("LOGFOLDER"):
                    lines[idx] = "LOGFOLDER=" + str(path[2]).replace('\\', '/').replace('//', '/') + '\n'
                elif line.startswith("MASTERFOLDER"):
                    lines[idx] = "MASTERFOLDER=" + str(path[3]).replace('\\', '/').replace('//', '/') + '\n'
                elif line.startswith("SLAVESFOLDER"):
                    lines[idx] = "SLAVESFOLDER=" + str(path[4]).replace('\\', '/').replace('//', '/') + '\n'
                elif line.startswith("RAWDATAFOLDER"):
                    lines[idx] = "RAWDATAFOLDER=" + str(path[5]).replace('\\', '/').replace('//', '/') + '\n'
                elif line.startswith("COREGFOLDER"):
                    lines[idx] = "COREGFOLDER=" + str(path[6]).replace('\\', '/').replace('//', '/') + '\n'
                elif line.startswith("IFGFOLDER"):
                    lines[idx] = "IFGFOLDER=" + str(path[7]).replace('\\', '/').replace('//', '/') + '\n'
                elif line.startswith("STAMPFOLDER"):
                    lines[idx] = "STAMPFOLDER=" + str(path[8]).replace('\\', '/').replace('//', '/') + '\n'
                elif line.startswith("CONFIG_PATH"):
                    lines[idx] = "CONFIG_PATH=" + str(path[9]).replace('\\', '/').replace('//', '/') + '\n'
                elif line.startswith("LONMIN"):
                    lines[idx] = "LONMIN=" + str(path[10]) + '\n'
                elif line.startswith("LATMIN"):
                    lines[idx] = "LATMIN=" + str(path[11]) + '\n'
                elif line.startswith("LONMAX"):
                    lines[idx] = "LONMAX=" + str(path[12]) + '\n'
                elif line.startswith("LATMAX"):
                    lines[idx] = "LATMAX=" + str(path[13]) + '\n'
                elif line.startswith("DOWNLOAD_CACHE"):
                    lines[idx] = "DOWNLOAD_CACHE=" + str(path[14]).replace('\\', '/').replace('//', '/') + '\n'
                elif line.startswith("BROKEN_CACHE"):
                    lines[idx] = "BROKEN_CACHE=" + str(path[15]).replace('\\', '/').replace('//', '/') + '\n'
                elif line.startswith("BASELINE_CACHE"):
                    lines[idx] = "BASELINE_CACHE=" + str(path[16]).replace('\\', '/').replace('//', '/') + '\n'
                elif line.startswith("DATALAKE"):
                    lines[idx] = "DATALAKE=" + str(path[17]).replace('\\', '/').replace('//', '/') + '\n'
                elif line.startswith("DIRECTION"):
                    lines[idx] = "DIRECTION=" + str(path[18]) + '\n'
                elif line.startswith("FRAME"):
                    lines[idx] = "FRAME=" + str(path[19]) + '\n'
                elif line.startswith("DATAFOLDER"):
                    lines[idx] = "DATAFOLDER=" + str(path[20]).replace('\\', '/').replace('//', '/') + '\n'
                elif line.startswith("COMSAR"):
                    lines[idx] = "COMSAR=" + str(path[21]) + '\n'
                elif line.startswith("GPTBIN_PATH"):
                    lines[idx] = "GPTBIN_PATH=" + str(path[22]).replace(str(path[2][0]), str(path[2][0])) + '\n'
                elif line.startswith("CPU"):
                    lines[idx] = "CPU=" + str(path[23]) + '\n'
                elif line.startswith("CACHE"):
                    lines[idx] = "CACHE=" + str(path[24]) + 'G\n'
            
        with open(config_file, "w") as file:
            file.writelines(lines)
            file.close()

if __name__ == "__main__":
    try:
        bbox = [106.6969, 10.7615, 106.7275, 10.7945]
        Initialize(bbox)
    except Exception as e:
        print(f"Engage project structure fails due to\n{e}\n")
        


