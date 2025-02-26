import os
import psutil
import platform

class Initialize:
    def __init__(self):
        super().__init__()
        plf = platform.system()
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
        config_file = project_folder + '/modules/snap2stamps/bin/project.conf'
        graphs_folder = config_file.replace("bin/project.conf", "graphs/")
        dc = project_folder + '/data/download_cache.txt'
        bc = project_folder + '/data/broken_cache.txt'
        bsc = project_folder + '/data/baseline_cache.txt'
        datalake = project_folder + '/data/lake.json'
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
                                    dc, bc, bsc, datalake, datafolder,
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
                elif line.startswith("DOWNLOAD_CACHE"):
                    lines[idx] = "DOWNLOAD_CACHE=" + str(path[10]).replace('\\', '/').replace('//', '/') + '\n'
                elif line.startswith("BROKEN_CACHE"):
                    lines[idx] = "BROKEN_CACHE=" + str(path[11]).replace('\\', '/').replace('//', '/') + '\n'
                elif line.startswith("BASELINE_CACHE"):
                    lines[idx] = "BASELINE_CACHE=" + str(path[12]).replace('\\', '/').replace('//', '/') + '\n'
                elif line.startswith("DATALAKE"):
                    lines[idx] = "DATALAKE=" + str(path[13]).replace('\\', '/').replace('//', '/') + '\n'
                elif line.startswith("DATAFOLDER"):
                    lines[idx] = "DATAFOLDER=" + str(path[14]).replace('\\', '/').replace('//', '/') + '\n'
                elif line.startswith("GPTBIN_PATH"):
                    lines[idx] = "GPTBIN_PATH=" + str(path[15]).replace(str(path[2][0]), str(path[2][0])) + '\n'
                elif line.startswith("CPU"):
                    lines[idx] = "CPU=" + str(path[16]) + '\n'
                elif line.startswith("CACHE"):
                    lines[idx] = "CACHE=" + str(path[17]) + 'G\n'
            
        with open(config_file, "w") as file:
            file.writelines(lines)
            file.close()

if __name__ == "__main__":
    try:
        Initialize()
    except Exception as e:
        print(f"Engage project structure fails due to\n{e}\n")
        
