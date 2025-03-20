import os
from pathlib import Path
import shutil
import sys
import glob
import subprocess
import time

class StaMPSExporter:
    def __init__(self, stamps_flag, project_result, renew_flag, to_remove=None):
        super().__init__()
        
        self.inputfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], "project.conf")
        self.bar_message = '\n#####################################################################\n'
        self.renew_flag = bool(int(renew_flag))
        self.stamps_flag = stamps_flag
        self._load_config()
        self.project_result = project_result
        if not os.path.exists(os.path.join(self.STAMPFOLDER, self.project_result)):
            os.makedirs(os.path.join(self.STAMPFOLDER, self.project_result), exist_ok=True)
            
        # Remove old runs with provided to_remove
        if to_remove:
            for idx in to_remove:
                folder_to_remove = [os.path.join(self.STAMPFOLDER, self.project_result, f) for f in os.listdir(os.path.join(self.STAMPFOLDER, self.project_result)) if f[-1] == str(idx)]
                if folder_to_remove:
                    if os.path.exists(folder_to_remove[0]):
                        shutil.rmtree(folder_to_remove[0])
                        print(f"OLD RUN REMOVE: Removed: {folder_to_remove[0]}")
                else:
                    print(f"OLD RUN REMOVE: Folder not found: INSAR_YYYYMMDD_v{idx}")
        
        self._setup_folders()
        self._setup_logging()
        
    def _load_config(self):
        with open(self.inputfile, 'r') as file:
            for line in file.readlines():
                key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                if key:
                    setattr(self, key, value)  # Dynamically set variables

    def _setup_folders(self):
        _, tail = os.path.split(self.MASTER)
        project_outputs = os.listdir(os.path.join(self.STAMPFOLDER, self.project_result))
        comsar = self.COMSAR
        if self.stamps_flag == "NORMAL":
            core = "NORMAL"
        elif comsar == "0" and self.stamps_flag != "NORMAL":
            core = "PSDS"
        elif comsar == "1" and self.stamps_flag != "NORMAL":
            core = "ComSAR"
        if project_outputs:
            project_outputs = sorted(project_outputs, key=lambda x: int(x.split("_")[-1][-1]))
            mark = int(project_outputs[-1][-1])+1
            if self.renew_flag:
                self.outputexportfolder = f"{self.STAMPFOLDER}{self.project_result}INSAR_{tail[:8]}_{core}_v{mark}"
            else:
                self.outputexportfolder = f"{self.STAMPFOLDER}{self.project_result}INSAR_{tail[:8]}_{core}_v{mark-1}"
        else:
            self.outputexportfolder = f"{self.STAMPFOLDER}{self.project_result}INSAR_{tail[:8]}_{core}_v1"

        os.makedirs(self.outputexportfolder, exist_ok=True)

    def _setup_logging(self):
        self.outlog = os.path.join(self.LOGFOLDER, 'export_proc_stdout.log')
        self.out_file = open(self.outlog, 'a')
        self.err_file = self.out_file
        
    def _check_diff(self):
        bases = glob.glob(self.outputexportfolder + "/diff0" + "/*.base")
        diff = glob.glob(self.outputexportfolder + "/diff0" + "/*.diff")
        par = glob.glob(self.outputexportfolder + "/diff0" + "/*.par")
        if len(bases) == len(diff) == len(par):
            print("-> Correct DIFF0 exports")
        else:
            print("-> EXPORT ERROR: Missing DIFF0 files")

            if len(bases) < len(diff) or len(bases) < len(par):
                print("-> Missing .base files\n")
            if len(diff) < len(bases) or len(diff) < len(par):
                print("-> Missing .diff files\n")
            if len(par) < len(bases) or len(par) < len(diff):
                print("-> Missing .par files\n")
                
    def _check_rslc(self):
        if int(len(glob.glob(self.outputexportfolder + "/diff0" + "/*.base"))) == int(len(glob.glob(self.outputexportfolder + "/rslc" + "/*.rslc")) - 1):
            print("-> Correct RSLC exports\n")
        else:
            print(int(len(glob.glob(self.outputexportfolder + "/rslc" + "/*.rslc")) - 1))
            print("-> EXPORT ERROR: Missing RSLC files. The data is doubted to be removed due to poor baseline\n")

    def export(self):
        self.out_file.write(self.bar_message)
        message = '## StaMPS PSI export started:\n'
        print(message)
        self.out_file.write(message)
        self.out_file.write(self.bar_message)

        sorted_dimfiles = sorted(glob.glob(self.COREGFOLDER + '/*' + self.IW1 + '.dim'))
        
        
        with open(self.BASELINE_CACHE, "r") as file:
            for line in file.readlines():
                print(f"-> {line.strip()}: Poor intergerogram due to invalid baseline checking...")
                try:
                    os.remove(os.path.join(self.outputexportfolder, "diff0", line.strip().replace("_IW1.dim", ".base")))
                    os.remove(os.path.join(self.outputexportfolder, "diff0", line.strip().replace("_IW1.dim", ".diff")))
                    os.remove(os.path.join(self.outputexportfolder, "diff0", line.strip().replace("_IW1.dim", ".diff.par")))
                except:
                    pass
                try:
                    os.remove(os.path.join(self.outputexportfolder, "rslc", line.strip().split(".")[0].split("_")[1]+'.rslc'))
                    os.remove(os.path.join(self.outputexportfolder, "rslc", line.strip().split(".")[0].split("_")[1]+'.rslc.par'))
                except:
                    pass
                print("-> Invalid baseline data detected. Relating files deleted\n")

        for k, dimfile in enumerate(sorted_dimfiles, start=1):
            _, tail = os.path.split(dimfile)
            message = f'[{k}] Exporting pair: master-slave pair {tail}'
            ifgdim = Path(self.IFGFOLDER + tail)

            if ifgdim.is_file():
                print(message)
                self.out_file.write(message)
                
                # Skip processing file where it's already processed
                if os.path.exists(os.path.join(self.outputexportfolder, "diff0")) and os.path.exists(os.path.join(self.outputexportfolder, "rslc")):
                    if tail.replace("_IW1.dim", ".base") in os.listdir(os.path.join(self.outputexportfolder, "diff0")) and \
                        tail.replace("_IW1.dim", ".diff") in os.listdir(os.path.join(self.outputexportfolder, "diff0")) and \
                        tail.replace("_IW1.dim", ".diff.par") in os.listdir(os.path.join(self.outputexportfolder, "diff0")):
                            print(f"-> Result of {tail} exported. Skipping...\n")
                            continue

                graphxml = self.GRAPHSFOLDER + 'export.xml'
                graph2run = self.GRAPHSFOLDER + 'export2run.xml'

                with open(graphxml, 'r') as file:
                    filedata = file.read()

                filedata = filedata.replace('COREGFILE', dimfile)
                filedata = filedata.replace('IFGFILE', str(ifgdim))
                filedata = filedata.replace('OUTPUTFOLDER', self.outputexportfolder)

                with open(graph2run, 'w') as file:
                    file.write(filedata)

                args = [self.GPTBIN_PATH, graph2run, '-c', self.CACHE, '-q', self.CPU]

                process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                timeStarted = time.time()
                try:
                    result = process.communicate()
                except:
                    print("-> Export fails. Skipping this export...")
                    if tail.replace("_IW1.dim", ".base") in os.listdir(os.path.join(self.outputexportfolder, "diff0")) and \
                        tail.replace("_IW1.dim", ".diff") in os.listdir(os.path.join(self.outputexportfolder, "diff0")) and \
                        tail.replace("_IW1.dim", ".diff.par") in os.listdir(os.path.join(self.outputexportfolder, "diff0")):
                        
                        try:
                            os.remove(os.path.join(self.outputexportfolder, "diff0", line.strip().replace("_IW1.dim", ".base")))
                            os.remove(os.path.join(self.outputexportfolder, "diff0", line.strip().replace("_IW1.dim", ".diff")))
                            os.remove(os.path.join(self.outputexportfolder, "diff0", line.strip().replace("_IW1.dim", ".diff.par")))
                        except:
                            pass
                        try:
                            os.remove(os.path.join(self.outputexportfolder, "rslc", line.strip().split(".")[0].split("_")[1]+'.rslc'))
                            os.remove(os.path.join(self.outputexportfolder, "rslc", line.strip().split(".")[0].split("_")[1]+'.rslc.par'))
                        except:
                            pass
                        
                        continue

                print(f'SNAP STDOUT: {result[0]}')
                timeDelta = time.time() - timeStarted
                print(f'[{k}] Finished process in {timeDelta} seconds.')
                self.out_file.write(f'[{k}] Finished process in {timeDelta} seconds.\n')

                if process.returncode != 0:
                    message = f'Error exporting {tail}\n'
                    self.err_file.write(message)
                else:
                    message = f'Stamps export of {tail} successfully completed.\n'
                    print(message)
                    self.out_file.write(message)

                print(self.bar_message)
                self.out_file.write(self.bar_message)

    def cleanup(self):
        time.sleep(1)
        try:
            if "target.data" in os.listdir(self.PROJECTFOLDER):
                shutil.rmtree(os.path.join(self.PROJECTFOLDER, "target.data"))
            if "target.dim" in os.listdir(self.PROJECTFOLDER):
                os.remove(os.path.join(self.PROJECTFOLDER, "target.dim"))
        except:
            pass
        
    def _update_config(self):
        with open(self.inputfile, 'r') as file:
            lines = file.readlines()
            for idx, line in enumerate(lines):
                if line.startswith("CURRENT_RESULT"):
                    lines[idx] = "CURRENT_RESULT=" + str(self.outputexportfolder).replace('\\', '/').replace('//', '/') + '\n'
                    
        with open(self.inputfile, "w") as file:
            file.writelines(lines)
            file.close()

    def process(self):
        self._update_config()
        self.export()
        self._check_diff()
        self._check_rslc()
        self.cleanup()
        self.out_file.close()
        self.err_file.close()

if __name__ == "__main__":
    try:
        start_time = time.time()
        exporter = StaMPSExporter("DEMOBaSon", 0)
        exporter.process()
        print(f"StaMPS Export executes in {(time.time() - start_time) / 60} minutes.")
    except Exception as e:
        print(f"StaMPS Export fails due to\n{e}")