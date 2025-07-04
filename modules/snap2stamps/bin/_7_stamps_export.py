# type: ignore
import os
from pathlib import Path
import shutil
import sys
import glob
import subprocess
import time

project_path = os.path.abspath(os.path.join(__file__, '../../../..')).replace("/config", "")
sys.path.append(project_path)
from config.parser import ConfigParser

class StaMPSExporter:
    def __init__(self, stamps_flag, project_result, renew_flag, to_remove=None, project_name="default"):
        super().__init__()
        self.project_name = project_name
        self.config_parser = ConfigParser(os.path.join(project_path, "config", "config.json"))
        self.config = self.config_parser.get_project_config(self.project_name)
        
        self.bar_message = '\n#####################################################################\n'
        self.renew_flag = bool(int(renew_flag))
        self.stamps_flag = stamps_flag
        self.project_result = project_result
        if not os.path.exists(os.path.join(self.config["project_definition"]["stamp_folder"], self.project_result)):
            os.makedirs(os.path.join(self.config["project_definition"]["stamp_folder"], self.project_result), exist_ok=True)
            
        # Remove old runs with provided to_remove
        if to_remove:
            for idx in to_remove:
                folder_to_remove = [os.path.join(self.config["project_definition"]["stamp_folder"], self.project_result, f) for f in os.listdir(os.path.join(self.config["project_definition"]["stamp_folder"], self.project_result)) if f[-1] == str(idx)]
                if folder_to_remove:
                    if os.path.exists(folder_to_remove[0]):
                        shutil.rmtree(folder_to_remove[0])
                        print(f"OLD RUN REMOVE: Removed: {folder_to_remove[0]}")
                else:
                    print(f"OLD RUN REMOVE: Folder not found: INSAR_YYYYMMDD_v{idx}")
        
        self._setup_folders()
        self._setup_logging()
        
    def _setup_folders(self):
        _, tail = os.path.split(self.config["project_definition"]["master"])
        project_outputs = os.listdir(os.path.join(self.config["project_definition"]["stamp_folder"], self.project_result))
        comsar = bool(int(self.config["api_flags"]["comsar"]))
        if self.stamps_flag == "NORMAL":
            self.core = "NORMAL"
        elif not comsar and self.stamps_flag != "NORMAL":
            self.core = "PSDS"
        elif comsar and self.stamps_flag != "NORMAL":
            self.core = "ComSAR"
        if project_outputs:
            project_outputs = sorted(project_outputs, key=lambda x: int(x.split("_v")[-1]))
            mark = int(project_outputs[-1].split("_v")[-1])+1
            if self.renew_flag:
                self.outputexportfolder = f"{self.config['project_definition']['stamp_folder']}{self.project_result}INSAR_{tail[:8]}_{self.core}_v{mark}"
            else:
                self.outputexportfolder = f"{self.config['project_definition']['stamp_folder']}{self.project_result}INSAR_{tail[:8]}_{self.core}_v{mark-1}"
        else:
            self.outputexportfolder = f"{self.config['project_definition']['stamp_folder']}{self.project_result}INSAR_{tail[:8]}_{self.core}_v1"

        os.makedirs(self.outputexportfolder, exist_ok=True)

    def _setup_logging(self):
        self.outlog = os.path.join(self.config['project_definition']['log_folder'], 'export_proc_stdout.log')
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

        sorted_dimfiles = sorted(glob.glob(self.config['project_definition']['coreg_folder'] + '/*' + self.config['processing_parameters']['iw1'] + '.dim'))
        
        with open(self.config['cache_files']['baseline_cache'], "r") as file:
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
            ifgdim = Path(self.config['project_definition']['ifg_folder'] + tail)

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

                graphxml = self.config["project_definition"]["graphs_folder"] + 'export.xml'
                graph2run = self.config["project_definition"]["graphs_folder"] + 'export2run.xml'

                with open(graphxml, 'r') as file:
                    filedata = file.read()

                filedata = filedata.replace('COREGFILE', dimfile)
                filedata = filedata.replace('IFGFILE', str(ifgdim))
                filedata = filedata.replace('OUTPUTFOLDER', self.outputexportfolder)

                with open(graph2run, 'w') as file:
                    file.write(filedata)

                args = [self.config["snap_gpt"]["gptbin_path"], graph2run, '-c', str(self.config["computing_resources"]["cache"]), '-q', str(self.config["computing_resources"]["cpu"])]

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
            if "target.data" in os.listdir(self.config["project_definition"]["project_folder"]):
                shutil.rmtree(os.path.join(self.config["project_definition"]["project_folder"], "target.data"))
            if "target.dim" in os.listdir(self.config["project_definition"]["project_folder"]):
                os.remove(os.path.join(self.config["project_definition"]["project_folder"], "target.dim"))
        except:
            pass
        
    def _update_config(self):
        self.config["processing_parameters"]["current_result"] = self.outputexportfolder
        self.config_parser.update_project_config(self.project_name, self.config)

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