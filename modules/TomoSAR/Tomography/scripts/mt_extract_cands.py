#!/usr/bin/env python3

import logging
import multiprocessing
import os
import platform
import subprocess
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Optional

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(project_path)

class MTExtractCands:
    """
    Extract candidate pixels data from patches
    """
    
    def __init__(self):
        """Initialize MTExtractCands class"""
        self.logger = self._setup_logger()
        self.project_conf_path = Path(os.path.join(project_path, "modules/snap2stamps/bin/project.conf"))
        self.work_dir = self._read_project_conf("CURRENT_RESULT")

        self.dophase = 1
        self.dolonlat = 1
        self.dodem = 1
        self.docands = 1
        self.precision = "f"
        self.byteswap = 1
        self.maskfile = None
        self.num_cores = 4
        
        # Set up log file
        log_dir = Path(self.work_dir) / "logs"
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_dir / f"mt_extract_cands_{timestamp}.log"
        self._setup_file_logger()
        
        # Set the executable path
        if platform.system() == "Linux":
            self.selsbc_path = Path(os.path.join(project_path, "modules/StaMPS/bin/selsbc_patch"))
            self.selpsc_path = Path(os.path.join(project_path, "modules/StaMPS/bin/selpsc_patch"))
            self.psclonlat = Path(os.path.join(project_path, "modules/StaMPS/bin/psclonlat"))
            self.pscdem = Path(os.path.join(project_path, "modules/StaMPS/bin/pscdem"))
            self.pscphase = Path(os.path.join(project_path, "modules/StaMPS/bin/pscphase"))
        else:
            self.selsbc_path = Path(os.path.join(project_path, "modules/StaMPS/src/selsbc_patch.exe"))
            self.selpsc_path = Path(os.path.join(project_path, "modules/StaMPS/src/selpsc_patch.exe"))
            self.psclonlat = Path(os.path.join(project_path, "modules/StaMPS/src/psclonlat.exe"))
            self.pscdem = Path(os.path.join(project_path, "modules/StaMPS/src/pscdem.exe"))
            self.pscphase = Path(os.path.join(project_path, "modules/StaMPS/src/pscphase.exe"))

    def _read_project_conf(self, key):
        """Read value from project configuration file"""
        try:
            with open(self.project_conf_path, 'r') as f:
                for line in f:
                    if key in line:
                        return line.split('=')[1].strip()
        except FileNotFoundError:
            self.logger.error(f"Could not open {self.project_conf_path}")
        return ''
        
    def _setup_logger(self):
        """Setup logging configuration"""
        logger = logging.getLogger('MTExtractCands')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _setup_file_logger(self):
        """Setup file logging configuration"""
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _run_command(self, cmd: List[str], patch: str) -> bool:
        """
        Run a command and log its output
        
        Args:
            cmd: Command to run
            patch: Current patch being processed
            
        Returns:
            bool: True if command succeeded, False otherwise
        """
        try:
            self.logger.info(f"Executing command in patch {patch}: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            stdout, stderr = process.communicate()
            
            if stdout:
                self.logger.info(f"Command output:\n{stdout}")
            if stderr:
                self.logger.warning(f"Command warnings/errors:\n{stderr}")
                
            if process.returncode != 0:
                self.logger.error(f"Command failed with return code {process.returncode}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error executing command: {str(e)}")
            return False

    def process_patch(self, 
                     patch: str,
                     dophase: bool = True,
                     dolonlat: bool = True,
                     dodem: bool = True,
                     docands: bool = True,
                     precision: str = "f",
                     byteswap: bool = False,
                     maskfile: Optional[str] = None) -> bool:
        """
        Process a single patch
        
        Args:
            patch: Patch directory name
            dophase: Whether to extract phase data
            dolonlat: Whether to extract lon/lat data
            dodem: Whether to extract DEM data
            docands: Whether to extract candidate data
            precision: Data precision ('s' for short, 'f' for float)
            byteswap: Whether to byteswap the data
            maskfile: Optional mask file path
            
        Returns:
            bool: True if processing succeeded, False otherwise
        """
        try:
            print(f"   -> Patch: {patch}")
            os.chdir(patch)
            
            # Process candidates
            if docands:
                if os.path.exists(f"{self.work_dir}/selsbc.in"):
                    # Select SB candidates
                    cmd = [
                        str(self.selsbc_path).replace('\\', '/'),
                        f"{self.work_dir[1:]}/selsbc.in",
                        "patch.in",
                        "pscands.1.ij",
                        "pscands.1.da",
                        "mean_amp.flt",
                        precision,
                        str(int(self.byteswap))
                    ]
                    if maskfile:
                        cmd.append(maskfile)
                else:
                    # Select PS candidates
                    cmd = [
                        str(self.selpsc_path).replace('\\', '/'),
                        f"{self.work_dir}/selpsc.in",
                        "patch.in",
                        "pscands.1.ij",
                        "pscands.1.da",
                        "mean_amp.flt",
                        precision,
                        str(int(self.byteswap))
                    ]
                    if maskfile:
                        cmd.append(maskfile)
                        
                with open(self.log_file, 'a') as log:
                    log.write(f"\n=== Processing {patch} with {'selsbc_patch' if os.path.exists(f'{self.work_dir}/selsbc.in') else 'selpsc_patch'} ===\n")
                    subprocess.run(' '.join(cmd), shell=True, stdout=log, stderr=log)
                    
            # Process lon/lat
            if dolonlat:
                cmd = [
                    str(self.psclonlat).replace('\\', '/'),
                    f"{self.work_dir}/psclonlat.in",
                    "pscands.1.ij",
                    "pscands.1.ll"
                ]
                with open(self.log_file, 'a') as log:
                    log.write(f"\n=== Processing {patch} with psclonlat ===\n")
                    subprocess.run(' '.join(cmd), shell=True, stdout=log, stderr=log)
                    
            # Process DEM
            if dodem:
                cmd = [
                    str(self.pscdem).replace('\\', '/'),
                    f"{self.work_dir}/pscdem.in",
                    "pscands.1.ij",
                    "pscands.1.hgt"
                ]
                with open(self.log_file, 'a') as log:
                    log.write(f"\n=== Processing {patch} with pscdem ===\n")
                    subprocess.run(' '.join(cmd), shell=True, stdout=log, stderr=log)
                    
            # Process phase
            if dophase:
                cmd = [
                    str(self.pscphase).replace('\\', '/'),
                    f"{self.work_dir}/pscphase.in",
                    "pscands.1.ij",
                    "pscands.1.ph"
                ]
                with open(self.log_file, 'a') as log:
                    log.write(f"\n=== Processing {patch} with pscphase ===\n")
                    subprocess.run(' '.join(cmd), shell=True, stdout=log, stderr=log)
            os.chdir("..")        
            return True
        except Exception as e:
            self.logger.error(f"Error processing patch {patch}: {str(e)}")
            return False
        
    def run(self, num_cores: Optional[int] = None) -> bool:
        """
        Run the extraction process for all patches in parallel
        
        Args:
            num_cores: Number of CPU cores to use (default: all available cores)
            
        Returns:
            bool: True if all processing succeeded, False otherwise
        """
        # Read patch list
        patch_list_file = f"{self.work_dir}/patch.list"
        try:
            with open(patch_list_file, 'r') as f:
                patches = [f"{str(self.work_dir)}/{line.strip()}" for line in f if line.strip()]
        except FileNotFoundError:
            self.logger.error(f"Patch list file {patch_list_file} not found")
            return False

        # Set number of cores to use
        if num_cores is None:
            num_cores = self.num_cores
        else:
            num_cores = min(num_cores, self.num_cores)
            
        # Create a partial function with the fixed parameters
        process_func = partial(
            self.process_patch,
            dophase=self.dophase,
            dolonlat=self.dolonlat,
            dodem=self.dodem,
            docands=self.docands,
            precision=self.precision,
            byteswap=self.byteswap,
            maskfile=self.maskfile
        )
            
        # Process patches in parallel
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.map(process_func, patches)
                
        # Check if all patches were processed successfully
        success = all(results)
        if not success:
            self.logger.error("Some patches failed to process")
                
        return success

if __name__ == "__main__":
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract candidate pixels data from patches"
    )
    parser.add_argument(
        "dophase",
        type=int,
        nargs="?",
        default=1,
        help="Whether to extract phase data (1) or not (0)"
    )
    parser.add_argument(
        "dolonlat",
        type=int,
        nargs="?",
        default=1,
        help="Whether to extract lon/lat data (1) or not (0)"
    )
    parser.add_argument(
        "dodem",
        type=int,
        nargs="?",
        default=1,
        help="Whether to extract DEM data (1) or not (0)"
    )
    parser.add_argument(
        "docands",
        type=int,
        nargs="?",
        default=1,
        help="Whether to extract candidate data (1) or not (0)"
    )
    parser.add_argument(
        "precision",
        type=str,
        nargs="?",
        default="f",
        choices=["s", "f"],
        help="Data precision ('s' for short, 'f' for float)"
    )
    parser.add_argument(
        "byteswap",
        type=int,
        nargs="?",
        default=1,
        help="Whether to byteswap the data (1) or not (0)"
    )
    parser.add_argument(
        "maskfile",
        type=str,
        nargs="?",
        default="",
        help="Optional mask file path"
    )
    parser.add_argument(
        "patch_list",
        type=str,
        nargs="?",
        default="patch.list",
        help="Path to patch list file"
    )
    parser.add_argument(
        "--cores",
        type=int,
        help="Number of CPU cores to use (default: all available cores)"
    )
    
    args = parser.parse_args()
    
    extractor = MTExtractCands()
    success = extractor.run(num_cores=args.cores)
    
    sys.exit(0 if success else 1)