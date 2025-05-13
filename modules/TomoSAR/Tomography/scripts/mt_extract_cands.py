#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import subprocess
import logging
from typing import Optional, List
import platform

class MTExtractCands:
    """
    Extract candidate pixels data from patches
    """
    
    def __init__(self):
        """Initialize MTExtractCands class"""
        self.logger = self._setup_logger()
        self.project_conf_path = Path(__file__).parent.parent.parent.parent / "snap2stamps" / "bin" / "project.conf"
        self.work_dir = self._read_project_conf("CURRENT_RESULT")

        self.dophase = 1
        self.dolonlat = 1
        self.dodem = 1
        self.docands = 1
        self.precision = "f"
        self.byteswap = 1
        self.maskfile = None
        
        # Set the executable path
        if platform.system() == "Linux":
            self.selsbc_path = Path(__file__).parent.parent.parent.parent / "StaMPS" / "bin" / "selsbc_patch"
            self.selpsc_path = Path(__file__).parent.parent.parent.parent / "StaMPS" / "bin" / "selpsc_patch"
            self.psclonlat = Path(__file__).parent.parent.parent.parent / "StaMPS" / "bin" / "psclonlat"
            self.pscdem = Path(__file__).parent.parent.parent.parent / "StaMPS" / "bin" / "pscdem"
            self.pscphase = Path(__file__).parent.parent.parent.parent / "StaMPS" / "bin" / "pscphase"
        else:
            self.selsbc_path = Path(__file__).parent.parent.parent.parent / "StaMPS" / "src" / "selsbc_patch.exe"
            self.selpsc_path = Path(__file__).parent.parent.parent.parent / "StaMPS" / "src" / "selpsc_patch.exe"
            self.psclonlat = Path(__file__).parent.parent.parent.parent / "StaMPS" / "src" / "psclonlat.exe"
            self.pscdem = Path(__file__).parent.parent.parent.parent / "StaMPS" / "src" / "pscdem.exe"
            self.pscphase = Path(__file__).parent.parent.parent.parent / "StaMPS" / "src" / "pscphase.exe"

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
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
            
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
        self.logger.info(f"   -> Patch: {patch}")
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
                    
            os.system(' '.join(cmd))
                
        # Process lon/lat
        if dolonlat:
            cmd = [
                str(self.psclonlat).replace('\\', '/'),
                f"{self.work_dir}/psclonlat.in",
                "pscands.1.ij",
                "pscands.1.ll"
            ]
            os.system(' '.join(cmd))
                
        # Process DEM
        if dodem:
            cmd = [
                str(self.pscdem).replace('\\', '/'),
                f"{self.work_dir}/pscdem.in",
                "pscands.1.ij",
                "pscands.1.hgt"
            ]
            os.system(' '.join(cmd))
                
        # Process phase
        if dophase:
            cmd = [
                str(self.pscphase).replace('\\', '/'),
                f"{self.work_dir}/pscphase.in",
                "pscands.1.ij",
                "pscands.1.ph"
            ]
            os.system(' '.join(cmd))
        os.chdir("..")        
        return True
        
    def run(self) -> bool:
        """
        Run the extraction process for all patches
        
        Args:
            dophase: Whether to extract phase data
            dolonlat: Whether to extract lon/lat data
            dodem: Whether to extract DEM data
            docands: Whether to extract candidate data
            precision: Data precision ('s' for short, 'f' for float)
            byteswap: Whether to byteswap the data
            maskfile: Optional mask file path
            patch_list: Path to patch list file (default: patch.list)
            
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
            
        # Process each patch
        success = True
        for patch in patches:
            success = self.process_patch(
                patch,
                dophase=self.dophase,
                dolonlat=self.dolonlat,
                dodem=self.dodem,
                docands=self.docands,
                precision=self.precision,
                byteswap=self.byteswap,
                maskfile=self.maskfile
            )
            if not success:
                break
                
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
    
    args = parser.parse_args()
    
    extractor = MTExtractCands()
    success = extractor.run(
        dophase=bool(args.dophase),
        dolonlat=bool(args.dolonlat),
        dodem=bool(args.dodem),
        docands=bool(args.docands),
        precision=args.precision,
        byteswap=bool(args.byteswap),
        maskfile=args.maskfile if args.maskfile else None,
        patch_list=args.patch_list
    )
    
    sys.exit(0 if success else 1)