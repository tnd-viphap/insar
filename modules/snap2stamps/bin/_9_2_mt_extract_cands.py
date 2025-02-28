import os
import platform
import subprocess


class MTExtractCands:
    def __init__(self, patch_list, sel_file):
        self.inputfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], "project.conf")
        self._load_config()
        
        self.dophase=1
        self.dolonlat=1
        self.dodem=1
        self.docands=1,
        self.precision="f"
        self.byteswap=1
        self.mask_file=""
        self.patch_list=patch_list
        self.sel_file = sel_file
        
        self.masterampfilename=""
        
        self.workdir = self.CURRENT_RESULT
        self.plf = platform.system()
        if self.plf == "Windows":
            self._add_executors()

    def _load_config(self):
        with open(self.inputfile, 'r') as file:
            for line in file.readlines():
                key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                if key:
                    setattr(self, key, value)  # Dynamically set variables
                    
    def _add_executors(self):
        folder_to_add = f"{self.PROJECTFOLDER}modules/StaMPS/src/"
        # Add temporarily
        os.environ["PATH"] += os.pathsep + folder_to_add

        # Add permanently using setx
        setx_command = f'setx PATH "%PATH%;{folder_to_add}"'
        subprocess.run(setx_command, shell=True)
        print("Executors added to PATH")
    
    '''
    def sel_parm_process(self, param_file):
        try:
            with open(param_file, "r") as parmfile:
                # Read dispersion threshold
                D_thresh = float(parmfile.readline().strip())
                self.D_thresh_sq = D_thresh * D_thresh
                self.pick_higher = 1 if D_thresh < 0 else 0

                # Read width
                self.width = int(parmfile.readline().strip())

                # Save position and count number of files
                savepos = parmfile.tell()
                self.num_files = sum(1 for _ in parmfile)  # Count remaining lines
                parmfile.seek(savepos)  # Reset to saved position
                
                # Read amplitude filenames and calibration factors
                self.amp_filenames = []
                self.calib_factor = []
                for _ in range(self.num_files):
                    line = parmfile.readline().strip().split()
                    if len(line) < 2:
                        print("[PARAM_PROCESS]: Invalid format in parameter file")
                        sys.exit(1)
                    self.amp_filenames.append(line[0])
                    self.calib_factor.append(float(line[1]))
                parmfile.close()
        except FileNotFoundError:
            print(f"[PARAM_PROCESS]: Error opening file {param_file}")
            print("\n")
            sys.exit(1)
            
    def sel_amplitude_process(self):
        # Process amplitude files
        magic_number = 0x59A66A95
        self.amp_files = []

        for i in range(self.num_files):
            try:
                ampfile = open(self.amp_filenames[i], "rb")  # Open binary file
                self.amp_files.append(ampfile)
                print(f"-> Opening {self.amp_filenames[i]}...")

                # Read the first 32 bytes
                header = ampfile.read(32)

                # Check for magic number
                if len(header) >= 4:
                    file_magic = struct.unpack("<I", header[:4])[0]  # Read as little-endian
                    if file_magic == magic_number:
                        print("-> Sun raster file. Skipping header...\n")
                    else:
                        ampfile.seek(0)  # Reset to beginning

            except FileNotFoundError:
                print(f"[AMPLITUDE_PROCESS]: Error opening file {self.amp_filenames[i]}")
                sys.exit(1)

        print("-> Number of amplitude files =", self.num_files)
        
    def sel_read_patch(self, patch_file):
        # Read patch file
        try:
            with open(patch_file, "r") as patchfile:
                self.rg_start = int(patchfile.readline().strip())
                self.rg_end = int(patchfile.readline().strip())
                self.az_start = int(patchfile.readline().strip())
                self.az_end = int(patchfile.readline().strip())
        except FileNotFoundError:
            print(f"[SEL_READ_PATCH]: Error opening file {patch_file}")
            print("\n")
            sys.exit(1)

        # Determine size of a patch
        self.patch_lines = self.az_end - self.az_start + 1
        self.patch_width = self.rg_end - self.rg_start + 1

        # Determine pixel size based on `prec`
        self.sizeoffloat = 4  # size of a float
        if self.precision[0] == 's':
            self.sizeofelement = 2  # size of short (16-bit integer)
        else:
            self.sizeofelement = 4  # size of float (32-bit float)

        # Bytes per line in amplitude files (SLCs)
        self.linebytes = self.width * self.sizeofelement * 2
        self.patch_linebytes = self.patch_width * self.sizeofelement * 2
        self.patch_amp_linebytes = self.patch_width * self.sizeofelement

        # Determine number of lines in the first amplitude file
        try:
            file_size = os.path.getsize(self.amp_filenames[0])  # Get file size in bytes
            numlines = file_size // (self.width * self.sizeofelement * 2)
            print("-> Number of lines per file =", numlines)
        except FileNotFoundError:
            print(f"[SEL_READ_PATCH]: Error opening file {self.amp_filenames[0]}")
            sys.exit(1)

        print("-> Patch lines =", self.patch_lines)
        print("-> Patch width =", self.patch_width)
        patchfile.close()
        
    def sel_core(self, pscandij, daoutname, meanoutname, maskfilename=None, masterampfilename=None):
        # Check if the mask file exists
        mask_exists = 0
        if os.path.exists(maskfilename):
            mask_exists = 1
            print(f"-> Opening {maskfilename}...")
            maskfile = open(maskfilename, "r")
            
        masteramp_exists=0
        if os.path.exists(masterampfilename):
            masteramp_exists = 1
            print(f"-> Opening {masterampfilename}...")
            masterampfile = open(masterampfilename, "r")

        # Open output files
        with open(pscandij, 'w') as ijfile, \
            open(pscandij+'.int', 'wb') as jifile, \
            open(pscandij+'0', 'w') as ijfile0, \
            open(daoutname, 'w') as daoutfile, \
            open(meanoutname, 'wb') as meanoutfile:
            
            # Allocate buffers
            buffer = bytearray(self.num_files * self.patch_linebytes)
            bufferf = np.frombuffer(buffer, dtype=np.complex64)
            buffers = np.frombuffer(buffer, dtype=np.int16)
            
            maskline = np.zeros(self.patch_width, dtype=np.uint8)
            
            masterampline = bytearray(self.patch_linebytes)
            masterlinef = np.frombuffer(masterampline, dtype=np.complex64)
            masterlines = np.frombuffer(masterampline, dtype=np.int16)
            
            for x in range(self.patch_width):
                if self.precision[0] == 's':
                    masterlines[x] = 1
                    if self.byteswap:
                        masterlines[x] = masterlines[x].byteswap()
                else:
                    masterlinef[x] = 1
                    if self.byteswap:
                        masterlinef[x] = masterlinef[x].byteswap()
            
            pix_start = (self.az_start - 1) * self.width + (self.rg_start - 1)
            pos_start = pix_start * self.sizeofelement * 2
            
            # Read first line from each amp file
            for i in range(self.num_files):
                self.amp_files[i].seek(pos_start)
                buffer[i * self.patch_linebytes:(i + 1) * self.patch_linebytes] = self.amp_files[i].read(self.patch_linebytes)
            
            if mask_exists:
                maskfile.seek(pix_start)
                maskline[:] = np.frombuffer(maskfile.read(self.patch_width), dtype=np.uint8)
            
            if masteramp_exists:
                masterampfile.seek(pos_start)
                masterampline[:] = masterampfile.read(self.patch_linebytes)
            
            y = 0
            pscid = 0
            
            while y < self.patch_lines and self.amp_files[1].readable():
                if y >= 0:
                    for x in range(self.patch_width):
                        sumamp = 0
                        sumampsq = 0
                        amp_0 = 0
                        
                        if self.precision[0] == 's':
                            master_amp = masterlines[x]
                        else:
                            master_amp = masterlinef[x]
                        
                        if abs(master_amp) == 0:
                            master_amp = 1
                        
                        for i in range(self.num_files):
                            if self.precision[0] == 's':
                                camp = buffers[i * self.patch_width + x]
                            else:
                                camp = bufferf[i * self.patch_width + x]
                            
                            if self.byteswap:
                                camp = camp.byteswap()
                            
                            amp = abs(camp) / self.calib_factor[i] / abs(master_amp)
                            
                            if amp <= 0.00005:
                                amp_0 = 1
                                sumamp = 0
                                continue
                            else:
                                sumamp += amp
                                sumampsq += amp ** 2
                        
                        meanoutfile.write(struct.pack('f', sumamp))
                        
                        if maskline[x] == 0 and sumamp > 0:
                            D_sq = self.num_files * sumampsq / (sumamp ** 2) - 1
                            
                            if (self.pick_higher == 0 and D_sq < self.D_thresh_sq) or \
                            (self.pick_higher == 1 and D_sq >= self.D_thresh_sq):
                                if amp_0 != 1:
                                    pscid += 1
                                    ijfile.write(f"{pscid} {self.az_start - 1 + y} {self.rg_start - 1 + x}\n")
                                    J = struct.pack('i', self.rg_start - 1 + x)
                                    I = struct.pack('i', self.az_start - 1 + y)
                                    jifile.write(J + I)
                                    
                                    D_a = np.sqrt(D_sq)
                                    daoutfile.write(f"{D_a}\n")
                                else:
                                    ijfile0.write(f"{pscid} {self.az_start - 1 + y} {self.rg_start - 1 + x}\n")
                
                y += 1
                
                for i in range(self.num_files):
                    self.amp_files[i].seek(self.linebytes - self.patch_linebytes, 1)
                    buffer[i * self.patch_linebytes:(i + 1) * self.patch_linebytes] = self.amp_files[i].read(self.patch_linebytes)
                
                if mask_exists:
                    maskfile.seek(self.width - self.patch_width, 1)
                    maskline[:] = np.frombuffer(maskfile.read(self.patch_width), dtype=np.uint8)
                
                if masteramp_exists:
                    masterampfile.seek(self.linebytes - self.patch_linebytes, 1)
                    masterampline[:] = masterampfile.read(self.patch_linebytes)
                
                if y % 100 == 0:
                    print(f"{y} lines processed")
                
        ijfile.close()
        jifile.close()
        ijfile0.close()
        daoutfile.close()
        meanoutfile.close()
        for file in self.amp_files:
            file.close()
        if mask_exists==1:
            maskfile.close()
        if masteramp_exists==1:
            masterampfile.close()

    def extract_lonlat(self, parm_file, pscands_ij, pscands_ll):
        # Open parameter file
        with open(parm_file, "r") as parmfile:
            width = int(parmfile.readline().strip())
            
            # Read interferogram file paths
            self.ifgfiles = []
            for _ in range(2):  # Assuming num_files = 2
                ifgfilename = parmfile.readline().strip()
                print(f"-> Opening {ifgfilename}...")
                try:
                    self.ifgfiles.append(open(ifgfilename, "rb"))
                except IOError:
                    print(f"[EXTRACT_LONLAT]: Error opening file {ifgfilename}")
                    return

        # Open PS candidates file
        try:
            psfile = open(pscands_ij, "r")
            print(f"-> Opening {pscands_ij}...")
        except IOError:
            print(f"[EXTRACT_LONLAT]: Error opening file {pscands_ij}")
            return

        # Open output file
        try:
            outfile = open(pscands_ll, "wb")
        except IOError:
            print(f"[EXTRACT_LONLAT]: Error opening file {pscands_ll}")
            return
        
        # Process PS candidates
        for line in psfile:
            parts = line.split()
            if len(parts) < 3:
                continue
            pscid, y, x = map(int, parts[:3])
            xyaddr = (y * width + x) * 4  # sizeof(float) = 4 bytes
            
            for ifgfile in self.ifgfiles:
                ifgfile.seek(xyaddr)
                ifg_pixel = ifgfile.read(4)
                outfile.write(ifg_pixel)
                
            if pscid % 100000 == 0:
                print(f"-> [{pscid}]: PS candidates processed")
        
        # Close files
        psfile.close()
        outfile.close()
        for f in self.ifgfiles:
            f.close()

    def extract_psc_height(self, parm_file, pscands_ij, pscands_hgt):
        with open(parm_file, "r") as parmfile:
            width = int(parmfile.readline().strip())
            ifgfilename = parmfile.readline().strip()
            
        nodem_sw = False
        try:
            ifgfile = open(ifgfilename, "rb")
            print(f"[EXTRACT_DEM]: Error opening {ifgfilename}...")
        except FileNotFoundError:
            nodem_sw = True
        
        sizeofpixel = 8 if self.precision == 'd' else 4
        
        with open(pscands_ij, "r") as psfile, open(pscands_hgt, "wb") as outfile:
            for line in psfile:
                parts = line.split()
                if len(parts) < 3:
                    continue
                pscid, y, x = map(int, parts[:3])
                xyaddr = (y * width + x) * sizeofpixel
                
                if nodem_sw:
                    data = b"\x00" * sizeofpixel  # Write zeros if no DEM file
                else:
                    ifgfile.seek(xyaddr)
                    data = ifgfile.read(sizeofpixel)
                    if not data:
                        data = b"\x00" * sizeofpixel
                
                outfile.write(data)
                
                if pscid % 100000 == 0:
                    print(f"-> [{pscid}]: PS candidates processed")
        
        if not nodem_sw:
            ifgfile.close()
        psfile.close()
        outfile.close()
        print("DEM Ready")
    '''
    
    def process(self):
        with open(self.patch_list, 'r') as f:
            patches = f.read().splitlines()
        
        for patch in patches:
            self.workdir = patch
            print("\nProcessing:", patch)
            
            # SELPSC or SELSBC
            patch_in = os.path.join(self.workdir, 'patch.in').replace("\\", "/")
            pscands_ij = os.path.join(self.workdir, 'pscands.1.ij').replace("\\", "/")
            pscands_da = os.path.join(self.workdir, 'pscands.1.da').replace("\\", "/")
            mean_amp_flt = os.path.join(self.workdir, 'mean_amp.flt').replace("\\", "/")
            
            if self.docands:
                sel_file = "selsbc.in" if 'sbc' in self.sel_file else "selpsc.in"
                print(f"######### {sel_file.split('.')[0].upper()} #########")
                sel_cmd = f"{sel_file.split('.')[0]}_patch {self.sel_file} {patch_in} {pscands_ij} {pscands_da} {mean_amp_flt} {self.precision} {self.byteswap}"
                os.system(sel_cmd)
                if self.mask_file:
                    sel_cmd += f" {os.path.join(self.workdir, self.mask_file)}"
            
            psclonlat = os.path.join(self.CURRENT_RESULT, 'psclonlat.in').replace("\\", "/")
            pscands_ll = os.path.join(self.workdir, 'pscands.1.ll').replace("\\", "/")
            
            if self.dolonlat:
                print(f"######### LON-LAT Preparation #########")
                lonlat_cmd = f"psclonlat {psclonlat} {pscands_ij} {pscands_ll}"
                os.system(lonlat_cmd)
                print("\n")
                
            pscdem = os.path.join(self.CURRENT_RESULT, 'pscdem.in').replace("\\", "/")
            pscands_hgt = os.path.join(self.workdir, 'pscands.1.hgt').replace("\\", "/")
            
            if self.dodem:
                print(f"######### DEM Preparation #########")
                dem_cmd = f"pscdem {pscdem} {pscands_ij} {pscands_hgt}"
                os.system(dem_cmd)
                print("\n")
                
            pscphase = os.path.join(self.CURRENT_RESULT, 'pscphase.in').replace("\\", "/")
            pscands_ph = os.path.join(self.workdir, 'pscands.1.ph').replace("\\", "/")
            
            if self.dophase:
                print(f"######### PHASE Preparation #########")
                phase_cmd = f"pscphase {pscphase} {pscands_ij} {pscands_ph}"
                os.system(phase_cmd)
                print("\n")