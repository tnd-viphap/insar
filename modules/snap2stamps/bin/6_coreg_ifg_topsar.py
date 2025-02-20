### Python script to use SNAP as InSAR processor compatible with StaMPS PSI processing
# Author Jose Manuel Delgado Blasco
# Date: 21/06/2018
# Version: 1.0

# Step 1 : preparing slaves in folder structure
# Step 2 : TOPSAR Splitting (Assembling) and Apply Orbit
# Step 3 : Coregistration and Interferogram generation
# Step 4 : StaMPS export

# Added option for CACHE and CPU specification by user
# Planned support for DEM selection and ORBIT type selection 

import os
from pathlib import Path
import sys
import glob
import subprocess
import shlex
import time
import xml.etree.ElementTree as ET
import shutil

start_time = time.time()

inputfile = sys.argv[1]

def retrieve_bandname(dimfile):
    tree = ET.parse(dimfile)
    root = tree.getroot()

    for bandname in root.findall(".//BAND_NAME"):
        if bandname.text == "elevation_VV":
            bandname.text = "elevation"
        elif bandname.text == "orthorectifiedLon_VV":
            bandname.text = "orthorectifiedLon"
        elif bandname.text == "orthorectifiedLat_VV":
            bandname.text = "orthorectifiedLat"
    
    tree.write(dimfile, encoding="UTF-8", xml_declaration=True)

bar_message = '\n#####################################################################\n'

# Getting configuration variables from inputfile
with open(inputfile, 'r') as file:
    for line in file.readlines():
        if "PROJECTFOLDER" in line:
            PROJECT = line.split('=')[1].strip()
            print(PROJECT)
        elif "IW1" in line:
            IW = line.split('=')[1].strip()
            print(IW)
        elif "MASTER" in line:
            MASTER = line.split('=')[1].strip()
            print(MASTER)
        elif "GRAPHSFOLDER" in line:
            GRAPH = line.split('=')[1].strip()
            print(GRAPH)
        elif "GPTBIN_PATH" in line:
            GPT = line.split('=')[1].strip()
            print(GPT)
        elif "LONMIN" in line:
            LONMIN = line.split('=')[1].strip()
        elif "LATMIN" in line:
            LATMIN = line.split('=')[1].strip()
        elif "LONMAX" in line:
            LONMAX = line.split('=')[1].strip()
        elif "LATMAX" in line:
            LATMAX = line.split('=')[1].strip()
        elif "CACHE" in line:
            CACHE = line.split('=')[1].strip()
        elif "CPU" in line:
            CPU = line.split('=')[1].strip()

polygon = 'POLYGON ((' + LONMIN + ' ' + LATMIN + ',' + LONMAX + ' ' + LATMIN + ',' + LONMAX + ' ' + LATMAX + ',' + LONMIN + ' ' + LATMAX + ',' + LONMIN + ' ' + LATMIN + '))'
######################################################################################
## TOPSAR Coregistration and Interferogram formation ##
######################################################################################
slavesplittedfolder = PROJECT + 'slaves'
outputcoregfolder = PROJECT + 'coreg'
outputifgfolder = PROJECT + 'ifg'
logfolder = PROJECT + 'logs'

for folder in [outputcoregfolder, outputifgfolder, logfolder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

outlog = logfolder + '/coreg_ifg_proc_stdout.log'

graphxml = GRAPH + 'coreg_ifg_computation_subset.xml'
print(graphxml)
graph2run = GRAPH + 'coreg_ifg2run.xml'

out_file = open(outlog, 'a')
err_file = out_file

print(bar_message)
out_file.write(bar_message)
message = '## Coregistration and Interferogram computation started:\n'
print(message)
out_file.write(message)
print(bar_message)
out_file.write(bar_message)

k = 0
sorted_slavesplittedfolder = []
for folder in os.listdir(slavesplittedfolder):
    for file in os.listdir(os.path.join(slavesplittedfolder, folder)):
        if file.endswith('.dim'):
            sorted_slavesplittedfolder.append(os.path.join(slavesplittedfolder, folder, file))
sorted_slavesplittedfolder = sorted(sorted_slavesplittedfolder)
tailm = MASTER.split('/')[-1].split('_')[0]
for dimfile in sorted_slavesplittedfolder:
    print(dimfile)
    k += 1
    head, tail = os.path.split(os.path.join(slavesplittedfolder, dimfile))
    if tail[0:8] != tailm:
        message = '[' + str(k) + '] Processing slave file :' + tail + '\n'
        print(message)
        out_file.write(message)
        outputname = tailm + '_' + tail[0:8] + '_' + IW
        
        with open(graphxml, 'r') as file:
            filedata = file.read()
        
        # Replace the target string
        filedata = filedata.replace('MASTER', MASTER)
        filedata = filedata.replace('SLAVE', dimfile)
        filedata = filedata.replace('OUTPUTCOREGFOLDER', outputcoregfolder)
        filedata = filedata.replace('OUTPUTIFGFOLDER', outputifgfolder)
        filedata = filedata.replace('OUTPUTFILE', outputname)
        filedata = filedata.replace('POLYGON', polygon)
        
        # Write the file out again
        with open(graph2run, 'w') as file:
            file.write(filedata)
        
        args = [GPT, graph2run, '-c', CACHE, '-q', CPU]
        
        # Launch the processing
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        timeStarted = time.time()
        stdout = process.communicate()[0]
        print('SNAP STDOUT:{}'.format(stdout))
        
        timeDelta = time.time() - timeStarted  # Get execution time.
        print('[' + str(k) + '] Finished process in ' + str(timeDelta) + ' seconds.')
        out_file.write('[' + str(k) + '] Finished process in ' + str(timeDelta) + ' seconds.\n')
        
        if process.returncode != 0:
            message = 'Error computing coregistration and interferogram generation of splitted slave ' + str(dimfile)
            err_file.write(message + '\n')
        else:
            message = 'Coregistration and Interferogram computation for data ' + str(tail) + ' successfully completed.\n'
            print(message)
            out_file.write(message)
        
        print(bar_message)
        out_file.write(bar_message)

        # Retrieve bandname
        #retrieve_bandname(f"{outputifgfolder}/{outputname}.dim")

out_file.close()
end_time = time.time()
print(f"Coregistration and Interferogram executes in {(end_time - start_time)/60} minutes.")

