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
import time

inputfile = sys.argv[1]

bar_message = '\n#####################################################################\n'

# Getting configuration variables from inputfile
try:
    with open(inputfile, 'r') as in_file:
        for line in in_file.readlines():
            if "PROJECTFOLDER" in line:
                PROJECT = line.split('=')[1].strip()
                print(PROJECT)
            elif "IW1" in line:
                IW = line.split('=')[1].strip()
                print(IW)
            elif "GRAPHSFOLDER" in line:
                GRAPH = line.split('=')[1].strip()
                print(GRAPH)
            elif "GPTBIN_PATH" in line:
                GPT = line.split('=')[1].strip()
                print(GPT)
            elif "CACHE" in line:
                CACHE = line.split('=')[1].strip()
            elif "CPU" in line:
                CPU = line.split('=')[1].strip()
except Exception as e:
    print("Error reading input file: " + e)
    sys.exit(1)

# Creating necessary folders
slavefolder = os.path.join(PROJECT, 'slaves')
splitfolder = os.path.join(PROJECT, 'split')
logfolder = os.path.join(PROJECT, 'logs')
graphfolder = os.path.join(PROJECT, 'graphs')

for folder in [splitfolder, logfolder, graphfolder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

graph2run = os.path.join(graphfolder, 'splitgraph2run.xml')
outlog = os.path.join(logfolder, 'split_proc_stdout.log')

with open(outlog, 'a') as out_file:
    out_file.write(bar_message)
    message = '## TOPSAR Splitting and Apply Orbit\n'
    out_file.write(message)
    out_file.write(bar_message)
    
    k = 0
    for acdatefolder in sorted(os.listdir(slavefolder)):
        k += 1
        print('[' + str(k) + '] Folder: ' + acdatefolder)
        out_file.write('[' + str(k) + '] Folder: ' + acdatefolder + '\n')

        folder_path = os.path.join(slavefolder, acdatefolder)
        files = glob.glob(folder_path + '/*.zip')

        print(files)
        out_file.write(str(files) + '\n')

        if not files:
            continue

        head, tail = os.path.split(files[0])
        date_str = tail[17:25]
        splitslavefolder = os.path.join(splitfolder, date_str)
        outputname = date_str + '_' + IW + '.dim'
        output_path = os.path.join(splitslavefolder, outputname)

        # Check if the file is already processed
        if os.path.exists(output_path):
            print("Slave image " + date_str + " is already processed.")
            out_file.write("Slave image " + date_str + " is already processed.\n")
            continue  # Skip to the next file

        # Create folder only if processing is needed
        if not os.path.exists(splitslavefolder):
            os.makedirs(splitslavefolder)

        # Selecting the correct graph
        if len(files) == 1:
            graphxml = os.path.join(GRAPH, 'slave_split_applyorbit.xml')
        else:
            graphxml = os.path.join(GRAPH, 'slaves_assemble_split_applyorbit.xml')

        with open(graphxml, 'r') as file:
            filedata = file.read()
            filedata = filedata.replace('INPUTFILE', files[0])
            filedata = filedata.replace('IWs', IW)
            filedata = filedata.replace('OUTPUTFILE', output_path)

        with open(graph2run, 'w') as file:
            file.write(filedata)

        args = [GPT, graph2run, '-c', CACHE, '-q', CPU]
        print(args)
        out_file.write(str(args) + '\n')

        # Launch the process
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        timeStarted = time.time()
        stdout, stderr = process.communicate()
        process.wait()

        print('SNAP STDOUT: ' + stdout)
        out_file.write(stdout + '\n')

        timeDelta = time.time() - timeStarted
        print('[' + str(k) + '] Finished process in ' + str(timeDelta) + ' seconds.')
        out_file.write('[' + str(k) + '] Finished process in ' + str(timeDelta) + ' seconds.\n')

        if process.returncode != 0:
            message = 'Error splitting slaves ' + str(files) + '\n' + stderr
            print(message)
            out_file.write(message + '\n')
        else:
            message = 'Split master ' + str(files) + ' successfully completed.\n'
            print(message)
            out_file.write(message)

        out_file.write(bar_message)