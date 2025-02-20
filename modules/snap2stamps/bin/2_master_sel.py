import os
import sys
import numpy as np
import shutil
import time

# Update config
def modify_master(config_file, outputname):
    lines = ''''''
    with open(config_file, 'r') as file:
        lines = file.readlines()
        
    with open(config_file, "w") as file:
        for idx, line in enumerate(lines):
            if line.startswith("MASTER"):
                lines[idx] = "MASTER=" + outputname
            file.write(line)
        file.close()
                
######################################

inputfile = 'bin/project.conf'

try:
    with open(inputfile, 'r') as in_file:
        for line in in_file.readlines():
           if "IW1" in line:
                IW = line.split('=')[1].strip()
except Exception as e:
    print("Error reading input file: " + e)
    sys.exit(1)
        
# Get file index
sel = None
files = [f[17:25] for f in os.listdir("./data/raw") if f.endswith(".zip")]
files.extend()
files.extend([f for f in os.listdir("./data/slaves") if os.path.isdir(f)])
files = list(sorted(files))
if len(files) >=2:
    sel=int(np.floor(len(files)/2)-1)
if sel:
    if files[sel] == [f for f in os.listdir("./data/master") if os.path.isdir(f)][0]:
        print('Keep current Master date')
    else:
        if files[sel] in os.listdir("./data/slaves"):
            print(f'New selected Master date among Slaves: ./data/slaves/{files[sel]}')
        elif files[sel] in os.listdir("./data/raw"):
            print(f'New selected Master date among Raw: {files[sel]}')

try:
    if sel is not None:
        head, tail = os.path.split(files[sel])
        if tail in os.listdir("./data/master/"):
            pass
        else:
            splitmasterfolder = os.path.join("./data/master/", tail[17:25]) 
            os.makedirs(splitmasterfolder, exist_ok=True)
            outputname = splitmasterfolder + '/' + tail.replace(".zip", ".dim")
        ## Moving files to MASTER folder
            shutil.move(files[sel], splitmasterfolder)
            print("Moved " + files[sel] + " to " + splitmasterfolder)
            modify_master(inputfile, outputname)
except:
    print('Master to be selected and prepared manually by the user')
    pass

# Move other files to slaves folder
time.sleep(1)
if sel:
    del files[sel]
if files:
    for file in files:
        shutil.move(file, "./data/slaves/")
time.sleep(1)
