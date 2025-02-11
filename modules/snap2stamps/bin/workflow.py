import os
import time
import subprocess

# List of Python files to execute
python_files = ["master_sel.py", "splitting_master.py", "slaves_prep.py", "splitting_slaves.py"]
    # "coreg_ifg_topsar.py", "stamps_export.py"
python_files = [os.path.join("bin", f) for f in python_files]
print("Workflow:", python_files)

failed_scripts = {}

for idx, script in enumerate(python_files):
    command = ["python3", script]  # Default command

    if idx == 0:
        time.sleep(2)
        master_files = os.listdir("master/")
        if master_files:
            master_file = os.path.join("master", master_files[0])
            command = ["python3", script, "bin/project.conf", master_file]
        else:
            print("[WARNING] No master file found, skipping " + script + "...")
            continue
    elif idx == 2:
        time.sleep(1)
        command = ["python3", script, "bin/project.conf"]
    elif idx in [1, 3]:
        time.sleep(1)
        command = ["python2", script, "bin/project.conf"]

    print("[INFO] Running: " + ' '.join(command))
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if stdout:
        print("[STDOUT] " + script + ":\n" + stdout.decode())
    if stderr:
        print("[STDERR] " + script + ":\n" + stderr.decode())

    if process.returncode != 0:
        failed_scripts[script] = stderr.decode()
        print("[ERROR] " + script + " failed with return code " + str(process.returncode))

if failed_scripts:
    print("\n====== Execution Summary ======")
    print("[ERROR] The following scripts failed:")
    for script, error in failed_scripts.items():
        print(" - " + script + ": " + error.strip())
else:
    print("\n[INFO] All scripts executed successfully.")