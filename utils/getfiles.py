import os

master_file = os.listdir("master/")
slave_files = os.listdir("slaves/")

with open("docs/files.txt", "w") as file:
    file.write(master_file[0])
    file.write("\n")
    for item in slave_files:
        file.write(item)
        file.write("\n")
    file.close()