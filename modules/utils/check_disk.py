import os

def get_disk_usage_statvfs(path="/"):
    stat = os.statvfs(path)
    total = stat.f_blocks * stat.f_frsize
    free = stat.f_bfree * stat.f_frsize
    percentage = float(free) / float(total) * 100
    if percentage > 10.0:
        print("Disk space is still available")

get_disk_usage_statvfs()
