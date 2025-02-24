import xml.etree.ElementTree as ET
import numpy as np
import sys
import os

def parse_dim_file(dim_file):
    """Parses a .dim XML file and extracts necessary metadata"""
    tree = ET.parse(dim_file)
    root = tree.getroot()
    
    # Extract bounding box coordinates
    bbox_params = {
        'first_near_lat': float(root.find(".//MDATTR[@name='first_near_lat']").text),
        'first_near_long': float(root.find(".//MDATTR[@name='first_near_long']").text),
        'first_far_lat': float(root.find(".//MDATTR[@name='first_far_lat']").text),
        'first_far_long': float(root.find(".//MDATTR[@name='first_far_long']").text),
        'last_near_lat': float(root.find(".//MDATTR[@name='last_near_lat']").text),
        'last_near_long': float(root.find(".//MDATTR[@name='last_near_long']").text),
        'last_far_lat': float(root.find(".//MDATTR[@name='last_far_lat']").text),
        'last_far_long': float(root.find(".//MDATTR[@name='last_far_long']").text)
    }
    
    return bbox_params

def find_optimal_patches(bbox, max_patches, overlap):
    """Finds optimal num_range_patches and num_azimuth_patches considering overlap"""
    lat_min = min(bbox['first_near_lat'], bbox['first_far_lat'], bbox['last_near_lat'], bbox['last_far_lat'])
    lat_max = max(bbox['first_near_lat'], bbox['first_far_lat'], bbox['last_near_lat'], bbox['last_far_lat'])
    lon_min = min(bbox['first_near_long'], bbox['first_far_long'], bbox['last_near_long'], bbox['last_far_long'])
    lon_max = max(bbox['first_near_long'], bbox['first_far_long'], bbox['last_near_long'], bbox['last_far_long'])
    
    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min
    
    best_ratio = float('inf')
    best_patches = (1, 1)
    
    for num_range_patches in range(1, max_patches + 1):
        for num_azimuth_patches in range(1, max_patches + 1):
            patch_lat_size = lat_span / num_azimuth_patches
            patch_lon_size = lon_span / num_range_patches
            
            overlap_lat = overlap[1] / num_azimuth_patches
            overlap_lon = overlap[0] / num_range_patches
            
            effective_lat_size = patch_lat_size + overlap_lat
            effective_lon_size = patch_lon_size + overlap_lon
            
            aspect_ratio = abs(effective_lat_size / effective_lon_size - 1)
            
            if aspect_ratio < best_ratio:
                best_ratio = aspect_ratio
                best_patches = (num_range_patches, num_azimuth_patches)
    
    return best_patches

if __name__ == "__main__":
    # Getting configuration variables from inputfile
    inputfile = sys.argv[1]
    try:
        in_file = open(inputfile, 'r')
        
        for line in in_file.readlines():
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
    finally:
        in_file.close()

    dimfile = MASTER
    max_num_patches = int(sys.argv[2])
    range_overlap = float(sys.argv[3])
    azimuth_overlap = float(sys.argv[4])
    
    bbox = parse_dim_file(dimfile)
    optimal_range_patches, optimal_azimuth_patches = find_optimal_patches(bbox, max_num_patches, [range_overlap, azimuth_overlap])
    
    print(f"Optimal number of patches: Range = {optimal_range_patches}, Azimuth = {optimal_azimuth_patches}")
