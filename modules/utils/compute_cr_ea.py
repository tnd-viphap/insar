import numpy as np
import pandas as pd
import pyproj
from pyproj import Geod


def gps2ecef_pyproj(lat, lon, alt):
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)
    return x, y, z

def shift_target_point(pta_data: str):
    
    # Use pyproj to convert from ENU → Geodetic (lat/lon)
    geod = Geod(ellps='WGS84')
    
    # Read PTA data
    pta = pd.read_csv(pta_data)
    pta['latitude_corrected_[deg]'] = None
    pta['longitude_corrected_[deg]'] = None
    
    for index, data in pta.iterrows():
        # Compute azimuth and range unit vectors in ENU
        az_rad = np.radians(90-180*data['squint_angle_[rad]']/np.pi)
        
        # Azimuth (along-track) direction: from North clockwise
        az_vec = np.array([
            np.sin(az_rad),  # East
            np.cos(az_rad),  # North
            0                # Up
        ])

        # Ground range (across-track) direction is perpendicular to azimuth
        rg_vec = np.array([
            np.cos(az_rad),  # East
            -np.sin(az_rad), # North
            0                # Up
        ])

        # Total shift in ENU
        shift_ENU = data['azimuth_localization_error_[m]'] * az_vec + data['ground_range_localization_error_[m]'] * rg_vec
        
        # Shift east and north
        new_lon, new_lat, _ = geod.fwd(data['longitude_deg'], data['latitude_deg'], np.degrees(np.arctan2(shift_ENU[0], shift_ENU[1])),
                                    np.linalg.norm(shift_ENU[:2]))
        
        pta.at[index, 'longitude_corrected_[deg]'] = new_lon
        pta.at[index, 'latitude_corrected_[deg]'] = new_lat
    new_pta = pta.copy()
    pta = None
    new_pta.to_csv(pta_data, index=False)
    return new_pta["latitude_corrected_[deg]"], new_pta["longitude_corrected_[deg]"]
    
if __name__ == "__main__":
    print(gps2ecef_pyproj(15.141927140853692,108.88546286343805, 13.0))
    #shift_target_point('process/pta/pta_results.csv')
