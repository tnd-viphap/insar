import numpy as np
import pandas as pd
from pyproj import Geod, Transformer, Proj


def gps2ecef_pyproj(lat, lon, alt):
    ecef = Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    transformer = Transformer.from_proj(lla, ecef)
    x, y, z = transformer.transform(lon, lat, alt, radians=False)
    return x, y, z

def shift_target_point(pta_data: str):
    """
    Shift target point coordinates based on azimuth and ground range errors.
    Handles cases where either latitude or longitude is fixed.
    
    Args:
        pta_data (str): Path to the PTA results CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing original and corrected coordinates
    """
    # Use pyproj to convert from ENU → Geodetic (lat/lon)
    geod = Geod(ellps='WGS84')
    
    # Read PTA data
    pta = pd.read_csv(pta_data)
    
    # Initialize corrected coordinates columns
    pta['latitude_corrected_[deg]'] = pta['latitude_deg']
    pta['longitude_corrected_[deg]'] = pta['longitude_deg']
    
    # Process each row
    for index, row in pta.iterrows():
        # Skip if any required values are NaN
        if pd.isna(row['squint_angle_[rad]']) or pd.isna(row['azimuth_localization_error_[m]']) or pd.isna(row['ground_range_localization_error_[m]']):
            continue
            
        try:
            # Get current coordinates
            current_lon = row['longitude_deg']
            current_lat = row['latitude_deg']
            
            # Compute azimuth and range unit vectors in ENU
            az_rad = np.radians(90 - 180 * row['squint_angle_[rad]'] / np.pi)
            
            # Azimuth (along-track) direction: from North clockwise
            az_vec = np.array([
                np.sin(az_rad),  # East
                np.cos(az_rad),  # North
                0                # Up
            ])

            # Ground range (across-track) direction is perpendicular to azimuth
            rg_vec = np.array([
                np.cos(az_rad),   # East
                -np.sin(az_rad),  # North
                0                 # Up
            ])

            # Calculate shifts separately for azimuth and ground range
            az_shift = row['azimuth_localization_error_[m]'] * az_vec
            gr_shift = row['ground_range_localization_error_[m]'] * rg_vec
            
            # Calculate total shift in ENU
            shift_ENU = az_shift + gr_shift
            
            # Calculate shift magnitude and direction
            shift_magnitude = np.linalg.norm(shift_ENU[:2])
            shift_azimuth = np.degrees(np.arctan2(shift_ENU[0], shift_ENU[1]))
            
            # Calculate new coordinates
            new_lon, new_lat, _ = geod.fwd(
                current_lon,
                current_lat,
                shift_azimuth,
                shift_magnitude
            )
            
            # Update corrected coordinates based on error criteria
            # Only update coordinates that don't meet their error criteria
            if abs(row['ground_range_localization_error_[m]']) > 2.3:
                pta.at[index, 'longitude_corrected_[deg]'] = new_lon
            else:
                pta.at[index, 'longitude_corrected_[deg]'] = current_lon
                
            if abs(row['azimuth_localization_error_[m]']) > 14.1:
                pta.at[index, 'latitude_corrected_[deg]'] = new_lat
            else:
                pta.at[index, 'latitude_corrected_[deg]'] = current_lat
            
        except Exception as e:
            print(f"-> Error processing row {index}: {str(e)}")
            continue
    
    # Save updated data
    pta.to_csv(pta_data, index=False)
    
    return pta
    
if __name__ == "__main__":
    print(gps2ecef_pyproj(15.141927140853692,108.88546286343805, 13.0))
    #shift_target_point('process/pta/pta_results.csv')
