import numpy as np
from skyfield.api import load
from datetime import datetime

def lst_estimator(longitude):
    """Compute Local Sidereal Time (LST) based on given longitude and UTC time."""
    sensing_time_utc = "2025-02-15T22:51:06"  # Example Sentinel-1 Sensing Time

    ts = load.timescale()
    dt = datetime.strptime(sensing_time_utc, "%Y-%m-%dT%H:%M:%S")
    t = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

    gmst = t.gmst  # Greenwich Mean Sidereal Time
    lst = (gmst + longitude / 15) % 24  # Convert longitude to hours & keep in 0-24 range

    #print(f"Local Sidereal Time (LST) at {longitude}°E: {lst:.2f} hours")
    return lst

def xe(lon, lat, elev):
    return (6378 + elev) * np.cos(np.radians(lat)) * np.cos(np.radians(lon))

def ye(lon, lat, elev):
    return (6378 + elev) * np.cos(np.radians(lat)) * np.sin(np.radians(lon))

def ze(lon, lat, elev):
    return (6378 + elev) * np.sin(np.radians(lat))

def ra(xe, ye):
    return np.degrees(np.arctan2(ye, xe)) / 15  # Convert from degrees to hours

def dec(ha, lat, elev):
    return np.degrees(np.arcsin(np.cos(np.radians(ha)) * np.cos(np.radians(lat)) - np.sin(np.radians(elev))))

def az(dec, lat, elev):
    return np.degrees(np.arccos((np.sin(np.radians(dec)) - np.sin(np.radians(lat)) * np.sin(np.radians(elev))) / 
                                (np.cos(np.radians(lat)) * np.cos(np.radians(elev)))))

device_db = {
    "CRRef": "",
    "CRVY": "",
    "CRMC": "",
    "CRNear": "",
    "CRFar": ""
}

# Coordinates
lons = [106.71300330161516, 106.71494522085693, 106.71307840346428, 106.71388224741904, 106.71404854437755]
lats = [20.87315469613859, 20.875570633905554, 20.873585758421523, 20.87457888918893, 20.874478642915218]
elevs = [0.004, 0.004, 0.003, 0.006, 0.006]

# Lists for computed values
lsts, xes, yes, zes, ras, ha, decs, azimuth = [], [], [], [], [], [], [], []

# COMPUTE LST
for lon in lons:
    lsts.append(lst_estimator(lon))

# COMPUTE X, Y, Z Coordinates ECI datum
for lon, lat, elev in zip(lons, lats, elevs):
    xes.append(xe(lon, lat, elev))
    yes.append(ye(lon, lat, elev))
    zes.append(ze(lon, lat, elev))

# COMPUTE Right Ascension
for x, y in zip(xes, yes):
    ras.append(ra(x, y))

# COMPUTE Hour Angle
for lst, ra_value in zip(lsts, ras):
    ha.append(lst - ra_value)

# COMPUTE Declination
for h, lat, elev in zip(ha, lats, elevs):
    decs.append(dec(h, lat, elev))

# COMPUTE Absolute Compass Direction (Azimuth)
for key, d, lat, elev in zip(device_db.keys(), decs, lats, elevs):
    azi = az(d, lat, elev)
    azimuth.append(azi)
    device_db[key] = str(azi)


print(f"Absolute Local Sidereal Time (LST) aross all longitudes (°E) in hours")
print(lsts)
print("\n")
for x, y, z in zip(xes, yes, zes):
    print(f"ECI coordinate of sites: {x, y, z}")
print("\n")
print("Right ascession:")
print(ras)
print("\n")
print("Hour Angle:")
print(ha)
print("\n")
print("Declination:")
print(decs)
print("\n")
print("Absolute compass direction:")
print(device_db)
print(f"Average compass direction = {np.mean(azimuth):.2f}°")