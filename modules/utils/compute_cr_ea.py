from skyfield.api import load, wgs84, EarthSatellite
from datetime import datetime, timezone

# === Step 1: Load TLE Data for Sentinel-1 ===
# Example TLE for Sentinel-1A (replace with current TLE for accuracy)
tle_lines = [
    "SENTINEL-1A",
    "1 39634U 14016A   24100.50000000  .00000000  00000-0  00000-0 0  9990",
    "2 39634  98.1800 123.0000 0001000  90.0000 270.0000 14.59100000000018"
]

ts = load.timescale()
satellite = EarthSatellite(tle_lines[1], tle_lines[2], tle_lines[0], ts)
# === Step 2: Define Observation Time ===
observation_time = ts.utc(2025, 3, 1, 22, 36, 10)

def converter(lat, lon, elev):

    # === Step 3: Define Observer Location ===
    location = wgs84.latlon(lat, lon, elevation_m=elev)

    # === Step 4: Compute Elevation Angle ===
    difference = satellite - location
    topocentric = difference.at(observation_time)
    alt, az, distance = topocentric.altaz()

    print("Azimuth angle: {:.2f} degrees".format(az.degrees))
    print("Elevation angle: {:.2f} degrees".format(alt.degrees))
    
if __name__ == "__main__":
    converter(None, None, None)
    converter(None, None, None)
