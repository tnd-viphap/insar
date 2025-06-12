import customtkinter as ctk
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
import math
import numpy as np
import os
import sys

project_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(project_path)

class CRTargetGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("CR Target Manager")
        self.geometry("1200x800")

        # Initialize entries dictionary first
        self.entries = {}
        self.target_rows = []

        # Create main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create scrollable frame for entries
        self.scrollable_frame = ctk.CTkScrollableFrame(self.main_frame)
        self.scrollable_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Create entry fields
        self.create_entry_fields()

        # Create buttons frame
        self.button_frame = ctk.CTkFrame(self.main_frame)
        self.button_frame.pack(fill="x", padx=5, pady=5)

        # Add buttons
        self.add_button = ctk.CTkButton(self.button_frame, text="Add New Target", command=self.add_new_target)
        self.add_button.pack(side="left", padx=5)

        self.save_button = ctk.CTkButton(self.button_frame, text="Save to CSV", command=self.save_to_csv)
        self.save_button.pack(side="left", padx=5)

    def llh_to_ecef(self, lat, lon, alt):
        """
        Convert latitude, longitude, altitude to ECEF coordinates
        lat, lon in degrees, alt in meters
        Returns x, y, z in meters
        """
        # WGS84 ellipsoid parameters
        a = 6378137.0  # semi-major axis in meters
        f = 1/298.257223563  # flattening
        b = a * (1 - f)  # semi-minor axis
        
        # Convert to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        
        # Calculate N (radius of curvature in the prime vertical)
        e2 = 1 - (b/a)**2
        N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
        
        # Calculate ECEF coordinates
        x = (N + alt) * math.cos(lat_rad) * math.cos(lon_rad)
        y = (N + alt) * math.cos(lat_rad) * math.sin(lon_rad)
        z = (N * (1 - e2) + alt) * math.sin(lat_rad)
        
        return x, y, z

    def compute_corner_angles(self, lat, lon, alt):
        """
        Compute corner azimuth and elevation angles based on target position
        """
        # Convert to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        
        # Earth radius in meters
        R = 6378137.0
        
        # Calculate target position vector
        x = (R + alt) * math.cos(lat_rad) * math.cos(lon_rad)
        y = (R + alt) * math.cos(lat_rad) * math.sin(lon_rad)
        z = (R + alt) * math.sin(lat_rad)
        
        # Calculate elevation angle (angle from horizontal plane)
        elevation = math.degrees(math.asin(z / math.sqrt(x**2 + y**2 + z**2)))
        
        # Calculate azimuth angle (angle from North, clockwise)
        azimuth = math.degrees(math.atan2(y, x))
        if azimuth < 0:
            azimuth += 360
            
        print(f"Computed angles - Azimuth: {azimuth:.2f}°, Elevation: {elevation:.2f}°")
        return azimuth, elevation

    def update_xyz_coordinates(self, *args):
        try:
            lat = float(self.entries["latitude_deg"].get())
            lon = float(self.entries["longitude_deg"].get())
            alt = float(self.entries["altitude_m"].get())
            
            print(f"Updating coordinates for lat: {lat}, lon: {lon}, alt: {alt}")
            
            # Update ECEF coordinates
            x, y, z = self.llh_to_ecef(lat, lon, alt)
            
            self.entries["x_coord_m"].delete(0, tk.END)
            self.entries["x_coord_m"].insert(0, f"{x:.6f}")
            
            self.entries["y_coord_m"].delete(0, tk.END)
            self.entries["y_coord_m"].insert(0, f"{y:.6f}")
            
            self.entries["z_coord_m"].delete(0, tk.END)
            self.entries["z_coord_m"].insert(0, f"{z:.6f}")
            
            # Update corner angles
            azimuth, elevation = self.compute_corner_angles(lat, lon, alt)
            
            # Make sure the corner angle entries exist
            if "corner_azimuth_deg" not in self.entries:
                self.entries["corner_azimuth_deg"] = ctk.CTkEntry(self.scrollable_frame)
                self.entries["corner_azimuth_deg"].grid(row=len(self.entries), column=1, padx=5, pady=5, sticky="ew")
            
            if "corner_elevation_deg" not in self.entries:
                self.entries["corner_elevation_deg"] = ctk.CTkEntry(self.scrollable_frame)
                self.entries["corner_elevation_deg"].grid(row=len(self.entries)+1, column=1, padx=5, pady=5, sticky="ew")
            
            # Update the corner angle values
            self.entries["corner_azimuth_deg"].delete(0, tk.END)
            self.entries["corner_azimuth_deg"].insert(0, f"{azimuth:.2f}")
            
            self.entries["corner_elevation_deg"].delete(0, tk.END)
            self.entries["corner_elevation_deg"].insert(0, f"{elevation:.2f}")
            
            print("Coordinates and angles updated successfully")
            
        except ValueError as e:
            print(f"Error updating coordinates: {str(e)}")
            # If any of the inputs are invalid, do nothing
            pass

    def create_entry_fields(self):
        # Define required fields
        required_fields = [
            "target_name", "target_type", "latitude_deg", "longitude_deg", 
            "altitude_m", "x_coord_m", "y_coord_m", "z_coord_m", "target_size_m"
        ]

        # Create labels and entries
        row = 0
        for field in required_fields:
            label = ctk.CTkLabel(self.scrollable_frame, text=field.replace("_", " ").title())
            label.grid(row=row, column=0, padx=5, pady=5, sticky="w")
            
            if field == "target_type":
                type_var = ctk.StringVar(value="CR")
                entry = ctk.CTkComboBox(self.scrollable_frame, 
                                      values=["CR", "TX"],
                                      variable=type_var)
            else:
                entry = ctk.CTkEntry(self.scrollable_frame)
                # Add trace for lat/lon/alt fields to update xyz and corner angles
                if field in ["latitude_deg", "longitude_deg", "altitude_m"]:
                    entry.bind('<KeyRelease>', self.update_xyz_coordinates)
            
            entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
            
            self.entries[field] = entry
            row += 1

        # Add target shape selection
        label = ctk.CTkLabel(self.scrollable_frame, text="Target Shape")
        label.grid(row=row, column=0, padx=5, pady=5, sticky="w")
        
        shape_var = ctk.StringVar(value="trihedral")
        shape_combo = ctk.CTkComboBox(self.scrollable_frame, 
                                    values=["trihedral", "dihedral"],
                                    variable=shape_var)
        shape_combo.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        self.entries["target_shape"] = shape_var
        row += 1

        # Add corner angles fields
        corner_fields = ["corner_azimuth_deg", "corner_elevation_deg"]
        for field in corner_fields:
            label = ctk.CTkLabel(self.scrollable_frame, text=field.replace("_", " ").title())
            label.grid(row=row, column=0, padx=5, pady=5, sticky="w")
            
            entry = ctk.CTkEntry(self.scrollable_frame)
            entry.insert(0, "0.0")
            entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
            
            self.entries[field] = entry
            row += 1

        # Add RCS fields
        rcs_fields = ["rcs_hh_dB", "rcs_hv_dB", "rcs_vv_dB", "rcs_vh_dB"]
        for field in rcs_fields:
            label = ctk.CTkLabel(self.scrollable_frame, text=field.replace("_", " ").title())
            label.grid(row=row, column=0, padx=5, pady=5, sticky="w")
            
            entry = ctk.CTkEntry(self.scrollable_frame)
            entry.insert(0, "35.0")
            entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
            
            self.entries[field] = entry
            row += 1

        # Add delay field
        label = ctk.CTkLabel(self.scrollable_frame, text="Delay (s)")
        label.grid(row=row, column=0, padx=5, pady=5, sticky="w")
        
        entry = ctk.CTkEntry(self.scrollable_frame)
        entry.insert(0, "0.0")
        entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        
        self.entries["delay_s"] = entry
        row += 1

        # Add date fields
        date_fields = ["measurement_date", "validity_start_date", "validity_stop_date"]
        default_dates = {
            "measurement_date": "2024-01-01 00:00:00.000",
            "validity_start_date": "2024-01-01 00:00:00.000",
            "validity_stop_date": "2099-01-01 00:00:00.000"
        }

        for field in date_fields:
            label = ctk.CTkLabel(self.scrollable_frame, text=field.replace("_", " ").title())
            label.grid(row=row, column=0, padx=5, pady=5, sticky="w")
            
            entry = ctk.CTkEntry(self.scrollable_frame)
            entry.insert(0, default_dates[field])
            entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
            
            self.entries[field] = entry
            row += 1

    def add_new_target(self):
        # Validate required fields
        for field in self.entries:
            if field in ["target_name", "target_type", "latitude_deg", "longitude_deg", 
                        "altitude_m", "x_coord_m", "y_coord_m", "z_coord_m", "target_size_m"]:
                if not self.entries[field].get():
                    messagebox.showerror("Error", f"{field.replace('_', ' ').title()} is required!")
                    return

        # Create new target data
        target_data = {
            "target_name": self.entries["target_name"].get(),
            "target_type": self.entries["target_type"].get(),
            "plate": "",
            "description": "",
            "latitude_deg": float(self.entries["latitude_deg"].get()),
            "longitude_deg": float(self.entries["longitude_deg"].get()),
            "altitude_m": float(self.entries["altitude_m"].get()),
            "x_coord_m": float(self.entries["x_coord_m"].get()),
            "y_coord_m": float(self.entries["y_coord_m"].get()),
            "z_coord_m": float(self.entries["z_coord_m"].get()),
            "drift_velocity_x_my": 0.0,
            "drift_velocity_y_my": 0.0,
            "drift_velocity_z_my": 0.0,
            "corner_azimuth_deg": float(self.entries["corner_azimuth_deg"].get()),
            "corner_elevation_deg": float(self.entries["corner_elevation_deg"].get()),
            "target_shape": self.entries["target_shape"].get(),
            "target_size_m": float(self.entries["target_size_m"].get()),
            "rcs_hh_dB": float(self.entries["rcs_hh_dB"].get()),
            "rcs_hv_dB": float(self.entries["rcs_hv_dB"].get()),
            "rcs_vv_dB": float(self.entries["rcs_vv_dB"].get()),
            "rcs_vh_dB": float(self.entries["rcs_vh_dB"].get()),
            "delay_s": float(self.entries["delay_s"].get()),
            "measurement_date": self.entries["measurement_date"].get(),
            "validity_start_date": self.entries["validity_start_date"].get(),
            "validity_stop_date": self.entries["validity_stop_date"].get()
        }

        self.target_rows.append(target_data)
        messagebox.showinfo("Success", "Target added successfully!")

        # Clear entries and reset to default values
        for field, entry in self.entries.items():
            if isinstance(entry, ctk.CTkEntry):
                entry.delete(0, tk.END)
                if field in ["rcs_hh_dB", "rcs_hv_dB", "rcs_vv_dB", "rcs_vh_dB"]:
                    entry.insert(0, "35.0")
                elif field == "delay_s":
                    entry.insert(0, "0.0")
                elif field == "measurement_date":
                    entry.insert(0, "2024-01-01 00:00:00.000")
                elif field == "validity_start_date":
                    entry.insert(0, "2024-01-01 00:00:00.000")
                elif field == "validity_stop_date":
                    entry.insert(0, "2099-01-01 00:00:00.000")
            elif isinstance(entry, ctk.StringVar):
                if field == "target_type":
                    entry.set("CR")
                elif field == "target_shape":
                    entry.set("trihedral")

    def save_to_csv(self):
        if not self.target_rows:
            messagebox.showerror("Error", "No targets to save!")
            return

        df = pd.DataFrame(self.target_rows)
        df.to_csv(os.path.join(project_path, "modules/pta/pta_target.csv"), index=False)
        messagebox.showinfo("Success", "Targets saved to CSV successfully!")

if __name__ == "__main__":
    app = CRTargetGUI()
    app.mainloop()