import concurrent.futures
import glob
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import scipy.io as sio
from matplotlib import pyplot as plt
from numpy.fft import fft, fft2, fftshift, ifft, ifft2, ifftshift
from numpy.linalg import lstsq
from scipy.interpolate import griddata
from scipy.signal import convolve, convolve2d, windows
from scipy.signal.windows import gaussian
from scipy.sparse import csr_matrix, lil_matrix, spdiags
from scipy.sparse.linalg import lsqr, svds
from scipy.spatial import Delaunay, cKDTree
from sklearn.utils import resample
from tqdm import tqdm

project_path = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(project_path)

from config.parser import ConfigParser
from dev.stamps.ps_plot import PSPlot
from modules.tomo.ps_parms import Parms


class StaMPSStep:
    def __init__(self, parms: Parms = None, est_gamma_flag=None, patch_list_file=None, stamps_PART_limitation=None):
        self.parms = parms
        self.config_parser = ConfigParser(os.path.join(project_path, 'config', 'config.json'))
        self.config = self.config_parser.get_project_config(self.parms.project_name)
        self._load_rslcpar()

        if platform.system() == 'Windows':
            self.triangle_path = os.path.join(project_path, 'modules/triangle/triangle.exe')
        else:
            self.triangle_path = os.path.join(project_path, 'modules/triangle/triangle')

        self.est_gamma_flag = est_gamma_flag
        if self.est_gamma_flag is None:
            self.est_gamma_flag = 0

        self.patch_list_file = patch_list_file
        if self.patch_list_file is None:
            self.patch_list_file = self.config["processing_parameters"]["current_result"] + '/patch.list'

        self.start_step = 1
        self.end_step = 8
        self.stamps_PART_limitation = stamps_PART_limitation
        if self.stamps_PART_limitation is None:
            self.stamps_PART_limitation = 0

        self.stamps_PART1_flag = True
        self.stamps_PART2_flag = False
        if self.stamps_PART_limitation == 1:
            self.stamps_PART2_flag = False
        if self.stamps_PART_limitation == 2:
            self.stamps_PART1_flag = False
        
        self.control_flow = {
            1: lambda: self._stamps_1(),
            2: lambda x=self.est_gamma_flag: self._stamps_2(x),
            3: lambda x: self._stamps_3(x),
            4: lambda: self._stamps_4(),
            5: lambda: self._stamps_5(),
            6: lambda x: self._stamps_6(x),
            7: lambda: self._stamps_7(),
            8: lambda: self._stamps_8(),
        }

        self.quick_est_gamma_flag = self.parms.get('quick_est_gamma_flag')
        self.reest_gamma_flag = self.parms.get('select_reest_gamma_flag')
        self.unwrap_method = self.parms.get('unwrap_method')
        self.unwrap_prefilter_flag = self.parms.get('unwrap_prefilter_flag')
        self.small_baseline_flag = self.parms.get('small_baseline_flag')
        self.insar_processor = self.parms.get('insar_processor')
        self.scn_kriging_flag = self.parms.get('scn_kriging_flag')

        self.psver = 1
        
        # Constants
        self.rho = 830000  # mean range
        self.grid_size = float(self.parms.get('filter_grid_size'))
        self.filter_weighting = self.parms.get('filter_weighting')
        self.n_win = int(self.parms.get('clap_win'))
        self.low_pass_wavelength = float(self.parms.get('clap_low_pass_wavelength'))
        self.clap_alpha = float(self.parms.get('clap_alpha'))
        self.clap_beta = float(self.parms.get('clap_beta'))
        self.max_topo_err = float(self.parms.get('max_topo_err'))
        self.lambda_ = float(self.parms.get('lambda'))
        self.gamma_change_convergence = float(self.parms.get('gamma_change_convergence'))
        self.gamma_max_iterations = int(self.parms.get('gamma_max_iterations'))
        self.small_baseline_flag = self.parms.get('small_baseline_flag')

    ########## Load initial gamma ##########
    
    def _handle_patches(self, patches_flag=None):
        """Handle patch processing logic similar to MATLAB's stamps.m"""
        patch_dirs = []
        
        # Check if patch list file exists
        if os.path.exists(self.patch_list_file):
            with open(self.patch_list_file, 'r') as f:
                patch_dirs = [line.strip() for line in f if line.strip()]
        else:
            # Find all PATCH_* directories except patch_noover.in
            patch_dirs = [d for d in glob.glob('PATCH_*') 
                        if os.path.isdir(d) and d != 'patch_noover.in']
        
        if not patch_dirs:
            patches_flag = False
        else:
            self.parms.initialize()
            patches_flag = True
            
        return patch_dirs, patches_flag
    
    def _update_psver(self, psver, patch_dir=None):
        self.psver = psver
        sio.savemat(os.path.join(self.config["processing_parameters"]["current_result"], f'psver.mat'), {'psver': self.psver})
        if patch_dir is not None:
            sio.savemat(os.path.join(self.config["processing_parameters"]["current_result"], patch_dir, f'psver.mat'), {'psver': self.psver})

    def _load_rslcpar(self):
        master_date = self.config["processing_parameters"]["current_result"].split('_')[1]
        with open(os.path.join(self.config["processing_parameters"]["current_result"], f'rslc/{master_date}.rslc.par').replace(os.sep, '/'), 'r') as file:
            for line in file.readlines():
                line = line.strip().split('\t')
                if line[0].startswith('range_pixel_spacing'):
                    self.parms.set('range_pixel_spacing', float(line[1]))
                elif line[0].startswith('near_range_slc'):
                    self.parms.set('near_range_slc', float(line[1]))
                elif line[0].startswith('sar_to_earth_center'):
                    self.parms.set('sar_to_earth_center', float(line[1]))
                elif line[0].startswith('earth_radius_below_sensor'):
                    self.parms.set('earth_radius_below_sensor', float(line[1]))
                elif line[0].startswith('center_range_slc'):
                    self.parms.set('center_range_slc', float(line[1]))
                elif line[0].startswith('azimuth_lines'):
                    self.parms.set('azimuth_lines', float(line[1]))
                elif line[0].startswith('prf'):
                    self.parms.set('prf', float(line[1]))
                elif line[0].startswith('heading'):
                    self.parms.set('heading', float(line[1]))
                elif line[0].startswith('radar_frequency'):
                    self.parms.set('lambda', 299792458 / float(line[1]))
                elif line[0].startswith('sensor'):
                    if 'ASAR' in line[1]:
                        self.parms.set('platform', 'ENVISAT')
                    else:
                        self.parms.set('platform', line[1])
        self.parms.save()
    
    def _fetch_baseline(self, basename):
        with open(basename, 'r') as f:
            values = []
            for line in f.readlines():
                if line.startswith('initial_baseline(TCN):'):
                    B_TCN = line.strip().split(':')[-1].strip().split('\t')[:3]
                    values.append(B_TCN)
                elif line.startswith('initial_baseline_rate:'):
                    BR_TCN = line.strip().split(':')[-1].strip().split('\t')[:3]
                    values.append(BR_TCN)
                elif line.startswith('precision_baseline(TCN):'):
                    P_TCN = line.strip().split(':')[-1].strip().split('\t')[:3]
                    values.append(P_TCN)
                elif line.startswith('precision_baseline_rate:'):
                    PR_TCN = line.strip().split(':')[-1].strip().split('\t')[:3]
                    values.append(PR_TCN)
                elif line.startswith('unwrap_phase_constant:'):
                    unwrap_phase_constant = line.strip().split(':')[-1].strip().split('\t')[:1]
                    values.append(unwrap_phase_constant)
        return values

    def _initialize_ps_info(self):
        """Initialize or load no_ps_info.mat file"""
        if not os.path.exists(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, 'no_ps_info.mat')):
            # Create zeros array for first 5 steps only
            self.stamps_step_no_ps = np.zeros((5, 1))
            # Save to mat file
            sio.savemat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, 'no_ps_info.mat'), {'stamps_step_no_ps': self.stamps_step_no_ps})
        else:
            # Load existing file
            self.stamps_step_no_ps = sio.loadmat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, 'no_ps_info.mat'))['stamps_step_no_ps']
    
    def _save_ps_info(self):
        sio.savemat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, 'no_ps_info.mat'), {'stamps_step_no_ps': self.stamps_step_no_ps})

    def _llh2local(self, llh, origin):
        """
        Converts from longitude and latitude to local coordinates given an origin.
        Ignores height. Both `llh` and `origin` are in degrees.
        Output `xy` is in kilometers.

        Parameters:
        - llh: np.ndarray with shape (3, N) — rows: lon, lat, height
        - origin: np.ndarray or list of shape (3,) — lon, lat, height

        Returns:
        - xy: np.ndarray with shape (2, N) — local east-north coordinates in km
        """

        # WGS84 ellipsoid constants
        a = 6378137.0  # semi-major axis (meters)
        e = 0.08209443794970  # eccentricity

        # Convert to radians
        llh = llh.astype(np.float64) * np.pi / 180
        origin = origin.astype(np.float64) * np.pi / 180

        z = llh[1, :] != 0

        dlambda = llh[0, z] - origin[0]

        phi = llh[1, z]

        # Meridian arc length (M)
        M = a * ((1 - e**2/4 - 3*e**4/64 - 5*e**6/256) * phi
                - (3*e**2/8 + 3*e**4/32 + 45*e**6/1024) * np.sin(2*phi)
                + (15*e**4/256 + 45*e**6/1024) * np.sin(4*phi)
                - (35*e**6/3072) * np.sin(6*phi))

        # Meridian arc at origin
        phi0 = origin[1]
        M0 = a * ((1 - e**2/4 - 3*e**4/64 - 5*e**6/256) * phi0
                - (3*e**2/8 + 3*e**4/32 + 45*e**6/1024) * np.sin(2*phi0)
                + (15*e**4/256 + 45*e**6/1024) * np.sin(4*phi0)
                - (35*e**6/3072) * np.sin(6*phi0))

        N = a / np.sqrt(1 - e**2 * np.sin(phi)**2)
        E = dlambda * np.sin(phi)

        xy = np.zeros(llh.shape)

        xy[0, z] = N * 1/np.tan(phi) * np.sin(E)
        xy[1, z] = M - M0 + N * 1/np.tan(phi) * (1 - np.cos(E))

        # Special case: latitude == 0
        dlambda = llh[0, ~z] - origin[0]
        xy[0, ~z] = a * dlambda
        xy[1, ~z] = -M0

        # Convert to kilometers
        xy = xy / 1000.0

        return xy

    def _ps_load_initial_gamma(self):
        """Load initial gamma data"""
        self.parms.load()
        phname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, 'pscands.1.ph')
        ijname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, 'pscands.1.ij')
        llname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, 'pscands.1.ll')
        # xyname = os.path.join(self.config["processing_parameters"]["current_result"], patch_dir, 'pscands.1.xy')
        hgtname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, 'pscands.1.hgt')
        daname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, 'pscands.1.da')
        # rscname = os.path.join(self.config["processing_parameters"]["current_result"], 'rsc.txt')
        pscname = os.path.join(self.config["processing_parameters"]["current_result"], 'pscphase.in')

        self._update_psver(1, self.patch_dir)
        
        # Read RSC file
        # try:
        #     with open(rscname, 'r') as f:
        #         rslcpar = f.readline().strip()
        # except FileNotFoundError:
        #     raise FileNotFoundError(f"{rscname} does not exist")

        # Read PSC file
        try:
            with open(pscname, 'r') as f:
                ifgs = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            raise FileNotFoundError(f"{pscname} does not exist")
        
        # Process IFGs
        ifgs = ifgs[1:]  # Skip first line
        nb = len(ifgs[0])
        master_day = int(ifgs[0][nb-22:nb-14])
        n_ifg = len(ifgs)
        n_image = n_ifg
        
        # Process days
        day = np.zeros(n_ifg)
        for i in range(n_ifg):
            day[i] = int(ifgs[i][nb-13:nb-5])

        # Convert dates
        year = np.floor(day/10000).astype(int)
        month = np.floor((day - year*10000)/100).astype(int)
        monthday = (day - year*10000 - month*100).astype(int)
        day = np.array([pd.Timestamp(f"{y}-{m:02d}-{d:02d}").toordinal() for y, m, d in zip(year, month, monthday)])
        
        # Process master day
        # master_day_yyyymmdd = master_day
        year = int(np.floor(master_day/10000))
        month = int(np.floor((master_day - year*10000)/100))
        monthday = int(master_day - year*10000 - month*100)
        master_day = pd.Timestamp(f"{year}-{month:02d}-{monthday:02d}").toordinal()

        # Find master index
        master_ix = np.sum(day < master_day)
        if day[master_ix-1] != master_day:
            master_master_flag = False  # no null master-master ifg provided
            day = np.concatenate([day[:master_ix], [master_day], day[master_ix:]])
        else:
            master_master_flag = True  # yes, null master-master ifg provided

        ij = np.loadtxt(ijname)
        n_ps = ij.shape[0]

        rps = self.parms.get('range_pixel_spacing')
        rgn = self.parms.get('near_range_slc')
        se = self.parms.get('sar_to_earth_center')
        re = self.parms.get('earth_radius_below_sensor')
        rgc = self.parms.get('center_range_slc')
        naz = self.parms.get('azimuth_lines')
        prf = self.parms.get('prf')

        mean_az = naz/2-0.5
        rg = rgn+ij[:, 2]*rps
        look = np.arccos((se**2+rg**2-re**2)/(2*se*rg))

        bperp_mat = np.zeros((n_ps, n_image), dtype=np.float32)
        for i in range(n_ifg):
            basename = ifgs[i][:nb-5]+'.base'
            B_TCN = [float(x) for x in self._fetch_baseline(basename)[0]]
            BR_TCN = [float(x) for x in self._fetch_baseline(basename)[1]]
            bc = B_TCN[1]+BR_TCN[1]*(ij[:, 2]-mean_az)/prf
            bn = B_TCN[2]+BR_TCN[2]*(ij[:, 2]-mean_az)/prf
            bperp_mat[:, i] = bc*np.cos(look)-bn*np.sin(look)

        bperp = np.mean(bperp_mat, axis=0)
        if master_master_flag:
            bperp_mat = bperp_mat[:, :master_ix, master_ix:]
        else:
            bperp = np.concatenate([bperp[:master_ix], [0], bperp[master_ix:]])

        inci = np.arccos((se**2-re**2-rg**2)/(2*re*rg))
        mean_incidence = np.mean(inci)
        mean_range = rgc

        # Read phase data
        dtype = np.dtype('>f4')  # or '<f4' depending on `endian`
        ph = np.zeros((n_ps, n_ifg), dtype=np.complex64)
        with open(phname, 'rb') as fid:
            for i in range(n_ifg):
                ph_bit = np.fromfile(fid, dtype=dtype, count=n_ps * 2)
                ph[:, i] = ph_bit[::2] + 1j * ph_bit[1::2]
        
        # Handle master-master IFG
        if master_master_flag:
            ph[:, master_ix] = 1
        else:
            ph = np.concatenate([ph[:, :master_ix], np.ones((n_ps, 1), dtype=ph.dtype), ph[:, master_ix:]], axis=1)
            n_ifg += 1
            n_image += 1

        # Read lonlat data
        with open(llname, 'r') as f:
            lonlat = np.loadtxt(f, dtype=np.float32).reshape(-1, 2)

        # Remove rows where lonlat has any NaNs
        ix_nan = np.isnan(lonlat).sum(axis=1) >= 1
        lonlat = lonlat[~ix_nan, :]
        ij = ij[~ix_nan, :]
        ph = np.delete(ph, ix_nan, axis=0)
        bperp_mat = np.delete(bperp_mat, ix_nan, axis=0)
        n_ps = lonlat.shape[0]

        # Calculate center coordinates
        ll0 = (np.amax(lonlat, axis=0) + np.amin(lonlat, axis=0)) / 2
        xy = self._llh2local(lonlat.T, ll0).T * 1000

        # Rotate coordinates
        heading = float(self.parms.get('heading'))
        theta = (180 - heading) * np.pi / 180
        if theta > np.pi:
            theta -= 2 * np.pi

        rotm = np.array([[np.cos(theta), np.sin(theta)], 
                        [-np.sin(theta), np.cos(theta)]])
        xy = xy.T
        xynew = rotm @ xy
        xynew = xynew.T

        # Check if rotation improves alignment
        if (np.max(xynew[0, :]) - np.min(xynew[0, :]) < np.max(xy[0, :]) - np.min(xy[0, :]) and
            np.max(xynew[1, :]) - np.min(xynew[1, :]) < np.max(xy[1, :]) - np.min(xy[1, :])):
            xy = xynew
            print(f"   -> Rotating by {theta * 180 / np.pi} degrees")

        # Convert to single precision and transpose
        xy = np.array(xy, dtype=np.float32)
        # Sort by y then x (MATLAB: sortrows(xy,[2,1]))
        sort_ix = np.lexsort((xy[:, 0], xy[:, 1]))
        xy = xy[sort_ix, :]
        # Add PS numbers (1-based indexing like MATLAB)
        xy = np.column_stack([np.arange(n_ps), xy])
        # Round to mm (MATLAB: round(xy(:,2:3)*1000)/1000)
        xy[:, 1:3] = np.round(xy[:, 1:] * 1000) / 1000

        # Update arrays with sorted indices
        ph = ph[sort_ix, :]
        ij = ij[sort_ix, :]
        ij[:, 0] = np.arange(n_ps)
        lonlat = lonlat[sort_ix, :]
        bperp_mat = bperp_mat[sort_ix, :]

        # Remove rows where ph has any NaNs
        ix_nan = np.isnan(ph).sum(axis=1) >= 1
        lonlat = lonlat[~ix_nan, :]
        ij = ij[~ix_nan, :]
        xy = xy[~ix_nan, :]
        ph = np.delete(ph, ix_nan, axis=0)
        bperp_mat = np.delete(bperp_mat, ix_nan, axis=0)
        n_ps = lonlat.shape[0]

        # Update the first column of ij and xy
        ij[:, 0] = np.arange(n_ps)
        xy[:, 0] = np.arange(n_ps)

        # Remove rows where lonlat has a 0 in the first column (i.e., lon = 0)
        ix_0 = np.where(lonlat[:, 0] == 0)[0]
        lonlat = np.delete(lonlat, ix_0, axis=0)
        ph = np.delete(ph, ix_0, axis=0)
        bperp_mat = np.delete(bperp_mat, ix_0, axis=0)
        ij = np.delete(ij, ix_0, axis=0)
        xy = np.delete(xy, ix_0, axis=0)

        n_ps = lonlat.shape[0]

        # Update the first column again
        ij[:, 0] = np.arange(n_ps)
        xy[:, 0] = np.arange(n_ps)

        # Plot phase data for all interferograms with lon/lat coordinates
        if not os.path.exists(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'1_phase_all_ifgs.png')):
            n_rows = int(np.ceil(np.sqrt(n_ifg)))
            n_cols = int(np.ceil(n_ifg / n_rows))
            plt.figure(figsize=(4*n_cols, 4*n_rows))
            
            for i in range(n_ifg):
                plt.subplot(n_rows, n_cols, i+1)
                sc = plt.scatter(lonlat[:, 0], lonlat[:, 1], c=np.angle(ph[:, i]), 
                            cmap='jet', s=1)
                plt.colorbar(sc, label='Phase (rad)')
                plt.title(f'Interferogram {i+1}')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.config["processing_parameters"]["current_result"], 
                                    self.patch_dir, f'1_phase_all_ifgs.png'))
            plt.clf()
            plt.close()

        # Save results
        sio.savemat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'ps{self.psver}.mat'), {
            'ij': ij,
            'lonlat': lonlat,
            'xy': xy,
            'bperp': bperp,
            'day': day,
            'master_day': master_day,
            'master_ix': master_ix,
            'n_ifg': n_ifg,
            'n_image': n_image,
            'n_ps': n_ps,
            'sort_ix': sort_ix,
            'll0': ll0,
            'mean_incidence': mean_incidence,
            'mean_range': mean_range
        })

        # Save psver
        sio.savemat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, 'psver.mat'), {'psver': self.psver})

        # Save phase data
        sio.savemat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'ph{self.psver}.mat'), {'ph': ph})

        # Save baseline data
        sio.savemat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'bp{self.psver}.mat'), {'bperp_mat': bperp_mat})

        # Save look angle data
        la = inci[sort_ix]
        la = np.delete(la, ix_nan, axis=0)
        la = np.delete(la, ix_0, axis=0)
        sio.savemat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'la{self.psver}.mat'), {'la': la})

        # Handle DA file if it exists
        if os.path.exists(daname):
            D_A = np.loadtxt(daname)
            D_A = D_A[sort_ix]
            D_A = np.delete(D_A, ix_nan, axis=0)
            D_A = np.delete(D_A, ix_0, axis=0)
            sio.savemat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'da{self.psver}.mat'), {'D_A': D_A})

        # Handle height file if it exists
        if os.path.exists(hgtname):
            with open(hgtname, 'rb') as f:
                hgt = np.fromfile(f, dtype='>f4').reshape(-1, 1)
                hgt = hgt[sort_ix]
                hgt = np.delete(hgt, ix_nan, axis=0)
                hgt = np.delete(hgt, ix_0, axis=0)
                sio.savemat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'hgt{self.psver}.mat'), {'hgt': hgt})

    ########## Estimate PS coherence, phase, and range error ##########
    
    def _ps_topofit(self, cpxphase, bperp, n_trial_wraps, plotflag=False, asym=None):
        """Find best-fitting range error
        Args:
            cpxphase: Complex phase
            bperp: Perpendicular baseline
            n_trial_wraps: Number of trial wraps
            plotflag: Whether to plot results
            asym: Asymmetry parameter (-1 to +1, default 0)
        Returns:
            K0: Best-fitting range error
            C0: Constant phase offset
            coh0: Coherence
            phase_residual: Phase residual
        """
        if asym is None:
            asym = 0

        if cpxphase.ndim > 1:
            cpxphase = cpxphase.T

        # # Ensure cpxphase is 1D
        cpxphase = np.squeeze(cpxphase)
        bperp = np.squeeze(bperp)

        # Get non-zero indices
        ix = cpxphase != 0
        cpxphase = cpxphase[ix]
        bperp = bperp[ix]
        bperp_range = np.amax(bperp) - np.amin(bperp)

        # Calculate wrapped phase
        wphase = np.angle(cpxphase)

        # Calculate trial phases
        trial_mult = np.arange(-np.ceil(8 * n_trial_wraps), np.ceil(8 * n_trial_wraps) + 1) + asym * 8 * n_trial_wraps
        n_trials = len(trial_mult)
        trial_phase = bperp / bperp_range * np.pi / 4
        trial_phase_mat = np.exp(-1j * trial_phase[:, np.newaxis] * trial_mult[np.newaxis, :])
        cpxphase_mat = np.tile(cpxphase, (n_trials, 1)).T
        phaser = trial_phase_mat * cpxphase_mat
        phaser_sum = np.sum(phaser, axis=0)
        C_trial = np.angle(phaser_sum)
        coh_trial = np.abs(phaser_sum) / np.sum(np.abs(cpxphase))

        # Find best fit
        coh_high_max_ix = np.argmax(coh_trial)
        K0 = np.pi / 4 / bperp_range * trial_mult[coh_high_max_ix]
        C0 = C_trial[coh_high_max_ix]
        coh0 = coh_trial[coh_high_max_ix]

        # Linearize and solve
        resphase = cpxphase * np.exp(-1j * (K0 * bperp))
        offset_phase = np.sum(resphase)
        resphase = np.angle(resphase * np.conj(offset_phase))
        weighting = np.abs(cpxphase)
        A = (weighting * bperp).reshape(-1, 1)
        b = (weighting * resphase).reshape(-1, 1)
        mopt = np.linalg.lstsq(A, b, rcond=None)[0]
        K0 = K0 + mopt
        phase_residual = cpxphase * np.exp(-1j * (K0 * bperp))
        mean_phase_residual = np.sum(phase_residual)
        C0 = np.angle(mean_phase_residual)
        coh0 = np.abs(mean_phase_residual) / np.sum(np.abs(phase_residual))

        if plotflag:
            plt.figure()
            plt.subplot(2, 1, 2)
            bvec = np.linspace(np.amin(bperp), np.amax(bperp), 200)
            wphase_hat = np.angle(np.exp(1j * (K0 * bvec + C0)))
            plt.plot(bvec, wphase_hat, 'r', linewidth=2)
            plt.plot(bperp, wphase, 'bo', linewidth=2)
            plt.ylim([-np.pi, np.pi])
            plt.ylabel('Wrapped Phase', fontdict={'fontsize': 12, 'fontweight': 'bold'})
            plt.xlabel('B_{\perp} (m)', fontdict={'fontsize': 12, 'fontweight': 'bold'})
            plt.subplot(2, 1, 1)
            plt.plot(np.pi/4/bperp_range*trial_mult, coh_trial, 'g')
            plt.ylabel('\\gamma_x', fontdict={'fontsize': 12, 'fontweight': 'bold'})
            plt.xlabel('Spatially uncorrelated look angle error (radians/m)', fontdict={'fontsize': 12, 'fontweight': 'bold'})
            plt.tight_layout()
            plt.ylim([0, 1])
            plt.show()

        return K0, C0, coh0, phase_residual

    def _clap_filt(self, ph_grid, alpha=0.5, beta=0.1, n_win=32, n_pad=0, low_pass=None):
        """Apply CLAP filter to phase grid"""
        n_win = int(n_win)
        n_pad = int(n_pad)

        if low_pass is None:
            low_pass = np.zeros((n_win+n_pad, n_win+n_pad))

        ph_out = np.zeros(ph_grid.shape, dtype=np.complex64)
        n_i, n_j = ph_grid.shape

        n_inc = int(np.floor(n_win/4))
        n_win_i = int(np.ceil(n_i/n_inc))-3
        n_win_j = int(np.ceil(n_j/n_inc))-3

        # Create window function
        x = np.arange(0, n_win//2)
        X, Y = np.meshgrid(x, x)
        X = X+Y
        wind_func = np.vstack([
            np.hstack([X, np.fliplr(X)]),
            np.flipud(np.hstack([X, np.fliplr(X)]))
        ])
        wind_func = wind_func + 1e-6

        # Set NaN values in ph to 0
        ph_grid[np.isnan(ph_grid)] = 0

        # Create Gaussian window filter
        gausswin_7 = scipy.signal.windows.gaussian(7, std=2.5)
        B = np.outer(gausswin_7, gausswin_7.T)

        n_win_ex = n_win + n_pad
        ph_bit = np.zeros((n_win_ex, n_win_ex), dtype=np.complex64)

        for ix1 in range(n_win_i):
            wf = wind_func
            i1 = ix1 * n_inc
            i2 = i1 + n_win
            if i2 > n_i:
                i_shift = i2 - n_i
                i2 = n_i
                i1 = n_i - n_win
                wf = np.vstack([np.zeros((i_shift, n_win)), wf[:n_win-i_shift, :]])
            
            for ix2 in range(n_win_j):
                wf2 = wf.copy()
                j1 = ix2 * n_inc
                j2 = j1 + n_win
                if j2 > n_j:
                    j_shift = j2 - n_j
                    j2 = n_j
                    j1 = n_j - n_win
                    wf2 = np.hstack([np.zeros((n_win, j_shift)), wf2[:, :n_win-j_shift]])
                
                ph_bit[:n_win, :n_win] = ph_grid[i1:i2, j1:j2]
                
                ph_fft = scipy.fft.fft2(ph_bit)
                H = np.abs(ph_fft)
                
                # Smooth response
                H = scipy.fft.ifftshift(scipy.ndimage.convolve(scipy.fft.fftshift(H), B, mode='constant'))
                
                meanH = np.median(H)
                if meanH != 0:
                    H = H / meanH
                
                H = H ** alpha
                H = H - 1  # set all values under median to zero
                H[H < 0] = 0  # set all values under median to zero
                G = H * beta + low_pass
                ph_filt = scipy.fft.ifft2(ph_fft * G)
                ph_filt = ph_filt[:n_win, :n_win] * wf2
                
                if np.isnan(ph_filt[0, 0]):
                    import pdb; pdb.set_trace()
                
                ph_out[i1:i2, j1:j2] = ph_out[i1:i2, j1:j2] + ph_filt
                
        return ph_out
    
    def _ps_est_gamma_quick(self, est_gamma_parm=None):
        """Estimate coherence of PS candidates
        Args:
            patch_dir: Directory containing patch data
            est_gamma_parm: Restart flag (0: new run, 1: restart, 2: restart patch only)
        """
        self.parms.load()
        if est_gamma_parm is None:
            est_gamma_parm = 0

        patch_dir = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir)

        # Set coherence threshold
        if self.small_baseline_flag == 'y':
            low_coh_thresh = 15
        else:
            low_coh_thresh = 31

        # Calculate low pass filter
        freq0 = 1 / self.low_pass_wavelength
        freq_i = np.arange(-(self.n_win-1)/self.grid_size/self.n_win/2, (self.n_win-1)/self.grid_size/self.n_win/2+1/self.grid_size/self.n_win, 1/self.grid_size/self.n_win)
        butter_i = 1 / (1 + (freq_i/freq0)**(2*5))
        self.low_pass = np.outer(butter_i, butter_i)
        self.low_pass = np.fft.fftshift(self.low_pass)

        # Load data
        psname = f'{patch_dir}/ps{self.psver}.mat'
        laname = f'{patch_dir}/la{self.psver}.mat'
        incname = f'{patch_dir}/inc{self.psver}.mat'
        pmname = f'{patch_dir}/pm{self.psver}.mat'
        daname = f'{patch_dir}/da{self.psver}.mat'
        phname = f'{patch_dir}/ph{self.psver}.mat'
        bpname = f'{patch_dir}/bp{self.psver}.mat'

        ps = sio.loadmat(psname)
        bp = sio.loadmat(bpname)

        # Load DA file if exists
        if os.path.exists(daname):
            da = sio.loadmat(daname)
            D_A = da['D_A']
            del da
        else:
            D_A = np.ones(ps['n_ps'][0])

        if os.path.exists(phname):
            ph = sio.loadmat(phname)
            ph = ph['ph']
        else:
            ph = ps['ph']

        # Handle zero phases
        null_i, null_j = np.where(ph == 0)
        null_i = np.unique(null_i)
        good_ix = np.ones((int(ps['n_ps'][0][0]), 1), dtype=bool)
        good_ix[null_i] = False

        # Handle small baseline flag
        if self.small_baseline_flag == 'y':
            bperp = ps['bperp'].flatten()
            n_ifg = ps['n_ifg'][0][0]
            n_image = ps['n_image'][0][0]
            ifgday_ix = ps['ifgday_ix']
        else:
            ph = np.delete(ph, ps['master_ix'][0][0], axis=1)
            bperp = np.delete(ps['bperp'], ps['master_ix'][0][0])
            n_ifg = ps['n_ifg'][0][0] - 1
        n_ps = ps['n_ps'][0][0]
        xy = ps['xy']
        del ps
        
        # Normalize phase
        A = np.abs(ph)
        A = np.float32(A)
        A[A == 0] = 1
        ph = ph / A

        # Get incidence angle
        print("   -> Getting incidence angle...")
        if os.path.exists(incname):
            inc = sio.loadmat(incname)
            inc_mean = np.mean(inc['inc'][inc['inc'] != 0])
            del inc
        else:
            if os.path.exists(laname):
                la = sio.loadmat(laname)
                inc_mean = np.mean(la['la']) + 0.052
                del la
            else:
                inc_mean = 21 * np.pi / 180

        # Calculate max K
        max_K = self.max_topo_err / (self.lambda_ * self.rho * np.sin(inc_mean) / 4 / np.pi)
        bperp_range = np.amax(bperp) - np.amin(bperp)
        n_trial_wraps = bperp_range * max_K / (2 * np.pi)
        print(f"   -> n_trial_wraps = {n_trial_wraps}")

        if est_gamma_parm > 0:
            print('   -> Restarting previous run...')
            pm = sio.loadmat(pmname)
            for key in pm.keys():
                if key not in locals():
                    locals()[key] = pm[key]
            if 'gamma_change_save' not in locals():
                gamma_change_save = 1
        else:
            print('   -> Initialising random distribution...')
            n_rand = n_ps  # number of simulated random phase pixels
            if self.small_baseline_flag == 'y':
                rand_image = 2 * np.pi * np.random.rand(n_rand, n_image)
                rand_ifg = np.zeros((n_rand, n_ifg))
                for i in range(n_ifg):
                    rand_ifg[:, i] = rand_image[:, ifgday_ix[i, 1]] - rand_image[:, ifgday_ix[i, 0]]
                del rand_image
            else:
                rand_ifg = 2 * np.pi * np.random.rand(n_rand, n_ifg)
            
            coh_rand = np.zeros(n_rand)
            for i in tqdm(list(reversed(range(0, n_rand, 1))), desc='      -> Computing progress', unit=' pixels'):
                _, _, coh_r, _ = self._ps_topofit(np.exp(1j * rand_ifg[i, :]), bperp.T, n_trial_wraps, False)
                coh_rand[i] = coh_r
            del rand_ifg

            coh_bins = np.linspace(0.005, 0.995, 100)
            step = 0.01
            half_step = step / 2
            edges = np.concatenate((
                [coh_bins[0] - half_step],
                coh_bins[:-1] + half_step,
                [coh_bins[-1] + half_step]
            ))
            Nr = np.histogram(coh_rand, bins=edges)[0]
            i = len(Nr)-1
            while Nr[i] == 0:
                i = i - 1
            Nr_max_nz_ix = i

            step_number = 1

            K_ps = np.zeros((n_ps, 1))
            C_ps = np.zeros((n_ps, 1))
            coh_ps = np.zeros((n_ps, 1))
            coh_ps_save = np.zeros((n_ps, 1))
            N_opt = np.zeros((n_ps, 1))
            self.ph_res = np.zeros((n_ps, n_ifg), dtype=np.complex64)
            self.ph_patch = np.zeros(ph.shape, dtype=np.complex64)
            grid_ij = np.zeros((n_ps, 2))
            grid_ij[:, 0] = np.ceil((xy[:, 2] - np.amin(xy[:, 2]) + 1e-6) / self.grid_size)
            grid_ij[grid_ij[:, 0] == np.amax(grid_ij[:, 0]), 0] = np.amax(grid_ij[:, 0]) - 1
            grid_ij[:, 1] = np.ceil((xy[:, 1] - np.amin(xy[:, 1]) + 1e-6) / self.grid_size)
            grid_ij[grid_ij[:, 1] == np.amax(grid_ij[:, 1]), 1] = np.amax(grid_ij[:, 1]) - 1 
            weighting = 1.0 / D_A
            gamma_change_save = 0

        n_i = int(np.amax(grid_ij[:, 0]))
        n_j = int(np.amax(grid_ij[:, 1]))

        # print(f"   -> {n_ps} PS candidates to process")
        self.loop_end_sw = 0
        i_loop = 1
        while self.loop_end_sw == 0:
            print(f"      -> Iteration #{i_loop}")
            print("      -> Calculating patch phases...")
            self.ph_grid = np.zeros((int(n_i), int(n_j), int(n_ifg)), dtype=np.complex64)
            ph_filt = self.ph_grid.copy()
            ph_weight = ph * np.exp(-1j * bp['bperp_mat'] * K_ps) * weighting.reshape(-1, 1)
            
            for i in range(n_ps):
                self.ph_grid[grid_ij[i, 0].astype(int)-1, grid_ij[i, 1].astype(int)-1, :] += ph_weight[i, :]

            for i in range(n_ifg):
                ph_filt[:, :, i] = self._clap_filt(self.ph_grid[:, :, i], self.clap_alpha, self.clap_beta, self.n_win*0.75, self.n_win*0.25, self.low_pass)

            for i in range(n_ps):
                self.ph_patch[i, 0:n_ifg] = np.squeeze(ph_filt[grid_ij[i, 0].astype(int)-1, grid_ij[i, 1].astype(int)-1, :])

            del ph_filt
            ix = self.ph_patch != 0
            self.ph_patch[ix] = self.ph_patch[ix] / np.abs(self.ph_patch[ix])

            if est_gamma_parm < 2:
                step_number = 2
                for i in tqdm(range(n_ps), desc='      -> Estimating topo error', unit=' pixels'):
                    psdph = ph[i, :] * np.conj(self.ph_patch[i, :])
                    if np.sum(psdph == 0) == 0:
                        Kopt, Copt, cohopt, ph_residual = self._ps_topofit(psdph, bp['bperp_mat'][i, :].reshape(-1, 1), n_trial_wraps, False)
                        K_ps[i] = Kopt
                        C_ps[i] = Copt
                        coh_ps[i] = cohopt
                        N_opt[i] = 1
                        self.ph_res[i, :] = np.angle(ph_residual)
                    else:
                        K_ps[i] = np.nan
                        coh_ps[i] = 0

                step_number = 1

                gamma_change_rms = np.sqrt(np.sum((coh_ps - coh_ps_save)**2) / n_ps)
                gamma_change_change = gamma_change_rms - gamma_change_save
                print(f"      -> gamma_change_change = {gamma_change_change}")
                gamma_change_save = gamma_change_rms
                coh_ps_save = coh_ps.copy()

                if abs(gamma_change_change) < self.gamma_change_convergence or i_loop >= self.gamma_max_iterations:
                    self.loop_end_sw = 1
                else:
                    i_loop += 1
                    if self.filter_weighting == 'P-square':
                        # Compute histogram of coherence values
                        Na = np.histogram(coh_ps, bins=edges)[0]  # Use edges instead of coh_bins
                        
                        # Scale Nr using low coherence values
                        scale_factor = np.sum(Na[:low_coh_thresh]) / np.sum(Nr[:low_coh_thresh])
                        Nr = Nr * scale_factor

                        # Avoid divide by zero
                        Na[Na == 0] = 1

                        # Calculate probability of being random
                        Prand = Nr / Na

                        # Apply fixed probability for low coherence
                        Prand[:low_coh_thresh] = 1
                        Prand[Nr_max_nz_ix + 1:] = 0
                        Prand[Prand > 1] = 1

                        # Smooth with Gaussian filter
                        gwin = np.expand_dims(np.hanning(7), axis=0)
                        Prand_padded = np.concatenate([np.ones(7), Prand])
                        Prand = convolve(Prand_padded, gwin.flatten(), mode='valid') / np.sum(gwin)

                        # Interpolate to 1000 samples
                        x = np.linspace(0, len(Prand)-1, len(Prand))
                        x_new = np.linspace(0, len(Prand)-1, 91)  # Match MATLAB's 91 points
                        Prand = np.interp(x_new, x, Prand)

                        # Assign probabilities to each PS
                        coh_ps_idx = np.clip(np.round(coh_ps * 1000).astype(int), 0, len(Prand)-1)
                        Prand_ps = Prand[coh_ps_idx]

                        # Compute weighting
                        weighting = (1 - Prand_ps) ** 2
                    else:
                        g = np.mean(A * np.cos(self.ph_res), axis=1)
                        sigma_n = np.sqrt(0.5 * (np.mean(A**2, axis=1) - g**2))
                        weighting[sigma_n == 0] = 0
                        weighting[sigma_n != 0] = g[sigma_n != 0] / sigma_n[sigma_n != 0] # snr
            else:
                self.loop_end_sw = 1

            # Save results
            sio.savemat(pmname, {
                'ph_patch': self.ph_patch,
                'K_ps': K_ps,
                'C_ps': C_ps,
                'coh_ps': coh_ps,
                'N_opt': N_opt,
                'ph_res': self.ph_res,
                'step_number': step_number,
                'ph_grid': self.ph_grid,
                'n_trial_wraps': n_trial_wraps,
                'grid_ij': grid_ij,
                'grid_size': self.grid_size,
                'low_pass': self.low_pass,
                'i_loop': i_loop,
                'ph_weight': ph_weight,
                'Nr': Nr,
                'Nr_max_nz_ix': Nr_max_nz_ix,
                'coh_bins': edges,
                'coh_ps_save': coh_ps_save,
                'gamma_change_save': gamma_change_save
            })

    ############ PS Select ############
    
    def _datenum_to_datetime(dn):
        """Convert MATLAB serial date number to Python datetime"""
        return datetime.fromordinal(int(dn)) + timedelta(days=dn%1) - timedelta(days=366)
    
    def _clap_filt_patch(self, ph, alpha=0.5, beta=0.1, low_pass=None):
        """Apply CLAP filter to a patch of phase data"""
        if low_pass is None:
            low_pass = np.zeros(ph.shape)
        
        ph[np.isnan(ph)] = 0
        gausswin_7 = scipy.signal.windows.gaussian(7, std=2.5)
        B = np.outer(gausswin_7, gausswin_7.T)

        ph_fft = scipy.fft.fft2(ph)
        H = np.abs(ph_fft)
        H = scipy.fft.ifftshift(scipy.ndimage.convolve(scipy.fft.fftshift(H), B, mode='constant'))
        
        meanH = np.median(H)
        if meanH != 0:
            H = H / meanH
        
        H = H ** alpha
        H = H - 1  # set all values under median to zero
        H[H < 0] = 0  # set all values under median to zero
        G = H * beta + low_pass
        ph_out = scipy.fft.ifft2(ph_fft * G)
        
        return ph_out

    def _ps_select(self, reselect=0, plot_flag=False):
        if self.psver > 1:
            self._update_psver(1, self.patch_dir)

        self.parms.load()
        self.slc_osf = self.parms.get('slc_osf')
        self.clap_alpha = self.parms.get('clap_alpha')
        self.clap_beta = self.parms.get('clap_beta')
        self.n_win = self.parms.get('clap_win')
        select_method = self.parms.get('select_method')
        if select_method == 'PERCENT':
            self.max_percent_rand = self.parms.get('percent_rand')
        else:
            self.max_density_rand = self.parms.get('density_rand')
        gamma_stdev_reject = self.parms.get('gamma_stdev_reject')
        small_baseline_flag = self.parms.get('small_baseline_flag')
        drop_ifg_index = self.parms.get('drop_ifg_index')

        if small_baseline_flag == 'y':
            self.low_coh_thresh = 15
        else:
            self.low_coh_thresh = 31

        psname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'ps{self.psver}.mat')
        phname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'ph{self.psver}.mat')
        pmname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'pm{self.psver}.mat')
        selectname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'select{self.psver}.m')
        daname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'da{self.psver}.mat')
        bpname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'bp{self.psver}.mat')

        ps = sio.loadmat(psname)
        ifg_index = []
        for idx in drop_ifg_index:
            if not idx - 1 in range(len(ps['day'])):
                ifg_index.append(idx)
        
        if os.path.exists(phname):
            phin=sio.loadmat(phname)
            ph=phin['ph']
            del phin
        else:
            ph=ps['ph']

        bperp=ps['bperp'].flatten()
        n_ifg=ps['n_ifg'][0][0]
        if not small_baseline_flag == 'y':
            master_ix=ps['master_ix'][0][0]
            no_master_ix=[f for f in list(range(n_ifg)) if f != master_ix]
            ifg_index=[f for f in ifg_index if f != master_ix]
            ifg_index=[f-1 for f in ifg_index if f > master_ix]
            ph=ph[:,no_master_ix]
            bperp=bperp[no_master_ix]
            n_ifg=len(no_master_ix)
        n_ps=ps['n_ps'][0][0]
        xy=ps['xy']

        pm=sio.loadmat(pmname)
        if os.path.exists(daname):
            da=sio.loadmat(daname)
            D_A=da['D_A'].reshape(-1, 1)[:, 0]
            del da
        else:
            D_A=np.array([])
        if not len(D_A) == 0 and len(D_A) >= 10000:
            D_A_sort = np.sort(D_A).flatten()  # Ensure 1D array
            if len(D_A) >= 50000:
                bin_size = 10000
            else:
                bin_size = 2000
        
            # Create D_A_max with consistent shapes
            # Use size instead of shape[0] for 1D array length
            middle_values = D_A_sort[bin_size::bin_size][:-1]  # Get values at bin_size intervals, excluding last
            D_A_max = np.concatenate(([0], middle_values, [D_A_sort[-1]]))
        
        else:
            D_A_max = np.array([[0], [1]])
            D_A = np.ones((n_ps, n_ifg))
        
        if not select_method == 'PERCENT':
            patch_area=np.prod(np.amax(xy[:, 1:], axis=0) - np.amin(xy[:, 1:], axis=0)) / 1e6 # in km2
            self.max_percent_rand=self.max_density_rand*patch_area/(len(D_A_max)-1)
        min_coh=np.zeros((len(D_A_max)-1, 1))
        D_A_mean=np.zeros((D_A_max.shape[0]-1, 1))
        Nr_dist=pm['Nr'].reshape(-1, 1)[:, 0]

        if reselect==3:
            coh_thresh=0
            coh_thresh_coeffs=[]
        else:
            for i in tqdm(range(len(D_A_max)-1), desc='   -> Filtering coherence', unit=' bins'):
                # Convert to numpy arrays and ensure proper shapes
                mask = (D_A > D_A_max[i]) & (D_A <= D_A_max[i+1])
                coh_chunk = pm['coh_ps'][mask]
                
                # Handle empty slices
                # if np.sum(mask) == 0:
                #     D_A_mean[i] = np.nan
                #     min_coh[i] = np.nan
                #     continue
                    
                D_A_mean[i] = np.mean(D_A[mask])
                coh_chunk = coh_chunk[coh_chunk != 0]
                Na = np.histogram(coh_chunk, pm['coh_bins'].flatten())[0]
                Nr = Nr_dist * np.sum(Na[:self.low_coh_thresh]) / np.sum(Nr_dist[:self.low_coh_thresh])
                # if i == len(D_A_max) - 2 and plot_flag:
                #     plt.figure()
                #     plt.plot(edges[:-1], Na, 'g')
                #     plt.plot(edges[:-1], Nr, 'r')
                #     plt.legend(['data', 'random'])
                #     plt.title('Before Gamma Reestimation')
                #     plt.show()
                
                Na[Na==0] = 1  # avoid divide by zero
                if select_method.upper() == 'PERCENT':
                    percent_rand = np.flip(np.cumsum(np.flip(Nr)) / np.cumsum(np.flip(Na)) * 100)
                else:
                    percent_rand = np.flip(np.cumsum(np.flip(Nr)))
                ok_ix = np.where(percent_rand < self.max_percent_rand)[0]
                if len(ok_ix) == 0:
                    min_coh[i] = 1
                else:
                    min_fit_ix = np.amin(ok_ix) - 3
                    if min_fit_ix <= 0:
                        min_coh[i] = np.nan
                    else:
                        max_fit_ix = np.amin(ok_ix) + 2
                        max_fit_ix[max_fit_ix > 100] = 100
                        coeffs = np.polyfit(percent_rand[min_fit_ix:max_fit_ix], np.arange(min_fit_ix*0.01, max_fit_ix*0.01, 0.01), 3)
                        min_coh[i] = np.polyval(coeffs, self.max_percent_rand)

            nonnanix = ~np.isnan(min_coh)
            if sum(nonnanix)[0] < 1:
                print('   -> Not enough random phase pixels to set gamma threshold - using default threshold of 0.3')
                coh_thresh = 0.3
                coh_thresh_coeffs = np.array([])
            else:
                min_coh = min_coh[nonnanix.flatten()]
                D_A_mean = D_A_mean[nonnanix.flatten()]
                if min_coh.shape[0] > 1:
                    coh_thresh_coeffs = np.polyfit(D_A_mean, min_coh, 1)
                    if coh_thresh_coeffs[0] > 0:
                        coh_thresh = np.polyval(coh_thresh_coeffs, D_A)
                    else:
                        coh_thresh = np.polyval(coh_thresh_coeffs, 0.35)
                        coh_thresh_coeffs = np.array([])
                else:
                    coh_thresh = min_coh
                    coh_thresh_coeffs = np.array([])
        coh_thresh = np.array(coh_thresh, ndmin=1)
        coh_thresh[coh_thresh < 0] = 0
        print(f'   -> Initial gamma threshold: {min(coh_thresh)} at D_A={min(D_A_mean)[0]} to {max(coh_thresh)} at D_A={max(D_A_mean)[0]}')
        
        # plot_flag = True
        # if plot_flag:
        #     plt.figure()
        #     plt.plot(D_A_mean.flatten(), min_coh.flatten(), '*')
        #     if not coh_thresh_coeffs is None:
        #         plt.plot(np.polyval(coh_thresh_coeffs, D_A_mean.flatten()), min_coh.flatten(), 'r')
        #     plt.ylabel('\gamma_{thresh}')
        #     plt.xlabel('D_A')
        #     plt.show()

        # Initial PS selection
        self.ix = np.where(pm['coh_ps'] > coh_thresh)[0]
        n_ps = len(self.ix)
        print(f'      -> {n_ps} PS selected initially')
        
        n_boot = 100
        if gamma_stdev_reject > 0:
            ph_res_cpx = np.exp(1j * pm['ph_res'][:, ifg_index])
            coh_std = np.zeros((len(self.ix), 1))
            for i in range(len(self.ix)):
                ph_values = ph_res_cpx[self.ix[i], ifg_index]
                boot_vals = np.array([np.abs(np.sum(resample(ph_values))) / len(ph_values)
                                      for _ in range(n_boot)])
                coh_std[i] = np.std(boot_vals)
            del ph_res_cpx
            self.ix = self.ix[coh_std < gamma_stdev_reject]
            n_ps = len(self.ix)
            print(f'      -> {n_ps} PS left after pps rejection')

        if reselect != 1:
            if reselect != 2:
                for i in range(len(drop_ifg_index)):
                    ifg_day = self._datenum_to_datetime(ps['day'].flatten()[drop_ifg_index[i] - 1])
                    if small_baseline_flag == 'y':
                        print(f'   -> {ifg_day.strftime("%Y-%m-%d")}-{ifg_day.strftime("%Y-%m-%d")} is dropped from noise re-estimation')
                    else:
                        print(f'   -> {ifg_day.strftime("%Y-%m-%d")} is dropped from noise re-estimation')
                
                pm.pop('ph_res', None)
                pm.pop('ph_patch', None)
                ph_patch2 = np.zeros((n_ps, n_ifg), dtype=np.complex64)
                ph_res2 = np.zeros((n_ps, n_ifg), dtype=np.complex64)
                ph = ph[self.ix, :]

                if len(coh_thresh) > 1:
                    coh_thresh = coh_thresh[self.ix]

                self.n_i = int(np.amax(pm['grid_ij'][:, 0]))
                self.n_j = int(np.amax(pm['grid_ij'][:, 1]))
                K_ps2 = np.zeros((n_ps, 1))
                C_ps2 = np.zeros((n_ps, 1))
                coh_ps2 = np.zeros((n_ps, 1))

                ph_filt = np.zeros((self.n_win, self.n_win, n_ifg), dtype=np.complex64)
                self.low_pass = pm['low_pass'][0][0]
                hw = self.n_win // 2  # half window size

                for i in tqdm(range(n_ps), desc="   -> Smoothing pixels", unit="pixels"):
                    i_center, j_center = pm['grid_ij'][self.ix[i], :].astype(int)

                    # Calculate patch bounds
                    i_min = max(i_center - hw, 0)
                    i_max = min(i_min + self.n_win, self.n_i)
                    if (i_max - i_min) < self.n_win:
                        i_min = max(i_max - self.n_win, 0)

                    j_min = max(j_center - hw, 0)
                    j_max = min(j_min + self.n_win, self.n_j)
                    if (j_max - j_min) < self.n_win:
                        j_min = max(j_max - self.n_win, 0)

                    # Check bounds
                    if i_min < 0 or j_min < 0 or i_max >= self.n_i or j_max >= self.n_j:
                        ph_patch2[i, :] = 0
                        continue

                    # PS index within patch
                    ps_bit_i = i_center - i_min
                    ps_bit_j = j_center - j_min

                    # Extract patch
                    ph_bit = pm['ph_grid'][i_min:i_max, j_min:j_max, :].copy()
                    ph_bit[ps_bit_i, ps_bit_j, :] = 0.0

                    # Optional: subtract weight
                    # ph_bit[ps_bit_i, ps_bit_j, :] -= pm['ph_weight'][i, :].astype(np.float32).reshape(1, -1)

                    # Oversampling window update
                    os_range = np.arange(-(self.slc_osf - 1), self.slc_osf)
                    ix_i = ps_bit_i + os_range
                    ix_j = ps_bit_j + os_range
                    ix_i = ix_i[(ix_i >= 0) & (ix_i < ph_bit.shape[0])]
                    ix_j = ix_j[(ix_j >= 0) & (ix_j < ph_bit.shape[1])]

                    # Zero surrounding oversample region
                    ph_bit[np.ix_(ix_i, ix_j)] = 0.0

                    # Apply filter
                    for i_ifg in range(n_ifg):
                        ph_filt[:, :, i_ifg] = self._clap_filt_patch(
                            ph_bit[:, :, i_ifg],
                            self.clap_alpha,
                            self.clap_beta,
                            self.low_pass
                        )

                    # Save result
                    ph_patch2[i, :] = ph_filt[ps_bit_i, ps_bit_j, :]

                # Re-estimate coherence
                pm.pop('ph_grid', None)
                bp = sio.loadmat(bpname)
                bperp_mat = bp['bperp_mat'][self.ix, :]
                del bp

                for i in tqdm(range(n_ps), desc='   -> Re-estimating PS coherence', unit=' pixels'):
                    psdph = ph[i, :] * np.conj(ph_patch2[i, :])
                    if sum(psdph == 0) == 0:
                        psdph = np.complex128(psdph) / np.abs(np.complex128(psdph))
                        Kopt, Copt, cohopt, ph_residual = self._ps_topofit(psdph, bperp_mat[i, :], pm['n_trial_wraps'][0][0], False)
                        K_ps2[i] = Kopt
                        C_ps2[i] = Copt
                        coh_ps2[i] = cohopt
                        ph_res2[i, :] = np.angle(ph_residual)
                    else:
                        K_ps2[i] = np.nan
                        coh_ps2[i] = np.nan

            else:
                sl = sio.loadmat(selectname)
                self.ix = sl['ix']
                coh_ps2 = sl['coh_ps2']
                K_ps2 = sl['K_ps2']
                C_ps2 = sl['C_ps2']
                ph_res2 = sl['ph_res2']
                ph_patch2 = sl['ph_patch2']

            pm['coh_ps'][self.ix] = coh_ps2
            for i in range(len(D_A_max) - 1):
                # Convert to numpy arrays and ensure proper shapes
                mask = (D_A > D_A_max[i]) & (D_A <= D_A_max[i+1])
                coh_chunk = pm['coh_ps'][mask]
                D_A_mean[i] = np.mean(D_A[mask])
                coh_chunk = coh_chunk[coh_chunk != 0]
                Na = np.histogram(coh_chunk, pm['coh_bins'][0])[0]
                Nr = Nr_dist * np.sum(Na[:self.low_coh_thresh]) / np.sum(Nr_dist[:self.low_coh_thresh])
                
                # if i == len(D_A_max) - 2 & plot_flag:
                #     plt.figure()
                #     plt.plot(edges[:-1], Na, 'g')
                #     plt.plot(edges[:-1], Nr, 'r')
                #     plt.legend(['data', 'random'])
                #     plt.title('After Gamma Reestimation')
                #     plt.show()
                
                Na[Na == 0] = 1  # avoid divide by zero
                if select_method.upper() == 'PERCENT':
                    percent_rand = np.flip(np.cumsum(np.flip(Nr)) / np.cumsum(np.flip(Na)))
                else:
                    percent_rand = np.flip(np.cumsum(np.flip(Nr)))
                
                ok_ix = np.where(percent_rand < self.max_percent_rand)[0]
                if len(ok_ix) == 0:
                    min_coh[i] = 1
                else:
                    min_fit_ix = np.amin(ok_ix) - 3
                    if min_fit_ix <= 0:
                        min_coh[i] = np.nan
                    else:
                        max_fit_ix = np.amin(ok_ix) + 2
                        max_fit_ix[max_fit_ix > 100] = 100
                        coeffs = np.polyfit(percent_rand[min_fit_ix:max_fit_ix], range(min_fit_ix * 0.01, max_fit_ix * 0.01, 0.01), 3)
                        min_coh[i] = np.polyval(coeffs, self.max_percent_rand)

            nonnanix = ~np.isnan(min_coh)
            if sum(nonnanix.flatten()) < 1:
                coh_thresh = np.array([0.3])
                coh_thresh_coeffs = np.array([])
            else:
                min_coh = min_coh[nonnanix]
                D_A_mean = D_A_mean[nonnanix]
                if min_coh.shape[0] > 1:
                    coh_thresh_coeffs = np.polyfit(D_A_mean, min_coh, 1)
                    if coh_thresh_coeffs[0] > 0:
                        coh_thresh = np.polyval(coh_thresh_coeffs, D_A[self.ix])
                    else:
                        coh_thresh = np.polyval(coh_thresh_coeffs, 0.35)
                        coh_thresh_coeffs = np.array([])
                else:
                    coh_thresh = min_coh
                    coh_thresh_coeffs = np.array([])

            coh_thresh[coh_thresh < 0] = 0
            print(f'   -> Reestimation gamma threshold: {np.amin(coh_thresh)} at D_A={np.amin(D_A_mean)} to {np.amax(coh_thresh)} at D_A={np.amax(D_A_mean)}')
            
            # Calculate final selection
            bperp_range = np.amax(bperp) - np.amin(bperp)
            keep_ix = (coh_ps2 > coh_thresh) & (np.abs(pm['K_ps'][self.ix] - K_ps2) < 2 * np.pi / bperp_range)
            
            del pm
            print(f'      -> {np.sum(keep_ix)} PS selected after re-estimation')
        else:
            pm.pop('ph_grid', None)
            ph_patch2 = pm['ph_patch'][self.ix, :]
            ph_res2 = pm['ph_res'][self.ix, :]
            K_ps2 = pm['K_ps'][self.ix]
            C_ps2 = pm['C_ps'][self.ix]
            coh_ps2 = pm['coh_ps'][self.ix]
            keep_ix = np.ones(len(self.ix), dtype=bool)

        if not os.path.exists(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, 'no_ps_info.mat')):
            stamps_step_no_ps = np.zeros((5, 1))
        else:
            stamps_step_no_ps = sio.loadmat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, 'no_ps_info.mat'))['stamps_step_no_ps']
            stamps_step_no_ps[2:] = 0
        
        if np.sum(keep_ix) == 0:
            print(' -> No PS points left. Updating the stamps log for this\n')
            stamps_step_no_ps[2] = 1
        sio.savemat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, 'no_ps_info.mat'), {'stamps_step_no_ps': stamps_step_no_ps})

        # if plot_flag:
        #     plt.figure()
        #     plt.plot(D_A_mean, min_coh, '*')
        #     if not coh_thresh_coeffs is None:
        #         plt.plot(D_A_mean, np.polyval(coh_thresh_coeffs, D_A_mean), 'r')
        #     plt.ylabel('\gamma_{thresh}')
        #     plt.xlabel('D_A')
        #     plt.show()
        
        sio.savemat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'select{self.psver}.mat'),
                    {'ix': self.ix,
                     'keep_ix': keep_ix,
                     'ph_patch2': ph_patch2,
                     'ph_res2': ph_res2,
                     'K_ps2': K_ps2,
                     'C_ps2': C_ps2,
                     'coh_ps2': coh_ps2,
                     'coh_thresh': coh_thresh, 
                     'coh_thresh_coeffs': coh_thresh_coeffs,
                     'clap_alpha': self.clap_alpha,
                     'clap_beta': self.clap_beta,
                     'n_win': self.n_win,
                     'max_percent_rand': self.max_percent_rand,
                     'gamma_stdev_reject': gamma_stdev_reject,
                     'small_baseline_flag': small_baseline_flag,
                     'ifg_index': ifg_index})
        
        """Plot phase patch for each interferogram"""
        try:
            # Load data
            selected = sio.loadmat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'select{self.psver}.mat'))
            ix = selected['ix'][0]
            ph_patch2 = selected['ph_patch2']
            ps = sio.loadmat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'ps{self.psver}.mat'))
            lonlat = ps['lonlat']

            # Get number of interferograms
            n_ifg = ph_patch2.shape[1]

            # Create subplot grid
            n_rows = int(np.ceil(np.sqrt(n_ifg)))
            n_cols = int(np.ceil(n_ifg / n_rows))

            plt.figure(figsize=(10*n_cols, 6*n_rows))

            # Plot phase for each interferogram
            for i in range(n_ifg):
                plt.subplot(n_rows, n_cols, i+1)
                
                # Get valid data points (non-zero/non-nan values)
                valid_idx = ~np.isnan(ph_patch2[:,i]) & (ph_patch2[:,i] != 0)
                
                plt.scatter(lonlat[ix[valid_idx],0], lonlat[ix[valid_idx],1], 
                        c=np.angle(ph_patch2[valid_idx,i]),
                        cmap='hsv', vmin=-np.pi, vmax=np.pi, s=10)
                plt.colorbar(label='Phase (rad)')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.title(f'Interferogram {i+1}')

            plt.tight_layout()
            plt.savefig(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'3_ps_select_{self.psver}.png'))
            plt.close()
        except:
            pass

    ########## Weed PS ##########
    
    def _ps_weed(self, all_da_flag=False, no_weed_adjacent=False, no_weed_noisy=False, use_triangle=False):
        print(f'   -> Weeding PS')
        if all_da_flag is None:
            all_da_flag = False

        self.parms.load()
        time_win = self.parms.get('weed_time_win')
        weed_standard_dev = self.parms.get('weed_standard_dev')
        weed_max_noise = self.parms.get('weed_max_noise')
        weed_zero_elevation = self.parms.get('weed_zero_elevation')
        weed_neighbours = self.parms.get('weed_neighbours')
        drop_ifg_index = self.parms.get('drop_ifg_index')
        small_baseline_flag = self.parms.get('small_baseline_flag')

        if weed_neighbours == 'y':
            no_weed_adjacent = False
        else:
            no_weed_adjacent = True

        if weed_standard_dev >= np.pi and weed_max_noise >= np.pi:
            no_weed_noisy = True
        else:
            no_weed_noisy = False
        
        self._update_psver(1, self.patch_dir)
        psname = f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/ps{self.psver}.mat'
        pmname = f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/pm{self.psver}.mat'
        phname = f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/ph{self.psver}.mat'
        selectname = f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/select{self.psver}.mat'
        hgtname = f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/hgt{self.psver}.mat'
        laname = f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/la{self.psver}.mat'
        incname = f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/inc{self.psver}.mat'
        bpname = f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/bp{self.psver}.mat'
        psothername = 'ps_other'
        pmothername = 'pm_other'
        selectothername = 'select_other'
        hgtothername = 'hgt_other'
        laothername = 'la_other'
        incothername = 'inc_other'
        bpothername = 'bp_other'

        ps = sio.loadmat(psname)
        ifg_index = []
        for idx in range(len(ps['day'][0])):
            if not (idx + 1) in drop_ifg_index:
                ifg_index.append(idx)

        sl = sio.loadmat(selectname)

        if os.path.exists(phname):
            phin = sio.loadmat(phname)
            ph = phin['ph']
            del phin
        else:
            ph = ps['ph']

        day = ps['day'][0]
        bperp = ps['bperp'][0]

        keep_ix = sl['keep_ix'].astype(bool)
        if 'keep_ix' in sl:
            ix2 = sl['ix'][0][keep_ix.flatten()]
            K_ps2 = sl['K_ps2'][:, 0][keep_ix.flatten()]
            C_ps2 = sl['C_ps2'][:, 0][keep_ix.flatten()]
            coh_ps2 = sl['coh_ps2'][:, 0][keep_ix.flatten()]
        else:
            ix2 = sl['ix2'][0]
            K_ps2 = sl['K_ps2'][0]
            C_ps2 = sl['C_ps2'][0]
            coh_ps2 = sl['coh_ps2'][0]

        ij2 = ps['ij'][ix2, :]
        xy2 = ps['xy'][ix2, :]
        ph2 = ph[ix2, :]
        lonlat2 = ps['lonlat'][ix2, :]

        pm = sio.loadmat(pmname)
        ph_patch2 = pm['ph_patch'][ix2, :]
        if 'ph_res2' in sl:
            ph_res2 = sl['ph_res2'][keep_ix.flatten(), :]
        else:
            ph_res2 = np.array([])
        del pm
        del sl
        del ph
        if 'ph' in ps:
            ps.pop('ph')

        for field in ['xy', 'ij', 'lonlat', 'sort_ix']:
            ps.pop(field, None)

        if all_da_flag:
            pso = sio.loadmat(psothername)
            slo = sio.loadmat(selectothername)
            ix_other = slo['ix_other']
            n_ps_other = np.sum(ix_other)
            K_ps_other2 = pso['K_ps_other'][ix_other]
            C_ps_other2 = pso['C_ps_other'][ix_other]
            coh_ps_other2 = pso['coh_ps_other'][ix_other]
            ph_res_other2 = pso['ph_res_other'][ix_other, :]
            ij2 = np.concatenate((ij2, pso['ij_other'][ix_other, :]), axis=0)
            xy2 = np.concatenate((xy2, pso['xy_other'][ix_other, :]), axis=0)
            ph2 = np.concatenate((ph2, pso['ph_other'][ix_other, :]), axis=0)
            lonlat2 = np.concatenate((lonlat2, pso['lonlat_other'][ix_other, :]), axis=0)

            del pso
            del slo

            pmo = sio.loadmat(pmothername)
            ph_patch_other2 = pmo['ph_patch_other'][ix_other, :]
            del pmo

            K_ps2 = np.concatenate((K_ps2, K_ps_other2), axis=0)
            C_ps2 = np.concatenate((C_ps2, C_ps_other2), axis=0)
            coh_ps2 = np.concatenate((coh_ps2, coh_ps_other2), axis=0)
            ph_patch2 = np.concatenate((ph_patch2, ph_patch_other2), axis=0)
            ph_res2 = np.concatenate((ph_res2, ph_res_other2), axis=0)
        else:
            n_ps_other = 0

        if os.path.exists(hgtname):
            ht = sio.loadmat(hgtname)
            hgt = ht['hgt'][ix2]
            del ht
            if all_da_flag:
                hto = sio.loadmat(hgtothername)
                hgt = np.concatenate((hgt, hto['hgt_other'][ix_other]), axis=0)
                del hto

        n_ps_low_D_A = len(ix2)
        n_ps = n_ps_low_D_A + n_ps_other
        ix_weed = np.ones((n_ps, 1), dtype=bool)
        print(f'      -> {n_ps_low_D_A} low D_A PS, {n_ps_other} high D_A PS')
        
        if no_weed_adjacent == False:
            print('      -> Removing adjacent PS')
            ij_shift = ij2[:, 1:3] + (np.array([2, 2]) - np.min(ij2[:, 1:3], axis=0))
            neigh_ix = np.zeros((int(np.max(ij_shift[:, 0])) + 2, int(np.max(ij_shift[:, 1])) + 2), dtype=int)
            miss_middle = np.ones((3, 3), dtype=bool)
            miss_middle[1, 1] = False

            for i in tqdm(range(n_ps), desc='         -> Generating neighbour matrix', unit=' ps'):
                neigh_this = neigh_ix[int(ij_shift[i, 0]-1):int(ij_shift[i, 0] + 2), int(ij_shift[i, 1]-1):int(ij_shift[i, 1] + 2)]
                neigh_this[neigh_this == 0 & miss_middle] = i
                neigh_ix[int(ij_shift[i, 0]-1):int(ij_shift[i, 0] + 2), int(ij_shift[i, 1]-1):int(ij_shift[i, 1] + 2)] = neigh_this
            
            neigh_ps = [[] for _ in range(n_ps)]
            for i in tqdm(range(n_ps), desc='         -> Finding neighbours', unit=' ps'):
                my_neigh_ix = neigh_ix[int(ij_shift[i, 0]), int(ij_shift[i, 1])]
                if my_neigh_ix != 0:
                    neigh_ps[my_neigh_ix].append(i)

            del neigh_ix
            for i in tqdm(range(n_ps), desc='         -> Select best PS', unit=' ps'):
                if len(neigh_ps[i]) > 0:
                    same_ps = [i]
                    i2 = 0
                    while i2 < len(same_ps):
                        ps_i = same_ps[i2]
                        same_ps.extend(neigh_ps[ps_i])
                        neigh_ps[ps_i] = []
                        i2 += 1
                    
                    same_ps = np.unique(same_ps)
                    high_coh_idx = np.argmax(coh_ps2[same_ps])
                    low_coh_idx = np.ones(same_ps.shape[0], dtype=bool)
                    low_coh_idx[high_coh_idx] = False
                    ix_weed[same_ps[low_coh_idx]] = False

            print(f'         -> {np.sum(ix_weed)} PS kept after dropping adjacent pixels')

        if weed_zero_elevation == 'y' and hgt in locals():
            sea_ix = hgt < 1e-6
            ix_weed[sea_ix] = False
            print(f'      -> {np.sum(ix_weed)} PS kept after weeding zero elevation')
        
        xy_weed = xy2[ix_weed.flatten(), :]
        n_ps = np.sum(ix_weed)
        
        # Find unique coordinates and their indices
        ix_weed_num = np.where(ix_weed)[0]
        unique_vals, unique_indices = np.unique(xy_weed[:, 1:3], axis=0, return_index=True)
        all_indices = np.arange(xy_weed.shape[0])
        dups = np.setdiff1d(all_indices, unique_indices)

        # Process each unique coordinate
        for i in tqdm(range(len(dups)), total=len(dups), desc='      -> Removing duplicate PS', unit=' groups'):
            lon_val = xy_weed[dups[i], 1]
            lat_val = xy_weed[dups[i], 2]
            matching_group = np.where((xy_weed[:, 1] == lon_val) & (xy_weed[:, 2] == lat_val))[0]
            global_indices = ix_weed_num[matching_group]
            group_coh = coh_ps2[matching_group]
            max_idx_relative = np.argmax(group_coh)
            keep_index = global_indices[max_idx_relative]
            mask = (global_indices != keep_index)
            ix_weed[global_indices[mask]] = False

        if dups.size > 0:
            xy_weed = xy2[ix_weed.flatten(), :]
            print(f'         -> {len(dups)} PS with duplicate coordinates removed')

        n_ps = len(xy_weed)
        ix_weed2 = np.ones((n_ps, 1), dtype=bool)
        ix_weed2[:, 0] = True

        ps_std = np.zeros((n_ps, 1))
        ps_max = np.zeros((n_ps, 1))

        use_triangle = False
        if n_ps != 0 and no_weed_noisy == False:
            if use_triangle:
                with open(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/psweed.1.node", "w") as f:
                    f.write(f"{n_ps} 2 0 0\n")
                    for i in range(n_ps):
                        f.write(f"{i+1} {xy_weed[i, 1]} {xy_weed[i, 2]}\n")
                    f.close()
                # try:
                #     shutil.copy(self.triangle_path, f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}")
                # except:
                #     pass
                # if platform.system() == 'Windows':
                #     self.triangle_path = os.path.join(self.config['processing_parameters']['current_result'], self.patch_dir, 'triangle.exe')
                # else:
                #     self.triangle_path = os.path.join(self.config['processing_parameters']['current_result'], self.patch_dir, 'triangle')

                os.chdir(f"{self.config['project_definition']['project_folder']}/modules/triangle")
                os.system(f"triangle.exe -e {self.config['processing_parameters']['current_result']}/{self.patch_dir}/psweed.1.node > {self.config['processing_parameters']['current_result']}/{self.patch_dir}/triangle_weed.log")
                os.chdir(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}")
                with open(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/psweed.2.edge", "r") as f:
                    header = list(map(int, f.readline().split()))
                    edgs = np.array([list(map(int, f.readline().split())) for _ in range(header[0])])[:, 1:] - 1
            else:
                # Assume xy_weed is a NumPy array of shape (n_points, 3)
                # xy_weed is a (n, 3) numpy array; use columns 1 and 2 like MATLAB
                points = xy_weed[:, 1:3]  # corresponds to xy_weed(:,2:3) in MATLAB

                # Perform Delaunay triangulation using scipy (triangle package doesn't directly expose delaunay)
                delaunay = Delaunay(points)
                triangles = delaunay.simplices  # shape (n_tri, 3), indices of triangle vertices

                # Now extract edges from triangles
                edges = set()
                for tri in triangles:
                    edges.update([
                        tuple(sorted((tri[0], tri[1]))),
                        tuple(sorted((tri[1], tri[2]))),
                        tuple(sorted((tri[2], tri[0])))
                    ])

                edgs = np.array(list(edges))  # convert set to numpy array

            n_edge = edgs.shape[0]
            # Calculate phase for weeded points with proper array shapes
            K_ps_weeded = K_ps2[ix_weed.flatten()].reshape(-1, 1)  # Ensure 2D array
            bperp_reshaped = bperp.reshape(1, -1)  # Ensure 2D array
            ph_weed = ph2[ix_weed.flatten(), :] * np.exp(-1j * (K_ps_weeded @ bperp_reshaped))
            ph_weed /= np.abs(ph_weed)
            
            if small_baseline_flag != 'y':
                ph_weed[:, ps['master_ix'][0][0]] = np.exp(1j * C_ps2[ix_weed.flatten()].flatten())
            
            edge_std = np.zeros((n_edge, 1))
            edge_max = np.zeros((n_edge, 1))
            dph_space = ph_weed[edgs[:, 1], :] * np.conj(ph_weed[edgs[:, 0], :])
            dph_space = dph_space[:, ifg_index]
            n_use = len(ifg_index)
            
            for i in range(len(drop_ifg_index)):
                print(f"      -> Dropping IFG {drop_ifg_index[i]} from noise estimation")

            if small_baseline_flag != 'y':
                print("      -> Estimating noise for all arcs...")
                dph_smooth = np.zeros((n_edge, n_use), dtype=np.complex64)
                dph_smooth2 = np.zeros_like(dph_smooth)

                for i1 in range(n_use):
                    time_diff = (day[ifg_index[i1]] - day[ifg_index]).astype(float)
                    weight_factor = np.exp(-0.5 * (time_diff / time_win) ** 2)
                    weight_factor /= np.sum(weight_factor)

                    dph_mean = np.sum(dph_space * np.tile(weight_factor, (n_edge, 1)), axis=1)
                    dph_mean_adj = np.angle(dph_space * np.conj(dph_mean)[:, np.newaxis])
                    G = np.column_stack([np.ones((n_use, 1)), time_diff])
                    m = lstsq(G.astype(np.float64) * weight_factor[:, None], dph_mean_adj.astype(np.float64).T * weight_factor[:, None], rcond=None)[0]
                    dph_mean_adj = np.angle(np.exp(1j * (dph_mean_adj - (G @ m).T)))
                    m2 = lstsq(G.astype(np.float64) * weight_factor[:, None], dph_mean_adj.astype(np.float64).T * weight_factor[:, None], rcond=None)[0]
                    dph_smooth[:, i1] = dph_mean * np.exp(1j * (m[0, :].T + m2[0, :].T))
                    weight_factor[i1] = 0
                    dph_smooth2[:, i1] = np.sum(dph_space * np.tile(weight_factor, (n_edge, 1)), axis=1)

                dph_noise = np.angle(dph_space * np.conj(dph_smooth))
                dph_noise2 = np.angle(dph_space * np.conj(dph_smooth2))
                ifg_var = np.var(dph_noise2, axis=0, ddof=1)
                K = lstsq(bperp[ifg_index][:, np.newaxis] / ifg_var[:, np.newaxis], dph_noise.T / ifg_var[:, np.newaxis], rcond=None)[0]
                dph_noise = dph_noise - K.reshape(-1, 1) * bperp[ifg_index].T
                del dph_space, dph_smooth, dph_smooth2, dph_noise2
                edge_std = np.std(dph_noise, axis=1, ddof=1)
                edge_max = np.amax(np.abs(dph_noise), axis=1)
                del dph_noise

            else:
                ifg_var = np.var(dph_space, axis=0, ddof=1)
                A = bperp[ifg_index][:, np.newaxis] / ifg_var[:, np.newaxis]  # (n_use, 1)
                B = dph_noise.T / ifg_var[:, np.newaxis]  # (n_use, n_edge)
                K = lstsq(A, B, rcond=None)[0]  # K: (1, n_edge)
                dph_space = dph_space - K.reshape(-1, 1) * bperp[ifg_index].T
                edge_std = np.std(np.angle(dph_space), axis=1, ddof=1)
                edge_max = np.amax(np.abs(np.angle(dph_space)), axis=1)
                del dph_space

            print("      -> Estimating max noise for all pixels...")
            ps_std = np.full(n_ps, np.inf, dtype=np.float32)
            ps_max = np.full(n_ps, np.inf, dtype=np.float32)

            for i in range(n_edge):
                ps_std[edgs[i]] = np.min([ps_std[edgs[i]], [edge_std[i], edge_std[i]]], axis=0)
                ps_max[edgs[i]] = np.min([ps_max[edgs[i]], [edge_max[i], edge_max[i]]], axis=0)
            ix_weed2 = (ps_std < weed_standard_dev) & (ps_max < weed_max_noise)
            ix_weed[ix_weed] = ix_weed2
            n_ps = np.sum(ix_weed)

            print(f"      -> {n_ps} PS kept after dropping noisy pixels")

        if not os.path.exists(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, 'no_ps_info.mat')):
            self.stamps_step_no_ps = np.zeros((5, 1))
        else:
            self.stamps_step_no_ps = sio.loadmat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, 'no_ps_info.mat'))['stamps_step_no_ps']
        
        if n_ps == 0:
            print('   -> No PS left. Updating the stamps log for this')
            self.stamps_step_no_ps[3] = 1
            
        self._save_ps_info()
            
        # Save the results
        weedname = f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/weed{self.psver}.mat'
        sio.savemat(weedname, {'ix_weed': ix_weed, 'ix_weed2': ix_weed2, 'ps_std': ps_std, 'ps_max': ps_max, 'ifg_index': ifg_index})

        coh_ps = coh_ps2[ix_weed.flatten()]
        K_ps = K_ps2[ix_weed.flatten()]
        C_ps = C_ps2[ix_weed.flatten()]
        ph_patch = ph_patch2[ix_weed.flatten(), :]
        if ph_res2.shape[1] > 0 :
            ph_res = ph_res2[ix_weed.flatten(), :]
        else:
            ph_res = ph_res2

        pmname = f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/pm{self.psver+1}.mat'
        sio.savemat(pmname, {'ph_patch': ph_patch, 'ph_res': ph_res, 'coh_ps': coh_ps, 'K_ps': K_ps, 'C_ps': C_ps})
        del ph_patch
        del ph_res
        del coh_ps
        del K_ps
        del C_ps
        del ph_patch2
        del ph_res2
        del coh_ps2
        del K_ps2
        del C_ps2

        ph2 = ph2[ix_weed.flatten(), :]
        ph = ph2
        phname = f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/ph{self.psver+1}.mat'
        sio.savemat(phname, {'ph': ph})
        del ph

        xy2 = xy2[ix_weed.flatten(), :]
        ij2 = ij2[ix_weed.flatten(), :]
        lonlat2 = lonlat2[ix_weed.flatten(), :]
        psname = f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/ps{self.psver+1}.mat'
        save_dict = {
            'xy': xy2,
            'ij': ij2,
            'lonlat': lonlat2,
            'n_ps': ph2.shape[0]
        }
        # Add remaining keys from ps
        for key in ps.keys():
            if key not in ['xy', 'ij', 'lonlat', 'n_ps']:
                save_dict[key] = ps[key]
        sio.savemat(psname, save_dict)
        del ps
        del xy2
        del ij2
        del lonlat2

        if os.path.exists(hgtname):
            hgt = hgt[ix_weed]
            hgtname = f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/hgt{self.psver+1}.mat'
            sio.savemat(hgtname, {'hgt': hgt})
            del hgt

        if os.path.exists(laname):
            la = sio.loadmat(laname)
            la = la['la'].reshape(-1, 1)[ix2]
            if all_da_flag:
                laothername = f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/la_other{self.psver+1}.mat'
                lao = sio.loadmat(laothername)
                la = np.concatenate([la, lao['la_other'][ix_other]])
                del lao
            la = la[ix_weed]
            sio.savemat(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/la{self.psver+1}.mat', {'la': la})
            del la

        if os.path.exists(incname):
            inc = sio.loadmat(incname)
            inc = inc['inc'][ix2]
            if all_da_flag:
                incothername = f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/inc_other{self.psver+1}.mat'
                inco = sio.loadmat(incothername)
                inc = np.concatenate([inc, inco['inc_other'][ix_other]])
                del inco
            inc = inc[ix_weed]
            sio.savemat(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/inc{self.psver+1}.mat', {'inc': inc})
            del inc

        if os.path.exists(bpname):
            bp = sio.loadmat(bpname)
            bperp_mat = bp['bperp_mat'][ix2, :]
            del bp
            if all_da_flag:
                bpo = sio.loadmat(bpothername)
                bperp_mat = np.concatenate([bperp_mat, bpo['bperp_other'][ix_other, :]])
                del bpo
            bperp_mat = bperp_mat[ix_weed.flatten(), :]
            sio.savemat(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/bp{self.psver+1}.mat', {'bperp_mat': bperp_mat})
            del bperp_mat

        if os.path.exists(f'scla_smooth{self.psver+1}.mat'):
            os.remove(f'scla_smooth{self.psver+1}.mat')
        if os.path.exists(f'scla{self.psver+1}.mat'):
            os.remove(f'scla{self.psver+1}.mat')
        if os.path.exists(f'scla_smooth_sb{self.psver+1}.mat'):
            os.remove(f'scla_smooth_sb{self.psver+1}.mat')
        if os.path.exists(f'scla_sb{self.psver+1}.mat'):
            os.remove(f'scla_sb{self.psver+1}.mat')
        if os.path.exists(f'aps{self.psver+1}.mat'):
            os.remove(f'aps{self.psver+1}.mat')
        if os.path.exists(f'scn{self.psver+1}.mat'):
            os.remove(f'scn{self.psver+1}.mat')

        self._update_psver(self.psver + 1, self.patch_dir)

        """Plot phase patch for each interferogram"""
        # Load data
        ph = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/ph{self.psver}.mat")['ph']
        ps = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/ps{self.psver}.mat")
        lonlat = ps['lonlat']

        # Get number of interferograms
        n_ifg = ph.shape[1]

        # Create subplot grid
        n_rows = int(np.ceil(np.sqrt(n_ifg)))
        n_cols = int(np.ceil(n_ifg / n_rows))

        plt.figure(figsize=(10*n_cols, 6*n_rows))

        # Plot phase for each interferogram
        for i in range(n_ifg):
            plt.subplot(n_rows, n_cols, i+1)
            
            # Get valid data points (non-zero/non-nan values)
            valid_idx = ~np.isnan(ph[:,i]) & (ph[:,i] != 0)
            
            plt.scatter(lonlat[:,0], lonlat[:,1], 
                    c=np.angle(ph[:, i]),
                    cmap='hsv', vmin=-np.pi, vmax=np.pi, s=3)
            plt.colorbar(label='Phase (rad)')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(f'Interferogram {i+1}')

        plt.tight_layout()
        plt.savefig(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/4_ps_ph_{self.psver}.png')
        plt.close()

    ########## Correct PS phase ##########
    
    def _ps_correct_phase(self):
        print("   -> Correcting PS phase for look angle error...")
        small_baseline_flag = self.parms.get('small_baseline_flag')

        self._update_psver(2, self.patch_dir)
        psname = f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/ps{self.psver}.mat'
        phname = f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/ph{self.psver}.mat'
        pmname = f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/pm{self.psver}.mat'
        rcname = f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/rc{self.psver}.mat'
        bpname = f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/bp{self.psver}.mat'

        ps = sio.loadmat(psname)
        pm = sio.loadmat(pmname)
        bp = sio.loadmat(bpname)

        if os.path.exists(phname):
            ph = sio.loadmat(phname)['ph']
        else:
            ph = ps['ph']

        K_ps = pm['K_ps']
        C_ps = pm['C_ps']
        master_ix = ps['master_ix'][0][0]

        if small_baseline_flag == 'y':
            ph_rc = ph * np.exp(-1j * (np.tile(K_ps, (1, ps['n_ifg'][0][0])) * bp['bperp_mat']))
            sio.savemat(rcname, {'ph_rc': ph_rc})
        else:
            # Insert a column of zeros at the master index in bperp_mat
            bperp_mat = np.hstack([
                bp['bperp_mat'][:, :master_ix],
                np.zeros((ps['n_ps'][0][0], 1), dtype=np.float32),
                bp['bperp_mat'][:, master_ix:]
            ])

            # Compute ph_rc (range-corrected phase)
            K_ps_tiled = np.tile(K_ps, (ps['n_ifg'][0][0], 1)).ravel().reshape(ps['n_ps'][0][0], ps['n_ifg'][0][0])
            C_ps_tiled = np.tile(C_ps, (ps['n_ifg'][0][0], 1)).ravel().reshape(ps['n_ps'][0][0], ps['n_ifg'][0][0])
            ph_rc = ph * np.exp(-1j * (K_ps_tiled * bperp_mat + C_ps_tiled))

            # Prepare ph_reref with 1s at master index
            ph_reref = np.hstack([
                pm['ph_patch'][:, :master_ix].astype(np.complex64),
                np.ones((ps['n_ps'][0][0], 1), dtype=np.complex64), 
                pm['ph_patch'][:, master_ix:].astype(np.complex64)
            ])

            # Save to .mat file
            sio.savemat(rcname, {'ph_rc': ph_rc, 'ph_reref': ph_reref})

        """Plot phase patch for each interferogram"""
        # Get number of interferograms
        ps2 = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/ps{self.psver}.mat")
        lonlat = ps2['lonlat']
        n_ifg = ph.shape[1]

        # Create subplot grid
        n_rows = int(np.ceil(np.sqrt(n_ifg)))
        n_cols = int(np.ceil(n_ifg / n_rows))

        if 'ph_rc' in locals():
            plt.figure(figsize=(10*n_cols, 6*n_rows))

            # Plot phase for each interferogram
            for i in range(n_ifg):
                plt.subplot(n_rows, n_cols, i+1)
                plt.scatter(lonlat[:,0], lonlat[:,1], 
                        c=np.angle(ph[:, i]),
                        cmap='hsv', vmin=-np.pi, vmax=np.pi, s=1)
                plt.colorbar(label='Phase (rad)')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.title(f'Interferogram {i+1}')

            plt.tight_layout()
            plt.savefig(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/5_ps_ph_rc_{self.psver}.png')
            plt.close()
        if 'ph_reref' in locals():
            plt.figure(figsize=(10*n_cols, 6*n_rows))

            # Plot phase for each interferogram
            for i in range(n_ifg):
                plt.subplot(n_rows, n_cols, i+1)
                plt.scatter(lonlat[:,0], lonlat[:,1], 
                        c=np.angle(ph[:, i]),
                        cmap='hsv', vmin=-np.pi, vmax=np.pi, s=1)
                plt.colorbar(label='Phase (rad)')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.title(f'Interferogram {i+1}')

            plt.tight_layout()
            plt.savefig(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/5_ps_ph_reref_{self.psver}.png')
            plt.close()

    ########## Merge patches ##########
    
    def _ps_merge_patches(self):
        print("   -> Merging patches...")
        self._update_psver(2)
        small_baseline_flag = self.parms.get('small_baseline_flag')
        grid_size = self.parms.get('merge_resample_size')
        merge_stdev = self.parms.get('merge_standard_dev')
        phase_accuracy = 10 * np.pi / 180 # minimum possible accuracy for a pixel
        min_weight = 1 / merge_stdev ** 2
        max_coh = np.abs(np.sum(np.exp(1j * np.random.randn(1000, 1) * phase_accuracy))) / 1000

        psname = f'ps{self.psver}.mat'
        phname = f'ph{self.psver}.mat'
        rcname = f'rc{self.psver}.mat'
        pmname = f'pm{self.psver}.mat'
        phuwname = f'phuw{self.psver}.mat'
        sclaname = f'scla{self.psver}.mat'
        sclasbname = f'scla_sb{self.psver}.mat'
        scnname = f'scn{self.psver}.mat'
        bpname = f'bp{self.psver}.mat'
        laname = f'la{self.psver}.mat'
        incname = f'inc{self.psver}.mat'
        hgtname = f'hgt{self.psver}.mat'
        
        if len(self.patch_dirs) > 1:
            if os.path.exists(f"{self.config['processing_parameters']['current_result']}/patch.list"):
                with open(f"{self.config['processing_parameters']['current_result']}/patch.list", 'r') as f:
                    dirname = [line.strip() for line in f.readlines()]
        else:
            dirname = [self.patch_dir]

        n_patch = len(dirname)
        remove_ix = np.array([], dtype=np.bool_)

        for i in tqdm(range(n_patch), desc="      -> Processing patches"):    
            if not os.path.exists(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}"):
                continue
            self.ps = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{psname}")
            n_ifg = self.ps['n_ifg'][0][0]
            if 'n_image' in self.ps:
                n_image = self.ps['n_image'][0][0]
            else:
                n_image = self.ps['n_ifg'][0][0]
                
            patch_ij = open(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/patch_noover.in", 'r').read().splitlines()
            patch_ij = [int(x) for x in patch_ij]
            ix = (
                (self.ps['ij'][:, 1] >= patch_ij[2] - 1) &
                (self.ps['ij'][:, 1] <= patch_ij[3] - 1) &
                (self.ps['ij'][:, 2] >= patch_ij[0] - 1) &
                (self.ps['ij'][:, 2] <= patch_ij[1] - 1)
            )
            if np.sum(ix) == 0:
                ix_no_ps = True
            else:
                ix_no_ps = False
            
            if grid_size == 0:
                ij = np.zeros_like(self.ps['ij'])[:, 1:]
                C, IA, IB = np.intersect1d(self.ps['ij'][ix, 1:], ij, return_indices=True)
                remove_ix = np.concatenate([remove_ix, IB])
                C, IA, IB = np.intersect1d(self.ps['ij'][:, 1:], ij, return_indices=True)
                ix_ex = np.ones(self.ps['n_ps'][0][0], dtype=np.bool_)
                ix_ex[IA] = False
                ix[ix_ex] = True
                
                # Update pixels coordinates
                ij = self.ps['ij'][ix, 1:3]
                lonlat = self.ps['lonlat'][ix]

                # Update phase
                if os.path.exists(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{phname}"):
                    phin = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{phname}")
                    ph_w = phin['ph']
                    del phin
                else:
                    ph_w = self.ps['ph']
                ph = ph_w[ix, :]
                rc = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{rcname}")
                ph_rc = rc['ph_rc'][ix,:]
                if small_baseline_flag != 'y':
                    ph_reref = rc['ph_reref'][ix,:]
                pm = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{pmname}")
                ph_patch = pm['ph_patch'][ix,:]
                if 'ph_res' in pm:
                    ph_res = pm['ph_res'][ix,:]
                if 'K_ps' in pm:
                    K_ps = pm['K_ps'].T[ix,:]
                if 'C_ps' in pm:
                    C_ps = pm['C_ps'].T[ix,:]
                if 'coh_ps' in pm:
                    coh_ps = pm['coh_ps'].T[ix,:]
                bp = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{bpname}")
                bperp_mat = bp['bperp_mat'][ix,:]
                if os.path.exists(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{laname}"):
                    lain = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{laname}")
                    la = lain['la'].T[ix,:]
                
                if os.path.exists(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{incname}"):
                    incin = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{incname}")
                    inc = incin['inc'].T[ix,:]
            
                if os.path.exists(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{hgtname}"):
                    hgtin = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{hgtname}")
                    hgt = hgtin['hgt'].T[ix,:]

                if os.path.exists(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{phuwname}"):
                    phuw = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{phuwname}")
                    if not len(C) == 0:
                        ph_uw_diff = np.mean(phuw['ph_uw'][IA,:] - ph_uw[IB,:], axis=1)
                        if small_baseline_flag != 'y':
                            ph_uw_diff = np.round(ph_uw_diff / 2 / np.pi) * 2 * np.pi
                    else:
                        ph_uw_diff = np.zeros((1, phuw['ph_uw'].shape[1]))
                    ph_uw = phuw['ph_uw'][ix,:] - np.tile(ph_uw_diff, (sum(ix), 1))
                    del phuw
                else:
                    ph_uw = np.zeros((sum(ix), n_image), dtype=np.float32)
                    
                if os.path.exists(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{sclaname}"):
                    scla = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{sclaname}")
                    if not len(C) == 0:
                        ph_scla_diff = np.mean(scla['ph_scla'][IA,:] - ph_scla[IB,:])
                        K_ps_diff = np.mean(scla['K_ps_uw'][IA,:] - K_ps_uw[IB,:])
                        C_ps_diff = np.mean(scla['C_ps_uw'][IA,:] - C_ps_uw[IB,:])
                    else:
                        ph_scla_diff = np.zeros((1, phuw['ph_uw'].shape[1]))
                        K_ps_diff = 0
                        C_ps_diff = 0
                    ph_scla = scla['ph_scla'][ix,:] - np.tile(ph_scla_diff, (sum(ix), 1))
                    K_ps_uw = scla['K_ps_uw'][ix,:] - np.tile(K_ps_diff, (sum(ix), 1))
                    C_ps_uw = scla['C_ps_uw'][ix,:] - np.tile(C_ps_diff, (sum(ix), 1))
                    del scla
        
                if small_baseline_flag == 'y':
                    if os.path.exists(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{sclasbname}"):
                        sclasb = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{sclasbname}")
                        ph_scla_diff = np.mean(sclasb['ph_scla'][IA,:] - ph_scla_sb[IB,:])
                        K_ps_diff = np.mean(sclasb['K_ps_uw'][IA,:] - K_ps_uw_sb[IB,:])
                        C_ps_diff = np.mean(sclasb['C_ps_uw'][IA,:] - C_ps_uw_sb[IB,:])
                        ph_scla_sb = sclasb['ph_scla'][ix,:] - np.tile(ph_scla_diff, (sum(ix), 1))
                        K_ps_uw_sb = sclasb['K_ps_uw'][ix,:] - np.tile(K_ps_diff, (sum(ix), 1))
                        C_ps_uw_sb = sclasb['C_ps_uw'][ix,:] - np.tile(C_ps_diff, (sum(ix), 1))
                        del sclasb
                
                if os.path.exists(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{scnname}"):
                    scn = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{scnname}")
                    if not len(C) == 0:
                        ph_scn_diff = np.mean(scn['ph_scn_slave'][IA,:] - ph_scn_slave[IB,:])
                    else:
                        ph_scn_diff = np.zeros((1, scn['ph_scn_slave'].shape[1]))
                    ph_scn_slave = scn['ph_scn_slave'][ix,:] - np.tile(ph_scn_diff, (sum(ix), 1))
                    del scn
            elif grid_size != 0 and ix_no_ps == False:
                # Initialize and compute grid indices
                g_ij = np.zeros((np.sum(ix), 2), dtype=int)
                xy_min = np.min(self.ps['xy'][ix, :], axis=0)
                g_ij[:, 0] = np.ceil((self.ps['xy'][ix, 2] - xy_min[2] + 1e-9) / grid_size).astype(int)
                g_ij[:, 1] = np.ceil((self.ps['xy'][ix, 1] - xy_min[1] + 1e-9) / grid_size).astype(int)

                # Unique grid positions
                g_ij_unique, indices, g_ix = np.unique(g_ij, axis=0, return_index=True, return_inverse=True)
                sort_ix = np.argsort(g_ix)
                g_ix = g_ix[sort_ix]
                ix = np.where(ix)[0][sort_ix]

                # Load patch residuals
                pm = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{pmname}")
                pm['ph_res'] = np.angle(np.exp(1j * (pm['ph_res'] - np.tile(pm['C_ps'], (pm['ph_res'].shape[1], 1)).reshape(pm['ph_res'].shape))))
                if small_baseline_flag != 'y':
                    pm['ph_res'] = np.hstack((pm['ph_res'], pm['C_ps'].T))

                sigsq_noise = np.var(pm['ph_res'], axis=1)
                coh_ps_all = np.abs(np.sum(np.exp(1j * pm['ph_res']), axis=1)) / n_ifg
                coh_ps_all[coh_ps_all > max_coh] = max_coh
                sigsq_noise[sigsq_noise < phase_accuracy**2] = phase_accuracy**2

                ps_weight = 1.0 / sigsq_noise[ix]
                ps_snr = 1.0 / (1.0 / coh_ps_all[ix]**2 - 1.0)

                # Grid filtering
                l_ix = np.append(np.where(np.diff(g_ix))[0], len(g_ix))
                f_ix = np.append([0], l_ix[:-1] + 1)
                n_ps_g = len(f_ix)

                weightsave = np.zeros((n_ps_g, 1), dtype=np.float32)
                for k in range(n_ps_g):
                    weights = ps_weight[f_ix[k]:l_ix[k] + 1]
                    weightsum = np.sum(weights)
                    weightsave[k] = weightsum
                    if weightsum < min_weight:
                        ix[f_ix[k]:l_ix[k] + 1] = 0

                # Final filtering
                g_ix = g_ix[ix > 0]
                if len(g_ix) == 0:
                    ix_no_ps = True  # All PS rejected due to low weight

                l_ix = np.append(np.where(np.diff(g_ix))[0], len(g_ix) - 1)
                f_ix = np.append([0], l_ix[:-1] + 1)
                ps_weight = ps_weight[ix > 0]
                ps_snr = ps_snr[ix > 0]
                ix = ix[ix > 0]
                n_ps_g = len(f_ix)

                # Update coordinates
                ij_g = np.zeros((n_ps_g, 2))
                lonlat_g = np.zeros((n_ps_g, 2))
                self.ps['ij'] = self.ps['ij'][ix]
                self.ps['lonlat'] = self.ps['lonlat'][ix]
                for k in range(n_ps_g):
                    weights = np.tile(ps_weight[f_ix[k]:l_ix[k] + 1][:, np.newaxis], (1, 2))
                    ij_g[k, :] = np.round(np.sum(self.ps['ij'][f_ix[k]:l_ix[k] + 1, 1:3] * weights, axis=0) / np.sum(weights[:, 0]))
                    lonlat_g[k, :] = np.sum(self.ps['lonlat'][f_ix[k]:l_ix[k] + 1] * weights, axis=0) / np.sum(weights[:, 0])

                ij = ij_g
                lonlat = lonlat_g
                
                if os.path.exists(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{phname}"):
                    phin = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{phname}")
                    ph_w = phin['ph']
                    del phin
                elif 'ph' in self.ps:
                    ph_w = self.ps['ph']

                ph_w = ph_w[ix, :]
                ph_g = np.zeros((n_ps_g, n_ifg), dtype=np.complex64)
                for k in range(n_ps_g):
                    weights = np.tile(ps_snr[f_ix[k]:l_ix[k] + 1][:, np.newaxis], (1, n_ifg))
                    ph_g[k, :] = np.sum(ph_w[f_ix[k]:l_ix[k] + 1, :] * weights, axis=0) / np.sum(weights[:, 0])
                ph = ph_g
                del ph_g
                
                rc = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{rcname}")
                if ix_no_ps == False:
                    rc['ph_rc'] = rc['ph_rc'][ix, :]
                    ph_g = np.zeros((n_ps_g, n_ifg), dtype=np.complex64)
                    if small_baseline_flag != 'y':
                        rc['ph_reref'] = rc['ph_reref'][ix, :]
                        ph_reref_g = np.zeros((n_ps_g, n_ifg), dtype=np.complex64)
                    for k in range(n_ps_g):
                        weights = np.tile(ps_snr[f_ix[k]:l_ix[k] + 1][:, np.newaxis], (1, n_ifg))
                        ph_g[k, :] = np.sum(rc['ph_rc'][f_ix[k]:l_ix[k] + 1, :] * weights, axis=0) / np.sum(weights[:, 0])
                        if small_baseline_flag != 'y':
                            ph_reref_g[k, :] = np.sum(rc['ph_reref'][f_ix[k]:l_ix[k] + 1, :] * weights, axis=0) / np.sum(weights[:, 0])
                    ph_rc = np.complex128(ph_g)
                    del ph_g
                    if small_baseline_flag != 'y':
                        ph_reref = ph_reref_g
                        del ph_reref_g
                del rc
                pm = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{pmname}")
                pm['ph_patch'] = pm['ph_patch'][ix, :]
                ph_g = np.zeros((n_ps_g, pm['ph_patch'].shape[1]), dtype=np.complex64)
                if 'ph_res' in pm:
                    pm['ph_res'] = pm['ph_res'][ix, :]
                    ph_res_g = ph_g
                if 'K_ps' in pm:
                    pm['K_ps'] = pm['K_ps'].T[ix, :]
                    K_ps_g = np.zeros((n_ps_g, 1), dtype=np.float32)
                if 'C_ps' in pm:
                    pm['C_ps'] = pm['C_ps'].T[ix, :]
                    C_ps_g = np.zeros((n_ps_g, 1), dtype=np.float32)
                if 'coh_ps' in pm:
                    pm['coh_ps'] = pm['coh_ps'].T[ix, :]
                    coh_ps_g = np.zeros((n_ps_g, 1), dtype=np.float32)
                
                if not os.path.exists(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/ph_g_temp.mat"):
                    def process_group(k, f_ix, l_ix, ps_snr, ps_weight, pm, ph_g, ph_res_g, coh_ps_g, K_ps_g, C_ps_g):
                        weights = np.tile(ps_snr[f_ix[k]:l_ix[k]+1][:, np.newaxis], (1, ph_g.shape[1]))
                        ph_g[k, :] = np.sum(pm['ph_patch'][f_ix[k]:l_ix[k]+1, :] * weights, axis=0) / np.sum(weights, axis=0)
                        if 'ph_res' in pm:
                            ph_res_g[k, :] = np.sum(pm['ph_res'][f_ix[k]:l_ix[k]+1, :] * weights, axis=0) / np.sum(weights, axis=0)
                        if 'coh_ps' in pm:
                            snr = np.sqrt(np.sum(weights[:, 0]**2, axis=0))
                            coh_ps_g[k] = np.sqrt(1./(1+1./snr))
                        weights = ps_weight[f_ix[k]:l_ix[k]+1][:, np.newaxis]
                        if 'K_ps' in pm:
                            K_ps_g[k] = np.sum(pm['K_ps'][f_ix[k]:l_ix[k]+1, :] * weights, axis=0) / np.sum(weights)
                        if 'C_ps' in pm:
                            C_ps_g[k] = np.sum(pm['C_ps'][f_ix[k]:l_ix[k]+1, :] * weights, axis=0) / np.sum(weights)

                        if np.sum(np.sum(np.isnan(C_ps_g)))>0 or np.sum(np.sum(np.isnan(weights)))>0 or np.sum(np.sum(np.isnan(np.abs(ph_g))))>0 or np.sum(np.sum(np.isnan(K_ps_g)))>0 or np.sum(np.sum(np.isnan(coh_ps_g)))>0 or np.sum(np.sum(np.isnan(snr)))>0:
                            import pdb
                            pdb.set_trace()

                    def process_batch(batch_indices, *args):
                        for k in batch_indices:
                            process_group(k, *args)

                    batch_size = 200  # Process groups in batches of 100
                    with ThreadPoolExecutor(max_workers=self.config["computing_resources"]["cpu"]) as executor:
                        process_args = (f_ix, l_ix, ps_snr, ps_weight, pm, ph_g, ph_res_g, coh_ps_g, K_ps_g, C_ps_g)
                        
                        # Create batches of indices
                        batches = [range(i, min(i + batch_size, n_ps_g)) for i in range(0, n_ps_g, batch_size)]
                        
                        # Process each batch in parallel
                        futures = [executor.submit(process_batch, batch, *process_args) for batch in batches]
                        
                        # Show progress
                        for _ in tqdm(concurrent.futures.as_completed(futures),
                                    total=len(batches),
                                    desc="         -> Collocating phase over groups",
                                    unit=" batch"):
                            pass
                    sio.savemat(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/ph_g_temp.mat", {'ph_g': ph_g})
                else:
                    ph_g = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/ph_g_temp.mat")['ph_g']

                ph_patch = ph_g
                del ph_g
                if 'ph_res' in pm:
                    ph_res = ph_res_g
                    del ph_res_g
                if 'K_ps' in pm:
                    K_ps = K_ps_g
                    del K_ps_g
                if 'C_ps' in pm:
                    C_ps = C_ps_g
                    del C_ps_g
                if 'coh_ps' in pm:
                    coh_ps = coh_ps_g
                    del coh_ps_g
                del pm

                bp = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{bpname}")
                bperp_g = np.zeros((n_ps_g, bp['bperp_mat'].shape[1]), dtype=np.float32)
                bp['bperp_mat'] = bp['bperp_mat'][ix,:]
                for k in tqdm(range(n_ps_g), desc="         -> Collocating bperp over groups", unit=" group"):
                    weights = np.tile(ps_weight[f_ix[k]:l_ix[k] + 1][:, np.newaxis], (1, bperp_g.shape[1]))
                    weights[weights == 0] = 1e-9
                    bperp_g[k,:] = np.sum(bp['bperp_mat'][f_ix[k]:l_ix[k] + 1, :] * weights, axis=0) / np.sum(weights)
                bperp_mat = bperp_g
                del bperp_g
                del bp

                if os.path.exists(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{laname}"):
                    lain = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{laname}")
                    la_g = np.zeros((n_ps_g, 1), dtype=np.float32)
                    lain['la'] = lain['la'].T[ix,:]
                    for k in tqdm(range(n_ps_g), desc="         -> Collocating line of sight over groups", unit=" group"):
                        weights = ps_weight[f_ix[k]:l_ix[k] + 1][:, np.newaxis]
                        la_g[k] = np.sum(lain['la'][f_ix[k]:l_ix[k] + 1] * weights, axis=0) / np.sum(weights)
                    la = la_g
                    del la_g
                    del lain

                if os.path.exists(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{incname}"):
                    incin = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{incname}")
                    inc_g = np.zeros((n_ps_g, 1), dtype=np.float32)
                    incin['inc'] = incin['inc'][ix,:]
                    for k in tqdm(range(n_ps_g), desc="         -> Collocating incidence over groups", unit=" group"):
                        weights = ps_weight[f_ix[k]:l_ix[k] + 1][:, np.newaxis]
                        inc_g[k] = np.sum(incin['inc'][f_ix[k]:l_ix[k] + 1] * weights, axis=0) / np.sum(weights)
                    inc = inc_g
                    del inc_g
                    del incin

                if os.path.exists(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{hgtname}"):
                    hgtin = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{dirname[i]}/{hgtname}")
                    hgt_g = np.zeros((n_ps_g, 1), dtype=np.float64)
                    hgtin['hgt'] = hgtin['hgt'].T[ix,:]
                    for k in tqdm(range(n_ps_g), desc="         -> Collocating height over groups", unit=" group"):
                        weights = ps_weight[f_ix[k]:l_ix[k] + 1][:, np.newaxis]
                        with np.errstate(divide='ignore', invalid='ignore'):
                            hgt_g[k] = np.sum(hgtin['hgt'][f_ix[k]:l_ix[k] + 1] * weights, axis=0) / np.sum(weights)
                    hgt = hgt_g
                    del hgt_g
                    del hgtin
                        
        ps_new = self.ps
        n_ps_orig = ij.shape[0]
        keep_ix = np.ones(n_ps_orig, dtype=np.bool_)
        keep_ix[remove_ix] = False
        lonlat_save = lonlat
        coh_ps_weed = coh_ps[keep_ix]
        lonlat = lonlat[keep_ix, :]

        I = np.unique(lonlat, axis=0)
        dups = np.setxor1d(I, np.arange(len(lonlat)).reshape(-1, 1)).astype(int)
        keep_ix_num = np.where(keep_ix)[0]

        for i in tqdm(range(len(dups)), desc="   -> Removing duplicate PS on merging", unit=" pixel"):
            dups_ix_weed = np.where((lonlat[:, 0] == lonlat[int(dups[i]), 0]) & (lonlat[:, 1] == lonlat[int(dups[i]), 1]))[0]
            dups_ix = keep_ix_num[dups_ix_weed]
            I = np.argmax(coh_ps_weed[dups_ix_weed])
            keep_ix[dups_ix[np.arange(len(dups_ix)) != I]] = False

        if len(dups) > 0:
            lonlat = lonlat_save[keep_ix, :]
            if np.sum(keep_ix==False) > 0:
                print(f"      -> {np.sum(keep_ix==False)} pixel with duplicate lon/lat dropped")

        del lonlat_save
        ll0 = (np.amax(lonlat, axis=0) + np.amin(lonlat, axis=0)) / 2
        xy = self._llh2local(lonlat.T, ll0).T * 1000

        heading = self.parms.get('heading')
        if heading is None:
            heading = 0
        theta = (180 - heading) * np.pi / 180
        if theta > np.pi:
            theta = theta - 2 * np.pi

        rotm = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        xy = xy.T
        xynew = rotm @ xy
        if max(xynew[0, :]) - min(xynew[0, :]) < max(xy[0, :]) - min(xy[0, :]) and max(xynew[1, :]) - min(xynew[1, :]) < max(xy[1, :]) - min(xy[1, :]):
            xy = xynew
            print(f"   -> Rotating xy by {theta * 180 / np.pi} degrees")
        del xynew

        xy = xy.T.astype(np.float32)
        sort_ix = np.lexsort((xy[:, 0], xy[:, 1]))
        xy = np.column_stack([np.arange(len(xy)).reshape(-1, 1), xy])
        xy[:, 1:] = np.round(xy[:, 1:] * 1000) / 1000
        lonlat = lonlat[sort_ix, :]

        all_ix = np.arange(len(ij)).T
        keep_ix = all_ix[keep_ix]
        sort_ix = keep_ix[sort_ix]

        self.n_ps = len(sort_ix)
        print(f"   -> Writing merged dataset (contains {self.n_ps} pixels)")

        ij = ij[sort_ix,:]
        ph_rc = ph_rc[sort_ix,:]
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            ph_rc[ph_rc!=0] = ph_rc[ph_rc!=0]/np.abs(ph_rc[ph_rc!=0])
        if small_baseline_flag != 'y':
            ph_reref = ph_reref[sort_ix,:]
        
        sio.savemat(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/{rcname}", {'ph_rc': ph_rc, 'ph_reref': ph_reref})
        del ph_rc
        del ph_reref

        if grid_size == 0:
            if ph_uw.shape[0] == n_ps_orig:
                ph_uw = ph_uw[sort_ix,:]
        else:
            ph_uw = np.array([])
        sio.savemat(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/{phuwname}", {'ph_uw': ph_uw})
        del ph_uw

        ph_patch = ph_patch[sort_ix,:]
        if ph_res.shape[0] == n_ps_orig:
            ph_res = ph_res[sort_ix,:]
        else:
            ph_res = np.array([])
        if K_ps.shape[0] == n_ps_orig:
            K_ps = K_ps[sort_ix,:]
        else:
            K_ps = np.array([])
        if C_ps.shape[0] == n_ps_orig:
            C_ps = C_ps[sort_ix,:]
        else:
            C_ps = np.array([])
        if coh_ps.shape[0] == n_ps_orig:
            coh_ps = coh_ps[sort_ix,:]
        else:
            coh_ps = np.array([])
        sio.savemat(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/{pmname}", {'ph_patch': ph_patch, 'ph_res': ph_res, 'K_ps': K_ps, 'C_ps': C_ps, 'coh_ps': coh_ps})
        del ph_patch
        del ph_res
        del K_ps
        del C_ps
        del coh_ps

        if grid_size == 0:
            if ph_scla.shape[0] == self.n_ps:
                ph_scla = ph_scla[sort_ix,:]
                K_ps_uw = K_ps_uw[sort_ix,:]
                C_ps_uw = C_ps_uw[sort_ix,:]
                sio.savemat(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/{sclaname}", {'ph_scla': ph_scla, 'K_ps_uw': K_ps_uw, 'C_ps_uw': C_ps_uw})
                del ph_scla
                del K_ps_uw
                del C_ps_uw

            if ph_scla_sb.shape[0] == self.n_ps and small_baseline_flag == "y":
                ph_scla = ph_scla_sb[sort_ix,:]
                K_ps_uw = K_ps_uw_sb[sort_ix,:]
                C_ps_uw = C_ps_uw_sb[sort_ix,:]
                sio.savemat(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/{sclasbname}", {'ph_scla': ph_scla, 'K_ps_uw': K_ps_uw, 'C_ps_uw': C_ps_uw})
                del ph_scla
                del K_ps_uw
                del C_ps_uw

            if ph_scn_slave.shape[0] == self.n_ps:
                ph_scn_slave = ph_scn_slave[sort_ix,:]
                sio.savemat(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/{scnname}", {'ph_scn_slave': ph_scn_slave})
                del ph_scn_slave
        else:
            ph_scla = np.zeros((len(sort_ix), n_ifg), dtype=np.float32)
            K_ps_uw = np.array((len(sort_ix), n_ifg), dtype=np.float32)
            C_ps_uw = np.array((len(sort_ix), n_ifg), dtype=np.float32)
            if small_baseline_flag != "y":
                sio.savemat(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/{sclaname}", {'ph_scla': ph_scla, 'K_ps_uw': K_ps_uw, 'C_ps_uw': C_ps_uw})
            else:
                sio.savemat(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/{sclasbname}", {'ph_scla': ph_scla, 'K_ps_uw': K_ps_uw, 'C_ps_uw': C_ps_uw})
            del ph_scla
            del K_ps_uw
            del C_ps_uw

            ph_scn_slave = np.array((len(sort_ix), n_ifg), dtype=np.float32)
            sio.savemat(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/{scnname}", {'ph_scn_slave': ph_scn_slave})
            del ph_scn_slave

        if ph.shape[0] == n_ps_orig:
            ph = ph[sort_ix,:]
        else:
            ph = np.array([])
        sio.savemat(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/{phname}", {'ph': ph})
        del ph

        if la.shape[0] == n_ps_orig:
            la = la[sort_ix,:]
        else:
            la = np.array([])
        sio.savemat(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/{laname}", {'la': la})
        del la

        if os.path.exists(f"{self.config['processing_parameters']['current_result']}/{incname}"):
            if inc.shape[0] == n_ps_orig:
                inc = inc[sort_ix,:]
            else:
                inc = np.array([])
            sio.savemat(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/{incname}", {'inc': inc})
            del inc

        if hgt.shape[0] == n_ps_orig:
            hgt = hgt[sort_ix,:]
        else:
            hgt = np.array([])
        sio.savemat(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/{hgtname}", {'hgt': hgt})
        del hgt

        bperp_mat = bperp_mat[sort_ix,:]
        sio.savemat(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/{bpname}", {'bperp_mat': bperp_mat})
        del bperp_mat

        ps_new['n_ps'] = self.n_ps
        ps_new['ij'] = np.hstack([np.arange(1, self.n_ps + 1).reshape(-1, 1), ij])
        ps_new['xy'] = xy
        ps_new['lonlat'] = lonlat
        ps_dict = {}
        for key, value in ps_new.items():
            ps_dict[key] = value
        sio.savemat(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/{psname}", ps_dict)
        del ps_new
        del ps_dict

        self._update_psver(2, self.patch_dir)
        
        if os.path.exists(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/mean_amp.flt"):
            os.remove(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/mean_amp.flt")
        if os.path.exists(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/amp_mean.mat"):
            os.remove(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/amp_mean.mat")

    def _ps_calc_ifg_std(self):
        print("   -> Estimating noise standard deviation (degrees)...")
        small_baseline_flag = self.parms.get('small_baseline_flag')
        patch_dir = os.path.join(self.config['processing_parameters']['current_result'], self.patch_dir)
        psver = self.psver
        psname = f'ps{psver}.mat'
        phname = f'ph{psver}.mat'
        pmname = f'pm{psver}.mat'
        bpname = f'bp{psver}.mat'
        ifgstdname = f'ifgstd{psver}.mat'

        ps = sio.loadmat(os.path.join(patch_dir, psname))
        pm = sio.loadmat(os.path.join(patch_dir, pmname))
        bp = sio.loadmat(os.path.join(patch_dir, bpname))
        
        if os.path.exists(os.path.join(patch_dir, phname)):
            ph = sio.loadmat(os.path.join(patch_dir, phname))['ph']
        else:
            ph = ps['ph']

        # Read MAX_PERP from config
        max_bperp = float(self.config['processing_parameters']['max_perp'])
        if max_bperp is None:
            print("WARNING: MAX_PERP not found in the file.")
            max_bperp = 150.0  # fallback

        bperp_values = ps['bperp'].flatten()
        toremove_bperp_indices = np.where(np.abs(bperp_values) >= max_bperp)[0]
        print(f"   -> Removing {len(toremove_bperp_indices)} interferograms with |bperp| ≥ {max_bperp}")

        n_ps = ps['xy'].shape[0]
        master_ix = np.sum(ps['master_day'] > ps['day']) + 1

        # Handle overflow by computing in chunks
        chunk_size = 1000
        n_chunks = (n_ps + chunk_size - 1) // chunk_size
        ph_diff = np.zeros((n_ps, ps['n_ifg'][0][0]), dtype=np.float64)

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_ps)
            
            if small_baseline_flag == 'y':
                # Process chunk
                ph_chunk = ph[start_idx:end_idx]
                ph_patch_chunk = pm['ph_patch'][start_idx:end_idx]
                K_ps_chunk = pm['K_ps'][start_idx:end_idx]
                
                # Split complex multiplication into real and imaginary parts
                ph_conj = ph_chunk * np.conj(ph_patch_chunk)
                exp_term = np.exp(-1j * (np.tile(K_ps_chunk, (1, ps['n_ifg'][0][0])) * bp['bperp_mat'][start_idx:end_idx]))
                ph_diff[start_idx:end_idx] = np.angle(ph_conj * exp_term)
                
            else:
                bperp_mat = np.hstack([
                    bp['bperp_mat'][start_idx:end_idx, :ps['master_ix'][0][0]],
                    np.zeros((end_idx-start_idx, 1), dtype=np.float32),
                    bp['bperp_mat'][start_idx:end_idx, ps['master_ix'][0][0]:]
                ])
                ph_patch = np.hstack([
                    pm['ph_patch'][start_idx:end_idx, :master_ix],
                    np.ones((end_idx-start_idx, 1), dtype=np.complex128),
                    pm['ph_patch'][start_idx:end_idx, master_ix:]
                ])
                
                # Split complex multiplication into real and imaginary parts
                ph_conj = ph[start_idx:end_idx] * np.conj(ph_patch)
                exp_term = np.exp(-1j * (
                    np.tile(pm['K_ps'][start_idx:end_idx], (1, ps['n_ifg'][0][0])) * bperp_mat + 
                    np.tile(pm['C_ps'][start_idx:end_idx], (1, ps['n_ifg'][0][0]))
                ))
                ph_diff[start_idx:end_idx] = np.angle(ph_conj * exp_term)

        ifg_std = np.sqrt(np.sum(ph_diff ** 2, axis=0) / n_ps) * 180 / np.pi

        # Print results
        if small_baseline_flag == 'y':
            for i in range(ps['n_ifg'][0][0]):
                print(f"      {i+1:3d} IFG {i+1:3d} {ifg_std[i]:3.2f}")
        else:
            for i in range(ps['n_ifg'][0][0]):
                date = datetime(1,1,1) + timedelta(days=int(ps['day'][0][i])-1)
                print(f"      {i+1:3d} {date.strftime('%Y-%b-%d')} {ifg_std[i]:3.2f}")

        # mean_std = np.mean(ifg_std)
        # high_std_indices = np.where(ifg_std > mean_std)[0]
        # print(f"-> Mean standard deviation: {mean_std:3.2f} degrees")
        # print("-> Interferograms with standard deviation > mean:")
        # for i in high_std_indices:
        #     print(f"   -> {i+1:3d} {ps['day'][0][i]} {ifg_std[i]:3.2f}")

        # # Append high bperp indices to drop_ifg_index
        # drop_ifg_index = np.unique(np.concatenate([high_std_indices, toremove_bperp_indices]))
        # print("drop_ifg_index values:")
        # print(' '.join(str(idx+1) for idx in drop_ifg_index))

        # # Set the updated drop_ifg_index parameter
        # self.parms.set('drop_ifg_index', ' '.join(str(idx+1) for idx in drop_ifg_index))
        # print(f"-> Set drop_ifg_index parameter with {len(drop_ifg_index)} interferograms (including those with |bperp| ≥ {max_bperp})")

        sio.savemat(os.path.join(self.config['processing_parameters']['current_result'], self.patch_dir, ifgstdname), {'ifg_std': ifg_std})
        
        # Plot phase patch for each interferogram
        # Load data
        ph = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/ph{self.psver}.mat")['ph']
        ps = sio.loadmat(f"{self.config['processing_parameters']['current_result']}/{self.patch_dir}/ps{self.psver}.mat")
        lonlat = ps['lonlat']

        # Get number of interferograms
        n_ifg = ph.shape[1]

        # Create subplot grid
        n_rows = int(np.ceil(np.sqrt(n_ifg)))
        n_cols = int(np.ceil(n_ifg / n_rows))

        plt.figure(figsize=(10*n_cols, 6*n_rows))

        # Plot phase for each interferogram
        for i in range(n_ifg):
            plt.subplot(n_rows, n_cols, i+1)
            
            # Get valid data points (non-zero/non-nan values)
            valid_idx = ~np.isnan(ph[:,i]) & (ph[:,i] != 0)
            
            plt.scatter(lonlat[:,0], lonlat[:,1], 
                    c=np.angle(ph[:, i]),
                    cmap='hsv', vmin=-np.pi, vmax=np.pi, s=1)
            plt.colorbar(label='Phase (rad)')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(f'Interferogram {i+1}')

        plt.tight_layout()
        plt.savefig(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/5_ps_ph_merge_{self.psver}.png')
        plt.close()
        
    ########## Unwraping phase ##########
    def _ps_plot_tca(self, aps, aps_flag):
        """Process and return the selected APS correction and a string describing the correction.
        
        Args:
            aps: Dictionary containing APS correction data
            aps_flag: String or integer indicating which APS correction to use
            
        Returns:
            tuple: (aps_corr, fig_name_tca, aps_flag)
                - aps_corr: The selected APS correction
                - fig_name_tca: String describing the correction
                - aps_flag: Integer flag indicating the correction type
        """
        # Convert string flags to integers if needed
        if isinstance(aps_flag, str):
            if aps_flag in ['a_linear', 'a_l']:
                aps_flag = 1  # aps topo correlated linear correction
            elif aps_flag in ['a_powerlaw', 'a_p']:
                aps_flag = 2  # aps topo correlated powerlaw correction
            elif aps_flag in ['a_meris', 'a_m']:
                aps_flag = 3  # aps topo correlated meris correction
            elif aps_flag in ['a_erai', 'a_e']:
                aps_flag = 4  # aps topo correlated ERA-I correction
            elif aps_flag in ['a_erai-h', 'a_eh']:
                aps_flag = 5  # aps hydrostatic ERA-I correction
            elif aps_flag in ['a_erai-w', 'a_ew']:
                aps_flag = 6  # aps topo correlated ERA-I correction
            elif aps_flag in ['a_wrf', 'a_w']:
                aps_flag = 7  # aps topo correlated WRF correction
            elif aps_flag in ['a_wrf-h', 'a_wh']:
                aps_flag = 8  # aps hydrostatic WRF correction
            elif aps_flag in ['a_wrf-w', 'a_ww']:
                aps_flag = 9  # aps topo correlated WRF correction
            elif aps_flag in ['a_meris-ni', 'a_mi']:
                aps_flag = 10  # aps topo correlated MERIS (non-interpolated)
            elif aps_flag in ['a_powerlaw-k', 'a_pk']:
                aps_flag = 11  # Spatial maps of the coefficient relating phase and tropo for power-law
            elif aps_flag in ['a_modis', 'a_M']:
                aps_flag = 12  # aps topo correlated modis correction
            elif aps_flag in ['a_modis-ni', 'a_MI']:
                aps_flag = 13  # aps topo correlated modis (non-interpolated)
            elif aps_flag in ['a_meris+a_erai-h', 'a_m+a_eh']:
                aps_flag = 14  # aps topo correlated MERIS plus a hydrostatic component from ERA-I
            elif aps_flag in ['a_meris-ni+a_erai-h', 'a_mi+a_eh']:
                aps_flag = 15  # aps topo correlated MERIS (non-interpolated) plus a hydrostatic component from ERA-I
            elif aps_flag in ['a_modis+a_erai-h', 'a_M+a_eh']:
                aps_flag = 16  # aps topo correlated modis plus a hydrostatic component from ERA-I
            elif aps_flag in ['a_modis-ni+a_erai-h', 'a_MI+a_eh']:
                aps_flag = 17  # aps topo correlated modis (non-interpolated) plus a hydrostatic component from ERA-I
            elif aps_flag in ['a_linear-man', 'a_lman']:
                aps_flag = 18  # aps topo correlated manually estimated
            elif aps_flag in ['a_recalmodis', 'a_RM']:
                aps_flag = 19  # aps topo correlated modis recalibrated correction
            elif aps_flag in ['a_recalmodis-ni', 'a_RMI']:
                aps_flag = 20  # aps topo correlated modis recalibrated (non-interpolated)
            elif aps_flag in ['a_recalmodis+a_erai-h', 'a_RM+a_eh']:
                aps_flag = 21  # aps topo correlated modis recalibrated plus a hydrostatic component from ERA-I
            elif aps_flag in ['a_recalmodis-ni+a_erai-h', 'a_RMI+a_eh']:
                aps_flag = 22  # aps topo correlated modis recalibrated (non-interpolated) plus a hydrostatic component from ERA-I
            elif aps_flag in ['a_meris+a_wrf-h', 'a_m+a_wh']:
                aps_flag = 23  # aps topo correlated MERIS plus a hydrostatic component from WRF
            elif aps_flag in ['a_meris-ni+a_wrf-h', 'a_mi+a_wh']:
                aps_flag = 24  # aps topo correlated MERIS (non-interpolated) plus a hydrostatic component from WRF
            elif aps_flag in ['a_modis+a_wrf-h', 'a_M+a_wh']:
                aps_flag = 25  # aps topo correlated modis plus a hydrostatic component from WRF
            elif aps_flag in ['a_modis-ni+a_wrf-h', 'a_MI+a_wh']:
                aps_flag = 26  # aps topo correlated modis (non-interpolated) plus a hydrostatic component from WRF
            elif aps_flag in ['a_recalmodis+a_wrf-h', 'a_RM+a_wh']:
                aps_flag = 27  # aps topo correlated modis recalibrated plus a hydrostatic component from WRF
            elif aps_flag in ['a_recalmodis-ni+a_wrf-h', 'a_RMI+a_wh']:
                aps_flag = 28  # aps topo correlated modis recalibrated (non-interpolated) plus a hydrostatic component from WRF
            elif aps_flag == 'a_merra':
                aps_flag = 29  # MERRA correction
            elif aps_flag == 'a_merra2':
                aps_flag = 30  # MERRA-2 correction
            elif aps_flag == 'a_merra-h':
                aps_flag = 31  # MERRA hydro correction
            elif aps_flag == 'a_merra2-h':
                aps_flag = 32  # MERRA-2 hydro correction
            elif aps_flag == 'a_merra-w':
                aps_flag = 33  # MERRA wet correction
            elif aps_flag == 'a_merra2-w':
                aps_flag = 34  # MERRA-2 wet correction
            elif aps_flag == 'a_gacos':
                aps_flag = 35  # GACOS correction
            elif aps_flag == 'a_narr':
                aps_flag = 36  # NARR correction
            elif aps_flag == 'a_narr-h':
                aps_flag = 37  # NARR hydro correction
            elif aps_flag == 'a_narr-w':
                aps_flag = 38  # NARR wet correction
            elif aps_flag in ['a_era5', 'a_e']:
                aps_flag = 39  # ERA5 correction
            elif aps_flag in ['a_era5-h', 'a_eh']:
                aps_flag = 40  # ERA5 hydro correction
            elif aps_flag in ['a_era5-w', 'a_ew']:
                aps_flag = 41  # ERA5 wet correction

        # Select the appropriate correction based on the flag
        if aps_flag == 1:  # linear correction
            aps_corr = aps['ph_tropo_linear']
            fig_name_tca = ' (linear)'
        elif aps_flag == 2:  # powerlaw correlation
            aps_corr = aps['ph_tropo_powerlaw']
            fig_name_tca = ' (powerlaw)'
        elif aps_flag == 3:  # meris correction
            aps_corr = aps['ph_tropo_meris']
            fig_name_tca = ' (meris)'
        elif aps_flag == 4:  # ERA-I & ERA5 correction
            aps_corr = aps['ph_tropo_era']
            fig_name_tca = ' (era)'
        elif aps_flag == 5:  # ERA-I & ERA5 correction
            aps_corr = aps['ph_tropo_era_hydro']
            fig_name_tca = ' (era hydro)'
        elif aps_flag == 6:  # ERA-I & ERA5 correction
            aps_corr = aps['ph_tropo_era_wet']
            fig_name_tca = ' (era wet)'
        elif aps_flag == 7:  # WRF correction
            aps_corr = aps['ph_tropo_wrf']
            fig_name_tca = ' (wrf)'
        elif aps_flag == 8:  # ERA-I correction
            aps_corr = aps['ph_tropo_wrf_hydro']
            fig_name_tca = ' (wrf hydro)'
        elif aps_flag == 9:  # ERA-I correction
            aps_corr = aps['ph_tropo_wrf_wet']
            fig_name_tca = ' (wrf wet)'
        elif aps_flag == 10:  # meris correction (not interpolated)
            aps_corr = aps['ph_tropo_meris_no_interp']
            fig_name_tca = ' (meris)'
        elif aps_flag == 11:  # power-law spatial maps of K
            aps_corr = aps['K_tropo_powerlaw']
            fig_name_tca = ' (powerlaw - spatial K map)'
        elif aps_flag == 12:  # modis correction
            aps_corr = aps['ph_tropo_modis']
            fig_name_tca = ' (modis)'
        elif aps_flag == 13:  # modis correction (not interpolated)
            aps_corr = aps['ph_tropo_modis_no_interp']
            fig_name_tca = ' (modis)'
        elif aps_flag == 14:  # meris correction + ERA hydro component
            ix_no_meris = np.where(np.sum(aps['ph_tropo_meris'], axis=0) == 0)[0]
            aps_corr = aps['ph_tropo_meris'] + aps['ph_tropo_era_hydro']
            aps_corr[:, ix_no_meris] = 0
            fig_name_tca = ' (meris + ERA hydro)'
        elif aps_flag == 15:  # meris correction (not interpolated) + ERA hydro component
            aps_corr = aps['ph_tropo_meris_no_interp'] + aps['ph_tropo_era_hydro']
            fig_name_tca = ' (meris + ERA hydro)'
        elif aps_flag == 16:  # modis correction + ERA hydro component
            ix_no_modis = np.where(np.sum(aps['ph_tropo_modis'], axis=0) == 0)[0]
            aps_corr = aps['ph_tropo_modis'] + aps['ph_tropo_era_hydro']
            aps_corr[:, ix_no_modis] = 0
            fig_name_tca = ' (modis + ERA hydro)'
        elif aps_flag == 17:  # modis correction (not interpolated)
            aps_corr = aps['ph_tropo_modis_no_interp'] + aps['ph_tropo_era_hydro']
            fig_name_tca = ' (modis + ERA hydro)'
        elif aps_flag == 18:  # current implementation of aps correction (manually estimated)
            aps_corr = aps['strat_corr']
            fig_name_tca = ' (linear)'
        elif aps_flag == 19:  # modis correction
            aps_corr = aps['ph_tropo_modis_recal']
            fig_name_tca = ' (modis recal)'
        elif aps_flag == 20:  # modis correction (not interpolated)
            aps_corr = aps['ph_tropo_modis_no_interp_recal']
            fig_name_tca = ' (modis recal)'
        elif aps_flag == 21:  # modis recal correction + ERA hydro component
            ix_no_modis = np.where(np.sum(aps['ph_tropo_modis'], axis=0) == 0)[0]
            aps_corr = aps['ph_tropo_modis_recal'] + aps['ph_tropo_era_hydro']
            aps_corr[:, ix_no_modis] = 0
            fig_name_tca = ' (modis recal + ERA hydro)'
        elif aps_flag == 22:  # modis recal correction (not interpolated)
            aps_corr = aps['ph_tropo_modis_no_interp_recal'] + aps['ph_tropo_era_hydro']
            fig_name_tca = ' (modis recal + ERA hydro)'
        elif aps_flag == 23:  # MERIS + WRF hydro component
            ix_no_meris = np.where(np.sum(aps['ph_tropo_meris'], axis=0) == 0)[0]
            aps_corr = aps['ph_tropo_meris'] + aps['ph_tropo_wrf_hydro']
            aps_corr[:, ix_no_meris] = 0
            fig_name_tca = ' (meris + WRF hydro)'
        elif aps_flag == 24:  # MERIS (non-interpolated) + WRF hydro component
            aps_corr = aps['ph_tropo_meris_no_interp'] + aps['ph_tropo_wrf_hydro']
            fig_name_tca = ' (meris + WRF hydro)'
        elif aps_flag == 25:  # modis + WRF hydro component
            ix_no_modis = np.where(np.sum(aps['ph_tropo_modis'], axis=0) == 0)[0]
            aps_corr = aps['ph_tropo_modis'] + aps['ph_tropo_wrf_hydro']
            aps_corr[:, ix_no_modis] = 0
            fig_name_tca = ' (modis + WRF hydro)'
        elif aps_flag == 26:  # modis (non-interpolated) + WRF hydro component
            aps_corr = aps['ph_tropo_modis_no_interp'] + aps['ph_tropo_wrf_hydro']
            fig_name_tca = ' (modis + WRF hydro)'
        elif aps_flag == 27:  # modis recalibrated + WRF hydro component
            ix_no_modis = np.where(np.sum(aps['ph_tropo_modis'], axis=0) == 0)[0]
            aps_corr = aps['ph_tropo_modis_recal'] + aps['ph_tropo_wrf_hydro']
            aps_corr[:, ix_no_modis] = 0
            fig_name_tca = ' (modis recal + WRF hydro)'
        elif aps_flag == 28:  # modis recalibrated (non-interpolated) + WRF hydro component
            aps_corr = aps['ph_tropo_modis_no_interp_recal'] + aps['ph_tropo_wrf_hydro']
            fig_name_tca = ' (modis recal + WRF hydro)'
        elif aps_flag == 29:  # MERRA correction
            aps_corr = aps['ph_tropo_merra']
            fig_name_tca = ' (MERRA)'
        elif aps_flag == 30:  # MERRA-2 correction
            aps_corr = aps['ph_tropo_merra2']
            fig_name_tca = ' (MERRA-2)'
        elif aps_flag == 31:  # MERRA hydro correction
            aps_corr = aps['ph_tropo_merra_hydro']
            fig_name_tca = ' (MERRA hydro)'
        elif aps_flag == 32:  # MERRA-2 hydro correction
            aps_corr = aps['ph_tropo_merra2_hydro']
            fig_name_tca = ' (MERRA-2 hydro)'
        elif aps_flag == 33:  # MERRA wet correction
            aps_corr = aps['ph_tropo_merra_wet']
            fig_name_tca = ' (MERRA wet)'
        elif aps_flag == 34:  # MERRA-2 wet correction
            aps_corr = aps['ph_tropo_merra2_wet']
            fig_name_tca = ' (MERRA-2 wet)'
        elif aps_flag == 35:  # GACOS correction
            aps_corr = aps['ph_tropo_gacos']
            fig_name_tca = ' (GACOS)'
        elif aps_flag == 36:  # NARR correction total
            aps_corr = aps['ph_tropo_narr']
            fig_name_tca = ' (NARR)'
        elif aps_flag == 37:  # NARR correction hydro
            aps_corr = aps['ph_tropo_narr_hydro']
            fig_name_tca = ' (NARR hydro)'
        elif aps_flag == 38:  # NARR correction wet
            aps_corr = aps['ph_tropo_narr_wet']
            fig_name_tca = ' (NARR wet)'
        elif aps_flag == 39:  # ERA5 correction
            aps_corr = aps['ph_tropo_era5']
            fig_name_tca = ' (era5)'
        elif aps_flag == 40:  # ERA5 hydro correction
            aps_corr = aps['ph_tropo_era5_hydro']
            fig_name_tca = ' (era hydro5)'
        elif aps_flag == 41:  # ERA5 wet correction
            aps_corr = aps['ph_tropo_era5_wet']
            fig_name_tca = ' (era wet5)'
        else:
            raise ValueError('not a valid APS option')

        return aps_corr, fig_name_tca, aps_flag
    
    def _wrap_filt(self, ph, n_win, alpha, n_pad=[], low_flag='n'):
        """
        Goldstein adaptive and lowpass filtering

        Parameters:
        ph : 2D ndarray
            Input wrapped phase image
        n_win : int
            Window size
        alpha : float
            Goldstein filter exponent
        n_pad : int, optional
            Padding size (defaults to 25% of window size)
        low_flag : str, optional
            If 'y', also compute lowpass filtered output

        Returns:
        ph_out : 2D ndarray
            Filtered phase image (Goldstein)
        ph_out_low : 2D ndarray or None
            Lowpass filtered phase image if low_flag=='y', else None
        """
        if len(n_pad) == 0 or n_pad is None:
            n_pad = round(n_win * 0.25)

        if low_flag is None:
            low_flag = 'n'

        n_i, n_j = ph.shape
        n_inc = n_win // 2
        n_win_i = int(np.ceil(n_i / n_inc)) - 1
        n_win_j = int(np.ceil(n_j / n_inc)) - 1

        ph_out = np.zeros_like(ph, dtype=complex)
        if low_flag.lower() == 'y':
            ph_out_low = np.zeros_like(ph, dtype=complex)
        else:
            ph_out_low = None

        # Create windowing function
        x = np.arange(1, n_win//2 + 1)
        X, Y = np.meshgrid(x, x)
        X = X + Y
        wind_func = np.block([[X, np.fliplr(X)], [np.flipud(X), np.flipud(np.fliplr(X))]])

        # Replace NaNs with zeros
        ph = np.nan_to_num(ph)
        
        # Create smoothing kernel
        B = np.outer(windows.gaussian(7, std=2.5), windows.gaussian(7, std=2.5))
        ph_bit = np.zeros((n_win + n_pad, n_win + n_pad), dtype=complex)

        # Create lowpass filter
        L = ifftshift(np.outer(windows.gaussian(n_win + n_pad, std=16), 
                             windows.gaussian(n_win + n_pad, std=16)))

        for ix1 in range(n_win_i):
            wf = wind_func.copy()
            i1 = ix1 * n_inc
            i2 = i1 + n_win
            
            if i2 > n_i:
                i_shift = i2 - n_i
                i2 = n_i
                i1 = n_i - n_win
                wf = np.vstack((np.zeros((i_shift, n_win)), wf[:(n_win-i_shift),:]))

            for ix2 in range(n_win_j):
                wf2 = wf.copy()
                j1 = ix2 * n_inc
                j2 = j1 + n_win
                
                if j2 > n_j:
                    j_shift = j2 - n_j
                    j2 = n_j
                    j1 = n_j - n_win
                    wf2 = np.hstack((np.zeros((n_win, j_shift)), wf2[:,:(n_win-j_shift)]))

                ph_bit[:n_win,:n_win] = ph[i1:i2,j1:j2]
                ph_fft = fft2(ph_bit)
                H = np.abs(ph_fft)
                H = ifftshift(convolve2d(fftshift(H), B, mode='same'))
                meanH = np.median(H)
                
                if meanH != 0:
                    H = H / meanH
                H = H ** alpha

                ph_filt = ifft2(ph_fft * H)
                ph_filt = ph_filt[:n_win,:n_win] * wf2

                if low_flag.lower() == 'y':
                    ph_filt_low = ifft2(ph_fft * L)
                    ph_filt_low = ph_filt_low[:n_win,:n_win] * wf2

                if np.isnan(ph_filt[0,0]):
                    raise ValueError('Filtered phase contains NaNs in _wrap_filt')

                ph_out[i1:i2,j1:j2] += ph_filt
                if low_flag.lower() == 'y':
                    ph_out_low[i1:i2,j1:j2] += ph_filt_low

        # Reset magnitude to original values
        ph_out = np.abs(ph) * np.exp(1j * np.angle(ph_out))
        if low_flag.lower() == 'y':
            ph_out_low = np.abs(ph) * np.exp(1j * np.angle(ph_out_low))
            return ph_out, ph_out_low
        else:
            return ph_out, None
    
    def _uw_grid_wrapped(self, ph_in, xy_in, options):
        print('      -> Resampling phase to grid...')
        if not 'grid_size' in options:
            options['grid_size'] = 200
        if not 'prefilt_win' in options:
            options['prefilt_win'] = 32
        if not 'goldfilt_flag' in options:
            options['goldfilt_flag'] = 'y'
        if not 'lowfilt_flag' in options:
            options['lowfilt_flag'] = 'y'
        if not 'gold_alpha' in options:
            options['gold_alpha'] = 0.8
        if not 'ph_uw_predef' in options:
            options['ph_uw_predef'] = []
        
        ph_in_predef = options['ph_uw_predef']
        if len(ph_in_predef) == 0:
            predef_flag = 'n'
        else:
            predef_flag = 'y'
            
        n_ps, n_ifg = ph_in.shape

        print(f"         Number of interferograms  : {n_ifg}")
        print(f"         Number of points per ifg  : {n_ps}")

        if not np.iscomplexobj(ph_in) and np.sum(ph_in == 0) > 0:
            raise ValueError("Some phase values are zero")

        xy_in[:, 0] = np.arange(0, n_ps)

        if options['grid_size'] == 0:
            grid_x_min = 0
            grid_y_min = 0
            n_i = int(np.max(xy_in[:, 2]))
            n_j = int(np.max(xy_in[:, 1]))
            grid_ij = np.column_stack((xy_in[:, 2], xy_in[:, 1]))
        else:
            grid_x_min = np.min(xy_in[:, 1])
            grid_y_min = np.min(xy_in[:, 2])

            grid_ij = np.zeros((n_ps, 2))
            grid_ij[:, 0] = np.ceil((xy_in[:, 2] - grid_y_min + 1e-3) / options['grid_size'])
            grid_ij[grid_ij[:, 0] == np.amax(grid_ij[:, 0]), 0] = np.amax(grid_ij[:, 0]) - 1
            grid_ij[:, 1] = np.ceil((xy_in[:, 1] - grid_x_min + 1e-3) / options['grid_size'])
            grid_ij[grid_ij[:, 1] == np.amax(grid_ij[:, 1]), 1] = np.amax(grid_ij[:, 1]) - 1

            n_i = int(np.amax(grid_ij[:, 0]))
            n_j = int(np.amax(grid_ij[:, 1]))

        ph_grid = np.zeros((n_i, n_j), dtype=complex)
        if predef_flag == 'y':
            ph_grid_uw = np.zeros((n_i, n_j), dtype=complex)
            N_grid_uw = np.zeros((n_i, n_j), dtype=float)

        if min(ph_grid.shape) < options['prefilt_win']:
            raise ValueError(f"Minimum dimension of resampled grid ({min(ph_grid.shape)}) is less than prefilter window size ({options['prefilt_win']})")

        for i1 in tqdm(range(n_ifg), desc="      -> Processing interferograms"):
            if np.isrealobj(ph_in):
                ph_this = np.exp(1j * ph_in[:, i1])
            else:
                ph_this = ph_in[:, i1]

            if predef_flag == 'y':
                ph_this_uw = ph_in_predef[:, i1]
                ph_grid_uw.fill(0)
                N_grid_uw.fill(0)

            ph_grid.fill(0)

            if options['grid_size'] == 0:
                row_idx = (xy_in[:, 2] - 1).astype(int)
                col_idx = (xy_in[:, 1] - 1).astype(int)
                # Check indices are within bounds
                valid_idx = (row_idx >= 0) & (row_idx < n_i) & (col_idx >= 0) & (col_idx < n_j)
                ph_grid[row_idx[valid_idx], col_idx[valid_idx]] = ph_this[valid_idx]
                if predef_flag == 'y':
                    ph_grid_uw[row_idx[valid_idx], col_idx[valid_idx]] = ph_this_uw[valid_idx]
            else:
                for i in range(n_ps):
                    ph_grid[int(grid_ij[i, 0] - 1), int(grid_ij[i, 1] - 1)] += ph_this[i]
                if predef_flag == 'y':
                    for i in range(n_ps):
                        if not np.isnan(ph_this_uw[i]):
                            ph_grid_uw[int(grid_ij[i, 0] - 1), int(grid_ij[i, 1] - 1)] += ph_this_uw[i]
                            N_grid_uw[int(grid_ij[i, 0] - 1), int(grid_ij[i, 1] - 1)] += 1
                    ph_grid_uw = ph_grid_uw / N_grid_uw

            if i1 == 0:
                nzix = ph_grid != 0
                n_ps_grid = np.sum(nzix)
                ph = np.zeros((n_ps_grid, n_ifg), dtype=complex)
                if options['lowfilt_flag'].lower() == 'y':
                    ph_lowpass = ph.copy()
                else:
                    ph_lowpass = np.array([])
                if predef_flag == 'y':
                    ph_uw_predef = np.zeros((n_ps_grid, n_ifg), dtype=float)
                else:
                    ph_uw_predef = np.array([])

            if options['goldfilt_flag'].lower() == 'y' or options['lowfilt_flag'].lower() == 'y':
                # Load PS2 data for lat/lon coordinates
                ps2_file = os.path.join(self.config["processing_parameters"]["current_result"], 
                                      self.patch_dir, 'ps2.mat')
                ps2 = sio.loadmat(ps2_file)
                
                # Get lat/lon from PS2
                lons = ps2['lonlat'][:,0]
                lats = ps2['lonlat'][:,1]
                
                # Create lat/lon grids matching ph_grid dimensions
                lon_grid = np.zeros(ph_grid.shape)
                lat_grid = np.zeros(ph_grid.shape)
                
                # Fill grids with lat/lon values where we have data
                for i in range(len(lons)):
                    if options['grid_size'] == 0:
                        row_idx = int(xy_in[i,2] - 1)
                        col_idx = int(xy_in[i,1] - 1)
                    else:
                        row_idx = int(grid_ij[i,0] - 1)
                        col_idx = int(grid_ij[i,1] - 1)
                    lon_grid[row_idx, col_idx] = lons[i]
                    lat_grid[row_idx, col_idx] = lats[i]

                # Plot original phase before Goldstein filtering
                plt.figure(figsize=(10,8))
                masked_orig = np.ma.masked_where(ph_grid == 0, np.angle(ph_grid))
                plt.scatter(lon_grid[lon_grid != 0], lat_grid[lat_grid != 0], 
                          c=masked_orig.compressed(), cmap='jet', s=1)
                plt.colorbar(label='Phase (rad)')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.title(f'Original Phase Before Filtering - Interferogram {i1+1}')
                plt.savefig(os.path.join(self.config["processing_parameters"]["current_result"],
                                       self.patch_dir, f'6_ph_before_gold_ifg{i1+1}.png'))
                plt.close()

                ph_this_gold, ph_this_low = self._wrap_filt(ph_grid, options['prefilt_win'], 
                                                          options['gold_alpha'], [], options['lowfilt_flag'])

                # Plot ph_this_gold with masked zeros
                plt.figure(figsize=(10,8))
                plt.scatter(lon_grid[lon_grid != 0], lat_grid[lat_grid != 0],
                          c=np.angle(ph_this_gold)[nzix], cmap='jet', s=1)
                plt.colorbar(label='Phase (rad)')
                plt.xlabel('Longitude') 
                plt.ylabel('Latitude')
                plt.title(f'Goldstein Filtered Phase - Interferogram {i1+1}')
                plt.savefig(os.path.join(self.config["processing_parameters"]["current_result"],
                                       self.patch_dir, f'6_ph_after_gold_ifg{i1+1}.png'))
                plt.close()
                
                if options['lowfilt_flag'].lower() == 'y':
                    ph_lowpass[:, i1] = ph_this_low[nzix]

            if options['goldfilt_flag'].lower() == 'y':
                ph[:, i1] = ph_this_gold[nzix]
            else:
                ph[:, i1] = ph_grid[nzix]

            if predef_flag == 'y':
                ph_uw_predef[:, i1] = ph_grid_uw[nzix]
                ix = ~np.isnan(ph_uw_predef[:, i1])
                ph_diff = np.angle(ph[ix, i1] * np.conj(np.exp(1j * ph_uw_predef[ix, i1])))
                ph_diff[np.abs(ph_diff) > 1] = np.nan
                ph_uw_predef[ix, i1] = ph_uw_predef[ix, i1] + ph_diff

        n_ps = n_ps_grid
        print(f"         Number of resampled points: {n_ps}")

        nz_i, nz_j = np.where(ph_grid != 0)
        ij = np.column_stack([nz_i, nz_j])

        if options['grid_size'] == 0:
            xy = xy_in
        else:
            xy = np.column_stack([
                np.arange(0, n_ps),
                (nz_j - 0.5) * options['grid_size'],
                (nz_i - 0.5) * options['grid_size']
            ])

        sio.savemat(os.path.join(self.config["processing_parameters"]["current_result"], 
                                self.patch_dir, 'uw_grid.mat'),
                   {'ph': ph, 'ph_in': ph_in, 'ph_lowpass': ph_lowpass,
                    'ph_uw_predef': ph_uw_predef, 'ph_in_predef': ph_in_predef,
                    'xy': xy, 'ij': ij, 'nzix': nzix,
                    'grid_x_min': grid_x_min, 'grid_y_min': grid_y_min,
                    'n_i': n_i, 'n_j': n_j, 'n_ifg': n_ifg, 'n_ps': n_ps,
                    'grid_ij': grid_ij, 'pix_size': options['grid_size']})

        time.sleep(1)

    def _uw_interp(self):
        print('      -> Interpolating grid...')

        uw = sio.loadmat(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/uw_grid.mat', squeeze_me=True)
        n_ps = int(uw['n_ps'])
        nzix = uw['nzix']
        nrow, ncol = nzix.shape

        y, x = np.where(nzix)
        xy = np.column_stack((np.arange(n_ps), x, y))

        ele = Delaunay(xy[:, [1, 2]]).simplices
        edges_set = set()
        for tri in ele:
            for i in range(3):
                edge = tuple(sorted((tri[i], tri[(i + 1) % 3])))
                edges_set.add(edge)
        edgs = np.array(list(edges_set))
        edgs = np.hstack((np.arange(len(edgs)).reshape(-1, 1), edgs))

        # Grid mesh
        X, Y = np.meshgrid(np.arange(ncol), np.arange(nrow))
        grid_points = np.vstack((X.ravel(), Y.ravel())).T
        xy_nodes = xy[:, 1:3]

        tree = cKDTree(xy_nodes)
        _, Z_flat = tree.query(grid_points)
        Z = Z_flat.reshape(nrow, ncol)

        # Construct grid edges
        Zvec = Z.ravel()
        grid_edges = np.column_stack((Zvec[:-nrow], Zvec[nrow:]))  # vertical edges
        Zvec_row = Z.T.ravel()
        grid_edges = np.vstack((grid_edges, np.column_stack((Zvec_row[:-ncol], Zvec_row[ncol:]))))  # horizontal edges

        sort_edges = np.sort(grid_edges, axis=1)
        edge_sign = np.where(grid_edges[:, 0] < grid_edges[:, 1], 1, -1)

        alledges, J2 = np.unique(sort_edges, axis=0, return_inverse=True)
        sameix = alledges[:, 0] == alledges[:, 1]
        alledges[sameix] = 0

        edgs_unique, J3 = np.unique(alledges, axis=0, return_inverse=True)
        edgs_final = edgs_unique[1:]  # skip dummy edge
        edgs_final = np.column_stack((np.arange(len(edgs_final)), edgs_final))

        gridedgeix = (J3[J2] - 1) * edge_sign

        colix = gridedgeix[:nrow * (ncol - 1)].reshape(nrow, ncol - 1)
        rowix = gridedgeix[nrow * (ncol - 1):].reshape(ncol, nrow - 1).T

        print(f"         Number of unique edges in grid: {len(edgs_final)}")

        sio.savemat(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/uw_interp.mat', {
            'edgs': edgs_final,
            'n_edge': len(edgs_final),
            'rowix': rowix,
            'colix': colix,
            'Z': Z
        })

    def _uw_sb_smooth_unwrap(self,bounds, OPTIONS, G, W, dph, x1, *args):

        def _eprob(temp, v):
            toobig = 708.3964185322641
            pdf = v / temp
            mpdf = np.max(pdf)
            if mpdf > toobig:
                scale = mpdf / toobig
                pdf = np.exp(-pdf / scale)
                pdf = (pdf / np.max(pdf)) ** scale
            else:
                pdf = np.exp(-pdf)
                pdf /= np.max(pdf)
            return pdf / np.sum(pdf)
        
        def _make_pdf(temp, v):
            bad = np.isnan(v)
            if not np.any(bad):
                return _eprob(temp, v), 0
            else:
                good = ~bad
                w = v[good]
                pdf = _eprob(temp, w)
                full_pdf = np.zeros_like(v)
                full_pdf[good] = pdf
                return full_pdf, np.where(bad)[0]

        if not OPTIONS:
            scale, runs, grid = 4, 3, 4
            ts = np.linspace(1.5, 2.5, runs)
            matrix, newton, talk = 0, 0, 1
        else:
            OPTIONS = list(OPTIONS) + [0] * (8 - len(OPTIONS))  # pad to length 8
            scale = OPTIONS[0] if OPTIONS[0] else 4
            runs = OPTIONS[1] if OPTIONS[1] else 3
            grid = OPTIONS[2] if OPTIONS[2] else 4
            ts = np.full(runs, OPTIONS[3]) if OPTIONS[3] else np.linspace(2, 3, runs)
            matrix, newton, talk = OPTIONS[4:7]

        if np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("All the values in the first column of bounds must be less than those in the second.")

        p = bounds.shape[0]
        n = G.shape[1]

        count = np.zeros(runs, dtype=int)
        energy = np.full((100 * scale * 9, runs), np.inf)
        vals = np.concatenate([2.0 ** -np.arange(1, grid + 1), [0], -2.0 ** -np.arange(1, grid + 1)])
        delta = 0.5 * np.abs(bounds[:, 0] - bounds[:, 1])
        mhat = np.zeros((p, runs))
        F = np.zeros(runs)
        model = {}

        for k in range(runs):
            c = 0
            bestmodel = np.random.rand(p, 100) * (bounds[:, 1] - bounds[:, 0])[:, None] + bounds[:, 0][:, None]
            O = np.zeros(100)

            for e in range(100):
                m = bestmodel[:, e]
                step_hat = m[0] * x1 + m[1] * n / 2 * np.sin(2 * np.pi / n * x1 - m[2]) + m[3] * n / 2 * np.sin(4 * np.pi / n * x1 - m[4])
                dph_hat = G @ step_hat
                dph_r = (dph - dph_hat) / (2 * np.pi)
                dph_r = W @ (np.abs(dph_r - np.round(dph_r))) * 2 * np.pi
                O[e] = np.sum(dph_r) + np.sum(np.abs(np.diff(step_hat))) / 5

            tc = np.log10(np.mean(O)) - ts[k]
            bestmodel = bestmodel[:, [np.argmin(O)]]

            if talk:
                print(f"\n\nBeginning run #{k + 1:02d}. Critical temperature at {10 ** tc:.2f}.")
                print("------------------------------------------------")
                print("f-Calls\t\tTemperature\tMinimum f-Value")
                print("------------------------------------------------")

            x = scale * np.array([1, 2, 4, 6, 10, 6, 4, 2, 1])
            t = np.sum(x)
            temp = np.logspace(tc + 1, tc - 1, 9)
            T = np.repeat(temp, x)

            for w in range(t):
                temperature = T[w]
                c += 1
                if talk and c % 10 == 0:
                    print(f"{count[k]:7d}\t\t{temperature:7.2f}\t\t{np.min(energy[:c - 1, k]):7.2f}")

                for px in range(p):
                    if delta[px] == 0:
                        continue
                    v = bestmodel[px] + vals * delta[px]
                    v = v[(v <= bounds[px, 1]) & (v >= bounds[px, 0])]
                    mm = np.tile(bestmodel, (1, len(v)))
                    mm[px, :] = v
                    NM = mm.shape[1]
                    count[k] += NM
                    O = np.zeros(NM)

                    for e in range(NM):
                        step_hat = mm[0, e] * x1 + mm[1, e] * n / 2 * np.sin(2 * np.pi / n * x1 - mm[2, e]) + \
                                    mm[3, e] * n / 2 * np.sin(4 * np.pi / n * x1 - mm[4, e])
                        dph_hat = G @ step_hat
                        dph_r = (dph - dph_hat) / (2 * np.pi)
                        dph_r = np.abs(W @ (dph_r - np.round(dph_r))) * 2 * np.pi
                        O[e] = np.sum(dph_r) + np.sum(np.abs(np.diff(step_hat))) / 5

                    dist, nanflag = _make_pdf(temperature, O)
                    if isinstance(nanflag, np.ndarray) and len(nanflag):
                        for u in nanflag:
                            print("Warning: NaN in cost function for model:", mm[:, u])

                    s = np.where(np.cumsum(dist) >= np.random.rand())[0]
                    if len(s) == 0:
                        raise RuntimeError("Probability distribution is empty.")

                    s = s[0]
                    bestmodel[px, 0] = mm[px, s]
                    energy[c - 1, k] = O[s]
                    model[(k, c)] = bestmodel[:, 0].copy()

            F[k] = np.min(energy[:, k])
            mhat[:, k] = model[(k, int(np.argmin(energy[:, k])))]

        i = np.argmin(F)
        m = mhat[:, i]
        dph_smooth_series = m[0] * x1 + m[1] * n / 2 * np.sin(2 * np.pi / n * x1 - m[2]) + m[3] * n / 2 * np.sin(4 * np.pi / n * x1 - m[4])

        return dph_smooth_series, F, model, energy, count
    
    def _gradient_filt(self, ph, n_win):
        """
        Determine 2-D gradient through FFT.

        Parameters:
            ph (ndarray): Phase grid (2D array)
            n_win (int): Window size

        Returns:
            ifreq (ndarray): i-direction gradient frequency map
            jfreq (ndarray): j-direction gradient frequency map
            ij (ndarray): Center positions of each window
            Hmag (ndarray): Normalized FFT magnitude per window
        """
        n_i, n_j = ph.shape
        n_inc = n_win // 4
        n_win_i = (n_i + n_inc - 1) // n_inc - 3
        n_win_j = (n_j + n_inc - 1) // n_inc - 3

        ph = np.nan_to_num(ph)
        ph_bit = np.zeros((n_win, n_win))

        Hmag = np.full((n_win_i, n_win_j), np.nan)
        ifreq = np.full_like(Hmag, np.nan)
        jfreq = np.full_like(Hmag, np.nan)
        ij = np.full((n_win_i * n_win_j, 2), np.nan)

        i = 0
        for ix1 in range(n_win_i):
            i1 = ix1 * n_inc
            i2 = i1 + n_win
            if i2 > n_i:
                i2 = n_i
                i1 = n_i - n_win
            for ix2 in range(n_win_j):
                j1 = ix2 * n_inc
                j2 = j1 + n_win
                if j2 > n_j:
                    j2 = n_j
                    j1 = n_j - n_win

                ph_bit[:, :] = ph[i1:i2, j1:j2]

                if np.sum(ph_bit != 0) < 6:
                    continue

                ph_fft = fft2(ph_bit)
                H = np.abs(ph_fft)
                Hmag_this = np.max(H)
                Hmag[ix1, ix2] = Hmag_this / np.mean(H)

                I = np.argmax(H)
                I1 = I % n_win
                I2 = I // n_win

                I1 = (I1 + n_win // 2) % n_win
                I1 = n_win if I1 == 0 else I1
                I2 = (I2 + n_win // 2) % n_win
                I2 = n_win if I2 == 0 else I2

                ifreq[ix1, ix2] = (I1 - n_win // 2 - 1) * 2 * np.pi / n_win
                jfreq[ix1, ix2] = (I2 - n_win // 2 - 1) * -2 * np.pi / n_win

                ij[i, :] = [(i1 + i2) / 2, (j1 + j2) / 2]
                i += 1

        return ifreq.T, jfreq.T, ij[:i, :], Hmag.T
    
    def _uw_sb_unwrap_space_time(self, day, ifgday_ix, bperp, options):
        print('      -> Unwrapping in time-space...')
        start_time = time.perf_counter()
        uw = sio.loadmat(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/uw_grid.mat', squeeze_me=True)
        ui = sio.loadmat(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/uw_interp.mat', squeeze_me=True)

        n_ifg = int(uw['n_ifg'])
        n_ps = int(uw['n_ps'])
        nzix = uw['nzix']
        ij = uw['ij']

        if uw['ph_uw_predef'].shape[0] == 0:
            predef_flag = False
        else:
            predef_flag = True

        n_image = day.shape[0]
        nrow, ncol = ui['Z'].shape

        day_pos_ix = np.where(day.flatten() > 0)[0]
        I = np.argmin(day[day_pos_ix])
        dph_space = uw['ph'][ui['edgs'][:, 2], :] * np.conj(uw['ph'][ui['edgs'][:, 1], :])
        dph_space = dph_space.astype(np.complex128)
        if predef_flag:
            dph_space_uw = uw['ph_uw_predef'][ui['edgs'][:, 2]] - uw['ph_uw_predef'][ui['edgs'][:, 1]]
            predef_ix = ~np.isnan(dph_space_uw)
            dph_space_uw = dph_space_uw[predef_ix]
        else:
            predef_ix = np.array([])

        del uw
        
        with np.errstate(divide='ignore', invalid='ignore'):
            dph_space = dph_space / np.abs(dph_space)
        ifreq_ij = []
        jfreq_ij = []

        G = np.zeros((n_ifg, n_image), dtype=np.float32)
        for i in range(n_ifg):
            G[i, int(ifgday_ix[i, 0])] = -1
            G[i, int(ifgday_ix[i, 1])] = 1
        
        nzc_ix = np.any(np.abs(G) != 0, axis=0)  # Non-zero columns
        day = day[nzc_ix]
        n_image = day.shape[0]
        G = G[:, nzc_ix]

        zc_ix = np.where(~nzc_ix)[0]
        zc_ix = np.sort(zc_ix)[::-1]  # Descending sort

        for idx in zc_ix:
            ifgday_ix[ifgday_ix > idx] -= 1
        n = G.shape[1]

        if len(options['temp']) != 0:
            temp_flag = True
        else:
            temp_flag = False
        
        if temp_flag:
            print(f"         -> Estimating temperature correlation (elapsed time={round(time.perf_counter() - start_time)}s)")
    
            ix = np.abs(bperp) < options['max_bperp_for_temp_est']
            temp_sub = options['temp'][ix]
            temp_range = np.max(options['temp']) - np.min(options['temp'])
            temp_range_sub = np.max(temp_sub) - np.min(temp_sub)
            dph_sub = dph_space[:, ix]  # only ifgs using ith image
            
            n_temp_wraps = options['n_temp_wraps'] * (temp_range_sub / temp_range)
            
            trial_mult = np.arange(-np.ceil(8 * n_temp_wraps), np.ceil(8 * n_temp_wraps)).astype(int)
            n_trials = len(trial_mult)
            trial_phase = temp_sub / temp_range_sub * (np.pi / 4)
            
            # Create a matrix of exp(-1j * trial_phase * trial_mult)
            # trial_phase is (N,), trial_mult is (M,)
            # Want a (N, M) matrix: each element exp(-1j * trial_phase[n] * trial_mult[m])
            trial_phase_mat = np.exp(-1j * np.outer(trial_phase, trial_mult))
            
            Kt = np.zeros(ui['n_edge'], dtype=np.float32)
            coh = np.zeros(ui['n_edge'], dtype=np.float32)
            
            for i in range(ui['n_edge']):
                cpxphase = dph_sub[i, :].conj().T  # complex conjugate transpose, (N,)
                cpxphase_mat = np.tile(cpxphase[:, np.newaxis], (1, n_trials))  # (N, n_trials)
                phaser = trial_phase_mat * cpxphase_mat  # element-wise multiply (N, n_trials)
                phaser_sum = np.sum(phaser, axis=0)      # sum over N for each trial, (n_trials,)
                
                coh_trial = np.abs(phaser_sum) / np.sum(np.abs(cpxphase))
                coh_max_ix = np.argmax(coh_trial)
                coh_max = coh_trial[coh_max_ix]
                
                # Find falling indices before peak
                diff_before_peak = np.diff(coh_trial[:coh_max_ix])
                falling_ix = np.where(diff_before_peak < 0)[0]
                if falling_ix.shape[0] > 0:
                    peak_start_ix = falling_ix[-1] + 1
                else:
                    peak_start_ix = 0
                
                # Find rising indices after peak
                diff_after_peak = np.diff(coh_trial[coh_max_ix:])
                rising_ix = np.where(diff_after_peak > 0)[0]
                if rising_ix.shape[0] > 0:
                    peak_end_ix = rising_ix[0] + coh_max_ix
                else:
                    peak_end_ix = n_trials
                
                # Zero out values between peak_start_ix and peak_end_ix (inclusive)
                coh_trial[peak_start_ix:peak_end_ix + 1] = 0
                
                if coh_max - np.max(coh_trial) > 0.1:  # check peak prominence
                    K0 = np.pi / 4 / temp_range_sub * trial_mult[coh_max_ix]
                    resphase = cpxphase * np.exp(-1j * (K0 * temp_sub))
                    
                    offset_phase = np.sum(resphase)
                    resphase = np.angle(resphase * np.conj(offset_phase))
                    
                    weighting = np.abs(cpxphase)
                    # weighted least squares: mopt = (w*temp_sub) \ (w*resphase)
                    # In Python: mopt = (w*temp_sub).T @ (w*resphase) / (w*temp_sub).T @ (w*temp_sub)
                    numerator = np.sum(weighting * temp_sub * weighting * resphase)
                    denominator = np.sum(weighting * temp_sub * weighting * temp_sub)
                    mopt = numerator / denominator if denominator != 0 else 0
                    
                    Kt[i] = K0 + mopt
                    
                    phase_residual = cpxphase * np.exp(-1j * (Kt[i] * temp_sub))
                    mean_phase_residual = np.sum(phase_residual)
                    coh[i] = np.abs(mean_phase_residual) / np.sum(np.abs(phase_residual))
            
            # Set unreliable correlations to zero
            Kt[coh < 0.31] = 0
            
            # Update dph_space by removing temperature phase term
            dph_space = dph_space * np.exp(-1j * np.outer(Kt, options['temp'].T))
            
            if predef_flag:
                dph_temp = np.dot(Kt, options['temp'].T)
                dph_space_uw = dph_space_uw - dph_temp[predef_ix]
                del dph_temp
            dph_sub = dph_sub * np.exp(-1j * np.outer(Kt, temp_sub.T))
        
        if options['la_flag'].lower() == 'y':
            print(f"         -> Estimating look angle error (elapsed time={round(time.perf_counter() - start_time)}s)")
            
            bperp_range = np.amax(bperp) - np.amin(bperp)
            ix = np.where(np.abs(np.diff(ifgday_ix, axis=1)) == 1)[0]
            if len(ix) >= len(day) - 1:
                print('         -> Using sequential daisy chain of interferograms')
                dph_sub = dph_space[:, ix]
                bperp_sub = bperp[ix]
                bperp_range_sub = np.amax(bperp_sub) - np.amin(bperp_sub)
                n_trial_wraps = options['n_trial_wraps'] * (bperp_range_sub / bperp_range)
            else:
                ifgs_per_image = np.sum(np.abs(G), axis=0)
                max_ifgs_per_image = np.amax(ifgs_per_image)
                max_ix = np.argmax(ifgs_per_image)
                if max_ifgs_per_image >= len(day) - 2:
                    print('         -> Using sequential daisy chain of interferograms')
                    # Select interferograms using the chosen master image
                    ix = G[:, max_ix] != 0
                    gsub = G[ix, max_ix]
                    sign_ix = -np.sign(gsub.astype(np.float32))

                    dph_sub = dph_space[:, ix]  # dph for selected interferograms
                    bperp_sub = bperp[ix].copy()

                    # Flip bperp sign for negative direction
                    bperp_sub[sign_ix == -1] = -bperp_sub[sign_ix == -1]

                    # Add master bperp (0) at the end
                    bperp_sub = np.concatenate([bperp_sub, [0]])

                    # Flip dph sign by taking conjugate where needed
                    sign_matrix = np.tile(sign_ix, (ui['n_edge'], 1))  # shape: (n_edge, num_selected_ifgs)
                    dph_sub[sign_matrix == -1] = np.conj(dph_sub[sign_matrix == -1])

                    # Add master interferogram with zero phase (mean magnitude)
                    dph_sub = np.hstack([dph_sub, np.mean(np.abs(dph_sub), axis=1, keepdims=True)])

                    # Get slave image indices relative to max_ix (master)
                    slave_ix = np.sum(ifgday_ix[ix, :], axis=1) - max_ix
                    slave_ix = slave_ix.astype(np.int32)
                    day_sub = day[np.concatenate([slave_ix, [max_ix]])]
                    sort_ix = np.argsort(day_sub, axis=0)
                    day_sub = np.sort(day_sub, axis=0)

                    # Sort by ascending day
                    dph_sub = dph_sub[:, sort_ix]
                    bperp_sub = bperp_sub[sort_ix]
                    # Compute perpendicular baseline differences between sequential image pairs
                    bperp_diff = np.diff(bperp_sub, axis=0)
                    bperp_range_sub = np.amax(bperp_diff) - np.amin(bperp_diff)
                    n_trial_wraps = options['n_trial_wraps'] * (bperp_range_sub / bperp_range)
                    n_trial_wraps = n_trial_wraps.ravel()[0]
                    n_sub = len(day_sub)

                    # Sequential phase differences between adjacent acquisitions
                    dph_seq = dph_sub[:, 1:] * np.conj(dph_sub[:, :-1])

                    # Normalize to unit magnitude
                    with np.errstate(divide='ignore', invalid='ignore'):
                        dph_seq /= np.abs(dph_seq)
                else:
                    dph_sub = dph_space
                    bperp_sub = bperp
                    bperp_range_sub = bperp_range
                    
            trial_mult = np.arange(-np.ceil(8 * n_trial_wraps), np.ceil(8 * n_trial_wraps)).astype(int)
            n_trials = len(trial_mult)
            trial_phase = bperp_sub / bperp_range_sub * np.pi / 4
            trial_phase_mat = np.exp(-1j * np.outer(trial_phase, trial_mult))
            K = np.zeros(ui['n_edge'], dtype=np.float32)
            coh = np.zeros(ui['n_edge'], dtype=np.float32)
            
            for i in tqdm(range(ui['n_edge']), desc='         -> Reestimating edges info', unit="edges"):
                cpxphase = dph_sub[i, :].conj().T
                cpxphase_mat = np.tile(cpxphase, (n_trials, 1)).T
                phaser = trial_phase_mat * cpxphase_mat
                phaser_sum = np.sum(phaser, axis=0)
                coh_trial = np.abs(phaser_sum) / np.sum(np.abs(cpxphase))
                coh_max_ix = np.argmax(coh_trial)
                coh_max = coh_trial[coh_max_ix]
                falling_ix = np.where(np.diff(coh_trial[:coh_max_ix]) < 0)[0]
                if falling_ix.shape[0] > 0:
                    peak_start_ix = falling_ix[-1] + 1
                else:
                    peak_start_ix = 0
                rising_ix = np.where(np.diff(coh_trial[coh_max_ix:]) > 0)[0]
                if rising_ix.shape[0] > 0:
                    peak_end_ix = rising_ix[0] + coh_max_ix
                else:
                    peak_end_ix = n_trials
                coh_trial[peak_start_ix:peak_end_ix + 1] = 0
                if coh_max - np.max(coh_trial) > 0.1:
                    K0 = np.pi / 4 / bperp_range_sub * trial_mult[coh_max_ix]
                    resphase = cpxphase * np.exp(-1j * (K0 * bperp_sub))
                    offset_phase = np.sum(resphase)
                    resphase = np.angle(resphase * np.conj(offset_phase))
                    weighting = np.abs(cpxphase)
                    mopt = np.sum(weighting * bperp_sub * weighting * resphase) / np.sum(weighting * bperp_sub * weighting * bperp_sub)
                    K[i] = K0 + mopt
                    phase_residual = cpxphase * np.exp(-1j * (K[i] * bperp_sub))
                    mean_phase_residual = np.sum(phase_residual)
                    coh[i] = np.abs(mean_phase_residual) / np.sum(np.abs(phase_residual))
                    
            del cpxphase_mat
            del trial_phase_mat
            del phaser
            del dph_sub
            K[coh < 0.31] = 0
            if temp_flag:
                dph_space[K==0, :] = dph_space[K==0, :] * np.exp(1j * Kt[K==0] * options['temp'].T)
                Kt[K==0] = 0
                K[Kt==0] = 0
            dph_space = dph_space * np.exp(-1j * np.outer(K, bperp.T))
            if predef_flag:
                dph_scla = K * bperp.T
                dph_space_uw = dph_space_uw - dph_scla[predef_ix]
                del dph_scla
        
        spread = lil_matrix((ui['n_edge'], n_ifg), dtype=np.float32)
        if options['unwrap_method'] == '2D':
            dph_space_uw = np.angle(dph_space)
            if options['la_flag'].lower() == 'y':
                dph_space_uw = dph_space_uw + K * bperp.T
            if temp_flag:
                dph_space_uw = dph_space_uw + Kt * options['temp'].T
            dph_noise = np.array([], dtype=np.complex128)
            sio.savemat(f'{self.config["processing_parameters"]["current_result"]}/uw_space_time.mat', {'dph_space_uw': dph_space_uw, 'spread': spread, 'dph_noise': dph_noise})
        elif options['unwrap_method'] == '3D_NO_DEF':
            dph_noise = np.angle(dph_space)
            dph_space_uw = np.angle(dph_space)
            if options['la_flag'].lower() == 'y':
                dph_space_uw = dph_space_uw + K * bperp.T
            if temp_flag:
                dph_space_uw = dph_space_uw + Kt * options['temp'].T
            sio.savemat(f'{self.config["processing_parameters"]["current_result"]}/uw_space_time.mat', {'dph_space_uw': dph_space_uw, 'dph_noise': dph_noise, 'spread': spread})
        else:
            print(f'         -> Smoothing in time (elapsed time={round(time.perf_counter() - start_time)}s)')
            if options['unwrap_method'] == '3D_FULL':
                # Initialize smoothed interferogram with NaNs
                dph_smooth_ifg = np.full_like(dph_space, np.nan, dtype=np.float32)

                for i in range(n_image):
                    ix = G[:, i] != 0
                    if np.sum(ix) >= n_image - 2:
                        gsub = G[ix, i]
                        dph_sub = dph_space[:, ix]

                        sign_ix = -np.sign(gsub.astype(np.float32))[np.newaxis, :]
                        sign_ix = np.tile(sign_ix, (ui['n_edge'], 1))
                        dph_sub = np.where(sign_ix == -1, np.conj(dph_sub), dph_sub)

                        slave_ix = np.sum(ifgday_ix[ix, :], axis=1) - i
                        slave_ix = slave_ix.astype(np.int32)
                        day_sub = day[slave_ix]

                        sort_ix = np.argsort(day_sub)
                        day_sub = day_sub[sort_ix]
                        dph_sub = dph_sub[:, sort_ix]
                        if dph_sub.ndim != 2:
                            dph_sub = dph_sub.squeeze()

                        dph_sub_angle = np.angle(dph_sub)
                        n_sub = len(day_sub)
                        dph_smooth = np.zeros((ui['n_edge'], n_sub), dtype=complex)

                        dph_smooth = np.zeros_like(dph_sub, dtype=np.complex128)

                        for i1 in range(n_sub):
                            time_diff = (day_sub[i1] - day_sub).astype(np.float64)  # shape: (n_sub,)
    
                            weight_factor = np.exp(-(time_diff**2) / (2 * options['time_win']**2))
                            weight_factor = weight_factor / np.sum(weight_factor)
                            weight_factor = weight_factor.reshape(-1, 1) # shape: (n_sub, 1)

                            dph_mean = np.sum(dph_sub * weight_factor.T, axis=1) # shape: (n_edge,)

                            # Subtract weighted phase mean (wrapped)
                            dph_mean_angle = np.angle(dph_mean)  # shape: (n_edge,)
                            dph_mean_adj = (dph_sub_angle - np.tile(dph_mean_angle, (n_sub, 1)).T + np.pi) % (2 * np.pi) - np.pi  # shape: (n_edge, n_sub)

                            GG = np.concatenate((np.ones((n_sub, 1)), time_diff.reshape(-1,1)), axis=1)  # shape: (n_sub, 1)

                            if GG.shape[0] > 1:
                                # Solve weighted least squares for each edge
                                m = lstsq(GG * weight_factor, dph_mean_adj.T, rcond=None)[0]
                            else:
                                m = np.zeros((GG.shape[0], ui['n_edge']))

                            # Reconstruct smoothed complex phase
                            dph_smooth[:, i1] = dph_mean * np.exp(1j * m[0, :])

                        phase_1 = np.angle(dph_smooth[:, 0])
                        delta_phase = np.angle(dph_smooth[:, 1:] * np.conj(dph_smooth[:, :-1]))
                        dph_smooth_sub = np.cumsum(np.hstack([phase_1[:, np.newaxis], delta_phase]), axis=1)

                        close_master_ix = np.where(slave_ix - i > 0)[0]
                        if len(close_master_ix) == 0:
                            close_master_ix = [n_sub - 1]
                        else:
                            close_master_ix = close_master_ix[0]
                            if close_master_ix > 0:
                                close_master_ix = [close_master_ix - 1, close_master_ix]
                            else:
                                close_master_ix = [close_master_ix]

                        dph_close_master = np.mean(dph_smooth_sub[:, close_master_ix], axis=1)
                        ref_angle = np.angle(np.exp(1j * dph_close_master))
                        dph_smooth_sub = dph_smooth_sub - dph_close_master[:, np.newaxis] + ref_angle[:, np.newaxis]
                        dph_smooth_sub *= sign_ix

                        already_sub_ix = np.where(~np.isnan(dph_smooth_ifg[0, ix]))[0]
                        ix_true = np.where(ix)[0]
                        already_ix = ix_true[already_sub_ix]

                        std_noise1 = np.std(np.angle(dph_space[:, already_ix] * np.exp(-1j * dph_smooth_ifg[:, already_ix])), axis=0)
                        std_noise2 = np.std(np.angle(dph_space[:, already_ix] * np.exp(-1j * dph_smooth_sub[:, already_sub_ix])), axis=0)

                        keep_ix = np.ones(n_sub, dtype=bool)
                        keep_ix[already_sub_ix[std_noise1 < std_noise2]] = False

                        dph_smooth_ifg[:, ix_true[keep_ix]] = dph_smooth_sub[:, keep_ix]

                # Final noise filtering
                dph_noise = np.angle(dph_space * np.exp(-1j * dph_smooth_ifg))
                noisy_rows = np.std(dph_noise, axis=1) > 1.2
                dph_noise[noisy_rows, :] = np.nan
                
            else:
                # Use dates for smoothing
                x = (day - day[0]) * (n - 1) / (day[-1] - day[0])

                if predef_flag:
                    n_dph = dph_space.shape[0]
                    dph_space_angle = np.angle(dph_space)
                    dph_space_angle[predef_ix] = dph_space_uw
                    dph_space_series = np.zeros((n, n_dph))
                    
                    for i in range(n_dph):
                        # Give more weight to predefined unwrapped
                        W = predef_ix[i,:] + 0.01
                        # Solve weighted least squares
                        dph_space_series[1:,i] = lstsq(G[:,1:] * W[:,np.newaxis], 
                                                    dph_space_angle[i,:].T, rcond=None)[0]
                else:
                    # Stack zeros with least squares solution
                    dph_space_series = np.vstack([
                        np.zeros((1, ui['n_edge'])),
                        lstsq(G[:,1:], np.angle(dph_space).T, rcond=None)[0]
                    ])

                dph_smooth_series = np.zeros((G.shape[1], ui['n_edge']), dtype=np.float32)

                for i1 in range(n):
                    time_diff_sq = (day[i1] - day)**2
                    weight_factor = np.exp(-time_diff_sq / (2 * options['time_win']**2))
                    weight_factor = weight_factor / np.sum(weight_factor)
                    dph_smooth_series[i1,:] = np.sum(dph_space_series * weight_factor[:,np.newaxis], axis=0)

                dph_smooth_ifg = (G @ dph_smooth_series).T
                dph_noise = np.angle(dph_space * np.exp(-1j * dph_smooth_ifg))

                if options['unwrap_method'] in ['3D_SMALL_DEF', '3D_QUICK']:
                    not_small_ix = np.where(np.std(dph_noise, axis=1) > 1.3)[0]
                    print(f'   {len(not_small_ix)} edges with high std dev in time (elapsed time={round(time.time()-start_time)}s)')
                    dph_noise[not_small_ix,:] = np.nan
                    
                else: # 3D
                    uw = sio.loadmat(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/uw_grid.mat')
                    ph_noise = np.angle(uw['ph'] * np.conj(uw['ph_lowpass']))
                    del uw
                    
                    dph_noise_sf = ph_noise[ui['edgs'][:,2],:] - ph_noise[ui['edgs'][:,1],:]
                    
                    m_minmax = np.array([-np.pi, np.pi])[np.newaxis,:] * np.array([0.5,0.25,1,0.25,1])[:,np.newaxis]
                    anneal_opts = np.array([1,15,0,0,0,0,0])
                    
                    covm = np.cov(dph_noise_sf.T) # Estimate covariance
                    try:
                        W = np.linalg.cholesky(np.linalg.inv(covm)).T # Weighting matrix
                    except:
                        W = np.diag(1/np.sqrt(np.diag(covm)))
                        
                    not_small_ix = np.where(np.std(dph_noise, axis=1) > 1)[0]
                    print(f'         -> Performing complex smoothing on {len(not_small_ix)} edges (elapsed time={round(time.time()-start_time)}s)')

                    n_proc = 0
                    for i in not_small_ix:
                        dph = np.angle(dph_space[i,:])
                        dph_smooth_series[:,i] = self._uw_sb_smooth_unwrap(m_minmax, anneal_opts, G, W, dph, x)
                        
                        n_proc += 1
                        if n_proc % 1000 == 0:
                            sio.savemat(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/uw_unwrap_time.mat', {'G': G, 'dph_space': dph_space, 'dph_smooth_series': dph_smooth_series})
                            print(f'{n_proc} edges of {len(not_small_ix)} reprocessed (elapsed time={round(time.time()-start_time)}s)')
                            
                    dph_smooth_ifg = (G @ dph_smooth_series).T
                    dph_noise = np.angle(dph_space * np.exp(-1j * dph_smooth_ifg))
                
            del dph_space
            dph_space_uw = dph_smooth_ifg + dph_noise
            del dph_smooth_ifg

            if options['la_flag'].lower() == 'y':
                dph_space_uw = dph_space_uw + K[:, np.newaxis] * bperp[np.newaxis, :]
            if temp_flag:
                dph_space_uw = dph_space_uw + Kt[:, np.newaxis] * options['temp'][np.newaxis, :]
            
            if options['scf_flag'].lower() == 'y':
                print(f"         -> Calculating local phase gradients (elapsed time={round(time.time() - start_time)}s)")

                ifreq_ij = np.full((n_ps, n_ifg), np.nan, dtype=np.float32)
                jfreq_ij = np.full((n_ps, n_ifg), np.nan, dtype=np.float32)
                ifgw = np.zeros((nrow, ncol))
                uw = sio.loadmat(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/uw_grid.mat')

                for i in range(n_ifg):
                    ifgw[nzix] = uw['ph'][:, i]
                    ifreq, jfreq, grad_ij, Hmag = self._gradient_filt(ifgw, options['prefilt_win'])

                    ix = ~np.isnan(ifreq) & (Hmag / (np.abs(ifreq) + 1) > 3)
                    if np.sum(ix) > 2:
                        ifreq_ij[:, i] = griddata((grad_ij[ix, 1], grad_ij[ix, 0]), ifreq[ix], (ij[:, 1], ij[:, 0]), method='linear')

                    ix = ~np.isnan(jfreq) & (Hmag / (np.abs(jfreq) + 1) > 3)
                    if np.sum(ix) > 2:
                        jfreq_ij[:, i] = griddata((grad_ij[ix, 1], grad_ij[ix, 0]), jfreq[ix], (ij[:, 1], ij[:, 0]), method='linear')

                del uw

                spread2 = np.zeros_like(spread, dtype=np.float32)
                dph_smooth_uw2 = np.full((ui['n_edge'], n_ifg), np.nan, dtype=np.float32)

                print(f"         -> Smoothing using local phase gradients (elapsed time={round(time.time() - start_time)}s)")

                for i in range(ui['n_edge']):
                    nodes_ix = ui['edgs'][i, 1:3]
                    ifreq_edge = np.nanmean(ifreq_ij[nodes_ix, :], axis=0)
                    jfreq_edge = np.nanmean(jfreq_ij[nodes_ix, :], axis=0)
                    diff_i = np.diff(ij[nodes_ix, 0])
                    diff_j = np.diff(ij[nodes_ix, 1])
                    dph_smooth_uw2[i, :] = diff_i * ifreq_edge + diff_j * jfreq_edge
                    spread2[i, :] = np.diff(ifreq_ij[nodes_ix, :], axis=0) + np.diff(jfreq_ij[nodes_ix, :], axis=0)

                print(f"         -> Choosing between time and phase gradient smoothing (elapsed time={round(time.time() - start_time)}s)")

                std_noise = np.nanstd(dph_noise, axis=1)
                dph_noise2 = np.angle(np.exp(1j * (dph_space_uw - dph_smooth_uw2)))
                std_noise2 = np.nanstd(dph_noise2, axis=1)
                dph_noise2[std_noise2 > 1.3, :] = np.nan

                shaky_ix = np.isnan(std_noise) | (std_noise > std_noise2)

                print(f"         -> {ui['n_edge'] - np.sum(shaky_ix)} arcs smoothed in time, {np.sum(shaky_ix)} in space (elapsed time={round(time.time() - start_time)}s)")

                dph_noise[shaky_ix, :] = dph_noise2[shaky_ix, :]
                dph_space_uw[shaky_ix, :] = dph_smooth_uw2[shaky_ix, :] + dph_noise2[shaky_ix, :]
                spread[shaky_ix, :] = spread2[shaky_ix, :]
            else:
                shaky_ix = np.array([])
                
            sio.savemat(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/uw_space_time.mat', {'dph_space_uw': dph_space_uw, 'dph_noise': dph_noise, 'G': G, 'spread': spread, 'ifreq_ij': ifreq_ij, 'jfreq_ij': jfreq_ij, 'shaky_ix': shaky_ix, 'predef_ix': predef_ix})

    def _writecpx(self, filename, data, precision='float32', endian='native'):
        interleaved = np.empty(data.shape + (2,), dtype=precision)
        interleaved[..., 0] = np.real(data)
        interleaved[..., 1] = np.imag(data)
        interleaved = interleaved.reshape(-1)
        interleaved.tofile(filename)

    # Move this function to top-level scope (outside any other function)
    @staticmethod
    def _process_chunk(args):
        chunk_indices, rowix_in, colix_in = args
        rowix_chunk = rowix_in.copy().astype(float)
        colix_chunk = colix_in.copy().astype(float)
        for i in chunk_indices:
            rowix_chunk[np.abs(rowix_in).astype(int) == i] = np.nan
            colix_chunk[np.abs(colix_in).astype(int) == i] = np.nan
        return rowix_chunk, colix_chunk

    def _uw_stat_costs(self, unwrap_method='3D', variance=None, subset_ifg_index=None):
        print('      -> Phase stats and costs...')

        costscale = 100
        nshortcycle = 200
        maxshort = 32000

        uw = sio.loadmat(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/uw_grid.mat')
        ui = sio.loadmat(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/uw_interp.mat')
        ut = sio.loadmat(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/uw_space_time.mat', squeeze_me=True)

        ph = uw['ph']
        nzix = uw['nzix'].astype(bool)
        pix_size = uw['pix_size'][0][0]
        n_ps = uw['n_ps'][0][0]
        n_ifg = uw['n_ifg'][0][0]

        if subset_ifg_index is None:
            subset_ifg_index = list([int(i) for i in range(ph.shape[1])])

        predef_flag = False
        if 'predef_ix' in ut and ut['predef_ix'].shape[0] > 0:
            predef_flag = True

        nrow, ncol = nzix.shape
        y, x = np.where(nzix)
        colix = np.asarray(ui['colix'], dtype=float)
        rowix = np.asarray(ui['rowix'], dtype=float)
        Z = ui['Z']

        grid_edges = np.concatenate((np.abs(colix[colix > 0]), np.abs(rowix[rowix > 0])))
        n_edges = np.histogram(np.abs(grid_edges), bins=np.arange(ui['n_edge'][0][0]+1))[0].T
        if unwrap_method.upper() == '2D':
            edge_length = np.sqrt(np.sum(np.diff([x[ui['edgs'][:, 1]], y[ui['edgs'][:, 1]]], axis=0) ** 2, axis=0))
            if len(variance) > 0:
                sigsq_noise = variance[ui['edgs'][:, 1]] + variance[ui['edgs'][:, 2]]
            else:
                sigsq_noise = np.zeros(edge_length.shape)

            if uw['pix_size'][0][0] == 0:
                pix_size = 5
            else:
                pix_size = uw['pix_size'][0][0]

            sigsq_aps = (2 * np.pi) ** 2
            aps_range = 20000
            sigsq_noise += sigsq_aps * (1 - np.exp(-edge_length * pix_size * 3 / aps_range))
            sigsq_noise /= 10
            dph_smooth = ut['dph_space_uw']
        else:
            sigsq_noise = (np.std(ut['dph_noise'], axis=1, ddof=1) / (2 * np.pi)) ** 2
            dph_smooth = ut['dph_space_uw'] - ut['dph_noise']

        ut.pop('dph_noise')
        if not os.path.exists(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/uw_nostats_ix.mat'):
            nostats_ix = np.where(np.isnan(sigsq_noise))[0]
            n_cores = 13 #self.config['computing_resources']['cpu']
            chunk_size = max(1, len(nostats_ix) // n_cores)
            chunks = [nostats_ix[i:i + chunk_size] for i in range(0, len(nostats_ix), chunk_size)]

            # Flatten and prepare args as tuples for each worker
            args = [(chunk, rowix, colix) for chunk in chunks]

            # Run in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as pool:
                results = list(tqdm(
                    pool.map(StaMPSStep._process_chunk, args),
                    total=len(chunks),
                    desc='         -> Removing edges with no stats',
                    unit='chunks'
                ))

            # Combine results
            for chunk_rowix, chunk_colix in results:
                rowix[np.isnan(chunk_rowix)] = np.nan
                colix[np.isnan(chunk_colix)] = np.nan

            sio.savemat(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/uw_nostats_ix.mat', {'rowix': rowix, 'colix': colix})
        else:
            no_stats_data = sio.loadmat(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/uw_nostats_ix.mat', squeeze_me=True)
            rowix = no_stats_data['rowix']
            colix = no_stats_data['colix']
            del no_stats_data

        with np.errstate(divide='ignore', invalid='ignore'):
            sigsq = np.int16(np.round((sigsq_noise * nshortcycle ** 2) / costscale * n_edges))

        sigsq[sigsq < 1] = 1

        rowcost = np.zeros(((nrow-1), ncol * 4), dtype=np.int16)
        colcost = np.zeros((nrow, (ncol-1) * 4), dtype=np.int16)

        nzrowix = np.abs(rowix) > 0
        nzcolix = np.abs(colix) > 0

        rowstdgrid = np.ones(rowix.shape, dtype=np.int16)
        colstdgrid = np.ones(colix.shape, dtype=np.int16)

        rowcost[:, 2::4] = maxshort
        colcost[:, 2::4] = maxshort

        stats_ix = ~np.isnan(rowix)
        rowcost[:, 3::4] = np.int16(stats_ix) * (-1 - maxshort) + 1
        stats_ix = ~np.isnan(colix)
        colcost[:, 3::4] = np.int16(stats_ix) * (-1 - maxshort) + 1

        ph_uw = np.zeros((n_ps, n_ifg), dtype=np.float32)
        msd = np.zeros((n_ifg, 1))

        Path(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/snaphu.conf').write_text(
            f"INFILE  {self.config['processing_parameters']['current_result']}/{self.patch_dir}/snaphu.in\n"
            f"OUTFILE {self.config['processing_parameters']['current_result']}/{self.patch_dir}/snaphu.out\n"
            f"COSTINFILE {self.config['processing_parameters']['current_result']}/{self.patch_dir}/snaphu.costinfile\n"
            "STATCOSTMODE  DEFO\n"
            "INFILEFORMAT  COMPLEX_DATA\n"
            "OUTFILEFORMAT FLOAT_DATA\n"
        )
        for i1 in tqdm(subset_ifg_index, total=len(subset_ifg_index), desc='         -> Processing IFGs', unit='IFGs'):
            spread = ut['spread'][:, i1].toarray().ravel()
            spread = np.int16(np.round((np.abs(spread) * nshortcycle**2) / 6 / costscale * n_edges))
            sigsqtot = sigsq + spread
            
            if predef_flag:
                sigsqtot[ut['predef_ix'][:, i1]] = 1

            rowstdgrid[nzrowix] = sigsqtot[np.abs(rowix[nzrowix].astype(int))]
            rowcost[:, 1::4] = rowstdgrid  # sigsq
            colstdgrid[nzcolix] = sigsqtot[np.abs(colix[nzcolix].astype(int))]
            colcost[:, 1::4] = colstdgrid  # sigsq

            offset_cycle = (np.angle(np.exp(1j * ut['dph_space_uw'][:, i1])) - dph_smooth[:, i1]) / (2 * np.pi)
            offset_cycle = np.real(offset_cycle)
            offset_cycle[np.isnan(offset_cycle)] = 0
            
            offgrid = np.zeros(rowix.shape, dtype=np.int16)
            offgrid[nzrowix] = np.round(offset_cycle[np.abs(rowix[nzrowix].astype(int))] * np.sign(rowix[nzrowix]) * nshortcycle)
            rowcost[:, 0::4] = -offgrid  # offset
            
            offgrid = np.zeros(colix.shape, dtype=np.int16)
            offgrid[nzcolix] = np.round(offset_cycle[np.abs(colix[nzcolix].astype(int))] * np.sign(colix[nzcolix]) * nshortcycle)
            colcost[:, 0::4] = offgrid  # offset

            with open(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/snaphu.costinfile', 'wb') as fid:
                fid.write(rowcost.astype(np.int16).tobytes())
                fid.write(colcost.astype(np.int16).tobytes())

            ifgw = ph[Z, i1].reshape(nrow, ncol)
           
            # Plot the wrapped phase for this interferogram
            grid_ij = uw['grid_ij']
            ps2 = sio.loadmat(os.path.join(self.config["processing_parameters"]["current_result"], 
                                        self.patch_dir, 'ps2.mat'))
            lonlat = ps2['lonlat']
            # --- Initialize lists for valid data ---
            valid_lons = []
            valid_lats = []
            valid_phases = []

            # --- Loop through each PS and check if it's in a valid nzix region ---
            for idx, (row, col) in enumerate(grid_ij):
                row = int(row)
                col = int(col)
                # Check bounds and validity
                if 0 <= row < nrow and 0 <= col < ncol and nzix[row, col]:
                    valid_lons.append(lonlat[idx, 0])
                    valid_lats.append(lonlat[idx, 1])
                    valid_phases.append(ifgw[row, col])

            # --- Convert to arrays ---
            valid_lons = np.array(valid_lons)
            valid_lats = np.array(valid_lats)
            valid_phases = np.array(valid_phases)

            # --- Plot wrapped phase ---
            plt.figure(figsize=(12, 8))
            plt.scatter(valid_lons, valid_lats, c=np.angle(valid_phases), cmap='jet', s=1)
            plt.colorbar(label='Wrapped Phase (rad)')
            plt.title(f'Wrapped Phase - Interferogram {i1+1}')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config["processing_parameters"]["current_result"],
                                    self.patch_dir, f'6_phase_ifg_{i1+1}_right_before_unwrap.png'),
                    dpi=300, bbox_inches='tight')
            plt.close()

            self._writecpx(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/snaphu.in', ifgw)
            os.chdir(f'{self.config["project_definition"]["project_folder"]}/modules/snaphu/bin/')
            cmdstr = f'snaphu -f {self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/snaphu.conf {ncol} > {self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/snaphu.log'
            subprocess.run(cmdstr, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.chdir(self.config['processing_parameters']['current_result']+f'/{self.patch_dir}')

            with open(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/snaphu.out', 'rb') as fid:
                ifguw = np.fromfile(fid, dtype=np.float32).reshape((nrow, ncol))

                # --- Initialize lists for valid unwrapped data ---
                valid_lons = []
                valid_lats = []
                valid_phases = []

                # --- Loop through each PS and check if it's in a valid nzix region ---
                for idx, (row, col) in enumerate(grid_ij):
                    row = int(row)
                    col = int(col)
                    # Check bounds and validity
                    if 0 <= row < nrow and 0 <= col < ncol and nzix[row, col]:
                        valid_lons.append(lonlat[idx, 0])
                        valid_lats.append(lonlat[idx, 1])
                        valid_phases.append(ifguw[row, col])

                # --- Convert to arrays ---
                valid_lons = np.array(valid_lons)
                valid_lats = np.array(valid_lats)
                valid_phases = np.array(valid_phases)

                # --- Plot unwrapped phase ---
                plt.figure(figsize=(12, 8))
                plt.scatter(valid_lons, valid_lats, c=valid_phases, cmap='jet', s=1)
                plt.colorbar(label='Unwrapped Phase (rad)')
                plt.title(f'Unwrapped Phase - Interferogram {i1+1}')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(self.config["processing_parameters"]["current_result"],
                                        self.patch_dir, f'6_phase_ifg_{i1+1}_right_after_unwrap.png'),
                        dpi=300, bbox_inches='tight')
                plt.close()

                ifg_diff1 = ifguw[:-1, :] - ifguw[1:, :]
                ifg_diff1 = ifg_diff1[ifg_diff1 != 0]
                ifg_diff2 = ifguw[:, :-1] - ifguw[:, 1:]
                ifg_diff2 = ifg_diff2[ifg_diff2 != 0]
                
                msd[i1] = (np.sum(ifg_diff1**2) + np.sum(ifg_diff2**2)) / (len(ifg_diff1) + len(ifg_diff2))
                ph_uw[:, i1] = ifguw[np.where(nzix == True)].ravel()

        sio.savemat(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/uw_phaseuw.mat', {'ph_uw': ph_uw, 'msd': msd})
    
    def _uw_unwrap_from_grid(self, xy, options):
        print('      -> Unwrapping from grid...')
        uw = sio.loadmat(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/uw_grid.mat')
        uu = sio.loadmat(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/uw_phaseuw.mat')
        
        n_ps, n_ifg = uw['ph_in'].shape
        gridix = np.zeros(uw['nzix'].shape)
        gridix[uw['nzix']==True] = np.arange(uw['n_ps'][0][0])

        ph_uw = np.zeros((n_ps, n_ifg), dtype=np.float32)
        for i in range(n_ps):
            # Check array bounds before indexing
            if (int(uw['grid_ij'][i, 0]) >= gridix.shape[0] or 
                int(uw['grid_ij'][i, 1]) >= gridix.shape[1]):
                ph_uw[i, :] = np.nan
                continue
                
            ix = gridix[int(uw['grid_ij'][i, 0]), int(uw['grid_ij'][i, 1])]
            if ix == 0:
                ph_uw[i, :] = np.nan # wrapped phase values were zero
            else:
                ph_uw_pix = uu['ph_uw'][int(ix), :]
                if np.all(np.isreal(np.asarray(uw['ph_in']))):
                    ph_uw[i, :] = ph_uw_pix + np.angle(np.exp(1j * (np.asarray(uw['ph_in'][i, :]) - ph_uw_pix)))
                else:
                    ph_uw[i, :] = ph_uw_pix + np.angle(np.asarray(uw['ph_in'][i, :]) * np.exp(-1j * ph_uw_pix))

        if len(uw['ph_in_predef']) > 0:
            predef_ix = ~np.isnan(uw['ph_in_predef'])
            meandiff = np.nanmean(ph_uw - uw['ph_in_predef'])
            meandiff = 2 * np.pi * np.round(meandiff / 2 / np.pi)
            uw['ph_in_predef'] = uw['ph_in_predef'] + np.tile(meandiff, (n_ps, 1))
            ph_uw[predef_ix] = uw['ph_in_predef'][predef_ix]

        msd = uu['msd']
        return ph_uw, msd
    
    def _uw_3d(self, ph_w, xy, day, ifgday_ix, bperp=np.array([]), options={}):
        """Perform 3D unwrapping of phase.
        
        This method performs 3D unwrapping of phase using the unwrap3d function.
        
        Args:

        """
        if len(ifgday_ix) == 0:
            single_master_flag = True
        else:
            single_master_flag = False
            
        valid_options = ['la_flag', 'scf_flag', 'master_day', 'grid_size', 'prefilt_win', 'time_win', 'unwrap_method', 'goldfilt_flag', 'lowfilt_flag', 'gold_alpha', 'n_trial_wraps', 'temp', 'n_temp_wraps', 'max_bperp_for_temp_est', 'variance', 'ph_uw_predef']
        invalid_options = set(options.keys()) - set(valid_options)
        if len(invalid_options) > 0:
            raise ValueError(f"Invalid options: {invalid_options}")
            
        if not 'master_day' in options:
            options['master_day'] = 0
            
        if not 'grid_size' in options:
            options['grid_size'] = 5
            
        if not 'prefilt_win' in options:
            options['prefilt_win'] = 16
            
        if not 'time_win' in options:
            options['time_win'] = 365

        if not 'unwrap_method' in options:
            if single_master_flag:
                options['unwrap_method'] = '3D'
            else:
                options['unwrap_method'] = '3D_FULL'

        if not 'goldfilt_flag' in options:
            options['goldfilt_flag'] = 'n'
            
        if not 'lowfilt_flag' in options:
            options['lowfilt_flag'] = 'n'
            
        if not 'gold_alpha' in options:
            options['gold_alpha'] = 0.8
            
        if not 'n_trial_wraps' in options:
            options['n_trial_wraps'] = 6
            
        if not 'la_flag' in options:
            options['la_flag'] = 'y'

        if not 'scf_flag' in options:
            options['scf_flag'] = 'y'

        if not 'temp' in options:
            options['temp'] = []
        else:
            if not np.sum(options['temp']) == 0 and len(options['temp']) != ph_w.shape[1]:
                raise ValueError('options.temp must be M x 1 vector where M is no. of ifgs')

        if not 'n_temp_wraps' in options:
            options['n_temp_wraps'] = 2

        if not 'max_bperp_for_temp_est' in options:
            options['max_bperp_for_temp_est'] = 100

        if not 'variance' in options:
            options['variance'] = []

        if not 'ph_uw_predef' in options:
            options['ph_uw_predef'] = []

        if xy.shape[1] == 2:
            xy[:, 1:3] = xy[:, 0:2]
        
        if day.shape[0] == 1:
            day = day.T

        if options['unwrap_method'] == '3D' or options['unwrap_method'] == '3D_NEW':
            if len(np.unique(ifgday_ix[:, 0])) == 1:
                options['unwrap_method'] = '3D_FULL'
            else:
                options['lowfilt_flag'] = 'y'

        # Plot wrapped phase values for each interferogram
        n_ifgs = ph_w.shape[1]

        # Calculate number of rows and columns for subplots
        n_rows = int(np.ceil(np.sqrt(n_ifgs)))
        n_cols = int(np.ceil(n_ifgs / n_rows))
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        axes = axes.flatten()
        
        for i in range(n_ifgs):
            # Get phase values for this interferogram
            ph = np.angle(ph_w[:,i])
            
            # Plot on white background
            axes[i].set_facecolor('white')
            
            # Scatter plot of phase values with smaller points
            scatter = axes[i].scatter(xy[:,1], xy[:,2], c=ph, cmap='jet', s=0.1)
            
            # Add colorbar
            plt.colorbar(scatter, ax=axes[i], label='Phase (rad)')
            
            axes[i].set_title(f'IFG {i+1}')
            axes[i].set_xlabel('X coordinate')
            axes[i].set_ylabel('Y coordinate')
            axes[i].axis('equal')
        
        # Remove empty subplots
        for i in range(n_ifgs, len(axes)):
            fig.delaxes(axes[i])
            
        plt.suptitle('Wrapped Phase Values and Unwrapping Grid')
        plt.tight_layout()
        
        # Save figure with high DPI for better clarity
        plt.savefig(os.path.join(self.config["processing_parameters"]["current_result"],
                                self.patch_dir, '6_wrapped_phase_before_unwrap_all.png'), 
                    dpi=300)
        plt.close()

        self._uw_grid_wrapped(ph_w, xy, options)
        del ph_w
        self._uw_interp()
        self._uw_sb_unwrap_space_time(day, ifgday_ix, bperp, options)
        self._uw_stat_costs(options['unwrap_method'], options['variance'], None)
        ph_uw, msd = self._uw_unwrap_from_grid(xy, options['grid_size'])
        return ph_uw, msd
    
    def _ps_unwrap(self):
        print("   -> Unwrapping...")
        small_baseline_flag = self.parms.get('small_baseline_flag')
        unwrap_patch_phase = self.parms.get('unwrap_patch_phase')
        scla_deramp = self.parms.get('scla_deramp')
        subtr_tropo = self.parms.get('subtr_tropo')
        aps_name = self.parms.get('tropo_method')

        self._update_psver(2, self.patch_dir)

        # Load PS version
        psname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'ps{self.psver}.mat')
        rcname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'rc{self.psver}.mat')
        pmname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'pm{self.psver}.mat')
        bpname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'bp{self.psver}.mat')

        if small_baseline_flag != 'y':
            sclaname = f'scla_smooth{self.psver}'
            apsname = f'tca{self.psver}'
            phuwname = f'phuw{self.psver}.mat'

        # Load PS data
        ps = sio.loadmat(psname)

        # Get drop_ifg_index and unwrap_ifg_index
        drop_ifg_index = self.parms.get('drop_ifg_index')
        if len(list(drop_ifg_index)) > 0:
            unwrap_ifg_index = []
            for idx in drop_ifg_index:
                if not idx - 1 in np.arange(ps['n_ifg'][0][0]):
                    unwrap_ifg_index.append(idx - 1)
        else:
            unwrap_ifg_index = list(np.arange(ps['n_ifg'][0][0]))

        # Load baseline data
        if os.path.exists(bpname):
            bp = sio.loadmat(bpname)
        else:
            bperp = ps['bperp']
            if small_baseline_flag != 'y':
                bperp = np.delete(bperp, ps['master_ix'][0][0])
            bp['bperp_mat'] = np.tile(bperp.T, (ps['n_ps'][0][0], 1))

        # Handle baseline matrix based on small baseline flag
        if small_baseline_flag != 'y':
            bperp_mat = np.hstack([bp['bperp_mat'][:, :ps['master_ix'][0][0]],
                                 np.zeros((ps['n_ps'][0][0], 1)),
                                 bp['bperp_mat'][:, ps['master_ix'][0][0]:]])
        else:
            bperp_mat = bp['bperp_mat']

        # Handle phase based on unwrap_patch_phase flag
        if unwrap_patch_phase == 'y':
            pm = sio.loadmat(pmname)
            ph_w = np.complex128(pm['ph_patch']) / np.abs(pm['ph_patch'])
            del pm
            if small_baseline_flag != 'y':
                ph_w = np.concatenate([ph_w[:, :ps['master_ix'][0][0]],
                                np.ones((ps['n_ps'][0][0], 1)),
                                ph_w[:, ps['master_ix'][0][0]:]])
        else:
            rc = sio.loadmat(rcname)
            ph_w = rc['ph_rc']
            del rc
            if os.path.exists(pmname):
                pm = sio.loadmat(pmname)
                if 'K_ps' in pm and not np.sum(pm['K_ps']) == 0:
                    ph_w = ph_w * np.exp(1j * (np.tile(pm['K_ps'], (1, ps['n_ifg'][0][0])) * bperp_mat))

        # Normalize phase
        ix = ph_w != 0
        ph_w[ix] = ph_w[ix] / np.abs(ph_w[ix])

        # Initialize flags
        self.scla_subtracted_sw = False
        self.ramp_subtracted_sw = False

        # Handle good values
        options = {'master_day': ps['master_day'][0][0]}
        unwrap_hold_good_values = self.parms['unwrap_hold_good_values']
        if small_baseline_flag != 'y' or not os.path.exists(phuwname):
            unwrap_hold_good_values = 'n'
            print('      -> Code to hold good values skipped')

        # Handle SCLA subtraction
        if small_baseline_flag != 'y' and os.path.exists(f'{os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, sclaname)}.mat'):
            print('      -> Subtracting SCLA and Master AOE...')
            scla = sio.loadmat(f'{os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, sclaname)}.mat')
            if scla['K_ps_uw'].shape[1] == ps['n_ps'][0][0]:
                self.scla_subtracted_sw = True
                with np.errstate(divide='ignore', invalid='ignore'):
                    ph_w = ph_w * np.exp(-1j * (np.tile(scla['K_ps_uw'], (ps['n_ifg'][0][0], 1)).T * bperp_mat))
                    ph_w = ph_w * np.tile(np.exp(-1j * scla['C_ps_uw']), (ps['n_ifg'][0][0], 1)).T
                if scla_deramp == 'y' and 'ph_ramp' in scla and scla['ph_ramp'].shape[0] == ps['n_ps'][0][0]:
                    self.ramp_subtracted_sw = True
                    ph_w = ph_w * np.exp(-1j * scla['ph_ramp'])
            else:
                print('      -> Wrong number of PS in SCLA - subtraction skipped...')
        # Handle APS subtraction
        if os.path.exists(f'{os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, apsname)}.mat') and subtr_tropo == 'y':
            print('      -> Subtracting Slave APS...')
            aps = sio.loadmat(f'{os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, apsname)}.mat')
            aps_corr = self._ps_plot_tca(aps, aps_name)[0]
            ph_w = ph_w * np.exp(-1j * aps_corr)

        # Set unwrapping options
        options.update({
            'time_win': self.parms['unwrap_time_win'],
            'unwrap_method': self.parms['unwrap_method'],
            'grid_size': self.parms['unwrap_grid_size'],
            'prefilt_win': self.parms['unwrap_gold_n_win'],
            'goldfilt_flag': self.parms['unwrap_prefilter_flag'],
            'gold_alpha': self.parms['unwrap_gold_alpha'],
            'la_flag': self.parms['unwrap_la_error_flag'],
            'scf_flag': self.parms['unwrap_spatial_cost_func_flag']
        })

        # Calculate max K
        max_topo_err = self.parms['max_topo_err']
        lambda_val = self.parms['lambda']
        rho = 830000  # mean range - need only be approximately correct

        if 'mean_incidence' in ps:
            inc_mean = ps['mean_incidence']
        else:
            laname = f'la{self.psver}.mat'
            if os.path.exists(laname):
                la = sio.loadmat(laname)
                inc_mean = np.mean(la['la']) + 0.052  # incidence angle approx equals look angle + 3 deg
                del la
            else:
                inc_mean = 21 * np.pi / 180  # guess the incidence angle

        max_K = max_topo_err / (lambda_val * rho * np.sin(inc_mean) / 4 / np.pi)
        bperp_range = np.amax(ps['bperp']) - np.amin(ps['bperp'])
        options['n_trial_wraps'] = bperp_range * max_K / (2 * np.pi)

        # Handle small baseline flag for unwrapping
        options['lowfilt_flag'] = 'n'
        ifgday_ix = np.column_stack([np.ones(ps['n_ifg'][0][0]) * ps['master_ix'][0][0],
                                    np.arange(ps['n_ifg'][0][0])])
        master_ix = np.sum(ps['master_day'] > ps['day'])
        unwrap_ifg_index = np.setdiff1d(unwrap_ifg_index, master_ix)
        day = ps['day'][0] - ps['master_day'][0][0]

        if unwrap_hold_good_values == 'y':
            options['ph_uw_predef'] = options['ph_uw_predef'][:, unwrap_ifg_index]

        # Perform unwrapping
        ph_uw_some, msd_some = self._uw_3d(ph_w[:, unwrap_ifg_index], ps['xy'], day,
                                            ifgday_ix[unwrap_ifg_index, :],
                                            ps['bperp'][:, unwrap_ifg_index][0], options)

        # Initialize output arrays
        ph_uw = np.zeros((ps['n_ps'][0][0], ps['n_ifg'][0][0]), dtype=np.float32)
        msd = np.zeros((ps['n_ifg'][0][0], 1), dtype=np.float32)
        ph_uw[:, unwrap_ifg_index] = ph_uw_some
        if 'msd_some' in locals():
            msd[unwrap_ifg_index] = msd_some

        # Add back SCLA and master AOE if subtracted
        if self.scla_subtracted_sw and small_baseline_flag != 'y':
            print('      -> Adding back SCLA and master AOE...')
            scla = sio.loadmat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'{sclaname}.mat'))
            with np.errstate(divide='ignore', invalid='ignore'):
                ph_uw = ph_uw + (np.tile(scla['K_ps_uw'], (ps['n_ifg'][0][0], 1)).T * bperp_mat)
                ph_uw = ph_uw + np.tile(scla['C_ps_uw'], (ps['n_ifg'][0][0], 1)).T
            if self.ramp_subtracted_sw:
                ph_uw = ph_uw + scla['ph_ramp']

        # Add back APS if subtracted
        if os.path.exists(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'{apsname}.mat')) and subtr_tropo == 'y':
            print('      -> Adding back slave APS...')
            aps = sio.loadmat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'{apsname}.mat'))
            aps_corr = self._ps_plot_tca(aps, aps_name)[0]
            ph_uw = ph_uw + aps_corr

        # Handle patch phase
        if unwrap_patch_phase == 'y':
            pm = sio.loadmat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'{pmname}.mat'))
            ph_w = pm['ph_patch'] / np.abs(pm['ph_patch'])
            del pm
            if small_baseline_flag != 'y':
                ph_w = np.hstack([ph_w[:, :ps['master_ix'][0][0]],
                                np.zeros((ps['n_ps'][0][0], 1)),
                                ph_w[:, ps['master_ix'][0][0]:]])
            rc = sio.loadmat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'{rcname}.mat'))
            ph_uw = ph_uw + np.angle(rc['ph_rc'] * np.conj(ph_w))

        # Set non-unwrapped interferograms to zero
        ph_uw[:, np.setdiff1d(np.arange(ps['n_ifg'][0][0]), unwrap_ifg_index)] = 0

        # Save results
        sio.savemat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, phuwname), {'ph_uw': ph_uw, 'msd': msd})

    def _aps_parms(self):
        """Set APS parameters to default values if not already set"""
        parent_flag = 0
        self.aps_parmfile = os.path.join(self.config["processing_parameters"]["current_result"], "parms_aps.json")
        
        # Load existing parameters if file exists
        if os.path.exists(self.aps_parmfile):
            with open(self.aps_parmfile, 'r') as f:
                self.aps_parms = json.load(f)
        else:
            self.aps_parms = {}
            
        # Set default parameters if not already set
        defaults = {
            # Basic flags
            'stamps_processed': 'y',  # Assuming we're in StaMPS environment
            'small_baseline_flag': 'n',
            
            # Sounding data parameters
            'sounding_data': 'n',
            'sounding_dir': os.path.join(os.getcwd(), 'sounding_data'),
            'sounding_time_stamp': ['00', '12'],
            'sounding_start_date': None,
            'sounding_end_date': None,
            'sounding_h0': 0,
            'sounding_error_promp': 'n',
            'sounding_h_alpha_thres': 4,
            'sounding_months': 1,
            'sounding_ifg_dates': 'n',
            'sounding_sensitivity': 'n',
            
            # Region specific parameters
            'region_res': 0.008333,
            'region_lon_range': None,
            'region_lat_range': None,
            'demfile': os.path.join(os.getcwd(), 'dummy.dem'),
            'dem_null': -32768,
            
            # Spectrometer parameters
            'spectrometer_scaleheight': 8340,
            'spectrometer_PIconversion': 6.2,
            
            # MERIS parameters
            'meris_perc_coverage': 80,
            'meris_datapath': None,
            
            # MODIS parameters
            'modis_datapath': None,
            'modis_calibration': 0.95,
            'modis_recalibrated': 'n',
            
            # ERA parameters
            'era_datapath': None,
            'era_data_type': 'ECMWF',
            
            # GACOS parameters
            'gacos_datapath': None,
            
            # MERRA parameters
            'merra_datapath': None,
            
            # NARR parameters
            'narr_datapath': None,
            
            # WRF parameters
            'wrf_datapath': None,
            
            # Powerlaw correction parameters
            'powerlaw_DEM_corr': 'n',
            'powerlaw_h0': 10,
            'powerlaw_n_patches': 50,
            'powerlaw_alpha': 1.6,
            'powerlaw_xy_res': [30, 30],
            'powerlaw_patch_overlap': 50,
            'powerlaw_all_bands': 'y',
            'powerlaw_spatial_bands': [[2000, 4000],
                                     [4000, 8000],
                                     [8000, 16000],
                                     [16000, 32000],
                                     [32000, 64000],
                                     [64000, 128000]],
            'powerlaw_ridge_constraint': 'n',
            'powerlaw_kept': 0,
            'powerlaw_plane_mode': 'y',
            
            # IFG specific parameters
            'crop_flag': 'n',
            'non_defo_flag': 'n',
            'save_folder_name': 'aps_estimation',
            'drop_ifg_index': [],
            
            # File paths
            'phuw_matfile': os.path.join(self.config["processing_parameters"]["current_result"], "PATCH_1", f'phuw{self.psver+1}.mat'),
            'hgt_matfile': os.path.join(self.config["processing_parameters"]["current_result"], "PATCH_1", f'hgt{self.psver+1}.mat'),
            'll_matfile': os.path.join(self.config["processing_parameters"]["current_result"], "PATCH_1", f'ps{self.psver+1}.mat'),
            'bperp_matfile': os.path.join(self.config["processing_parameters"]["current_result"], "PATCH_1", f'ps{self.psver+1}.mat'),
        }
        
        # Update parameters with defaults if not already set
        for key, value in defaults.items():
            if key not in self.aps_parms:
                self.aps_parms[key] = value
                
        # Save parameters
        with open(self.aps_parmfile, 'w') as f:
            json.dump(self.aps_parms, f, indent=4)
    
    def _aps_linear(self):
        """
        Compute the tropospheric delay map from a linear relation between phase and topography.
        The computed tropospheric delay is in the same units as the input interferogram phases.
        """
        print("   -> Computing linear tropospheric correction...")
        self._aps_parms()
        
        # Get parameters
        small_baseline_flag = self.aps_parms.get('small_baseline_flag')
        non_defo_flag = self.aps_parms.get('non_defo_flag', 'n')
        powerlaw_DEM_corr = self.aps_parms.get('powerlaw_DEM_corr', 'n')
        
        # Load required data
        psver = 2
        save_path = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir)
        
        # Load phase data
        phuw_file = os.path.join(save_path, f'phuw{psver}.mat' if small_baseline_flag != 'y' else f'phuw_sb{psver}.mat')
        phuw = sio.loadmat(phuw_file)['ph_uw']
        
        # Load lonlat data
        ll_file = os.path.join(save_path, f'ps{psver}.mat')
        ps_data = sio.loadmat(ll_file)
        lonlat = ps_data['lonlat'] if 'lonlat' in ps_data else None
        
        # Load height data
        hgt_file = os.path.join(save_path, f'hgt{psver}.mat')
        hgt = sio.loadmat(hgt_file)['hgt']
        
        # Initialize variables
        n_points = phuw.shape[0]
        n_dates = phuw.shape[1]
        
        # Convert heights to km if in meters
        if np.max(hgt) > 10:  # heights are in meters
            hgt = hgt / 1000  # convert to km
            
        # Handle non-deforming area if specified
        if non_defo_flag == 'y':
            non_defo_file = os.path.join(save_path, 'non_defo.mat')
            if os.path.exists(non_defo_file):
                non_defo = sio.loadmat(non_defo_file)
                poly = non_defo['poly']
                # Find points within non-deforming polygon
                from matplotlib.path import Path
                polygon = Path(poly)
                ix_points = polygon.contains_points(lonlat)
                ix_points = np.where(ix_points)[0]
            else:
                print("Warning: non_defo.mat not found, using all points")
                ix_points = np.arange(n_points)
        else:
            ix_points = np.arange(n_points)
            
        # Correct for DEM error if specified
        if powerlaw_DEM_corr == 'y' and n_dates > 5:
            # Load perpendicular baseline information
            bperp_file = os.path.join(save_path, f'bp{psver}.mat')
            bperp = sio.loadmat(bperp_file)['bperp']
            if bperp.shape[1] > 1:
                bperp = bperp.T
            if bperp.shape[1] > 1:
                raise ValueError('bperp is not a vector')
                
            # Estimate and remove DEM correlated errors
            DEM_corr_e = np.linalg.lstsq(bperp, phuw.T, rcond=None)[0].T
            phuw = phuw - np.tile(bperp.T, (n_points, 1)) * np.tile(DEM_corr_e, (1, n_dates))
        else:
            if powerlaw_DEM_corr == 'y' and n_dates <= 5:
                print('Not enough interferograms to make a reliable estimate for the DEM error')
            DEM_corr_e = np.zeros((n_points, 1))
            
        # Compute linear relation between phase and topography
        ph_tropo_linear = np.zeros((n_points, n_dates))
        
        for k in range(n_dates):
            # Remove NaN phases from consideration
            valid_points = ix_points[~np.isnan(phuw[ix_points, k])]
            
            # Set up design matrix
            A = np.column_stack((hgt[valid_points], np.ones_like(hgt[valid_points])))
            
            # Compute linear relation
            coeff = np.linalg.lstsq(A, phuw[valid_points, k], rcond=None)[0]
            
            # Compute delay for all points
            ph_tropo_linear[:, k] = (hgt * coeff[0] + coeff[1]).flatten()
            
            # Set NaN phases back to NaN
            ph_tropo_linear[np.isnan(phuw[:, k]), k] = np.nan
            
        # Save the results
        save_name = os.path.join(save_path, f'tca{psver}.mat' if small_baseline_flag != 'y' else f'tca_sb{psver}.mat')
        sio.savemat(save_name, {'ph_tropo_linear': ph_tropo_linear})
        
        return ph_tropo_linear
            
    def _ps_deramp(self, ps, ph_all, degree=1):
        """
        Deramps the inputted data and returns deramped data and ramp. Needs ps struct information.
        
        Parameters
        ----------
        ps : dict
            Dictionary containing PS information including xy coordinates
        ph_all : ndarray
            Phase data to deramp
        degree : float, optional
            Degree of polynomial to use for deramping (default=1)
            
        Returns
        -------
        ph_all : ndarray
            Deramped phase data
        ph_ramp : ndarray 
            Estimated ramp that was removed
        """
        print("   -> Deramping computed on the fly...")
        
        # Handle case where ph_all dimensions don't match ps.n_ifg
        if ps['n_ifg'][0][0] != ph_all.shape[1]:
            ps['n_ifg'][0][0] = ph_all.shape[1]
            
        # Set up design matrix based on degree
        if degree == 1:
            # z = ax + by + c
            A = np.column_stack((ps['xy'][:,1:3]/1000, np.ones((ps['n_ps'][0][0], 1))))
            print("      **** z = ax + by + c")
            
        elif degree == 1.5:
            # z = ax + by + cxy + d
            A = np.column_stack((ps['xy'][:,1:3]/1000, 
                               (ps['xy'][:,1]/1000 * ps['xy'][:,2]/1000).reshape(-1,1),
                               np.ones((ps['n_ps'][0][0], 1))))
            print("      **** z = ax + by + cxy + d")
            
        elif degree == 2:
            # z = ax^2 + by^2 + cxy + d
            A = np.column_stack(((ps['xy'][:,1:3]/1000)**2,
                               (ps['xy'][:,1]/1000 * ps['xy'][:,2]/1000).reshape(-1,1),
                               np.ones((ps['n_ps'][0][0], 1))))
            print("      **** z = ax^2 + by^2 + cxy + d")
            
        elif degree == 3:
            # z = ax^3 + by^3 + cx^2y + dy^2x + ex^2 + fy^2 + gxy + h
            A = np.column_stack(((ps['xy'][:,1:3]/1000)**3,
                               (ps['xy'][:,1]/1000)**2 * ps['xy'][:,2]/1000,
                               (ps['xy'][:,2]/1000)**2 * ps['xy'][:,1]/1000,
                               (ps['xy'][:,1:3]/1000)**2,
                               (ps['xy'][:,1]/1000 * ps['xy'][:,2]/1000).reshape(-1,1),
                               np.ones((ps['n_ps'][0][0], 1))))
            print("      **** z = ax^3 + by^3 + cx^2y + dy^2x + ex^2 + fy^2 + gxy + h")

        ph_ramp = np.full(ph_all.shape, np.nan)
        
        for k in range(ps['n_ifg'][0][0]):
            ix = np.isnan(ph_all[:,k])
            if ps['n_ps'][0][0] - np.sum(ix) > 5:
                coeff = np.linalg.lstsq(A[~ix,:], ph_all[~ix,k], rcond=None)[0]
                ph_ramp[:,k] = A @ coeff
                ph_all[:,k] = ph_all[:,k] - ph_ramp[:,k]
            else:
                print(f"Ifg {k} is not deramped")
                
        return ph_all, ph_ramp
    
    def _ps_calc_scla(self, use_small_baselines = False, coest_mean_vel=False):
        print("   -> Calculating SCLA...")
        small_baseline_flag = self.parms['small_baseline_flag']
        scla_method = self.parms['scla_method']
        subtr_tropo = self.parms['subtr_tropo']
        tropo_method = self.parms['tropo_method']
        
        if use_small_baselines:
            if small_baseline_flag != 'y':
                return
        if not use_small_baselines:
            scla_drop_index = self.parms['scla_drop_index']
        else:
            scla_drop_index = self.parms['sb_scla_drop_index']
            
        self.psver = 2
        psname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'ps{self.psver}.mat')
        bpname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'bp{self.psver}.mat')
        meanvname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'mv{self.psver}.mat')
        ifgstdname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'ifgstd{self.psver}.mat')
        
        if not use_small_baselines:
            phuwname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'phuw{self.psver}.mat')
            sclaname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'scla{self.psver}.mat')
            apsname_old = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'aps{self.psver}.mat')
            apsname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'tca{self.psver}.mat')
        else:
            phuwname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'phuw_sb{self.psver}.mat')
            sclaname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'scla_sb{self.psver}.mat')
            apsname_old = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'aps_sb{self.psver}.mat')
            apsname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'tca_sb{self.psver}.mat')
        
        if not use_small_baselines:
            if os.path.exists(meanvname):
                os.unlink(meanvname)

        ps = sio.loadmat(psname)
        if os.path.exists(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'bp{self.psver}.mat')):
            bp = sio.loadmat(bpname)
        else:
            bperp = ps['bperp']
            if small_baseline_flag != 'y':
                bperp = bperp[:ps['master_ix'], ps['master_ix']+1:]
            bp["bperp_mat"] = np.tile(bperp, (ps['n_ps'][0][0], 1))

        uw = sio.loadmat(phuwname)
        if subtr_tropo == 'y':
            aps = sio.loadmat(apsname)
            aps_corr, fig_name_tca, tropo_method = self._ps_plot_tca(aps, tropo_method)
            uw["ph_uw"] = uw["ph_uw"] - aps_corr
        else:
            ph_ramp = np.array([])

        unwrap_ifg_index = np.arange(ps['n_image'][0][0])
        unwrap_ifg_index = np.setdiff1d(unwrap_ifg_index, scla_drop_index)

        if os.path.exists(apsname_old):
            if subtr_tropo == 'y':
                print(f"         -> You are removing atmosphere twice. Do not do this, either do:\n         use {apsname_old} with subtr_tropo='n'\n         remove {apsname_old} use subtr_tropo='y'\n")
            aps = sio.loadmat(apsname_old)
            uw["ph_uw"] = uw["ph_uw"] - aps["ph_aps_slave"]

            ref_ps = self._ps_setref()
            uw["ph_uw"] = uw["ph_uw"] - np.tile(np.nanmean(uw["ph_uw"][ref_ps,:], axis=0), (ps["n_ps"][0][0], 1))

        if not use_small_baselines:
            bperp_mat = np.hstack((bp["bperp_mat"][:,:ps["master_ix"][0][0]-1], 
                                    np.zeros((ps["n_ps"][0][0],1), dtype='float32'),
                                    bp["bperp_mat"][:,ps["master_ix"][0][0]-1:]))
            
            day = np.diff(ps["day"][0][unwrap_ifg_index])
            ph = np.diff(uw["ph_uw"][:,unwrap_ifg_index], axis=1).astype('float64')
            bperp = np.diff(bperp_mat[:,unwrap_ifg_index], axis=1)
        else:
            bperp_mat = bp["bperp_mat"]
            bperp = bperp_mat[:,unwrap_ifg_index]
            day = ps["ifgday"][unwrap_ifg_index,1] - ps["ifgday"][unwrap_ifg_index,0]
            ph = uw["ph_uw"][:,unwrap_ifg_index].astype('float64')
        del bp

        bprint = np.mean(bperp, axis=0)
        print(f"      -> {ph.shape[1]} ifgs used in estimation:")

        for i in range(ph.shape[1]):
            if use_small_baselines:
                print(f"         {ps['ifgday'][unwrap_ifg_index[i],0]} to {ps['ifgday'][unwrap_ifg_index[i],1]} {day[i]:5d} days {round(bprint[i]):5d} m")
            else:
                print(f"         {datetime.fromordinal(ps['day'][0][unwrap_ifg_index[i]]).strftime('%Y/%m/%d')} to {datetime.fromordinal(ps['day'][0][unwrap_ifg_index[i]+1]).strftime('%Y/%m/%d')} {day[i]:5d} days {round(bprint[i]):5d} m")

        K_ps_uw = np.zeros((ps["n_ps"][0][0], 1))

        if not coest_mean_vel or len(unwrap_ifg_index) < 4:
            G = np.column_stack((np.ones(ph.shape[1]), np.mean(bperp, axis=0)))
        else:
            G = np.column_stack((np.ones(ph.shape[1]), np.mean(bperp, axis=0), day))

        if os.path.exists(ifgstdname):
            ifgstd = sio.loadmat(ifgstdname)
            ifg_vcm = np.diag((ifgstd["ifg_std"] * np.pi/180)**2).astype('float64')

        m = np.linalg.lstsq(G, ph.T, rcond=None)[0]
        K_ps_uw = m[1,:]

        if scla_method == 'L1':
            from scipy.optimize import fmin
            for i in tqdm(range(ps["n_ps"][0][0]), desc="      -> L1-norm SCLA estimation"):
                d = ph[i,:].reshape(-1,1)
                m2 = m[:,i].reshape(-1,1)
                m2 = fmin(lambda x: np.sum(np.abs(d - G @ x.reshape(-1,1))), m2, disp=False)
                K_ps_uw[i] = m2[1]

        ph_scla = np.tile(K_ps_uw.reshape(-1,1), (1,bperp_mat.shape[1])) * bperp_mat

        if not use_small_baselines:
            unwrap_ifg_index = np.setdiff1d(unwrap_ifg_index, ps["master_ix"][0][0])
            if not coest_mean_vel:
                C_ps_uw = np.mean(uw["ph_uw"][:,unwrap_ifg_index] - ph_scla[:,unwrap_ifg_index], axis=1)
            else:
                G = np.column_stack((np.ones(len(unwrap_ifg_index)), 
                                ps["day"][0][unwrap_ifg_index] - ps["day"][0][ps["master_ix"][0][0]]))
                m = np.linalg.lstsq(G, (uw["ph_uw"][:,unwrap_ifg_index] - ph_scla[:,unwrap_ifg_index]).T, rcond=None)[0]
                C_ps_uw = m[0,:]
        else:
            C_ps_uw = np.zeros(ps["n_ps"][0][0])

        oldscla = glob.glob(f"{sclaname}.mat")
        if oldscla:
            oldscla = oldscla[0]
            olddatenum = os.path.getmtime(oldscla)
            olddate = datetime.fromtimestamp(olddatenum).strftime('%Y%m%d_%H%M%S')
            os.rename(sclaname + '.mat', f'tmp_{os.path.basename(sclaname)[3:]}_{olddate}.mat')

        sio.savemat(sclaname, {'ph_scla': ph_scla, 'K_ps_uw': K_ps_uw, 
                            'C_ps_uw': C_ps_uw, 'ph_ramp': ph_ramp, 
                            'ifg_vcm': ifg_vcm})

    def _ps_smooth_scla(self, use_small_baselines=False):
        print("   -> Smoothing SCLA...")
        psver = 2
        psname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'ps{psver}.mat')
        bpname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'bp{psver}.mat')
        
        if not use_small_baselines:
            sclaname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'scla{psver}.mat')
            sclasmoothname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'scla_smooth{psver}.mat')
        else:
            sclaname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'scla_sb{psver}.mat')
            sclasmoothname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'scla_smooth_sb{psver}.mat')

        ps = sio.loadmat(psname)
        scla = sio.loadmat(sclaname)
        K_ps_uw = scla['K_ps_uw'][0]
        C_ps_uw = scla['C_ps_uw'][0]
        ph_ramp = scla['ph_ramp']
        
        n_ps = ps['n_ps'][0][0]

        print(f"      -> Number of points per ifg: {n_ps}")

        xy = ps['xy'].astype('float64')
        tri = Delaunay(xy[:,1:3]).simplices
        edgs = np.unique(np.sort(np.vstack((tri[:,[0,1]], 
                                            tri[:,[1,2]], 
                                            tri[:,[2,0]])), axis=1), axis=0)
        n_edge = edgs.shape[0]

        print(f"      -> Number of arcs per ifg: {n_edge}")

        Kneigh_min = np.full(n_ps, np.inf, dtype='float32')
        Kneigh_max = np.full(n_ps, -np.inf, dtype='float32')
        Cneigh_min = np.full(n_ps, np.inf, dtype='float32')
        Cneigh_max = np.full(n_ps, -np.inf, dtype='float32')

        for i in tqdm(range(n_edge), desc="      -> Processing arcs"):
            ix = edgs[i,:2]
            ix1, ix2 = ix[0], ix[1]
            Kneigh_min[ix1] = min(Kneigh_min[ix1], K_ps_uw[ix2])
            Kneigh_min[ix2] = min(Kneigh_min[ix2], K_ps_uw[ix1])
            Kneigh_max[ix1] = max(Kneigh_max[ix1], K_ps_uw[ix2]) 
            Kneigh_max[ix2] = max(Kneigh_max[ix2], K_ps_uw[ix1])
            Cneigh_min[ix1] = min(Cneigh_min[ix1], C_ps_uw[ix2])
            Cneigh_min[ix2] = min(Cneigh_min[ix2], C_ps_uw[ix1])
            Cneigh_max[ix1] = max(Cneigh_max[ix1], C_ps_uw[ix2])
            Cneigh_max[ix2] = max(Cneigh_max[ix2], C_ps_uw[ix1])

        # Reduce outliers
        ix1 = K_ps_uw > Kneigh_max
        ix2 = K_ps_uw < Kneigh_min
        K_ps_uw[ix1] = Kneigh_max[ix1.flatten()]
        K_ps_uw[ix2] = Kneigh_min[ix2.flatten()]

        ix1 = C_ps_uw > Cneigh_max
        ix2 = C_ps_uw < Cneigh_min
        C_ps_uw[ix1] = Cneigh_max[ix1]
        C_ps_uw[ix2] = Cneigh_min[ix2]

        bp = sio.loadmat(bpname)
        if not use_small_baselines:
            master_ix = ps['master_ix'][0][0]-1
            bperp_mat = np.hstack((bp['bperp_mat'][:,:master_ix],
                                    np.zeros((ps['n_ps'][0][0],1)),
                                    bp['bperp_mat'][:,master_ix:]))
        else:
            bperp_mat = bp['bperp_mat']

        with np.errstate(invalid='ignore'):
            ph_scla = np.multiply(bperp_mat.T, K_ps_uw).T
        ph_scla = np.nan_to_num(ph_scla, nan=0.0)

        sio.savemat(sclasmoothname, {'K_ps_uw': K_ps_uw, 'C_ps_uw': C_ps_uw,
                                    'ph_scla': ph_scla, 'ph_ramp': ph_ramp})

    def _ps_scn_filt(self):
        print("   -> Estimating other spatially-correlated noise...")
        # Get parameters
        psver = 2
        time_win = self.parms['scn_time_win']
        deramp_ifg = self.parms['scn_deramp_ifg']
        scn_wavelength = self.parms['scn_wavelength']
        drop_ifg_index = self.parms['drop_ifg_index']
        small_baseline_flag = self.parms['small_baseline_flag']

        # Load data files
        ps = sio.loadmat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'ps{psver}.mat'))
        uw = sio.loadmat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'phuw{psver}.mat'))

        if small_baseline_flag == 'y':
            unwrap_ifg_index = np.arange(ps['n_image'][0][0])
        else:
            unwrap_ifg_index = np.setdiff1d(np.arange(ps['n_ifg'][0][0]), drop_ifg_index)

        day = ps['day'][0][unwrap_ifg_index]
        master_ix = np.sum(ps['master_day'][0][0] > ps['day'][0]) 
        n_ifg = len(unwrap_ifg_index)
        n_ps = ps['n_ps'][0][0]

        ph_all = uw['ph_uw'][:,unwrap_ifg_index].astype('float32')
        
        sclaname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'scla{psver}.mat')
        if os.path.exists(sclaname):
            scla = sio.loadmat(sclaname)
            ph_all = ph_all - scla['ph_scla'][:,unwrap_ifg_index].astype('float32')
            ph_all = ph_all - np.tile(scla['C_ps_uw'].astype('float32'), (ph_all.shape[1], 1)).T
            if scla['ph_ramp'].shape[1] > 0:
                ph_all = ph_all - scla['ph_ramp'][:,unwrap_ifg_index].astype('float32')

        ph_all[np.isnan(ph_all)] = 0

        print(f"      -> Number of points per ifg: {n_ps}")

        # Use Delaunay triangulation
        points = ps['xy'][:,1:3]  # Get x,y coordinates
        tri = Delaunay(points)
        
        # Get edges from triangulation
        edges = set()
        for simplex in tri.simplices:
            edges.add(tuple(sorted([simplex[0], simplex[1]])))
            edges.add(tuple(sorted([simplex[1], simplex[2]])))
            edges.add(tuple(sorted([simplex[2], simplex[0]])))
            
        # Convert edges to array format
        edges_nz = np.array([[i+1, e[0]+1, e[1]+1, 0] for i,e in enumerate(edges)])

        # Deramp interferograms
        if isinstance(deramp_ifg, str) and deramp_ifg.lower() == 'all':
            deramp_ifg = np.arange(ps['n_ifg'][0][0])
        deramp_ifg = np.intersect1d(deramp_ifg, unwrap_ifg_index)
        deramp_ix = np.zeros(deramp_ifg.shape)
        ph_ramp = np.zeros((n_ps, len(deramp_ifg)))

        if len(deramp_ifg) > 0:
            print('      -> Deramping selected ifgs...')
            G = np.column_stack((np.ones(n_ps), ps['xy'][:,1:3]))

            for i in range(len(deramp_ifg)):
                i3 = np.where(unwrap_ifg_index == deramp_ifg[i])[0][0]
                deramp_ix[i] = i3
                d = ph_all[:,i3]
                m = np.linalg.lstsq(G, d, rcond=None)[0]
                ph_this_ramp = G @ m
                ph_all[:,i3] = ph_all[:,i3] - ph_this_ramp
                ph_ramp[:,i] = ph_this_ramp

            scnname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'scn{psver}.mat')
            sio.savemat(scnname, {'ph_ramp': ph_ramp})

        # Smooth in time using gaussian moving window
        isnanix = np.isnan(uw['ph_uw'])
        uw['ph_uw'][isnanix] = 0
        dph = ph_all[edges_nz[:,2].astype(int)-1,:] - ph_all[edges_nz[:,1].astype(int)-1,:]
        dph_lpt = np.zeros(dph.shape)
        n_edges = dph.shape[0]

        print('      -> Low-pass filtering pixel-pairs in time...')

        for i1 in range(n_ifg):
            time_diff_sq = (day[i1] - day)**2
            weight_factor = np.exp(-time_diff_sq/(2*time_win**2))
            weight_factor[master_ix] = 0  # leave out master
            weight_factor = weight_factor/np.sum(weight_factor)
            dph_lpt[:,i1] = np.sum(dph * np.tile(weight_factor, (n_edges,1)), axis=1)

        dph_hpt = dph - dph_lpt

        ph_hpt = np.zeros((n_ps-1, n_ifg))
        ref_ix = 0

        # Build sparse matrix A
        rows = np.concatenate([np.arange(n_edges), np.arange(n_edges)])
        cols = np.concatenate([edges_nz[:,1].astype(int)-1, edges_nz[:,2].astype(int)-1])
        data = np.concatenate([-np.ones(n_edges), np.ones(n_edges)])
        A = sparse.csr_matrix((data, (rows, cols)))
        A = A[:,np.concatenate([np.arange(ref_ix), np.arange(ref_ix+1,n_ps)])]

        for i in tqdm(range(n_ifg), desc="         Processing ifgs"):
            ph_hpt[:,i] = sparse.linalg.spsolve(A.T @ A, A.T @ dph_hpt[:,i])

        ph_hpt = np.vstack((ph_hpt[:ref_ix,:], np.zeros((1,n_ifg)), ph_hpt[ref_ix:,:]))

        ph_hpt[:,deramp_ix.astype(int)] = ph_hpt[:,deramp_ix.astype(int)] + ph_ramp
        ph_hpt = ph_hpt.astype('float32')

        sigma_sq_times_2 = 2 * scn_wavelength**2
        ph_scn = np.full((n_ps, n_ifg), np.nan)
        patch_dist = scn_wavelength * 4
        patch_dist_sq = patch_dist**2
        ix_range = int(np.ceil(n_ps/(np.max(ps['xy'][:,2]) - np.min(ps['xy'][:,2])) * patch_dist * 0.2))
        ix1 = 0
        ix2 = ix_range
        ps['xy'] = np.column_stack((np.arange(n_ps), ps['xy']))

        print('      -> Low-pass filtering in space...')

        for i in tqdm(range(n_ps), desc="         Processing pixels"):
            x_min = ps['xy'][i,1] - patch_dist
            x_max = ps['xy'][i,1] + patch_dist
            y_min = ps['xy'][i,2] - patch_dist
            y_max = ps['xy'][i,2] + patch_dist

            ix1 = min(ix1 + ix_range, n_ps-1)
            while ix1 > 0 and ps['xy'][ix1-1,2] >= y_min:
                ix1 = max(ix1 - ix_range, 0)

            ix2 = max(ix2 - ix_range, 0)
            while ix2 < n_ps-1 and ps['xy'][ix2+1,2] <= y_max:
                ix2 = min(ix2 + ix_range, n_ps-1)

            xy_near = ps['xy'][ix1:ix2+1,:]
            mask = ((xy_near[:,1] >= x_min) & (xy_near[:,1] <= x_max) & 
                   (xy_near[:,2] >= y_min) & (xy_near[:,2] <= y_max))
            xy_near = xy_near[mask]

            dist_sq = (xy_near[:,1] - ps['xy'][i,1])**2 + (xy_near[:,2] - ps['xy'][i,2])**2
            in_range_ix = dist_sq < patch_dist_sq
            xy_near = xy_near[in_range_ix]
            dist_sq = dist_sq[in_range_ix]

            weight_factor = np.exp(-dist_sq/sigma_sq_times_2)
            weight_factor = weight_factor/np.sum(weight_factor)

            ph_scn[i,:] = weight_factor @ ph_hpt[xy_near[:,0].astype(int),:]

        # Re-reference to first PS
        ph_scn = ph_scn - np.tile(ph_scn[0,:], (n_ps,1))
        ph_scn_slave = np.zeros(uw['ph_uw'].shape)
        ph_scn_slave[:,unwrap_ifg_index] = ph_scn
        ph_scn_slave[:,master_ix] = 0

        scnname = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'scn{psver}.mat')
        sio.savemat(scnname, {'ph_scn_slave': ph_scn_slave, 'ph_hpt': ph_hpt, 'ph_ramp': ph_ramp})

    def _stamps_1(self):
        print("-> Loading PS candidates...")
        self.parms.load()
        self.small_baseline_flag = self.parms.get('small_baseline_flag')
        if self.start_step >= 1 and self.end_step < 2:
            if self.small_baseline_flag == "y":
                # stamps.m: sb_load_initial_gamma;
                None
            else:
                self._ps_load_initial_gamma()
            self.stamps_step_no_ps[0:] = 0
            self._save_ps_info()
        elif self.start_step <= 4:
            self._update_psver(1, self.patch_dir)
        
    def _stamps_2(self, est_gamma_flag=None):
        print("-> Estimating coherence of PS candidates...")
        self.parms.load()
        self.quick_est_gamma_flag = self.parms.get('quick_est_gamma_flag')
        if self.start_step <= 2 and self.end_step >= 2:
            self.stamps_step_no_ps[1:] = 0
            self._save_ps_info()
            if self.stamps_step_no_ps[0] == 0:
                if self.quick_est_gamma_flag == 'y':
                    self._ps_est_gamma_quick(est_gamma_flag)
            else:
                self.stamps_step_no_ps[1] = 1
                print("-> No PS left in step 1, so will skip step 2")
        self._save_ps_info()

    def _stamps_3(self, plot_flag=False):
        self.parms.load()
        self.quick_est_gamma_flag = self.parms.get('quick_est_gamma_flag')
        self.reest_gamma_flag = self.parms.get('reest_gamma_flag')
        if self.start_step <= 3 and self.end_step >= 3:
            self.stamps_step_no_ps[2:] = 0
            self._save_ps_info()
            if self.stamps_step_no_ps[1] == 0:
                if self.quick_est_gamma_flag == 'y' and self.reest_gamma_flag == 'y':
                    self._ps_select(plot_flag=plot_flag)
                else:
                    self._ps_select(reselect=1, plot_flag=plot_flag)
            else:
                self.stamps_step_no_ps[2] = 1
                print("-> No PS left in step 2, so will skip step 3")
        self._save_ps_info()

    def _stamps_4(self):
        self.parms.load()
        self.small_baseline_flag = self.parms.get('small_baseline_flag')
        if self.start_step <= 4 and self.end_step >= 4:
            self.stamps_step_no_ps[3:] = 0
            self._save_ps_info()
            if self.stamps_step_no_ps[2] == 0:
                if self.small_baseline_flag == 'y':
                    self._ps_weed(True, False)
                else:
                        self._ps_weed()
            else:
                self.stamps_step_no_ps[3] = 1
                print("-> No PS left in step 3, so will skip step 4")
        self._save_ps_info()

    def _stamps_5(self):
        if self.stamps_PART1_flag == True and self.stamps_PART2_flag == False:
            self.parms.load()
            if self.start_step <= 5 and self.end_step >= 5:
                self.stamps_step_no_ps[4:] = 0
                self._save_ps_info()
                if self.stamps_step_no_ps[3] == 0:
                    self._ps_correct_phase()
                else:
                    self.stamps_step_no_ps[4] = 1
                    print("-> No PS left in step 4, so will skip step 5")
            self._save_ps_info()
        elif self.stamps_PART1_flag == False and self.stamps_PART2_flag == True:
            self.parms.load()
            abord_flag = False
            self._ps_merge_patches()
            if self.n_ps == 0:
                abord_flag = True
            if abord_flag == False:
                self._ps_calc_ifg_std()
            else:
                print("No PS left in step 4, so will skip step 5")
            # self._ps_calc_ifg_std()

    def _stamps_6(self, aps_flag=False):
        if self.start_step <= 6 and self.end_step >= 6:
            self._ps_unwrap()
            if aps_flag:
                self._aps_linear()

    def _stamps_7(self):
        if self.start_step <= 7 and self.end_step >= 7:
            if self.small_baseline_flag == 'y':
                self._ps_calc_scla(True, True)
                self._ps_smooth_scla(True)
                self._ps_calc_scla(False, True)
            else:
                self._ps_calc_scla(False, True)
                self._ps_smooth_scla()

        psver = 2
        uw = sio.loadmat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'phuw{psver}.mat'))
        ps = sio.loadmat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'ps{psver}.mat'))
        v_dao, fig_name = self._ps_plot_v_dao(uw['ph_uw'])

        # Initialize lists for valid data
        valid_lons = []
        valid_lats = []
        valid_v_dao = []

        # Get valid points
        for i in range(len(v_dao)):
            if not np.isnan(v_dao[i]):
                valid_lons.append(ps['lonlat'][i,0])
                valid_lats.append(ps['lonlat'][i,1]) 
                valid_v_dao.append(v_dao[i])

        # Convert to arrays
        valid_lons = np.array(valid_lons)
        valid_lats = np.array(valid_lats)
        valid_v_dao = np.array(valid_v_dao)

        # Plot
        plt.figure(figsize=(12, 8))
        plt.scatter(valid_lons, valid_lats, c=valid_v_dao, cmap='jet', s=1)
        plt.colorbar(label='Velocity (mm/yr)')
        plt.title('PS Velocity Map')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config["processing_parameters"]["current_result"],
                                self.patch_dir, f'7_v_dao.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _ps_setref(self, ps2):
        """Find reference PS.
        
        Parameters
        ----------
        ps2 : dict, optional
            Dictionary containing PS information. If not provided, loads from ps{psver}.mat
            
        Returns
        -------
        ref_ps : ndarray
            Array of reference PS indices
        """
        if ps2 is None:
            psver = 2
            psname = os.path.join(self.config["processing_parameters"]["current_result"], 
                                self.patch_dir, f'ps{psver}.mat')
            ps2 = sio.loadmat(psname)
        else:
            psver = 2
            psname = os.path.join(self.config["processing_parameters"]["current_result"], 
                                self.patch_dir, f'ps{psver}.mat')
            ps_temp = sio.loadmat(psname)
            ps2['ll0'] = ps_temp['ll0']
            ps2['n_ps'] = ps2['lonlat'].shape[0]

        ref_lon = self.parms['ref_lon']
        ref_lat = self.parms['ref_lat']
        ref_centre_lonlat = self.parms['ref_centre_lonlat']
        ref_radius = self.parms['ref_radius']

        if ref_radius == -np.inf:
            ref_ps = 0
        else:
            ref_ps = np.where((ps2['lonlat'][:,0] > ref_lon[0]) &
                            (ps2['lonlat'][:,0] < ref_lon[1]) &
                            (ps2['lonlat'][:,1] > ref_lat[0]) &
                            (ps2['lonlat'][:,1] < ref_lat[1]))[0]
            
            if ref_radius < np.inf:
                ref_xy = self._llh2local(ref_centre_lonlat.T, ps2['ll0']) * 1000
                xy = self._llh2local(ps2['lonlat'][ref_ps,:].T, ps2['ll0']) * 1000
                dist_sq = (xy[0,:] - ref_xy[0])**2 + (xy[1,:] - ref_xy[1])**2
                ref_ps = ref_ps[dist_sq <= ref_radius**2]

        if len(ref_ps) == 0:
            if ps2 is not None:
                print('None of your external data points have a reference, all are set as reference.')
                ref_ps = np.arange(ps2['n_ps'][0][0])
        
        if ps2 is None:
            if ref_ps == 0:
                print('No reference set')
            else:
                print(f'{len(ref_ps)} ref PS selected')

        return ref_ps

    def _ps_plot_v_dao(self, ph_uw, ts_flag=False):
        """Process v-dao case for phase plotting.
        
        Args:
            ph_uw: Unwrapped phase data
            
        Returns:
            Processed phase data and figure name
        """
        psver = 2
        scla = sio.loadmat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'scla{psver}.mat'))
        aps = sio.loadmat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'tca{psver}.mat'))
        ps = sio.loadmat(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, f'ps{psver}.mat'))
        
        # Get APS correction and figure name
        aps_corr, fig_name_tca, aps_flag = self._ps_plot_tca(aps, self.parms["tropo_method"])
        
        fig_name = f'v-dao{fig_name_tca}'
        
        # Apply corrections
        ph_uw = ph_uw - scla['ph_scla'] - aps_corr
        
        # Deramp interferograms
        ph_uw, ph_ramp = self._ps_deramp(ps, ph_uw)
        ph_uw = np.nan_to_num(ph_uw, nan=0)

        if ts_flag:
            ph_uw = ph_uw - np.tile(scla['C_ps_uw'].astype('float32'), (ph_uw.shape[1], 1)).T
        v_dao = np.zeros((ps['n_ps'][0][0], 1))
        ref_ps = self._ps_setref(ps)
        unwrap_ifg_index = np.setdiff1d(np.arange(ps['n_ifg'][0][0]), ps['master_ix'][0])
        ph_uw = ph_uw[:, unwrap_ifg_index]
        ph_uw = ph_uw - np.tile(np.nanmean(ph_uw[ref_ps,:],1), (ph_uw.shape[1], 1)).T
        G = np.column_stack((np.ones((len(ps['day'][0][unwrap_ifg_index]),1)), ps['day'][0][unwrap_ifg_index]))
        lambda_ = self.parms['lambda']
        m = np.linalg.lstsq(G, ph_uw.astype('float64').T*lambda_/(4*np.pi), rcond=None)[0]
        v_dao = -m[1,:].T*365.25/(4*np.pi)*lambda_*1000
        # Save v_dao to ph_disp in ps_plot_v-dao.mat
        sio.savemat(os.path.join(self.config["processing_parameters"]["current_result"], 
                                self.patch_dir, 'ps_plot_v-dao.mat'), 
                   {'ph_disp': v_dao})
        # Save ph_uw to ph_mm in ps_plot_ts_v-dao.mat
        sio.savemat(os.path.join(self.config["processing_parameters"]["current_result"],
                                self.patch_dir, 'ps_plot_ts_v-dao.mat'),
                   {'ph_mm': ph_uw*lambda_/(4*np.pi)*1000,
                    'day': ps['day'][0][unwrap_ifg_index]})
        
        return v_dao, fig_name

    def _stamps_8(self):
        if self.start_step <= 8 and self.end_step >= 8:
            self._ps_scn_filt()

    def run(self, start_step=None, end_step=None, patches_flag=False, plot_flag=False, aps_flag=False):
        if start_step is None:
            start_step = self.start_step
        else:
            self.start_step = start_step
        if end_step is None:
            end_step = self.end_step
        else:
            self.end_step = end_step
        if patches_flag == False:
            self.stamps_PART1_flag = True
            self.stamps_PART2_flag = False
        else:
            self.stamps_PART1_flag = False
            self.stamps_PART2_flag = True

        # Get patch directories
        self.patch_dirs, patches_flag = self._handle_patches(patches_flag)

        # Process each patch
        if self.stamps_PART1_flag:
            for patch_dir in self.patch_dirs:
                if patch_dir != self.config["processing_parameters"]["current_result"]:
                    self.patch_dir = patch_dir
                
                # Initialize or load no_ps_info.mat
                self._initialize_ps_info()

                if self.start_step == 0:
                    # Check processing stage of stamps for all patches
                    # Step 4 find a ps_weed file
                    # Step 3 find a ps_select file
                    # Step 2 find a pm file
                    # Step 1 find a ps file
                    if os.path.exists(os.path.join(self.config["processing_parameters"]["current_result"], patch_dir, 'weed1.mat')):
                        self.start_step = 5
                        self._update_psver(2, self.patch_dir)
                    elif os.path.exists(os.path.join(self.config["processing_parameters"]["current_result"], patch_dir, 'select1.mat')):
                        self.start_step = 4
                    elif os.path.exists(os.path.join(self.config["processing_parameters"]["current_result"], patch_dir, 'pm1.mat')):
                        self.start_step = 3
                    elif os.path.exists(os.path.join(self.config["processing_parameters"]["current_result"], patch_dir, 'ps1.mat')):
                        self.start_step = 2
                    else:
                        self.start_step = 1
                    
                    if self.start_step > self.end_step:
                        print(f"   -> {patch_dir}: already up to end stage {self.end_step}")
                    else:
                        print(f"   -> {patch_dir}: complete up to stage {self.start_step - 1}")

                for step in range(self.start_step, self.end_step + 1):
                    if step == 3:
                        self.control_flow[step](plot_flag)
                    else:
                        self.control_flow[step]()

        if self.stamps_PART2_flag:
            if len(self.patch_dirs) > 1:
                for patch_dir in self.patch_dirs:
                    self.patch_dir = patch_dir
                    if patches_flag:
                        # Create new patch list file
                        with open(f'{self.config["processing_parameters"]["current_result"]}/patch.list_new', 'w') as f:
                            # Process patches in reverse order
                            for patch_dir in reversed(self.patch_dirs):
                                # Check PS information file
                                ps_info_file = os.path.join(self.config["processing_parameters"]["current_result"], patch_dir, 'no_ps_info.mat')
                                
                                # Assume to keep patch by default for backward compatibility
                                keep_patch = True
                                if os.path.exists(ps_info_file):
                                    ps_info = sio.loadmat(ps_info_file)
                                    if np.sum(ps_info['stamps_step_no_ps']) >= 1:
                                        keep_patch = False
                                
                                # Update patch list
                                if keep_patch:
                                    f.write(f"{patch_dir}\n")
                        
                        # Update patch list files
                        if os.path.exists(os.path.join(self.config["processing_parameters"]["current_result"], 'patch.list')):
                            if os.path.exists(os.path.join(self.config["processing_parameters"]["current_result"], 'patch.list_old')):
                                os.remove(os.path.join(self.config["processing_parameters"]["current_result"], 'patch.list_old'))
                            time.sleep(1)
                            os.rename(os.path.join(self.config["processing_parameters"]["current_result"], 'patch.list'), os.path.join(self.config["processing_parameters"]["current_result"], 'patch.list_old'))
                        time.sleep(1)
                        os.rename(os.path.join(self.config["processing_parameters"]["current_result"], 'patch.list_new'), os.path.join(self.config["processing_parameters"]["current_result"], 'patch.list'))

                    for step in range(5, self.end_step + 1):
                        if step == 6:
                            self.control_flow[step](aps_flag)
                        else:
                            self.control_flow[step]()
            else:
                self.patch_dir = "PATCH_1"
                # Create new patch list file
                with open(f'{self.config["processing_parameters"]["current_result"]}/{self.patch_dir}/patch.list_new', 'w') as f:
                    ps_info_file = os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, 'no_ps_info.mat')
                    keep_patch = True
                    if os.path.exists(ps_info_file):
                        ps_info = sio.loadmat(ps_info_file)
                        if np.sum(ps_info['stamps_step_no_ps']) >= 1:
                            keep_patch = False
                    
                    # Update patch list
                    if keep_patch:
                        f.write(f"{self.patch_dir}\n")
                    
                # Update patch list files
                if os.path.exists(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, 'patch.list')):
                    if os.path.exists(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, 'patch.list_old')):
                        os.remove(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, 'patch.list_old'))
                    time.sleep(1)
                    os.rename(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, 'patch.list'), os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, 'patch.list_old'))
                time.sleep(1)
                os.rename(os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, 'patch.list_new'), os.path.join(self.config["processing_parameters"]["current_result"], self.patch_dir, 'patch.list'))

                for step in range(self.start_step, self.end_step + 1):
                    if step == 6:
                        self.control_flow[step](aps_flag)
                    else:
                        self.control_flow[step]()

    def run_normal(self):
        with open("in.json", "r") as f:
            in_json = json.load(f)
        self.project_name = in_json['project_name']
        parms = Parms(self.project_name)
        parms.initialize()
        parms.load()
        parms.set('max_topo_err', in_json['max_topo_err'])
        parms.set('gamma_change_convergence', in_json['gamma_change_convergence'])
        parms.set('filter_grid_size', in_json['filter_grid_size'])
        parms.set('select_method', in_json['select_method'])
        parms.set('percent_rand', in_json['percent_rand'])
        parms.set('quick_est_gamma_flag', in_json['quick_est_gamma_flag'])
        parms.set('reest_gamma_flag', in_json['reest_gamma_flag'])
        parms.set('weed_zero_elevation', in_json['weed_zero_elevation'])
        parms.set('weed_neighbours', in_json['weed_neighbours'])
        parms.set('merge_resample_size', in_json['merge_resample_size'])
        parms.set('unwrap_grid_size', in_json['unwrap_grid_size'])
        parms.set('unwrap_time_win', in_json['unwrap_time_win'])
        parms.set('unwrap_gold_n_win', in_json['unwrap_gold_n_win'])
        parms.set('scla_deramp', in_json['scla_deramp'])
        parms.set('scn_time_win', in_json['scn_time_win'])
        parms.save()
        parms.load()
        stamps_step = StaMPSStep(parms)
        
        stamps_step.run(1, 1)
        stamps_step.run(2, 2)
        stamps_step.run(3, 3, plot_flag=True)
        stamps_step.run(4, 4)
        stamps_step.run(5, 5, False)
        stamps_step.run(5, 5, True)
        stamps_step.run(6, 6, True, aps_flag=True)
        stamps_step.run(7, 7, True)
        stamps_step.run(6, 6, True)
        stamps_step.run(7, 7, True)
        stamps_step.run(8, 8, True)

if __name__ == "__main__":
    stamps_step = StaMPSStep()
    stamps_step.run_normal()