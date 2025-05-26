import glob
import multiprocessing as mp
import os
import platform
import sys
from datetime import datetime, timedelta
from functools import partial

import numpy as np
import pandas as pd
import scipy
import scipy.io as sio
from matplotlib import pyplot as plt
from numpy.linalg import lstsq
from scipy.spatial import Delaunay
from sklearn.utils import resample
from tqdm import tqdm

project_path = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(project_path)

from modules.tomo.ps_parms import Parms


class StaMPSStep:
    def __init__(self, parms: Parms, est_gamma_flag=None, patch_list_file=None, stamps_PART_limitation=None):
        self.parms = parms
        self.input_file = self.parms.project_conf_path
        self._load_config()
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
            self.patch_list_file = self.CURRENT_RESULT + '/patch.list'

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
            6: lambda: self._stamps_6(),
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

    def _load_config(self):
        with open(self.input_file, 'r') as file:
            for line in file.readlines():
                key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                if key:
                    setattr(self, key, value)  # Dynamically set variables

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

    def _load_rslcpar(self):
        master_date = self.CURRENT_RESULT.split('_')[1]
        with open(os.path.join(self.CURRENT_RESULT, f'rslc/{master_date}.rslc.par'), 'r') as file:
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
        if not os.path.exists(os.path.join(self.CURRENT_RESULT, self.patch_dir, 'no_ps_info.mat')):
            # Create zeros array for first 5 steps only
            self.stamps_step_no_ps = np.zeros((5, 1))
            # Save to mat file
            sio.savemat(os.path.join(self.CURRENT_RESULT, self.patch_dir, 'no_ps_info.mat'), {'stamps_step_no_ps': self.stamps_step_no_ps})
        else:
            # Load existing file
            self.stamps_step_no_ps = sio.loadmat(os.path.join(self.CURRENT_RESULT, self.patch_dir, 'no_ps_info.mat'))['stamps_step_no_ps']
    
    def _save_ps_info(self):
        sio.savemat(os.path.join(self.CURRENT_RESULT, self.patch_dir, 'no_ps_info.mat'), {'stamps_step_no_ps': self.stamps_step_no_ps})
    
    def _setpsver(self, value):
        """Set psver variable"""
        self.psver = value
        sio.savemat(os.path.join(self.CURRENT_RESULT, self.patch_dir, 'psver.mat'), {'psver': self.psver})

    def _llh2local(self, llh, origin):
        """
        Convert longitude and latitude to local coordinates given an origin.
        
        Args:
            llh: Array of [longitude, latitude, height] in decimal degrees
            origin: Origin point [longitude, latitude] in decimal degrees
            
        Returns:
            xy: Local coordinates in kilometers
        """
        # WGS84 ellipsoid constants
        a = 6378137.0  # semi-major axis in meters
        e = 0.08209443794970  # eccentricity

        # Convert to radians and handle NaN values
        llh = np.array(llh, dtype=np.float64) * np.pi / 180
        origin = np.array(origin, dtype=np.float64) * np.pi / 180
        # Initialize output array
        xy = np.zeros((2, llh.shape[1]))

        # Handle non-zero latitude points
        z = llh[1, :] != 0
        
        dlambda = llh[0, z] - origin[0]

        # Calculate M for points
        M = a * ((1 - e**2/4 - 3*e**4/64 - 5*e**6/256) * llh[1, z] -
                (3*e**2/8 + 3*e**4/32 + 45*e**6/1024) * np.sin(2*llh[1, z]) +
                (15*e**4/256 + 45*e**6/1024) * np.sin(4*llh[1, z]) -
                (35*e**6/3072) * np.sin(6*llh[1, z]))

        # Calculate M0 for origin
        M0 = a * ((1 - e**2/4 - 3*e**4/64 - 5*e**6/256) * origin[1] -
                    (3*e**2/8 + 3*e**4/32 + 45*e**6/1024) * np.sin(2*origin[1]) +
                    (15*e**4/256 + 45*e**6/1024) * np.sin(4*origin[1]) -
                    (35*e**6/3072) * np.sin(6*origin[1]))

        # Calculate N and E
        N = a / np.sqrt(1 - e**2 * np.sin(llh[1, z])**2)
        E = dlambda * np.sin(llh[1, z])

        # Calculate local coordinates
        xy[0, z] = N * np.cos(llh[1, z]) * np.sin(E)
        xy[1, z] = M - M0 + N * np.cos(llh[1, z]) * (1 - np.cos(E))

        # Handle zero latitude points
        dlambda = llh[0, ~z] - origin[0]
        xy[0, ~z] = a * dlambda
        xy[1, ~z] = -M0

        # Convert to kilometers
        xy = xy / 1000

        return xy

    def _ps_load_initial_gamma(self):
        """Load initial gamma data"""
        self.parms.load()
        phname = os.path.join(self.CURRENT_RESULT, self.patch_dir, 'pscands.1.ph')
        ijname = os.path.join(self.CURRENT_RESULT, self.patch_dir, 'pscands.1.ij')
        llname = os.path.join(self.CURRENT_RESULT, self.patch_dir, 'pscands.1.ll')
        # xyname = os.path.join(self.CURRENT_RESULT, patch_dir, 'pscands.1.xy')
        hgtname = os.path.join(self.CURRENT_RESULT, self.patch_dir, 'pscands.1.hgt')
        daname = os.path.join(self.CURRENT_RESULT, self.patch_dir, 'pscands.1.da')
        # rscname = os.path.join(self.CURRENT_RESULT, 'rsc.txt')
        pscname = os.path.join(self.CURRENT_RESULT, 'pscphase.in')

        self.psver = 1
        
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
            master_master_flag = 0  # no null master-master ifg provided
            day = np.concatenate([day[:master_ix], [master_day], day[master_ix:]])
        else:
            master_master_flag = 1  # yes, null master-master ifg provided

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

        bperp_mat = np.zeros((n_ps, n_image))
        for i in range(n_ifg):
            basename = ifgs[i][:nb-5]+'.base'
            B_TCN = [float(x) for x in self._fetch_baseline(basename)[0]]
            BR_TCN = [float(x) for x in self._fetch_baseline(basename)[1]]
            bc = B_TCN[1]+BR_TCN[1]*(ij[:, 2]-mean_az)/prf
            bn = B_TCN[2]+BR_TCN[2]*(ij[:, 2]-mean_az)/prf
            bperp_mat[:, i] = bc*np.sin(look)-bn*np.cos(look)

        bperp = np.mean(bperp_mat, axis=0)
        if master_master_flag == 1:
            bperp_mat = bperp_mat[:, :master_ix, master_ix:]
        else:
            bperp = np.concatenate([bperp[:master_ix], [0], bperp[master_ix:]])

        inci = np.arccos((se**2-re**2-rg**2)/(2*re*rg))
        mean_incidence = np.mean(inci)
        mean_range = rgc

        # Read phase data
        try:
            with open(phname, 'rb') as f:
                # Initialize phase array
                ph = np.zeros((n_ps, n_ifg), dtype=np.complex64)
                byte_count = n_ps * 2
                
                # Read phase data for each IFG
                for i in range(n_ifg):
                    # Read float32 values
                    ph_bit = np.fromfile(f, dtype=np.float32, count=byte_count)
                    # Convert to complex numbers (real, imag pairs)
                    real_part = ph_bit[::2]
                    imag_part = ph_bit[1::2]
                    # Handle NaN values
                    real_part = np.nan_to_num(real_part, nan=0.0)
                    imag_part = np.nan_to_num(imag_part, nan=0.0)
                    ph[:, i] = real_part + 1j * imag_part
        except FileNotFoundError:
            raise FileNotFoundError(f"{phname} does not exist")
        except Exception as e:
            raise Exception(f"Error reading phase data: {str(e)}")

        # Handle zero phases
        # zero_ph = np.sum(ph == 0, axis=1)
        # nonzero_ix = zero_ph <= 1  # if more than 1 phase is zero, drop node

        # Handle master-master IFG
        if master_master_flag == 1:
            ph[:, master_ix] = 1
        else:
            ph = np.column_stack([ph[:, :master_ix], np.ones((n_ps, 1)), ph[:, master_ix:]])
            n_ifg += 1
            n_image += 1

        # Read lonlat data
        with open(llname, 'rb') as f:
            lonlat = np.fromfile(f, dtype='>f4').astype(np.float32).reshape(-1, 2)
        # Calculate local coordinates
        ll0 = (np.max(lonlat, axis=0) + np.min(lonlat, axis=0)) / 2
        xy = self._llh2local(lonlat.T, ll0).T * 1000
        
        # Calculate scene corners
        # sort_x = xy[np.argsort(xy[:, 0])]
        # sort_y = xy[np.argsort(xy[:, 1])]
        # n_pc = int(round(n_ps * 0.001))
        # bl = np.mean(sort_x[:n_pc], axis=0)  # bottom left corner
        # tr = np.mean(sort_x[-n_pc:], axis=0)  # top right corner
        # br = np.mean(sort_y[:n_pc], axis=0)  # bottom right corner
        # tl = np.mean(sort_y[-n_pc:], axis=0)  # top left corner

        # Rotate coordinates
        heading = self.parms.get('heading')
        theta = (180 - heading) * np.pi / 180
        if theta > np.pi:
            theta -= 2 * np.pi

        rotm = np.array([[np.cos(theta), np.sin(theta)], 
                        [-np.sin(theta), np.cos(theta)]])
        xy = xy.T
        xynew = rotm @ xy

        # Check if rotation improves alignment
        if (np.max(xynew[0, :]) - np.min(xynew[0, :]) < np.max(xy[0, :]) - np.min(xy[0, :]) and
            np.max(xynew[1, :]) - np.min(xynew[1, :]) < np.max(xy[1, :]) - np.min(xy[1, :])):
            xy = xynew
            print(f"   -> Rotating by {theta * 180 / np.pi} degrees")

        # Convert to single precision and transpose
        xy = np.array(xy.T, dtype=np.float32)
        
        # Sort by y then x (MATLAB: sortrows(xy,[2,1]))
        sort_ix = np.lexsort((xy[:, 0], xy[:, 1]))
        xy = xy[sort_ix, :]
        
        # Add PS numbers (1-based indexing like MATLAB)
        xy = np.column_stack([np.arange(0, n_ps).T, xy])
        
        # Round to mm (MATLAB: round(xy(:,2:3)*1000)/1000)
        xy[:, 1:3] = np.round(xy[:, 1:] * 1000) / 1000

        # Update arrays with sorted indices
        ph = ph[sort_ix, :]
        ij = ij[sort_ix, :]
        ij[:, 0] = np.arange(0, n_ps)
        lonlat = lonlat[sort_ix, :]
        bperp_mat = bperp_mat[sort_ix, :]

        # Remove NaN values
        ix_nan = np.any(np.isnan(lonlat), axis=1) | np.any(np.isnan(ph), axis=1)
        lonlat = lonlat[~ix_nan]
        ij = ij[~ix_nan]
        xy = xy[~ix_nan]
        n_ps = lonlat.shape[0]
        ij[:, 0] = np.arange(0, n_ps).T
        xy[:, 0] = np.arange(0, n_ps).T

        # Remove zero values
        ix_0 = lonlat[:, 0] == 0
        lonlat = lonlat[~ix_0]
        ij = ij[~ix_0]
        xy = xy[~ix_0]
        n_ps = lonlat.shape[0]
        ij[:, 0] = np.arange(0, n_ps).T
        xy[:, 0] = np.arange(0, n_ps).T

        # Save results
        sio.savemat(os.path.join(self.CURRENT_RESULT, self.patch_dir, f'ps{self.psver}.mat'), {
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
        sio.savemat(os.path.join(self.CURRENT_RESULT, self.patch_dir, 'psver.mat'), {'psver': self.psver})

        # Save phase data
        ph = ph[~ix_nan]
        ph = ph[~ix_0]
        sio.savemat(os.path.join(self.CURRENT_RESULT, self.patch_dir, f'ph{self.psver}.mat'), {'ph': ph})

        # Save baseline data
        bperp_mat = bperp_mat[~ix_nan]
        bperp_mat = bperp_mat[~ix_0]
        sio.savemat(os.path.join(self.CURRENT_RESULT, self.patch_dir, f'bp{self.psver}.mat'), {'bperp_mat': bperp_mat})

        # Save look angle data
        la = inci[sort_ix]
        la = la[~ix_nan]
        la = la[~ix_0]
        sio.savemat(os.path.join(self.CURRENT_RESULT, self.patch_dir, f'la{self.psver}.mat'), {'la': la})

        # Handle DA file if it exists
        if os.path.exists(daname):
            D_A = np.loadtxt(daname)
            D_A = D_A[sort_ix]
            D_A = D_A[~ix_nan]
            D_A = D_A[~ix_0]
            sio.savemat(os.path.join(self.CURRENT_RESULT, self.patch_dir, f'da{self.psver}.mat'), {'D_A': D_A})

        # Handle height file if it exists
        if os.path.exists(hgtname):
            with open(hgtname, 'rb') as f:
                hgt = np.fromfile(f, dtype='>f4').reshape(-1, 1)
                hgt = hgt[sort_ix]
                hgt = hgt[~ix_nan]
                hgt = hgt[~ix_0]
                sio.savemat(os.path.join(self.CURRENT_RESULT, self.patch_dir, f'hgt{self.psver}.mat'), {'hgt': hgt})

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

        # Ensure cpxphase is 1D
        cpxphase = np.squeeze(cpxphase)
        bperp = np.squeeze(bperp)

        # Get non-zero indices
        ix = cpxphase != 0
        cpxphase = cpxphase[ix]
        bperp = bperp[ix]
        n_ix = len(ix)
        bperp_range = np.max(bperp) - np.min(bperp)

        # Calculate wrapped phase
        wphase = np.angle(cpxphase)

        # Calculate trial phases
        trial_mult = np.arange(-np.ceil(8 * n_trial_wraps), np.ceil(8 * n_trial_wraps) + 1) + asym * 8 * n_trial_wraps
        n_trials = len(trial_mult)
        trial_phase = bperp / bperp_range * np.pi / 4
        trial_phase_mat = np.exp(-1j * trial_phase[:, np.newaxis] * trial_mult[np.newaxis, :])
        cpxphase_mat = np.tile(cpxphase[:, np.newaxis], (1, n_trials))
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
        # Ensure 2D arrays for lstsq
        A = (weighting * bperp).reshape(-1, 1)
        b = (weighting * resphase).reshape(-1, 1)
        mopt = np.linalg.lstsq(A, b, rcond=None)[0]
        K0 = K0 + mopt[0, 0]  # Extract scalar value
        phase_residual = cpxphase * np.exp(-1j * (K0 * bperp))
        mean_phase_residual = np.sum(phase_residual)
        C0 = np.angle(mean_phase_residual)
        coh0 = np.abs(mean_phase_residual) / np.sum(np.abs(phase_residual))

        if plotflag:
            plt.figure()
            plt.subplot(2, 1, 2)
            bvec = np.linspace(np.min(bperp), np.max(bperp), 200)
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
        gausswin_7 = scipy.signal.windows.gaussian(7, std=1.2)
        B = np.outer(gausswin_7, gausswin_7)

        n_win_ex = n_win + n_pad
        ph_bit = np.zeros((n_win_ex, n_win_ex), dtype=np.complex64)

        for ix1 in range(n_win_i):
            wf = wind_func.copy()
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

        patch_dir = os.path.join(self.CURRENT_RESULT, self.patch_dir)

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
        null_i, _ = np.where(ph == 0)
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
        A = np.abs(ph) + 1e-10
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
        bperp_range = np.max(bperp) - np.min(bperp)
        n_trial_wraps = bperp_range * max_K / (2 * np.pi)

        if est_gamma_parm > 0:
            print('   -> Restarting previous run...')
            pm = sio.loadmat(pmname)
            if 'gamma_change_save' not in pm:
                gamma_change_save = 1
            else:
                gamma_change_save = pm['gamma_change_save']
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
            
            coh_rand = np.zeros((n_rand, 1))
            for i in tqdm(list(reversed(range(0, n_rand, 1))), desc='   -> Computing progress', unit=' pixels'):
                _, _, coh_r, _ = self._ps_topofit(np.exp(1j * rand_ifg[i, :]), bperp.reshape(-1, 1), n_trial_wraps, False)
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
            ph_res = np.zeros((n_ps, n_ifg), dtype=np.float32)
            ph_patch = np.zeros(ph.shape, dtype=np.complex64)
            N_patch = np.zeros((n_ps, 1))
            grid_ij = np.zeros((n_ps, 2))
            grid_ij[:, 0] = np.ceil((xy[:, 2] - np.min(xy[:, 2]) + 1e-6) / self.grid_size)
            grid_ij[grid_ij[:, 0] == np.max(grid_ij[:, 0]), 0] = np.max(grid_ij[:, 0]) - 1
            grid_ij[:, 1] = np.ceil((xy[:, 1] - np.min(xy[:, 1]) + 1e-6) / self.grid_size)
            grid_ij[grid_ij[:, 1] == np.max(grid_ij[:, 1]), 1] = np.max(grid_ij[:, 1]) - 1
            i_loop = 1
            weighting = 1.0 / D_A
            weighting_save = weighting
            gamma_change_save = 0

        n_i = int(max(grid_ij[:, 0]))
        n_j = int(max(grid_ij[:, 1]))

        # print(f"   -> {n_ps} PS candidates to process")
        xy[:, 0] = np.arange(0, n_ps).T
        loop_end_sw = 0

        while loop_end_sw == 0:
            print(f"      -> Iteration #{i_loop}")
            print("      -> Calculating patch phases...")

            ph_grid = np.zeros((int(n_i), int(n_j), int(n_ifg)), dtype=np.complex64)
            ph_filt = ph_grid.copy()
            ph_weight = ph * np.exp(-1j * bp['bperp_mat'] * np.tile(K_ps.reshape(-1, 1), (1, n_ifg))) * np.tile(weighting.reshape(-1, 1), (1, n_ifg))
            
            for i in range(n_ps):
                ph_grid[int(grid_ij[i, 0])-1, int(grid_ij[i, 1])-1, :] += ph_weight[i, :]

            for i in range(n_ifg):
                ph_filt[:, :, i] = self._clap_filt(ph_grid[:, :, i], self.clap_alpha, self.clap_beta, self.n_win*0.75, self.n_win*0.25, self.low_pass)

            for i in range(n_ps):
                ph_patch[i, 0:n_ifg] = np.squeeze(ph_filt[int(grid_ij[i, 0])-1, int(grid_ij[i, 1])-1, :])

            del ph_filt
            ix = ph_patch != 0
            ph_patch[ix] = ph_patch[ix] / np.abs(ph_patch[ix])

            if est_gamma_parm < 2:
                step_number = 2
                for i in tqdm(range(n_ps), desc='      -> Estimating topo error', unit=' pixels'):
                    psdph = ph[i, :] * np.conj(ph_patch[i, :])
                    if np.sum(psdph == 0) == 0:
                        Kopt, Copt, cohopt, ph_residual = self._ps_topofit(psdph, bp['bperp_mat'][i, :].reshape(-1, 1), n_trial_wraps, False)
                        K_ps[i] = Kopt
                        C_ps[i] = Copt
                        coh_ps[i] = cohopt
                        N_opt[i] = 1
                        ph_res[i, :] = np.angle(ph_residual)
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
                    loop_end_sw = 1
                else:
                    i_loop += 1
                    if self.filter_weighting == 'P-square':
                        # Use same bin edges for both histograms
                        Na = np.histogram(coh_ps, bins=edges)[0]
                        Nr = Nr * np.sum(Na[:low_coh_thresh]) / np.sum(Nr[:low_coh_thresh])
                        Na[Na == 0] = 1
                        Prand = Nr / Na
                        Prand[:low_coh_thresh] = 1
                        Prand[Nr_max_nz_ix+1:] = 0
                        Prand[Prand > 1] = 1
                        # Apply Gaussian window filter
                        Prand = np.convolve(np.concatenate([np.ones(7), Prand]), np.hanning(7), mode='valid') / np.sum(np.hanning(7))
                        Prand = Prand[7:]
                        # Interpolate to 100 samples
                        Prand = np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(Prand)), Prand)
                        Prand = Prand[:-9]
                        # Get probability for each PS
                        Prand_ps = Prand[np.clip(np.round(coh_ps * 1000).astype(int), 0, len(Prand)-1)]
                        weighting = (1 - Prand_ps)**2
                    else:
                        g = np.mean(A * np.cos(ph_res), axis=1)
                        sigma_n = np.sqrt(0.5 * (np.mean(A**2, axis=1) - g**2))
                        weighting[sigma_n == 0] = 0
                        weighting[sigma_n != 0] = g[sigma_n != 0] / sigma_n[sigma_n != 0]
            else:
                loop_end_sw = 1

            # Save results
            sio.savemat(pmname, {
                'ph_patch': ph_patch,
                'K_ps': K_ps,
                'C_ps': C_ps,
                'coh_ps': coh_ps,
                'N_opt': N_opt,
                'ph_res': ph_res,
                'step_number': step_number,
                'ph_grid': ph_grid,
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
    
    @staticmethod
    def _clap_filt_patch(ph, alpha=0.5, beta=0.1, low_pass=None):
        """Apply CLAP filter to a patch of phase data"""
        if low_pass is None:
            low_pass = np.zeros(ph.shape)
        
        ph[np.isnan(ph)] = 0
        gausswin_7 = scipy.signal.windows.gaussian(7, std=1.2)
        B = np.outer(gausswin_7, gausswin_7)

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
    
    @staticmethod
    def _process_chunk_clap(chunk_data, n_win, n_i, n_j, slc_osf, clap_alpha, clap_beta, low_pass, ph_grid):
        """Process a chunk of PS points in parallel"""
        chunk_ps_ij, chunk_indices = chunk_data
        n_ifg = ph_grid.shape[2]
        ph_filt = np.zeros((n_win, n_win, n_ifg), dtype=np.complex64)
        ph_patch2 = np.zeros((len(chunk_indices), n_ifg), dtype=np.complex64)
        
        for i, (ps_ij, idx) in enumerate(zip(chunk_ps_ij, chunk_indices)):
            i_min = int(max(ps_ij[0] - n_win // 2, 0))
            i_max = int(i_min + n_win - 1)
            if i_max > n_i:
                i_min = i_min - i_max + n_i
                i_max = n_i
                
            j_min = int(max(ps_ij[1] - n_win // 2, 0))
            j_max = int(j_min + n_win - 1)
            if j_max > n_j:
                j_min = j_min - j_max + n_j
                j_max = n_j
                
            if j_min < 0 or i_min < 0:
                ph_patch2[i, :] = 0
            else:
                ps_bit_i = int(ps_ij[0] - i_min) - 1
                ps_bit_j = int(ps_ij[1] - j_min) - 1
                ph_bit = ph_grid[i_min:i_max + 1, j_min:j_max + 1, :]
                ph_bit[ps_bit_i, ps_bit_j, :] = 0

                ix_i = np.arange(ps_bit_i - (slc_osf - 1), ps_bit_i + (slc_osf - 1) + 1)
                ix_i = ix_i[(ix_i >= 0) & (ix_i < ph_bit.shape[0])]
                ix_j = np.arange(ps_bit_j - (slc_osf - 1), ps_bit_j + (slc_osf - 1) + 1)
                ix_j = ix_j[(ix_j >= 0) & (ix_j < ph_bit.shape[1])]
                ph_bit[np.ix_(ix_i, ix_j)] = 0

                for i_ifg in range(n_ifg):
                    ph_filt[:, :, i_ifg] = StaMPSStep._clap_filt_patch(ph_bit[:, :, i_ifg], clap_alpha, clap_beta, low_pass)

                ph_patch2[i, :] = np.squeeze(ph_filt[ps_bit_i, ps_bit_j, :])
                
        return chunk_indices, ph_patch2

    def _ps_select(self, reselect=0, plot_flag=False):
        if self.psver > 1:
            self._setpsver(1)

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

        psname = os.path.join(self.CURRENT_RESULT, self.patch_dir, f'ps{self.psver}.mat')
        phname = os.path.join(self.CURRENT_RESULT, self.patch_dir, f'ph{self.psver}.mat')
        pmname = os.path.join(self.CURRENT_RESULT, self.patch_dir, f'pm{self.psver}.mat')
        selectname = os.path.join(self.CURRENT_RESULT, self.patch_dir, f'select{self.psver}.m')
        daname = os.path.join(self.CURRENT_RESULT, self.patch_dir, f'da{self.psver}.mat')
        bpname = os.path.join(self.CURRENT_RESULT, self.patch_dir, f'bp{self.psver}.mat')

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

        bperp=ps['bperp'][0]
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
            D_A=da['D_A']
            del da
        else:
            D_A=[]
        if not len(D_A[0]) == 0 and D_A[0].shape[0] >= 10000:
            D_A_sort = np.sort(D_A).flatten()  # Ensure 1D array
            if D_A[0].shape[0] >= 50000:
                bin_size = 10000
            else:
                bin_size = 2000
        
            # Create D_A_max with consistent shapes
            # Use size instead of shape[0] for 1D array length
            middle_values = D_A_sort[bin_size::bin_size][:-1]  # Get values at bin_size intervals, excluding last
            D_A_max = np.concatenate(([0], middle_values, [D_A_sort[-1]]))
        
        else:
            D_A_max = np.array([0, 1])
            D_A = np.ones((n_ps, n_ifg))
        
        if not select_method == 'PERCENT':
            patch_area=np.prod(np.max(xy[:, 1:], axis=0) - np.min(xy[:, 1:], axis=0)) / 1e6 # in km2
            self.max_percent_rand=self.max_density_rand*patch_area/(len(D_A_max)-1)
        min_coh=np.zeros((len(D_A_max)-1, 1))
        D_A_mean=np.zeros((len(D_A_max)-1, 1))
        Nr_dist=pm['Nr'][0]

        if reselect==3:
            coh_thresh=0
            coh_thresh_coeffs=[]
        else:
            step = 0.01
            half_step = step / 2
            for i in tqdm(range(len(D_A_max)-1), desc='   -> Filtering coherence', unit=' bins'):
                # Convert to numpy arrays and ensure proper shapes
                mask = (D_A[0] > D_A_max[i]) & (D_A[0] <= D_A_max[i+1])
                coh_chunk = pm['coh_ps'][mask]
                
                # Handle empty slices
                # if np.sum(mask) == 0:
                #     D_A_mean[i] = np.nan
                #     min_coh[i] = np.nan
                #     continue
                    
                D_A_mean[i] = np.mean(D_A[0][mask])
                coh_chunk = coh_chunk[coh_chunk != 0]
                Na = np.histogram(coh_chunk, pm['coh_bins'][0])[0]
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
                    min_fit_ix = np.min(np.array(ok_ix)) - 3
                    if min_fit_ix <= 0:
                        min_coh[i] = np.nan
                    else:
                        max_fit_ix = np.max(np.array(ok_ix)) + 2
                        max_fit_ix = min(max_fit_ix, 100)
                        coeffs = np.polyfit(percent_rand[min_fit_ix:max_fit_ix+1], np.arange(min_fit_ix*0.01, max_fit_ix*0.01, 0.01), 3)
                        min_coh[i] = np.polyval(coeffs, self.max_percent_rand)

            nonnanix = ~np.isnan(min_coh)
            if all(nonnanix.flatten()) == False:
                print('   -> Not enough random phase pixels to set gamma threshold - using default threshold of 0.3')
                coh_thresh = 0.3
                coh_thresh_coeffs = []
            else:
                min_coh = np.array(min_coh)[nonnanix]
                D_A_mean = np.array(D_A_mean)[nonnanix]
                if min_coh.shape[0] > 1:
                    coh_thresh_coeffs = np.polyfit(D_A_mean, min_coh, 1)
                    if coh_thresh_coeffs[0] > 0:
                        coh_thresh = np.polyval(coh_thresh_coeffs, D_A)
                    else:
                        coh_thresh = np.polyval(coh_thresh_coeffs, 0.35)
                        coh_thresh_coeffs = []
                else:
                    coh_thresh = min_coh
                    coh_thresh_coeffs = []
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
        self.ix = np.where(pm['coh_ps'].flatten() > coh_thresh)[0]
        n_ps = len(self.ix)
        print(f'      -> {n_ps} PS selected initially')
        
        n_boot = 100
        if gamma_stdev_reject > 0:
            ph_res_cpx = np.exp(1j * pm['ph_res'][:, ifg_index])
            coh_std = np.zeros(len(self.ix))
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

                # Store necessary variables as instance attributes for parallel processing
                self.n_i = int(max(pm['grid_ij'][:, 0]))-1
                self.n_j = int(max(pm['grid_ij'][:, 1]))-1
                K_ps2 = np.zeros((n_ps, 1))
                C_ps2 = np.zeros((n_ps, 1))
                coh_ps2 = np.zeros((n_ps, 1))

                grid_ij = pm['grid_ij'][self.ix, :]
                self.low_pass = pm['low_pass'][0][0]
                ph_grid = pm['ph_grid']
                
                # Prepare data for parallel processing
                chunk_size = max(1, n_ps // int(self.CPU))
                chunks = []
                
                for i in range(0, n_ps, chunk_size):
                    chunk_indices = np.arange(i, min(i + chunk_size, n_ps))
                    chunk_ps_ij = grid_ij[chunk_indices]
                    chunks.append((chunk_ps_ij, chunk_indices))
                
                # Create partial function with fixed arguments
                process_chunk_partial = partial(StaMPSStep._process_chunk_clap,
                                              n_win=self.n_win,
                                              n_i=self.n_i,
                                              n_j=self.n_j,
                                              slc_osf=self.slc_osf,
                                              clap_alpha=self.clap_alpha,
                                              clap_beta=self.clap_beta,
                                              low_pass=self.low_pass,
                                              ph_grid=ph_grid)
                
                # Process chunks in parallel
                with mp.Pool(processes=int(self.CPU)) as pool:
                    results = list(tqdm(pool.imap(process_chunk_partial, chunks),
                                      total=len(chunks),
                                      desc='   -> Re-estimating PS patches',
                                      unit = ' chunks'))
                
                # Combine results
                ph_patch2 = np.zeros((n_ps, n_ifg), dtype=np.complex64)
                for chunk_indices, chunk_ph_patch2 in results:
                    ph_patch2[chunk_indices] = chunk_ph_patch2

                # Re-estimate coherence
                pm.pop('ph_grid', None)
                bp = sio.loadmat(bpname)
                bperp_mat = bp['bperp_mat'][self.ix, :]
                del bp

                for i in range(bperp_mat.shape[1]):
                    if i in ifg_index:
                        bperp_mat = np.delete(bperp_mat, i, axis=1)

                for i in tqdm(range(n_ps), desc='   -> Re-estimating PS coherence', unit=' pixels'):
                    psdph = ph[i, :] * np.conj(ph_patch2[i, :])
                    for j in range(psdph.shape[0]):
                        if j in ifg_index:
                            psdph = np.delete(psdph, j, axis=1)
                            ph_res2 = np.delete(ph_res2, j, axis=1)
                    if np.where(psdph == 0)[0].shape[0] == 0:
                        psdph = psdph / np.abs(psdph)
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
                mask = (D_A[0] > D_A_max[i]) & (D_A[0] <= D_A_max[i+1])
                coh_chunk = pm['coh_ps'][mask]
                D_A_mean[i] = np.mean(D_A[0][mask])
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
                    min_fit_ix = np.min(np.array(ok_ix)) - 3
                    if min_fit_ix <= 0:
                        min_coh[i] = np.nan
                    else:
                        max_fit_ix = np.min(np.array(ok_ix)) + 2
                        max_fit_ix[max_fit_ix > 100] = 100
                        coeffs = np.polyfit(percent_rand[min_fit_ix:max_fit_ix + 1], range(min_fit_ix * 0.01, max_fit_ix * 0.01, 0.01), 3)
                        min_coh[i] = np.polyval(coeffs, self.max_percent_rand)

            nonnanix = ~np.isnan(min_coh)
            if all(nonnanix.flatten()) == False:
                coh_thresh = 0.3
                coh_thresh_coeffs = []
            else:
                min_coh = min_coh[nonnanix]
                D_A_mean = D_A_mean[nonnanix]
                if min_coh.shape[0] > 1:
                    coh_thresh_coeffs = np.polyfit(D_A_mean, min_coh, 1)
                    if coh_thresh_coeffs[0] > 0:
                        coh_thresh = np.polyval(coh_thresh_coeffs, D_A[0][self.ix])
                    else:
                        coh_thresh = np.polyval(coh_thresh_coeffs, 0.35)
                        coh_thresh_coeffs = []
                else:
                    coh_thresh = min_coh
                    coh_thresh_coeffs = []
                
            coh_thresh = np.array(coh_thresh, ndmin=1)
            coh_thresh[coh_thresh < 0] = 0
            print(f'   -> Reestimation gamma threshold: {min(coh_thresh)} at D_A={min(D_A_mean)[0]} to {max(coh_thresh)} at D_A={max(D_A_mean)[0]}')
            
            # Calculate final selection
            bperp_range = np.max(bperp) - np.min(bperp)
            coh_ps2_reshaped = coh_ps2.reshape(-1)
            coh_thresh_reshaped = coh_thresh.reshape(-1)
            K_diff = abs(pm['K_ps'][self.ix].reshape(-1) - K_ps2.reshape(-1))
            
            # Match MATLAB's logic exactly
            keep_ix = np.logical_and(
                coh_ps2_reshaped > coh_thresh_reshaped,
                K_diff < 2 * np.pi / bperp_range
            )
            
            # Ensure keep_ix is boolean array
            keep_ix = keep_ix.astype(bool)
            
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

        if not os.path.exists(os.path.join(self.CURRENT_RESULT, self.patch_dir, 'no_ps_info.mat')):
            stamps_step_no_ps = np.zeros((5, 1))
        else:
            stamps_step_no_ps = sio.loadmat(os.path.join(self.CURRENT_RESULT, self.patch_dir, 'no_ps_info.mat'))['stamps_step_no_ps']
            stamps_step_no_ps[2:] = 0
        
        if np.sum(keep_ix) == 0:
            print(' -> No PS points left. Updating the stamps log for this\n')
            stamps_step_no_ps[2] = 1
        sio.savemat(os.path.join(self.CURRENT_RESULT, self.patch_dir, 'no_ps_info.mat'), {'stamps_step_no_ps': stamps_step_no_ps})

        # if plot_flag:
        #     plt.figure()
        #     plt.plot(D_A_mean, min_coh, '*')
        #     if not coh_thresh_coeffs is None:
        #         plt.plot(D_A_mean, np.polyval(coh_thresh_coeffs, D_A_mean), 'r')
        #     plt.ylabel('\gamma_{thresh}')
        #     plt.xlabel('D_A')
        #     plt.show()
        
        sio.savemat(os.path.join(self.CURRENT_RESULT, self.patch_dir, f'select{self.psver}.mat'),
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

    ########## Weed PS ##########
    
    def _ps_weed(self, all_da_flag=False, no_weed_adjacent=False, no_weed_noisy=False, use_triangle=False):
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
        
        self.psver = 1
        psname = f'{self.CURRENT_RESULT}/{self.patch_dir}/ps{self.psver}.mat'
        pmname = f'{self.CURRENT_RESULT}/{self.patch_dir}/pm{self.psver}.mat'
        phname = f'{self.CURRENT_RESULT}/{self.patch_dir}/ph{self.psver}.mat'
        selectname = f'{self.CURRENT_RESULT}/{self.patch_dir}/select{self.psver}.mat'
        hgtname = f'{self.CURRENT_RESULT}/{self.patch_dir}/hgt{self.psver}.mat'
        laname = f'{self.CURRENT_RESULT}/{self.patch_dir}/la{self.psver}.mat'
        incname = f'{self.CURRENT_RESULT}/{self.patch_dir}/inc{self.psver}.mat'
        bpname = f'{self.CURRENT_RESULT}/{self.patch_dir}/bp{self.psver}.mat'
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
        master_day = ps['master_day'][0][0]

        keep_ix = sl['keep_ix'].astype(bool)
        if 'keep_ix' in sl:
            ix2 = sl['ix'][keep_ix]
            K_ps2 = sl['K_ps2'][keep_ix.flatten(), :]
            C_ps2 = sl['C_ps2'][keep_ix.flatten(), :]
            coh_ps2 = sl['coh_ps2'][keep_ix.flatten(), :]
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
            ph_res2 = []
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
            if all_da_flag != 0:
                hto = sio.loadmat(hgtothername)
                hgt = np.concatenate((hgt, hto['hgt_other'][ix_other]), axis=0)
                del hto
        else:
            hgt = np.zeros((len(ij2), 1))

        n_ps_low_D_A = len(ix2)
        n_ps = n_ps_low_D_A + n_ps_other
        ix_weed = np.ones((n_ps, 1), dtype=bool)
        print(f'   -> {n_ps_low_D_A} low D_A PS, {n_ps_other} high D_A PS')
        
        no_weed_adjacent = False
        if no_weed_adjacent == False:
            print('   -> Removing adjacent PS')
            ij_shift = ij2[:, 1:3] + (np.array([2, 2]) - np.min(ij2[:, 1:3], axis=0))
            neigh_ix = np.zeros((int(np.max(ij_shift[:, 0])) + 2, int(np.max(ij_shift[:, 1])) + 2), dtype=int)
            miss_middle = np.ones((3, 3), dtype=bool)
            miss_middle[1, 1] = False

            for i in tqdm(range(n_ps), desc='      -> Generating neighbour matrix', unit=' ps'):
                neigh_this = neigh_ix[int(ij_shift[i, 0]-1):int(ij_shift[i, 0] + 2), int(ij_shift[i, 1]-1):int(ij_shift[i, 1] + 2)]
                neigh_this[neigh_this == 0 & miss_middle] = i
                neigh_ix[int(ij_shift[i, 0]-1):int(ij_shift[i, 0] + 2), int(ij_shift[i, 1]-1):int(ij_shift[i, 1] + 2)] = neigh_this
            
            neigh_ps = [[] for _ in range(n_ps)]
            for i in tqdm(range(n_ps), desc='      -> Finding neighbours', unit=' ps'):
                my_neigh_ix = neigh_ix[int(ij_shift[i, 0]), int(ij_shift[i, 1])]
                if my_neigh_ix != 0:
                    neigh_ps[my_neigh_ix].append(i)

            del my_neigh_ix
            for i in tqdm(range(n_ps), desc='      -> Weeding neighbours', unit=' ps'):
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
                    to_remove = np.delete(same_ps, high_coh_idx)
                    ix_weed[to_remove] = False

            print(f'      -> {np.sum(ix_weed)} PS kept after dropping adjacent pixels')

        self.parms.load()
        weed_zero_elevation = self.parms.get('weed_zero_elevation')
        if weed_zero_elevation == 'y' and hgt is not None:
            sea_ix = hgt < 1e-6
            ix_weed[sea_ix] = False
            print(f'   -> {np.sum(ix_weed)} PS kept after weeding zero elevation')
        
        xy_weed = xy2[ix_weed.flatten(), :]
        n_ps = len(xy_weed)
        ix_weed_num = np.where(ix_weed)[0]

        # Find unique coordinates and their indices
        unique_coords, inverse_indices = np.unique(xy_weed[:, 1:], axis=0, return_inverse=True)
        
        # Process each unique coordinate
        for i in tqdm(range(len(unique_coords)), desc='   -> Removing duplicate PS', unit=' groups'):
            # Find all PS points with this coordinate
            dup_mask = (inverse_indices == i)
            if np.sum(dup_mask) > 1:  # Only process if there are duplicates
                # Get global indices of duplicates
                dup_indices = ix_weed_num[dup_mask]
                
                # Find the one with highest coherence
                best_idx = np.argmax(coh_ps2[dup_indices])
                
                # Mark all others as False in ix_weed
                to_remove = np.delete(dup_indices, best_idx)
                ix_weed[to_remove] = False

        # Update xy_weed after removing duplicates
        xy_weed = xy2[ix_weed.flatten(), :]
        n_dups_removed = n_ps - len(xy_weed)
        if n_dups_removed > 0:
            print(f'      -> {n_dups_removed} PS with duplicate coordinates removed')

        n_ps = len(xy_weed)
        ix_weed2 = np.ones((n_ps, 1), dtype=bool)

        ps_std = np.zeros((n_ps, 1))
        ps_max = np.zeros((n_ps, 1))

        if n_ps != 0 and no_weed_noisy == False:
            if use_triangle:
                with open(f"{self.CURRENT_RESULT}/{self.patch_dir}/psweed.1.node", "w") as f:
                    f.write(f"{n_ps} 2 0 0\n")
                    for i in range(n_ps):
                        f.write(f"{i+1} {xy_weed[i, 1]} {xy_weed[i, 2]}\n")  # 0-based to 1-based

                os.system(f"{self.triangle_path} -e {self.CURRENT_RESULT}/{self.patch_dir}/psweed.1.node > {self.CURRENT_RESULT}/{self.patch_dir}/triangle_weed.log")

                with open(f"{self.CURRENT_RESULT}/{self.patch_dir}/psweed.2.edge", "r") as f:
                    header = list(map(int, f.readline().split()))
                    edgs = np.array([list(map(int, f.readline().split())) for _ in range(header[0])])[:, 1:] - 1
            else:
                points = xy_weed[:, 1:3].astype(float)
                tri = Delaunay(points)
                edges_set = set()
                for simplex in tri.simplices:
                    simplex = np.sort(simplex)
                    edges_set.update([(simplex[0], simplex[1]),
                                    (simplex[0], simplex[2]),
                                    (simplex[1], simplex[2])])

                # Convert set to sorted NumPy array
                edgs = np.array(list(edges_set))

            n_edge = edgs.shape[0]
            
            # Calculate phase for weeded points with proper array shapes
            K_ps_weeded = K_ps2[ix_weed.flatten()].reshape(-1, 1)  # Ensure 2D array
            bperp_reshaped = bperp.reshape(1, -1)  # Ensure 2D array
            ph_weed = ph2[ix_weed.flatten(), :] * np.exp(-1j * (K_ps_weeded @ bperp_reshaped))
            ph_weed /= np.abs(ph_weed)
            
            if small_baseline_flag != 'y':
                ph_weed[:, ps['master_ix'][0][0]] = np.exp(1j * C_ps2[ix_weed.flatten()].flatten())

            dph_space = ph_weed[edgs[:, 1], :] * np.conj(ph_weed[edgs[:, 0], :])
            dph_space = dph_space[:, ifg_index]
            n_use = len(ifg_index)
            for drop_ix in drop_ifg_index:
                print(f"   -> Dropping IFG {drop_ix} from noise estimation")

            if small_baseline_flag != 'y':
                print("   -> Estimating noise for all arcs...")
                dph_smooth = np.zeros((n_edge, n_use), dtype=np.complex64)
                dph_smooth2 = np.zeros_like(dph_smooth)

                for i1 in range(n_use):
                    time_diff = (day[ifg_index[i1]] - day[ifg_index]).astype(float)
                    weight_factor = np.exp(-0.5 * (time_diff / time_win) ** 2)
                    weight_factor /= weight_factor.sum()

                    dph_mean = np.sum(dph_space * weight_factor[np.newaxis, :], axis=1)

                    dph_mean_adj = np.angle(dph_space * np.conj(dph_mean)[:, np.newaxis])
                    G = np.column_stack([np.ones((n_use, 1)), time_diff])

                    m, _, _, _ = lstsq(G * weight_factor[:, None], dph_mean_adj.T * weight_factor[:, None], rcond=None)

                    adj1 = np.angle(np.exp(1j * (dph_mean_adj - (G @ m).T)))
                    m2, _, _, _ = lstsq(G * weight_factor[:, None], adj1.T * weight_factor[:, None], rcond=None)

                    dph_smooth[:, i1] = dph_mean * np.exp(1j * (m[0] + m2[0]))

                    weight_factor[i1] = 0
                    dph_smooth2[:, i1] = np.sum(dph_space * weight_factor[np.newaxis, :], axis=1)

                dph_noise = np.angle(dph_space * np.conj(dph_smooth))
                dph_noise2 = np.angle(dph_space * np.conj(dph_smooth2))
                ifg_var = np.var(dph_noise2, axis=0)

                A = bperp[ifg_index][:, np.newaxis]  # (n_use, 1)
                B = dph_noise.T / ifg_var[:, np.newaxis]  # (n_use, n_edge)
                K, _, _, _ = lstsq(A, B, rcond=None)  # K: (1, n_edge)
                fit = (A @ K).T  # shape (n_edge, n_use)

                dph_noise -= fit

                edge_std = np.std(dph_noise, axis=1)
                edge_max = np.max(np.abs(dph_noise), axis=1)

            else:
                ifg_var = np.var(dph_space, axis=0)
                A = bperp[ifg_index][:, np.newaxis]  # (n_use, 1)
                B = dph_noise.T / ifg_var[:, np.newaxis]  # (n_use, n_edge)
                K, _, _, _ = lstsq(A, B, rcond=None)  # K: (1, n_edge)
                fit = (A @ K).T  # shape (n_edge, n_use)
                dph_noise -= fit

                edge_std = np.std(np.angle(dph_space), axis=1)
                edge_max = np.max(np.abs(np.angle(dph_space)), axis=1)

            print("   -> Estimating max noise for all pixels...")
            ps_std = np.full(n_ps, np.inf, dtype=np.float32)
            ps_max = np.full(n_ps, np.inf, dtype=np.float32)

            for i in range(n_edge):
                for j in edgs[i]:
                    ps_std[j] = min(ps_std[j], edge_std[i])
                    ps_max[j] = min(ps_max[j], edge_max[i])

            ix_weed2 = (ps_std < weed_standard_dev) & (ps_max < weed_max_noise)
            
            # Update the original ix_weed using the original indices
            ix_weed[ix_weed] = ix_weed2.flatten()
            n_ps = np.sum(ix_weed)

            print(f"   -> {n_ps} PS kept after dropping noisy pixels")

            if not os.path.exists(os.path.join(self.CURRENT_RESULT, self.patch_dir, 'no_ps_info.mat')):
                self.stamps_step_no_ps = np.zeros((5, 1))
            else:
                self.stamps_step_no_ps = sio.loadmat(os.path.join(self.CURRENT_RESULT, self.patch_dir, 'no_ps_info.mat'))['stamps_step_no_ps']
            
            if n_ps == 0:
                print('   -> No PS left. Updating the stamps log for this')
                self.stamps_step_no_ps[3] = 1
            
            self._save_ps_info()
            
            # Save the results
            weedname = f'{self.CURRENT_RESULT}/{self.patch_dir}/weed{self.psver}.mat'
            sio.savemat(weedname, {'ix_weed': ix_weed, 'ix_weed2': ix_weed2, 'ps_std': ps_std, 'ps_max': ps_max, 'ifg_index': ifg_index})

            coh_ps = coh_ps2[ix_weed]
            K_ps = K_ps2[ix_weed]
            C_ps = C_ps2[ix_weed]
            ph_patch = ph_patch2[ix_weed.flatten(), :]
            if ph_res2.shape[1] > 0 :
                ph_res = ph_res2[ix_weed.flatten(), :]
            else:
                ph_res = ph_res2

            pmname = f'{self.CURRENT_RESULT}/{self.patch_dir}/pm{self.psver+1}.mat'
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
            phname = f'{self.CURRENT_RESULT}/{self.patch_dir}/ph{self.psver+1}.mat'
            sio.savemat(phname, {'ph': ph})
            del ph

            xy2 = xy2[ix_weed.flatten(), :]
            ij2 = ij2[ix_weed.flatten(), :]
            lonlat2 = lonlat2[ix_weed.flatten(), :]
            ps['xy'] = xy2
            ps['ij'] = ij2
            ps['lonlat'] = lonlat2
            ps['n_ps'] = ph2.shape[0]
            psname = f'{self.CURRENT_RESULT}/{self.patch_dir}/ps{self.psver+1}.mat'
            sio.savemat(psname, {'ps': ps})
            del ps
            del xy2
            del ij2
            del lonlat2

            if os.path.exists(hgtname):
                hgt = hgt[ix_weed]
                hgtname = f'{self.CURRENT_RESULT}/{self.patch_dir}/hgt{self.psver+1}.mat'
                sio.savemat(hgtname, {'hgt': hgt})
                del hgt

            if os.path.exists(laname):
                la = sio.loadmat(laname)
                la = la['la'].reshape(-1, 1)[ix2]
                if all_da_flag:
                    laothername = f'{self.CURRENT_RESULT}/{self.patch_dir}/la_other{self.psver+1}.mat'
                    lao = sio.loadmat(laothername)
                    la = np.concatenate([la, lao['la_other'][ix_other]])
                    del lao
                la = la[ix_weed]
                sio.savemat(f'{self.CURRENT_RESULT}/{self.patch_dir}/la{self.psver+1}.mat', {'la': la})
                del la

            if os.path.exists(incname):
                inc = sio.loadmat(incname)
                inc = inc['inc'][ix2]
                if all_da_flag:
                    incothername = f'{self.CURRENT_RESULT}/{self.patch_dir}/inc_other{self.psver+1}.mat'
                    inco = sio.loadmat(incothername)
                    inc = np.concatenate([inc, inco['inc_other'][ix_other]])
                    del inco
                inc = inc[ix_weed]
                sio.savemat(f'{self.CURRENT_RESULT}/{self.patch_dir}/inc{self.psver+1}.mat', {'inc': inc})
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
                sio.savemat(f'{self.CURRENT_RESULT}/{self.patch_dir}/bp{self.psver+1}.mat', {'bperp_mat': bperp_mat})
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

            self._setpsver(self.psver + 1)

    ########## Correct PS phase ##########
    
    def _ps_correct_phase(self):
        None

    ########## Merge patches ##########
    
    def _ps_merge_patches(self):
        None

    def _ps_calc_ifg_std(self):
        None
    
    def _ps_unwrap(self):
        None

    def _sb_invert_uw(self):
        None

    def _ps_calc_scla(self, value1, value2):
        None

    def _ps_smooth_scla(self, value):
        None

    def _ps_scn_filt(self):
        None

    def _ps_scn_filt_krig(self):
        None

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
            self._setpsver(1)
        

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

    def _stamps_5(self, patches_flag=False):
        if self.stamps_PART1_flag:
            if self.start_step <= 5 and self.end_step >= 5:
                self.stamps_step_no_ps[4:] = 0
                self._save_ps_info()
                if self.stamps_step_no_ps[3] == 0:
                    self._ps_correct_phase()
                else:
                    self.stamps_step_no_ps[4] = 1
                    print("-> No PS left in step 4, so will skip step 5")
            self._save_ps_info()
            self.stamps_PART1_flag = False
            self.stamps_PART2_flag = True
        elif self.stamps_PART2_flag:
            abord_flag = 0
            if patches_flag:
                self._ps_merge_patches()
            else:
                if os.path.exists(os.path.join(self.CURRENT_RESULT, self.patch_dir, 'no_ps_info.mat')):
                    self.stamps_step_no_ps = sio.loadmat(os.path.join(self.CURRENT_RESULT, self.patch_dir, 'no_ps_info.mat'))['stamps_step_no_ps']
                    if np.sum(self.stamps_step_no_ps) >= 1:
                        abord_flag = 1
            if abord_flag == 0:
                self._ps_calc_ifg_std()
            else:
                print("No PS left in step 4, so will skip step 5")

    def _stamps_6(self):
        if self.start_step <= 6 and self.end_step >= 6:
            self._ps_unwrap()
            if self.small_baseline_flag == 'y':
                self._sb_invert_uw()

    def _stamps_7(self):
        if self.start_step <= 7 and self.end_step >= 7:
            if self.small_baseline_flag == 'y':
                self._ps_calc_scla(1, 1)
                self._ps_smooth_scla(1)
                self._ps_calc_scla(0, 1)
            else:
                self._ps_calc_scla(0, 1)
                self._ps_smooth_scla()

    def _stamps_8(self):
        if self.start_step <= 8 and self.end_step >= 8:
            if self.scn_kriging_flag == 'y':
                self._ps_scn_filt_krig()
            else:
                self._ps_scn_filt()

    def run(self, start_step=None, end_step=None, patches_flag=False, plot_flag=False):
        if start_step is None:
            start_step = self.start_step
        else:
            self.start_step = start_step
        if end_step is None:
            end_step = self.end_step
        else:
            self.end_step = end_step
        if patches_flag is None:
            if start_step < 6:
                patches_flag = False
            else:
                patches_flag = True

        if self.start_step < 5 and self.end_step >= 5:
            self.stamps_PART1_flag = True
            self.stamps_PART2_flag = False
        elif self.start_step >= 5:
            self.stamps_PART1_flag = False
            self.stamps_PART2_flag = True

        # Get patch directories
        patch_dirs, patches_flag = self._handle_patches(patches_flag)

        # Process each patch
        if self.stamps_PART1_flag:
            for patch_dir in patch_dirs:
                if patch_dir != self.CURRENT_RESULT:
                    self.patch_dir = patch_dir
                
                # Initialize or load no_ps_info.mat
                self._initialize_ps_info()

                if self.start_step == 0:
                    # Check processing stage of stamps for all patches
                    # Step 4 find a ps_weed file
                    # Step 3 find a ps_select file
                    # Step 2 find a pm file
                    # Step 1 find a ps file
                    if os.path.exists(os.path.join(self.CURRENT_RESULT, patch_dir, 'weed1.mat')):
                        self.start_step = 5
                        self._setpsver(2)
                    elif os.path.exists(os.path.join(self.CURRENT_RESULT, patch_dir, 'select1.mat')):
                        self.start_step = 4
                    elif os.path.exists(os.path.join(self.CURRENT_RESULT, patch_dir, 'pm1.mat')):
                        self.start_step = 3
                    elif os.path.exists(os.path.join(self.CURRENT_RESULT, patch_dir, 'ps1.mat')):
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
                
                if patch_dir != '.':
                    os.chdir('..')

        if self.stamps_PART2_flag:
            if patches_flag:
                # Create new patch list file
                with open('patch.list_new', 'w') as f:
                    # Process patches in reverse order
                    for patch_dir in reversed(patch_dirs):
                        # Check PS information file
                        ps_info_file = os.path.join(self.CURRENT_RESULT, patch_dir, 'no_ps_info.mat')
                        
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
                if os.path.exists(os.path.join(self.CURRENT_RESULT, 'patch.list')):
                    os.rename(os.path.join(self.CURRENT_RESULT, 'patch.list'), os.path.join(self.CURRENT_RESULT, 'patch.list_old'))
                os.rename(os.path.join(self.CURRENT_RESULT, 'patch.list_new'), os.path.join(self.CURRENT_RESULT, 'patch.list'))

            for step in range(5, self.end_step + 1):
                self.control_flow[step]()


if __name__ == "__main__":
    project_path = os.path.join(os.path.dirname(__file__), '../..', "modules/snap2stamps/bin/project.conf")
    parms = Parms(project_path)
    parms.load()
    parms.set('max_topo_err', 10)
    parms.set('gamma_change_convergence', 0.005)
    parms.set('filter_grid_size', 50)
    parms.save()
    parms.load()
    parms.set('select_method', 'PERCENT')
    parms.set('percent_rand', 80)
    parms.save()
    parms.load()
    stamps_step = StaMPSStep(parms)
    
    # stamps_step.run(1, 1)
    # stamps_step.run(2, 2)
    # stamps_step.run(3, 3, plot_flag=True)
    stamps_step.run(4, 4)