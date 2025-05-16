import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import label
from tqdm import tqdm

from modules.tomo.bwstest import BWS


class SHP:
    def __init__(self, slcstack, calwin=[15, 15], alpha=0.05):
        """
        Identify statistically homogeneous pixels (SHPs) in a stack.

        Args:
            mlistack (np.ndarray): 3D input matrix (nlines, nwidths, npages)
            CalWin (tuple): Window size (rows, cols), default (15, 15)
            Alpha (float): Significance level for the statistical test, default 0.05

        Returns:
            dict: Contains PixelInd (bool array), BroNum (SHP count), CalWin (tuple)
        """

        self.slcstack = np.abs(slcstack)
        self.CalWin = calwin
        self.Alpha = alpha

        if self.slcstack.ndim != 3:
            raise ValueError("Input must be a 3D matrix.")
        
        self.inputfile = os.path.join(os.path.split(os.path.abspath(__file__))[0].split("modules")[0], "modules/snap2stamps/bin/project.conf")
        self._load_config()
        
    def _load_config(self):
        with open(self.inputfile, 'r') as file:
            for line in file.readlines():
                key, value = (line.split('=')[0].strip(), line.split('=')[1].strip()) if '=' in line else (None, None)
                if key:
                    setattr(self, key, value)  # Dynamically set variables

    @staticmethod
    def process_batch(task_batch):
        results = []
        for block, ref, calwin, alpha, init_row, init_col in task_batch:
            npages = block.shape[2]
            matrix_reshaped = block.reshape(-1, npages).T
            ref_repeated = np.tile(ref, (matrix_reshaped.shape[1], 1)).T
            T = BWS(ref_repeated, matrix_reshaped, alpha).run()
            seed_mask = np.reshape(~T, calwin)
            labeled, _ = label(seed_mask)
            mask = (labeled == labeled[init_row, init_col])
            results.append(mask.flatten())
        return results

    def run(self):
        start_time = time.time()
        nlines, nwidths, npages = self.slcstack.shape
        mlistack = self.slcstack.astype(np.float32)

        RadiusRow = (self.CalWin[0] - 1) // 2
        RadiusCol = (self.CalWin[1] - 1) // 2
        InitRow = (self.CalWin[0]+1) // 2
        InitCol = (self.CalWin[1]+1) // 2

        # Edge padding
        mlistack_padded = np.pad(mlistack,
                            ((RadiusRow, RadiusRow), (RadiusCol, RadiusCol), (0, 0)),
                            mode='symmetric')
        meanmli = np.mean(mlistack_padded, axis=2)
        nlines_EP, nwidths_EP = meanmli.shape[:2]

        # Prepare task list
        tasks = []
        for kk in range(InitCol-1, nwidths_EP - RadiusCol):
            for ll in range(InitRow-1, nlines_EP - RadiusRow):
                block = mlistack_padded[ll - RadiusRow:ll + RadiusRow + 1,
                                        kk - RadiusCol:kk + RadiusCol + 1, :]
                ref = block[RadiusRow, RadiusCol, :]
                tasks.append((block, ref, self.CalWin, self.Alpha, RadiusRow, RadiusCol))

        total_pixels = len(tasks)
        _PixelInd = np.zeros((self.CalWin[0] * self.CalWin[1], total_pixels), dtype=bool)

        # Split tasks into chunks of ~10,000
        block_size = self.CalWin[0] * self.CalWin[1] * self.slcstack.shape[2]
        bytes_per_task = block_size * 4  # float32
        memory_limit = 100 * 1024 * 1024  # 100 MB
        chunk_size = max(1, memory_limit // bytes_per_task)
        task_batches = [tasks[i:i + chunk_size] for i in range(0, total_pixels, chunk_size)]

        print("-> SHP family parallel computation started...")
        idx = 0
        with ProcessPoolExecutor(max_workers=int(self.CPU)) as executor, tqdm(total=total_pixels, desc="-> SHP Computation", unit="pixel") as pbar:
            for result_batch in executor.map(SHP.process_batch, task_batches):
                for mask in result_batch:
                    _PixelInd[:, idx] = mask
                    idx += 1
                pbar.update(len(result_batch))

        _BroNum = np.sum(_PixelInd, axis=0).reshape((nlines, nwidths)).astype(np.float32) - 1
        
        end_time = time.time()
        print(f"-> SHP Progress finished in {(end_time - start_time)/60.0} minutes")
        return {
            'PixelInd': _PixelInd,
            'BroNum': _BroNum,
            'CalWin': self.CalWin
        }