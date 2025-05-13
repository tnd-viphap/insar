import numpy as np
from scipy.stats import rankdata

class BWS:

    """
    Accelerated Baumgartner-Weiß-Schindler test for multiple sample columns.
    
    Args:
        Xarray (np.ndarray): (n, m) matrix — sample set 1
        Yarray (np.ndarray): (n, m) matrix — sample set 2
        Alpha (float): Significance level; supported: 0.05 (default), 0.01
    
    Returns:
        np.ndarray: Boolean array (size m,) where True indicates rejection of the null hypothesis
    """

    def __init__(self, x, y, alpha=0.05):
        self.Xarray = x
        self.Yarray = y
        self.Alpha = alpha

    def run(self):
        n, m = self.Xarray.shape
        ranks = np.apply_along_axis(rankdata, 0, np.vstack((self.Xarray, self.Yarray)))
        
        xrank = np.sort(ranks[:n, :], axis=0)
        yrank = np.sort(ranks[n:, :], axis=0)
        
        temp = np.arange(1, n + 1).reshape(-1, 1) @ np.ones((1, m))
        tempx = (xrank - 2 * temp) ** 2
        tempy = (yrank - 2 * temp) ** 2
        denom = (temp / (n + 1)) * (1 - temp / (n + 1)) * 2 * n
        
        BX = (1 / n) * np.sum(tempx / denom, axis=0)
        BY = (1 / n) * np.sum(tempy / denom, axis=0)
        
        B = 0.5 * (BX + BY)

        # Determine critical value
        if self.Alpha == 0.05:
            if n == 5:
                b = 2.533
            elif n == 6:
                b = 2.552
            elif n == 7:
                b = 2.620
            elif n == 8:
                b = 2.564
            elif n == 9:
                b = 2.575
            elif n == 10:
                b = 2.583
            else:
                b = 2.493  # Table 1 from the paper
        else:  # Alpha == 0.01
            b = 3.880

        # Return H as a boolean array: True = reject null
        H = B >= b
        return H