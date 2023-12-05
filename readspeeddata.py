import numpy as np
import scipy.io as sio
from scipy.ndimage import convolve1d


FRAME_RATE = 30  # Imaging frame rate, Hz
DATAPOINTS_PER_FRAME = 1000 # Speed datapoints per imaging frame

# ---------------------------FUNCTION FOR extract speed data from wheel from MATLAB file--------------------------------------------
def read_speed_data(speed_raw_data, speed_rate=FRAME_RATE*DATAPOINTS_PER_FRAME):
    """
    Read and smooth speed data from a MATLAB file.
    
    Parameters:
        speed_raw_data (str): Path to the speed data file.
        speed_rate (int): Speed rate set in Thorsync. Default is 30 kHz.
        
    Returns:
        ndarray: Smoothed speed data (full recorded data, hasn't pick out trial time yet).
    """
    thorsync_data = sio.loadmat(speed_raw_data)
    speed_data = thorsync_data['speedEncoder'][:, 0]
    return convolve1d(speed_data, weights=np.ones(int(speed_rate))) / int(speed_rate)
