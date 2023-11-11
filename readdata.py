import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.ndimage import convolve1d
from scipy.stats import zscore

# get rid of first 5 seconds of data

def read_Suite2p(directory_path, gcamp_type='6f', save_fig='tif', printInfo=True):
    """
    Read the data from Suite2p outputs.

    Parameters:
        directory_path (str): Folder of the image files.
        gcamp_type (str): Type of Gcamp ('6f', '7s', or '8m').
        printInfo (bool): Flag to print info or not. Default is True.

    Returns:
        tuple: F, Fneu, iscell, ops, spks, stat, roi_loc, stat_orig, ind_cells, num_roi
    """
    # Set NEUROPIL_FACTOR based on gcamp_type
    # r = 0.7 for GCaMP6f (Chen et al. 2013) and GCaMP7s(Dana et al. 2019) 
    # r = 0.8 for jGCaMP8m(Zhang et al. 2023; Pettit et al. 2022)
    if gcamp_type in ['6f', '7s']: 
        NEUROPIL_FACTOR = 0.7
    elif gcamp_type == '8m':
        NEUROPIL_FACTOR = 0.8
    else:
        raise ValueError("Invalid gcamp_type specified. Choose '6f', '7s', or '8m'.")

    # Define the path for the suite2p plane0 directory
    plane0_path = os.path.join(directory_path, 'suite2p', 'plane0')

    # Read in all output .npy files from plane0 folder
    suite2p_output = {filename: np.load(os.path.join(plane0_path, filename), allow_pickle=True)
                      for filename in os.listdir(plane0_path) if filename.endswith('.npy')}

    # Extract required arrays from suite2p_output dictionary
    F, Fneu, iscell, ops, spks, stat, stat_orig = [suite2p_output.get(key) for key in [
        "F.npy", "Fneu.npy", "iscell.npy", "ops.npy", "spks.npy", "stat.npy", "stat_orig.npy"]]
    if isinstance(ops, np.ndarray):
        ops = ops.tolist()
    
    # Compute fluorescence
    dF_F = F - Fneu * NEUROPIL_FACTOR

    # Print info if required
    if printInfo:
        print('output files = ', suite2p_output.keys())
        print("shape of 'F': ", F.shape)
        print("shape of 'Fneu': ", Fneu.shape)
        print("shape of 'iscell': ", iscell.shape)
        print("type of 'ops': ", type(ops))
        print("shape of 'spks': ", spks.shape)

    # Extract ROI information ['med': (y,x) center of cell]
    ind_cells = np.where(iscell[:, 0] == 1)
    num_roi = len(ind_cells[0])
    roi_loc = np.array([stat[roi_ind]['med'] for roi_ind in ind_cells[0]])

    # Print ROI info
    if printInfo:
        print("index of detected cells: ind_cells (ROI) = ", ind_cells[0])
        print("num_roi/cells' = ", num_roi)
        print("(y,x) location of detected cells: 'roi_loc' = ", roi_loc)

    # Plot projection images
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    titles = ["Registered Image, Max Projection", "Registered Image, Max Projection with ROI#",
              "Mean registered image", "High-pass filtered Mean registered image"]
    imgs = [ops['max_proj'], ops['max_proj'], ops['meanImg'], ops['meanImgE']]
    for ax, img, title in zip(axs.ravel(), imgs, titles):
                # Setting the vmin and vmax based on the specific subplot
        vmin_val, vmax_val = (0, 255) if title != "High-pass filtered Mean registered image" else (None, None)
        ax.imshow(img, vmin=vmin_val, vmax=vmax_val, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    # Add text for ROI numbers only on the "Registered Image, Max Projection with ROI#" subplot
    for cell in range(num_roi):
        axs[0,1].text(roi_loc[cell, 1] + 5, roi_loc[cell, 0],
                ind_cells[0][cell], fontsize=11, color='magenta')

    # Save figures
    if save_fig == 'svg':
        fig.savefig(os.path.join(directory_path, 'Projection_Image.svg'),
                    dpi=300, transparent=True)
    if save_fig == 'tif':
        fig.savefig(os.path.join(directory_path, 'Projection_Image.tif'),
                    dpi=300, transparent=True)

    return F, Fneu, iscell, ops, spks, stat, roi_loc, stat_orig, ind_cells, num_roi


def read_StimuliFile(stimuli_file, printInfo=True):
    """
    Read stimuli information from the provided file.
    
    Parameters:
        stimuli_file (str): Path to the stimuli file.
        printInfo (bool): Flag to print info or not. Default is True.
        
    Returns:
        tuple: stimuli_seq (all the stimuli in the file), stimuli_list (all stimuli types)
    """
    stimuli_seq = np.loadtxt(stimuli_file, delimiter='\t')
    stimuli_list = np.unique(stimuli_seq, axis=0)

    if printInfo:
        print("Stimuli Information")
        print("stimuli sequence used = 'stimuli_file' = ", stimuli_file)
        print("stimuli in the file: 'stimuli_seq' = ", stimuli_seq)
        print("shape of 'stimuli_seq' = ", stimuli_seq.shape)
        print("unique stimuli: 'stimuli_list' = ", stimuli_list)

    return stimuli_seq, stimuli_list


def read_speed_data(speed_raw_data, speed_rate=30*1000):
    """
    Read and process speed data.
    
    Parameters:
        speed_raw_data (str): Path to the speed data file.
        speed_rate (int): Speed rate set in Thorsync. Default is 30 kHz.
        
    Returns:
        ndarray: Smoothed speed data (full recorded data, 
        hasn't pick out trial time yet).
    """
    thorsync_data = sio.loadmat(speed_raw_data)
    speed_data = thorsync_data['speedEncoder'][:, 0]
    return convolve1d(speed_data, weights=np.ones(int(speed_rate))) / int(speed_rate)


def read_spike_R(filename, num_roi, spike=1):
    """
    Read spike or calcium activity data from the provided file: convol_dF_F_data_Rcode.R result - a txt file.
    
    Parameters:
        filename (str): Path to the data file.
        num_roi (int): Number of regions of interest.
        spike (int): If 1, reads deconvolved spike. If 0, reads calcium activity. Default is 1.
        
    Returns:
        dict: Spike or calcium activity data.
    """
    with open(filename) as f:
        spikes_data = f.readlines()
    
    return {
        cell: [int(val) if spike == 1 else float(val) for val in line.rstrip().split(',')]
        for cell, line in enumerate(spikes_data)
    }


def read_video(filename):
    """
    Read video data from Facemap output.

    Parameters:
        filename (str): Path to the video data file.
        
    Returns:
        tuple: video_data_motion, video_data_motSVD, video_data_running, video_data_blink, video_data_pupil
            motion: list of absolute motion energies across time - 
                first is "multivideo" motion energy (empty if not computed)
            motSVD: list of motion SVDs - 
                first is "multivideo SVD" (empty if not computed) - 
                each is nframes x components
            pupil: list of pupil ROI outputs - 
                each is a dict with 'area', 'area_smooth', and 'com' (center-of-mass)
            blink: list of blink ROI outputs - 
                each is nframes, the blink area on each frame
            running: list of running ROI outputs - 
                each is nframes x 2, for X and Y motion on each frame
    """
    video = np.load(filename, allow_pickle=True).item()
    
    # Extract required data from video dictionary
    # remove 1st - first is "multivideo SVD" (empty if not computed)
    video_data_motion = video['motion'][1] # shape (13960,)
    video_data_motSVD = video['motSVD'][1:][0] # shape (13960, 500)
    # shape (13960, 2), X&Y fourier space
    video_data_running = video['running'][0]
    video_data_blink = video['blink'][0] # shape (13960,)
    video_data_pupil = video['pupil'][0]
    # replace the blink frames with NaNs
    # video_data_pupil[video_data_blink > 0] = np.nan
    # pupil zscores using the sample standard deviation (ddof=1)
    # pupil_zscored = zscore(video_data_pupil, ddof=1, nan_policy='omit')
    pupil_zscored = zscore(video_data_pupil)

    print("pupil - dict_keys(['area', 'com', 'axdir', 'axlen', 'area_smooth', 'com_smooth'])")
    return video_data_motion, video_data_motSVD, video_data_running, video_data_blink, pupil_zscored
