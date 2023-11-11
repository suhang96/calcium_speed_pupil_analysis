import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
from scipy.ndimage import convolve1d, median_filter
from scipy.spatial import distance_matrix
from scipy.stats import sem, bootstrap
from scipy import signal


def configure_plotting_styles():
    """
    Configure global plotting styles for consistency across all plots.
    """
    # Define size parameters
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    # Apply configuration
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

class Spontaneous:
    """
    A class to analyze two-photon calcium imaging data from cells
    and correlate with the behavior of mice (pupil size and locomotion speed).
    
    Attributes
    ----------
    speed_data_smooth : np.array
        Smoothed data representing the locomotion speed of the mice.
    F : np.array
        Raw fluorescence data from the cells.
    Fneu : np.array
        Raw fluorescence data from the neuropil.
    ind_cells : list or np.array
        Indices or identifiers for cells.
    num_roi : int
        Number of Regions of Interest (ROIs), which are cells.
    roi_loc : list or np.array
        Locations of the ROIs/cells on the image region.
    directory_path : str
        Path to the directory where the data is stored.
    """
    NEUROPIL_FACTOR = 0.8 # Default is 0.8 for jGcamp8m # from papers [Harvey lab 2022; jGcamp8 biorxiv])
    DATAPOINTS_PER_FRAME = 1000 # Speed datapoints per frame
    FRAME_RATE = 30  # Imaging frame rate, Hz
    SPEED_TIME = 730 # Recording time of speed data, second
    MEDIAN_FILTER_WINDOW = 180 
    PIX_SIZE = 0.805 # the field of view measured 512 × 512 pixels with pixel size = 0.805 um, end up with 412.07 um X 412.07 um FOV.
    RANDOM_NUM = 42 # random state
    VIDEO_FRAMERATE = 19.12 # Hz
    RUN_SPEED_THRESHOLD = 1  # cm/s

    def __init__(self, 
                 speed_data_smooth: np.array, 
                 F: np.array, 
                 Fneu: np.array, 
                 ind_cells: list, 
                 num_roi: int, 
                 roi_loc: list, 
                 directory_path: str,
                 datapoints_per_frame: int = DATAPOINTS_PER_FRAME,
                 neuropil_factor: float = NEUROPIL_FACTOR,
                 frame_rate: float = FRAME_RATE,
                 speed_time: int = SPEED_TIME,
                 median_filter_window: int = MEDIAN_FILTER_WINDOW,
                 pix_size: float = PIX_SIZE,
                 random_num: int = RANDOM_NUM,
                 video_framerate: float = VIDEO_FRAMERATE):
        """
        Initializes the Spontaneous class with imaging and behavior data.

        Parameters
        ----------
        speed_data_smooth : np.array
            Smoothed data representing the locomotion speed of the mice.
        F : np.array
            Raw fluorescence data from the cells.
        Fneu : np.array
            Raw fluorescence data from the neuropil.
        ind_cells : list or np.array
            Indices or identifiers for cells.
        num_roi : int
            Number of Regions of Interest (ROIs), which are cells.
        roi_loc : list or np.array
            Locations of the ROIs/cells on the image region.
        directory_path : str
            Path to the directory where the data is stored.
        """
        self.speed_data_smooth = speed_data_smooth
        self.datapoints_per_frame = datapoints_per_frame
        self.F = F
        self.Fneu = Fneu
        self.ind_cells = ind_cells
        self.num_roi = num_roi
        self.neuropil_factor = neuropil_factor
        self.frame_rate = frame_rate
        self.speed_time =  speed_time
        self.median_filter_window = median_filter_window
        self.directory_path = directory_path
        self.roi_loc = roi_loc
        self.pix_size = pix_size
        self.random_num = random_num 
        self.video_framerate = video_framerate

    def binning_speed(self, add_one: bool = True) -> Tuple[np.array, np.array]:
        """
        Bin speed data to the same time resolution as the imaging data (30 Hz).
        
        Transforms the recording value to speed (cm/s). Note that in some spontaneous recordings, 
        there's 1 frame less, so an additional frame may need to be added at the start for shape.
        
        Parameters
        ----------
        add_one : bool, optional
            A flag to indicate whether to add an additional frame at the start, by default True
        
        Returns
        -------
        Tuple[np.array, np.array]
            binned_speed: 1D array representing speed after binning.
            binned_speed_abs: 1D array representing absolute speed after binning.
        """
        # Validation
        if not isinstance(add_one, bool):
            raise ValueError("add_one must be a boolean value.")
        
        # Data Preparation
        speed_data_smooth_data = (
            np.hstack((self.speed_data_smooth, self.speed_data_smooth[0])) if add_one 
            else np.hstack((self.speed_data_smooth))
        )
        
        # Binning and Transformation
        binned_speed_raw = np.average(
            speed_data_smooth_data.reshape(-1, int(self.datapoints_per_frame)), 
            axis=1
        ).flatten()
        binned_speed = (binned_speed_raw-1.6604)* 287.22 - 0.24 # measeured from wheel
        binned_speed_abs = abs(binned_speed)

        # Output Shape Logging
        print(f"shape of 'binned_speed' or 'binned_speed_abs'= {binned_speed.shape}")
        
        return binned_speed, binned_speed_abs

    def get_F(self) -> np.array:
        """
        Retrieve and process F, subtracting Fneu for the entire recording session.
        
        Utilizes F and Fneu (activity and neurophil activity extracted from ROIs),
        ind_cells (indexes of ROIs from suite2p), num_roi (number of ROIs),
        and other attributes to compute F_data.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        np.array
            F_data: Activity data after subtracting neurophil response. 
            The shape of F_data is logged to the console.
        """
        # Data Preparation & Processing
        F_subset = self.F[
            self.ind_cells, :int(self.speed_time * self.frame_rate)
        ].reshape(self.num_roi, -1)
        Fneu_subset = self.Fneu[
            self.ind_cells, :int(self.speed_time * self.frame_rate)
        ].reshape(self.num_roi, -1)
        
        # Compute F_data
        F_data = F_subset - self.neuropil_factor * Fneu_subset
        
        # Output Shape Logging
        print(f"shape of 'F_data' = {F_data.shape}")
        
        return F_data

    def get_dF_F(self, F_data: np.array) -> np.array:
        """
        Compute dF/F, using F after subtracting Fneu and applying a median filter window 
        for baseline calculation. Method used: DJ Millman et al., 2020 by Allen Brain Institute.
        
        Parameters
        ----------
        F_data : np.array
            Activity data after neurophil subtraction, obtained from (F - neuropil_factor * Fneu).
            Expected shape: (num_roi, frames).
        
        Returns
        -------
        np.array
            dF_F: The computed dF/F data. The shape of dF_F is logged to the console.
        """
        # Validation
        if not isinstance(F_data, np.ndarray):
            raise ValueError("F_data must be a NumPy array.")
        
        # Data Preparation & Processing
        median_filter_frames = int(self.median_filter_window * self.frame_rate)
        dF_F_baseline = np.zeros_like(F_data)
        
        for cell in range(self.num_roi):
            dF_F_baseline[cell] = median_filter(F_data[cell], median_filter_frames)
        
        # Compute dF/F
        dF_F = (F_data - dF_F_baseline) / dF_F_baseline
        
        # Output Shape Logging
        print(f"shape of dF_F = {dF_F.shape}")
        
        return dF_F
    
    def smoothing_dF_F(self, dF_F: np.array) -> np.array:
        """
        Smooth the dF/F traces using a 1D convolution (scipy.ndimage.convolve1d).
        
        Parameters
        ----------
        dF_F : np.array
            Activity data with shape: (num_roi, frames).
        
        Returns
        -------
        np.array
            dF_F_smooth: dF/F data after smoothing with frame_rate. The shape is logged to the console.
        """
        # Validation
        if not isinstance(dF_F, np.ndarray):
            raise ValueError("dF_F must be a NumPy array.")
        
        # Smoothing
        weights = np.ones(int(self.frame_rate)) / int(self.frame_rate)
        dF_F_smooth = convolve1d(dF_F, weights, axis=1, mode='nearest')
        
        # Output Shape Logging
        print(f"shape of dF_F_smooth = {dF_F_smooth.shape}")
        
        return dF_F_smooth
    
    def get_run_period(self, binned_speed: np.array, dF_F_smooth: np.array) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Identify running and static periods based on locomotion speed, and calculate 
        averaged dF/F for running and static frames.
        
        Parameters
        ----------
        binned_speed : np.array
            1D array of binned locomotion speeds.
        dF_F_smooth : np.array
            2D array of smoothed dF/F values, shape: (num_roi, frames).

        Returns
        -------
        tuple
            run_frames: 1D array of frame indices where speed > 1 cm/s or < -1 cm/s.
            run_ave_dF_F: 1D array of averaged dF/F values across running frames.
            stat_frames: 1D array of frame indices where -1 cm/s < speed < 1 cm/s.
            stat_ave_dF_F: 1D array of averaged dF/F values across static frames.
        """
        # Validate inputs
        if not (isinstance(binned_speed, np.ndarray) and isinstance(dF_F_smooth, np.ndarray)):
            raise ValueError("Inputs binned_speed and dF_F_smooth must be NumPy arrays.")
        
        # Identify running and static frames
        run_frames = np.where(np.abs(binned_speed) >= 1)[0]
        stat_frames = np.where(np.abs(binned_speed) < 1)[0]
        
        # Calculate average dF/F for running and static frames
        run_ave_dF_F = np.mean(dF_F_smooth[:, run_frames], axis=1)
        stat_ave_dF_F = np.mean(dF_F_smooth[:, stat_frames], axis=1)
        
        return run_frames, run_ave_dF_F, stat_frames, stat_ave_dF_F

    def plot_overview(self, dF_F_smooth: np.array, binned_speed: np.array,
                      run_frames: np.array, video_data_pupil: dict,
                      video_data_blink: np.array, video_data_running: np.array,
                      video_data_motSVD: np.array, plot_factor: int = 1):
        """
        Plot individual neurons' activity + averaged population mean + locomotion spped
        
        Parameters:
            dF_F_smooth (np.ndarray): dF/F after smoothing with frame_rate.
            binned_speed (np.ndarray): 1D speed after binning, with the same rate as imaging.
            run_frames (np.ndarray): Frames that mouse runs > 1 cm/s.
            video_data_pupil (dict): video_data_pupil['area_smooth'] is the size of pupil
            video_data_blink (np.ndarray): blinking frames should be removed
            video_data_running (np.ndarray): running speed from video 
            video_data_motSVD (np.ndarray): motionSVD data from video at the nose/face/whisker area
            plot_factor (int, optional): Scale for plot, default = 1.
        
        Returns:
            None
            Two figures 'dF_F_Overview' and 'dF_F_Overview_cell' are saved to the directory_path.
        """
        
        # Derive time vectors once
        time_dF_F = np.arange(dF_F_smooth.shape[1]) / self.frame_rate
        time_speed = np.arange(binned_speed.shape[0]) / self.frame_rate

        # Figure 1
        fig1, ax = plt.subplots(figsize=(30, self.num_roi))
        self.plot_dF_F_smooth(ax, dF_F_smooth, time_dF_F, plot_factor)
        self._save_plot(fig1, 'dF_F_Overview_cell')

        # Figure 2
        fig2, axs = plt.subplots(7, 1, figsize=(30, 15))
        
        # Plotting subplots
        self.plot_mean_population_dF_F(axs[0], dF_F_smooth, time_dF_F)
        self.plot_locomotion_speed(axs[1], binned_speed, run_frames, time_speed)
        self.plot_absolute_locomotion_speed(axs[2], binned_speed, run_frames, time_speed)
        self.plot_pupil_smooth(axs[3], video_data_pupil)
        self.plot_blink_smooth(axs[4], video_data_blink)
        self.plot_running(axs[5], video_data_running)
        self.plot_nose(axs[6], video_data_motSVD)

        # Save and show the plot
        self._save_plot(fig2, 'dF_F_Overview')
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------------------------------------------------------------------
    # plot helper functions
    def plot_dF_F_smooth(self, ax, dF_F_smooth, time_vector, plot_factor):
        for cell in range(self.num_roi):
            ax.plot(time_vector, dF_F_smooth[cell] + plot_factor * cell)
        ax.set_xlabel('Time(s)')
        ax.set_ylabel('dF/F')
        ax.set_ylim([-1, self.num_roi * plot_factor + 3])
        ax.set_title(self.directory_path)    
    
    def plot_mean_population_dF_F(self, ax, dF_F_smooth, time_vector):
        ax.plot(time_vector, np.average(dF_F_smooth, axis=0))
        ax.set_xlabel('Time(s)')
        ax.set_ylabel('Mean Population dF/F')

    def plot_locomotion_speed(self, ax, binned_speed, run_frames, time_vector):
        ax.plot(time_vector, binned_speed, c='black')
        ax.axhline(y=1, color='dimgray',linestyle='dashed', linewidth=0.5) # 1cm/s
        ax.axhline(y=-1, color='dimgray',linestyle='dashed', linewidth=0.5) # -1cm/s
        ax.eventplot(run_frames/self.frame_rate, linelengths=0.5, lineoffsets=1+np.max(binned_speed), color='red')
        ax.set_xlabel('Time(s)')
        ax.set_ylabel('Locomotion \n Speed (cm/s)')

    def plot_absolute_locomotion_speed(self, ax, binned_speed, run_frames, time_vector):
        ax.plot(time_vector, np.abs(binned_speed), c='black')
        ax.axhline(y=1, color='dimgray', linestyle='dashed', linewidth=0.5) # 1cm/s
        ax.axhline(y=-1, color='dimgray', linestyle='dashed', linewidth=0.5) # -1cm/s
        ax.eventplot(run_frames/self.frame_rate, linelengths=0.5, lineoffsets=1+np.max(np.abs(binned_speed)), color='red')
        ax.set_xlabel('Time(s)')
        ax.set_ylabel('Absolute Locomotion \n Speed (cm/s)')

    def plot_pupil_smooth(self, ax, video_data_pupil):
        pupil_smooth = self._video_convolve_smooth(video_data_pupil['area_smooth'])
        ax.plot((np.arange(pupil_smooth.shape[0])/self.video_framerate), pupil_smooth, color='black')
        ax.set_xlabel('Time(s)')
        ax.set_ylabel('Pupil Area_smooth')
    
    def plot_blink_smooth(self, ax, video_data_blink):
        blink_smooth = self._video_convolve_smooth(video_data_blink)
        ax.plot((np.arange(blink_smooth.shape[0])/self.video_framerate), blink_smooth, color='darkgrey')
        ax.set_xlabel('Time(s)')
        ax.set_ylabel('Eyelid (blink)')

    def plot_running(self, ax, video_data_running):
        for dim in range(2):  # First two dimensions
            run_smooth = self._video_convolve_smooth(video_data_running[:, dim])
            ax.plot((np.arange(run_smooth.shape[0])/self.video_framerate), run_smooth, alpha=0.5)
        ax.set_xlabel('Time(s)')
        ax.set_ylabel('Running x-y \n from Video')

    def plot_nose(self, ax, video_data_motSVD):
        for dim in range(3):  # First three dimensions
            mot_smooth = self._video_convolve_smooth(video_data_motSVD[:, dim])
            ax.plot((np.arange(mot_smooth.shape[0])/self.video_framerate), mot_smooth, alpha=0.5)
        ax.set_xlabel('Time(s)')
        ax.set_ylabel('Nose/Groom \n first three dimensions')

    def _video_convolve_smooth(self, data):
        return convolve1d(data[:int(self.speed_time * self.video_framerate)],
                          weights=np.ones(int(self.video_framerate)))/int(self.video_framerate)   
    
    def _save_plot(self, figure, filename_stem):
        """Save the plot in multiple formats."""
        # Ensure directory exists
        os.makedirs(self.directory_path, exist_ok=True)

        # Save the figure
        for ext in ['svg', 'tif']:
            path = os.path.join(self.directory_path, f"{filename_stem}.{ext}")
            figure.savefig(path, dpi=300, transparent=True)

    # end of plot helper functions
    # --------------------------------------------------------------------------------------------------------------------

    def downsample_array(self, data: np.ndarray, target_size: int) -> np.ndarray:
        """
        Downsample a 1D numpy array by averaging adjacent values.
        [https://stackoverflow.com/questions/36284100/how-we-can-down-sample-a-1d-array-values-by-averaging-method-using-float-and-int]

        Parameters:
            data (np.ndarray): The 1D array to be downsampled.
            target_size (int): The target size of the downsampled array.
            
        Returns:
            np.ndarray: The downsampled array.
            
        Raises:
            ValueError: If data is not a 1D numpy array or if target_size is not a positive integer.
        """
        # Input validation
        if not isinstance(data, np.ndarray) or data.ndim != 1:
            raise ValueError("Input data must be a 1D numpy array.")
        if not isinstance(target_size, int) or target_size <= 0:
            raise ValueError("Target size must be a positive integer.")
        
        # Downsampling ...
        M = data.size
        res = np.empty(target_size, data.dtype)
        carry = 0
        m = 0
        for n in range(target_size):
            sum_val = carry
            while m * target_size - n * M < M:
                sum_val += data[m]
                m += 1
            carry = (m - (n + 1) * M / target_size) * data[m - 1]
            sum_val -= carry
            res[n] = sum_val * target_size / M
        
        return res
    
    def compute_cross_correlation(self, signals: np.ndarray, reference_signal: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the cross-correlation between each row (ROI/cell) of 'signals' and 'reference_signal'.
        If a positive value of mov_lag is observed, it means that the corresponding signal in 'signals' lags behind 'reference_signal'. Conversely, a negative value indicates that the signal in 'signals' leads 'reference_signal'. 

        Parameters:
        - signals: 2D numpy array where each row represents a distinct signal. Shape = (num_roi, datapoints).
        - reference_signal: 1D numpy array representing the reference signal (behavior) against which the cross-correlation is computed. Shape = datapoints.
        - sample_rate: Sampling rate of the signals.
        
        Returns:
        - zero_time_correlation: Cross-correlation value at zero lag for each signal.
        - max_corr_lag: Lag (in seconds) at which the maximum correlation occurs for each signal.
        - correlations: Cross-correlation values for each signal.
        - lags: Lags (in sample points) corresponding to the correlation values for each signal.
        """
        
        num_signals, signal_length = signals.shape
        correlation_length = int(signal.correlate(signals[0, :], reference_signal).shape[0])
        
        correlations = np.zeros((num_signals, correlation_length))
        lags = np.zeros((num_signals, correlation_length))
        zero_time_correlation = np.zeros(num_signals)
        max_corr_lag = np.zeros(num_signals)

        for idx, signal_row in enumerate(signals):
            normalized_signal = signal_row - np.mean(signal_row)
            normalized_reference = reference_signal - np.mean(reference_signal)
            
            correlations[idx] = signal.correlate(normalized_signal, normalized_reference)
            correlations[idx] /= signal_length * normalized_signal.std() * normalized_reference.std()
            
            lags[idx] = signal.correlation_lags(len(signal_row), len(reference_signal))
            
            zero_time_correlation[idx] = correlations[idx][np.where(lags[idx] == 0)]
            max_corr_sample_lag = lags[idx][np.argmax(correlations[idx])]
            max_corr_lag[idx] = max_corr_sample_lag / sample_rate

        return zero_time_correlation, max_corr_lag, correlations, lags
  
    def mov_cross_corr(self, 
                    dF_F_smooth: np.ndarray, 
                    binned_speed: np.ndarray, 
                    video_data_pupil: Dict[str, np.ndarray], 
                    downsample_rate: int = 10) -> Tuple[List[np.ndarray], ...]:
        """
        Compute and plot the cross-correlation of speed or pupil vs dF/F for each neuron binned to a specified rate (all downsampled to 10Hz) .
        
        Parameters:
            dF_F_smooth (np.ndarray): dF/F data smoothed with frame rate.
            binned_speed (np.ndarray): Speed data after binning.
            video_data_pupil (Dict[str, np.ndarray]): Dictionary containing video data for the pupil. 
                                                    Expected key: 'area_smooth'.
            downsample_rate (int, optional): Rate for downsampling. Defaults to 10Hz.
            
        Returns:
            Tuple[List[np.ndarray], ...]: A tuple containing cross-correlation results of dF/F0 vs speed + dF/F0 vs pupil area + pupil area vs speed.
        """
        
        # Downsample the provided data
        size_downsampled = int(self.speed_time * downsample_rate)
        
        dF_F_smooth_downsampled = np.array([self.downsample_array(cell_data, size_downsampled) 
                                for cell_data in dF_F_smooth[:, :int(self.speed_time * self.frame_rate)]])
        
        binned_speed_downsampled = self.downsample_array(binned_speed[:int(self.speed_time * self.frame_rate)], size_downsampled)
        
        video_data_pupil_area_smooth_downsampled = self.downsample_array(video_data_pupil['area_smooth'][:int(self.speed_time * self.video_framerate)], size_downsampled)

        # Plot setup
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        
        # Compute and plot dF/F vs movement/locomotion cross-correlation returns zero_time_correlation, max_corr_lag, correlations, lags
        mov_results = self.compute_cross_correlation(dF_F_smooth_downsampled, binned_speed_downsampled, downsample_rate)
        for mov_cell_corr, mov_cell_lags in zip(*mov_results[2:]):
            ax1.plot(mov_cell_lags / downsample_rate, mov_cell_corr, 'lightgrey')
        ax1.plot(mov_cell_lags / downsample_rate, np.average(mov_results[2], axis=0), color= 'red')
        ax1.text(0.55, 0.97, ' mean_zero_corr: {:.2f} \n mean_max_corr: {:.2f} \n mean_lag: {:.2f} s'.format(np.mean(mov_results[0]), np.max(np.mean(mov_results[2], axis=0)), mov_cell_lags[np.argmax(np.average(mov_results[2], axis=0))] / downsample_rate), verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color='red')
        ax1.set_title('dF/F smooth vs locomotion')
        
        # Compute and plot dF/F vs pupil area cross-correlation returns zero_time_correlation, max_corr_lag, correlations, lags
        pupil_results = self.compute_cross_correlation(dF_F_smooth_downsampled, video_data_pupil_area_smooth_downsampled, downsample_rate)
        for pupil_cell_corr, pupil_cell_lags in zip(*pupil_results[2:]):
            ax2.plot(pupil_cell_lags / downsample_rate, pupil_cell_corr, 'lightgrey')
        ax2.plot(pupil_cell_lags / downsample_rate, np.average(pupil_results[2],axis=0), color= 'red')
        ax2.text(0.55, 0.97, ' mean_zero_corr: {:.2f} \n mean_max_corr: {:.2f} \n mean_lag: {:.2f} s'.format(np.mean(pupil_results[0]), np.max(np.mean(pupil_results[2], axis=0)), pupil_cell_lags[np.argmax(np.average(pupil_results[2], axis=0))] / downsample_rate), verticalalignment='top', horizontalalignment='left', transform=ax2.transAxes, color='red')
        ax2.set_title('dF/F smooth vs pupil area')
        
        # Compute and plot pupil area vs locomotion cross-correlation (1d vs 1d)
        pupil_mov_results = self.compute_cross_correlation(video_data_pupil_area_smooth_downsampled[np.newaxis,:], binned_speed_downsampled, downsample_rate)
        ax3.plot(pupil_mov_results[-1][-1] / downsample_rate, np.average(pupil_mov_results[2],axis=0), color= 'black')
        ax3.text(0.55, 0.97, ' mean_zero_corr: {:.2f} \n mean_max_corr: {:.2f} \n mean_lag: {:.2f} s'.format(np.mean(pupil_mov_results[0]), np.max(np.mean(pupil_mov_results[2], axis=0)), np.mean(pupil_mov_results[1])), verticalalignment='top', horizontalalignment='left', transform=ax3.transAxes, color='black')
        ax3.set_title('pupil area vs locomotion')
        
        # formatting ax1, ax2, ax3
        for ax in [ax1,ax2,ax3]:
            ax.axvline(x=0, color='dimgray',linestyle='dashed', linewidth = 0.5)
            ax.set_ylabel('Cross Correlation (zero-time)')
            ax.set_xlabel('Time(s)')
            ax.set_xlim([-40,40])
            ax.set_ylim([-1,1.1])

        ax4.scatter(binned_speed_downsampled, video_data_pupil_area_smooth_downsampled, c='lightgray', s=3)
        ax4.set_xlabel('Locomotion Speed')
        ax4.set_ylabel('Pupal Area')
        ax4.set_title('pupil area vs locomotion')

        plt.tight_layout()
        fig.subplots_adjust(top=0.95)
        fig.savefig(f"{self.directory_path}/cross_correlation.svg", dpi=300, transparent=True)
        fig.savefig(f"{self.directory_path}/cross_correlation.tif", dpi=300, transparent=True)
        
        return mov_results + pupil_results + pupil_mov_results

    def compute_1d_cross_correlation(self, cellA_data: np.ndarray, cellB_data: np.ndarray) -> np.ndarray:
        """
        Calculate the pairwise cross correlation of the ROIs' activities.
        """
        return signal.correlate(cellA_data - np.mean(cellA_data), cellB_data - np.mean(cellB_data))
    
    def get_all_roi_pairs_distance(self) -> np.ndarray:
        """
        Calculate the pairwise pixel distance of the ROIs and convert to physical distance in um.
        """
        return distance_matrix(self.roi_loc, self.roi_loc) * self.pix_size
    
    def roi_cross_corr(self, dF_F_smooth: np.ndarray) -> tuple:
        '''
        Computes and plots cross-correlation of dF/F within population. 
        
        Returns:
            Tuple containing various processed data including zero-time cross correlation, lags, etc.
        '''

        fig, axes = plt.subplots(self.num_roi, self.num_roi, figsize=(2*self.num_roi, 2*self.num_roi))
        num_samples = dF_F_smooth.shape[1]
        correlation_shape = int(signal.correlate(dF_F_smooth[0], dF_F_smooth[0]).shape[0])
        
        correlations = np.zeros((self.num_roi, self.num_roi, correlation_shape))
        lags_matrix = np.zeros_like(correlations)
        lag = np.zeros((self.num_roi, self.num_roi))
        zero_time_cross = np.zeros((self.num_roi, self.num_roi))
        lags = signal.correlation_lags(num_samples, num_samples)
        
        for cellA in range(self.num_roi):
            for cellB in range(self.num_roi):
                correlations[cellA, cellB] = self.compute_1d_cross_correlation(dF_F_smooth[cellA], dF_F_smooth[cellB])
                correlations[cellA, cellB] /= num_samples * dF_F_smooth[cellA].std() * dF_F_smooth[cellB].std() # correct on 8/3/22, earlier were wrong by /=max(corr)
                lags_matrix[cellA, cellB] = lags
                zero_time_cross[cellA, cellB] = correlations[cellA, cellB][np.where(lags == 0)]
                lag[cellA, cellB] = lags[np.argmax(correlations[cellA, cellB])]
                
                ax = axes[cellA, cellB]
                ax.plot(lags / self.frame_rate, correlations[cellA, cellB], 'black')
                ax.axvline(x=0, color='dimgray', linestyle='dashed', linewidth=0.5)
                ax.set_xlim([-40, 40])
                ax.set_ylim([-1.1, 1.1])
                ax.set_title(f'CellA: {cellA + 1}, ROIA: {self.ind_cells[0][cellA]} \n CellB: {cellB + 1}, ROIB: {self.ind_cells[0][cellB]}')
                if cellB == 0:
                    ax.set_ylabel('Cross Correlation \n (zero-time)')
                if cellA == self.num_roi - 1:
                    ax.set_xlabel('Time(s)')
        lag /= self.frame_rate

        # Saving the main cross-correlation figure
        fig.suptitle('dF/F smooth vs dF/F smooth')
        # fig.subplots_adjust(top=0.80)
        plt.tight_layout()
        fig.savefig(f"{self.directory_path}/roi_cross_correlation.svg", dpi=300, transparent=True)
        fig.savefig(f"{self.directory_path}/roi_cross_correlation.tif", dpi=300, transparent=True)

        # Additional visualization: Heatmap of zero-time cross-correlations
        fig2, ax1 = plt.subplots(1, 1, figsize=(self.num_roi, int(0.75 * self.num_roi)))
        corr_mask = np.triu(zero_time_cross, k=1)
        sns.heatmap(zero_time_cross, annot=True, cmap="viridis", ax=ax1, mask=corr_mask)
        ax1.set_title('Cross Correlation (zero-time)')
        fig2.savefig(f"{self.directory_path}/roi_zero_cross_correlation.svg", dpi=300, transparent=True)
        fig2.savefig(f"{self.directory_path}/roi_zero_cross_correlation.tif", dpi=300, transparent=True)

        # relationship between corr and distance
        dis_matrix = self.get_all_roi_pairs_distance()
        dis_pairs = self.get_matrix_lower_tri(dis_matrix)
        corr_pairs = self.get_matrix_lower_tri(zero_time_cross)
        
        # plot the corr_distance relationship
        fig3, ((ax2, ax3, ax8, ax9), (ax4, ax5, ax6, ax7)) = plt.subplots(2,4,figsize=(15,6))
        ax2.scatter(dis_pairs, corr_pairs, s=3, c='black', marker = "o", alpha=0.65)
        ax2.set_title('correlation vs distance of ROI pairs') 
        ax2.set_xlabel('Distance (um)')
        ax2.set_ylabel('Cross Correlation \n (zero-time)')
        ax2.set_xlim([0,413])
        ax2.axhline(y=0.15, color='dimgray',linestyle='dashed', linewidth = 0.5)

        # plot the corr_distance relationship with ylim
        ax3.scatter(dis_pairs, corr_pairs, s=3, c='black', marker = "o", alpha=0.65)
        ax3.set_title('correlation vs distance of ROI pairs') 
        ax3.set_xlim([0,413])
        ax3.axhline(y=0.15, color='dimgray',linestyle='dashed', linewidth = 0.5)

        # plot summary of data dots
        ax8.bar(0.5, np.mean(corr_pairs), width=0.1, yerr=sem(corr_pairs), edgecolor='black', color='None')
        ax8.set_xlim([0,1])
        ax8.set_xticks([0.5], ['Cell Type'])
        ax8.set_ylim([0,1])
        ax8.set_ylabel('Mean ± SEM \n Cross Correlation \n (zero-time)')
        ax8.text(0.5, 0.1 + np.mean(corr_pairs), 'mean ± SEM = {:.2f} ± {:.2f}'.format(np.mean(corr_pairs), sem(corr_pairs)), verticalalignment='top', horizontalalignment='center', color='red')

        n, bins, patches = ax9.hist(corr_pairs, weights = np.ones_like(corr_pairs) / len(corr_pairs), bins = int(len(corr_pairs)/2), histtype='step', color = 'black')
        ax9.set_xlim([-0.5,1])
        ax9.set_xlabel('Cross Correlation \n (zero-time)')
        ax9.set_ylabel('Fraction')
        ax9.set_title('n = {} pairs'.format(len(corr_pairs)))

        # plot the Corr_distance relationship with combined correlation
        corr_dis_group = self.group_distance_corr(dis_pairs,corr_pairs) 
        group_dis = ['≤100','100-200','200-300','>300']
        # plot scatter dots
        for i in range(4):
            ax4.scatter(i+0.5*np.ones(len(corr_dis_group[i])), corr_dis_group[i], s=1, c='grey')
        ax4.set_xticks([0.5,1.5,2.5,3.5], group_dis)
        ax4.set_title('Data Points')

        # get stats for ploting
        # get mean
        corr_dis_mean = np.zeros(4)
        for i in range(4):
                corr_dis_mean[i] = np.mean(corr_dis_group[i])
        # get variation stats
        # 95% CI 
        corr_dis_ci = np.zeros((4,2))
        for i in range(4):
            if len(corr_dis_group[i]) <2: # if no enough datapoint
               corr_dis_ci[i] = [np.nan, np.nan]
            else: 
               corr_dis_ci[i] = bootstrap((corr_dis_group[i],), np.std, axis=-1, confidence_level=0.95, method='percentile', n_resamples=1000, random_state=self.random_num).confidence_interval
        corr_dis_ci = corr_dis_ci.T
        ax5.errorbar(['≤100','100-200','200-300','>300'], corr_dis_mean, yerr=corr_dis_ci)
        ax5.set_title('SD CI 95%')

        # SEM and sd
        corr_dis_sem, corr_dis_sd = np.zeros(4), np.zeros(4)
        for i in range(4):
            corr_dis_sem[i] = sem(corr_dis_group[i])
            corr_dis_sd[i] = np.std(corr_dis_group[i])
        ax6.errorbar(['≤100','100-200','200-300','>300'], corr_dis_mean, yerr=corr_dis_sem)
        ax7.errorbar(['≤100','100-200','200-300','>300'], corr_dis_mean, yerr=corr_dis_sd)
        ax6.set_title('SEM')
        ax7.set_title('SD')


        for ax in [ax3, ax4, ax5, ax6, ax7]:
            ax.axhline(y=0.15, color='dimgray',linestyle='dashed', linewidth = 0.5)
            ax.set_xlabel('Distance (um)')
            ax.set_ylabel('Correlation \n (zero-time)')
            ax.set_ylim([-0.5,1])
        plt.tight_layout()
        # fig3.subplots_adjust(top=0.85)

        fig3.savefig(self.directory_path + '\\roi_corr_pair_vs_distance.svg', dpi=300, transparent=True)
        fig3.savefig(self.directory_path + '\\roi_corr_pair_vs_distance.tif', dpi=300, transparent=True)

        return zero_time_cross, lag, correlations, lags_matrix, dis_matrix, dis_pairs, corr_pairs, group_dis, corr_dis_group, corr_dis_mean, corr_dis_ci, corr_dis_sem, corr_dis_sd


    def group_distance_corr(self,dis_pairs,corr_pairs):
        '''
        Returns correlation pairs in nested list (<=100um, 100-200, 200-300, >300um).
        '''
        dis_A, dis_B, dis_C, dis_D = [], [], [], []
        for i, dis in enumerate(dis_pairs):
            if dis <= 100:
                dis_A.append(corr_pairs[i])
            if dis > 100:
                if dis <= 200:
                    dis_B.append(corr_pairs[i])
                if dis > 200:
                    if dis <= 300:
                        dis_C.append(corr_pairs[i])
                    if dis > 300:
                        dis_D.append(corr_pairs[i])
            corr_dis_group = [dis_A, dis_B, dis_C, dis_D] # list
        return corr_dis_group

    def get_matrix_lower_tri(self, X):
        '''
        This function append all lower triangle value of a matrix.
        '''
        Y=[]
        for row_num in range(len(X)):
            for col_num in range(row_num):
                Y.append(X[row_num][col_num])
        return Y


    # def coherence():
    #     scipy.signal.coherence

    # def compute_corr_coef(self, dF_F_smooth): 
    #     '''
    #     This function computes correlation coefficient and covariance.
    #     '''
    #     # Covariance shows you how the two variables differ, 
    #     # whereas correlation shows you how the two variables are related.
    #     # The correlation coefficient is determined by dividing the covariance 
    #     # by the product of the two variables' standard deviations. 
    #     # Standard deviation is a measure of the dispersion of data from its average. 
    #     # Covariance is a measure of how two variables change together.

    #     # compute corrcoef 
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    #     corr_coef = np.corrcoef(dF_F_smooth)
    #     g1 = sns.heatmap(corr_coef, ax=ax1, cmap="viridis", cbar=True, annot=True)
    #     g1.xaxis.set_ticks_position("top")
    #     g1.set_title('correlation coefficient')
    #     # compute covarience 
    #     covar = np.cov(dF_F_smooth)
    #     g2 = sns.heatmap(covar, ax=ax2, cmap="viridis", cbar=True, annot=True)
    #     g2.xaxis.set_ticks_position("top")
    #     g2.set_title('covarience')
    #     fig.suptitle(self.directory_path)
    #     plt.tight_layout()
    #     fig.subplots_adjust(top=0.80)
    #     fig.savefig(self.directory_path + '\\correlation map.svg', dpi=300, transparent=True)
    #     fig.savefig(self.directory_path + '\\correlation map.tif', dpi=300, transparent=True)

    #     return corr_coef, covar

    def compute_response_speed_scatter(self, dF_F_smooth, binned_speed, col=5):
        '''
        This function computes speed vs dF/F for each neuron
        Inputs:
            directory_path: image data storage address, save plots and precessed data to this folder
            dF_F_smooth: dF/F after smoothing with frate_rate
            binned_speed: 1d speed after binning, with the same rate as imaging
            num_roi: number of ROIs
        '''
        fig, ax = plt.subplots(int(np.ceil(self.num_roi/col)),col, figsize=(5*col, int(self.num_roi*5/col)), squeeze=True)
        i = 0
        for cell in range(self.num_roi):
            ax[cell//col,cell%col].scatter(binned_speed, dF_F_smooth[cell], s=0.1, c='black')
            ax[cell//col,cell%col].set_ylabel('dF/F')
            ax[cell//col,cell%col].set_xlabel('Speed (cm/s)')
            ax[cell//col,cell%col].set_title('Cell: {}, ROI: {}'.format(cell + 1, self.ind_cells[0][cell]))
            i = i + 1
        fig.suptitle(self.directory_path)
        plt.tight_layout()
        fig.subplots_adjust(top=0.80)
        fig.savefig(self.directory_path + '\\Speed vs response.svg', dpi=300, transparent=True)
        fig.savefig(self.directory_path + '\\Speed vs response.tif', dpi=300, transparent=True)

    def compute_response_abs_speed_scatter(self, dF_F_smooth, binned_speed_abs, col=5):
        '''
        This function computes speed vs dF/F for each neuron
        Inputs:
            directory_path: image data storage address, save plots and precessed data to this folder
            dF_F_smooth: dF/F after smoothing with frate_rate
            binned_speed_abs: 1d speed after binning, with the same rate as imaging, absolute value
            num_roi: number of ROIs
        '''
        fig, ax = plt.subplots(int(np.ceil(self.num_roi/col)), col, figsize=(5*col, int(self.num_roi*5/col)), squeeze=True)
        i = 0
        for cell in range(self.num_roi):
            ax[cell//col,cell%col].scatter(binned_speed_abs, dF_F_smooth[cell], s=0.1, c='black')
            ax[cell//col,cell%col].set_ylabel('dF/F')
            ax[cell//col,cell%col].set_xlabel('Speed_Absolute (cm/s)')
            ax[cell//col,cell%col].set_title('Cell: {}, ROI: {}'.format(cell+1, self.ind_cells[0][cell]))
            i = i + 1
        fig.suptitle(self.directory_path)
        plt.tight_layout()
        fig.subplots_adjust(top=0.80)
        fig.savefig(self.directory_path + '\\Speed_Absolte vs response.svg', dpi=300, transparent=True)
        fig.savefig(self.directory_path + '\\Speed_Absolte vs response.tif', dpi=300, transparent=True)

