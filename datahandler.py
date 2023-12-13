# Description: Data handler for calcium imaging data, speed data, and behavior data.

import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.signal import convolve
from scipy.signal.windows import hann
from scipy.interpolate import CubicSpline


class CalciumDataHandler:
    '''
    A class for processing calcium imaging data from Suite2p output. It is specifically designed
    to handle different types of Gcamp fluorescence markers. Key functionalities include reading
    and processing Suite2p output data, computing standardized dF/F0 values, and generating
    various visualizations.

    Attributes:
        NEUROPIL_FACTORS (dict): Correction factors for different Gcamp types. Keys are Gcamp
                                 types ('6f', '7s', '8m'), and values are the corresponding
                                 neuropil factors.
        FRAME_RATE (int): Imaging frame rate in Hertz.

    Methods:
        __init__(...): Initializes a new instance of the class.
        read_Suite2p(): Reads and processes Suite2p output data for the specified Gcamp type.
        standard_dF_F0(...): Processes raw fluorescence data to compute standardized dF/F0 values.
    '''
    # PARAMETERS
    # r = 0.7 for GCaMP6f (Chen et al. 2013) and GCaMP7s(Dana et al. 2019) 
    # r = 0.8 for jGCaMP8m(Zhang et al. 2023; Pettit et al. 2022)
    NEUROPIL_FACTORS = {'6f': 0.7, '7s': 0.7, '8m': 0.8}
    FRAME_RATE = 30  # Imaging frame rate, Hz

    def __init__(self, directory_path: str, filename: str, 
                 base_depth: float, delta_z: float, 
                 gcamp_type: str, 
                 print_info: bool = True, 
                 file_format: str = 'tif'):
        """
        Initialize the CalciumDataHandler with specific parameters for data processing.

        Parameters:
            directory_path (str): Path to the directory containing Suite2p output.
            filename (str): Name of the Suite2p output folder.
            base_depth (float): Baseline depth in micrometers (um).
            delta_z (float): Depth variation in micrometers (um).
            gcamp_type (str): Type of Gcamp used ('6f', '7s', or '8m').
            print_info (bool): Flag to indicate whether to print processing information. Defaults to True.
            file_format (str): File format for saving plots. Defaults to 'tif'.
        """
        self.validate_gcamp_type(gcamp_type)
        
        self.directory_path = directory_path
        self.filename = filename
        self.base_depth = base_depth
        self.delta_z = delta_z
        self.gcamp_type = gcamp_type
        self.print_info = print_info
        self.file_format = file_format

    def validate_gcamp_type(self, gcamp_type: str):
        if gcamp_type not in self.NEUROPIL_FACTORS:
            raise ValueError(f"Invalid gcamp_type '{gcamp_type}'. Choose from '6f', '7s', or '8m'.")

    # ---------------------------[SUPPORT FUNCTION FOR read_Suite2p]---------------------------------------------------
    def read_Suite2p(self) -> tuple:
        """
        Read and process Suite2p output data for the specified Gcamp type. This method loads the
        Suite2p output, applies neuropil correction, extracts ROI information, and computes the
        fluorescence of cells.

        Parameters:
            directory_path (str): Folder of the image files.
            filename (str): Name of the image file.
            base_depth (float): The depth at the center diagonal of the image, usually negative in um. The diagnal from top-left to bottom-right is the center diagonal.
            delta_z: The change in depth from the center diagonal to the edges (e.g. bottom-left or top-right), in um
            gcamp_type (str): Type of Gcamp ('6f', '7s', or '8m').
            printInfo (bool): Flag to print info or not. Default is True.
            file_format (str): The file format for saving the figure ('svg', 'tif', 'png', etc.).

        Returns:
        tuple: A tuple containing processed data arrays. This includes corrected fluorescence data,
               ROI locations, ROI indexed, number of ROIs, and other relevant information.
        """
        self.neuropil_factor = self.NEUROPIL_FACTORS.get(self.gcamp_type)

        plane0_path = os.path.join(self.directory_path + self.filename, 'suite2p', 'plane0')
        suite2p_output = self.load_suite2p_files(plane0_path)

        # Extract required arrays from suite2p_output dictionary
        # Note: in many data, I didn't use Suite2p for spike deconvolution, so spks may be empty
        self.FwithFneu, self.Fneu, self.iscell, ops, spks, stat, stat_orig = self.extract_arrays(suite2p_output)
        self.ops = ops.tolist()

        if self.print_info:
            self.print_suite2p_info(suite2p_output, spks)
            print("spike deconvolution is not used in all data, so 'spks' maybe empty or miscalculated.")

        # Extract ROI information ['med': (y,x) center of cell]
        self.ind_cells, self.num_roi, self.roi_loc = self.extract_roi_info(stat)

        depths = np.array([self.calculate_cell_depth(y, x, image_size=512) for y, x in self.roi_loc])
        self.roi_loc_with_depths = np.hstack((self.roi_loc, depths[:, np.newaxis]))

        # Compute fluorescence of cells
        self.F = self.compute_fluorescence() # F = F_raw - F_neu * neuropil_factor

        if self.print_info:
            print("shape of 'F': ", self.F.shape)
            self.print_roi_info()

        # Plot projection images
        self.plot_projection_images()

        return self.F, self.FwithFneu, self.Fneu, self.iscell, self.ops, spks, stat, self.roi_loc_with_depths, stat_orig, self.ind_cells, self.num_roi

    def load_suite2p_files(self, plane0_path: str) -> dict:
        """
        Load all output .npy files from the Suite2p plane0 folder.

        Parameters:
            plane0_path (str): Path to the Suite2p plane0 folder.

        Returns:
            dict: Dictionary of loaded .npy files.
        """
        return {filename: np.load(os.path.join(plane0_path, filename), allow_pickle=True)
                for filename in os.listdir(plane0_path) if filename.endswith('.npy')}

    def extract_arrays(self, suite2p_output: dict) -> tuple:
        """
        Extract required arrays from suite2p output dictionary.

        Parameters:
            suite2p_output (dict): Dictionary containing Suite2p output data.

        Returns:
            tuple: Extracted data arrays (F, Fneu, iscell, ops, spks, stat, stat_orig).
        """
        keys = ["F.npy", "Fneu.npy", "iscell.npy", "ops.npy", "spks.npy", "stat.npy", "stat_orig.npy"]
        return tuple(suite2p_output.get(key) for key in keys)

    def compute_fluorescence(self) -> np.ndarray:
        """
        Compute fluorescence data of detected cells.

        Parameters:
            FwithFneu (np.ndarray): Raw fluorescence data.
            Fneu (np.ndarray): Neuropil fluorescence data.
            neuropil_factor (float): Neuropil correction factor.

        Returns:
            np.ndarray: Corrected fluorescence data.
        """
        return self.FwithFneu[self.ind_cells] - self.Fneu[self.ind_cells] * self.neuropil_factor

    def print_suite2p_info(self, suite2p_output: dict, spks: np.ndarray) -> None:
        """
        Print information about the Suite2p output data.

        Parameters:
            suite2p_output (dict): Dictionary containing Suite2p output data.
            self.FwithFneu (np.ndarray), self.Fneu (np.ndarray), self.iscell (np.ndarray), self.ops (dict), spks (np.ndarray): Data arrays from Suite2p output.
        """
        print('Output files:', suite2p_output.keys())
        print("Shape of 'FwithFneu':", self.FwithFneu.shape)
        print("Shape of 'Fneu':", self.Fneu.shape)
        print("Shape of 'iscell':", self.iscell.shape)
        print("Type of 'ops':", type(self.ops))
        print("Shape of 'spks':", spks.shape)

    def extract_roi_info(self, stat: np.ndarray) -> tuple:
        """
        Extract ROI information from iscell and stat arrays.

        Parameters:
            self.iscell (np.ndarray): Array indicating cell presence.
            stat (np.ndarray): Array containing statistical information about cells.

        Returns:
            tuple: Indices of cells, number of ROI, and ROI locations.
        """
        self.ind_cells = np.where(self.iscell[:, 0] == 1)
        self.num_roi = len(self.ind_cells[0])
        self.roi_loc = np.array([stat[roi_ind]['med'] for roi_ind in self.ind_cells[0]])
        return self.ind_cells, self.num_roi, self.roi_loc

    def print_roi_info(self) -> None:
        """
        Print information about the ROI.

        Parameters:
            num_roi (int): Number of ROIs.
            ind_cells (np.ndarray): Indices of cells.
            roi_loc (np.ndarray): Locations of ROIs.
        """
        print("Number of ROI/cells: num_roi =", self.num_roi)
        print("Index of detected cells (ROIs): ind_cells = \n", self.ind_cells[0])
        print("location of detected cells (y,x,depth): 'roi_loc_with_depths' = \n", self.roi_loc_with_depths)

    def calculate_cell_depth(self, y, x, image_size: int = 512) -> float:
        """
        Calculate the cell depth. Left-upper corner is upper than center depth; right-bottom corner is lower than center depth.
        
        Parameters:
            y: y-coordinate of the cell (row index, starting from top-left corner)
            x: x-coordinate of the cell (column index, starting from top-left corner)
            base_depth (float): The depth at the center diagonal of the image, usually negative in um. 
                                The diagnal from top-right to bottom-left is the center diagonal.
            delta_z: The change in depth from the center diagonal to the edges (e.g. bottom-right or top-left), in um
            image_size: The height/width of the square image, here is 512 pixels.
        
        Returns:
            float: The depth of the cell in um.
        """
        # Calculate alpha, the angle between the vertical axis and the line to the top-left corner
        alpha = math.atan2(x, y)
        theta = math.radians(45) - alpha 
        # Calculate L, the distance from the cell to the top-right corner
        L = math.sqrt(x**2 + y**2)    
        # Calculate m, the projection length from the cell onto the axis perpendicular to the diagonal
        m = L * math.cos(theta)
        # Calculate the diagonal_ratio as the normalized depth along the diagonal
        diagonal_ratio = m / (math.sqrt(2) * image_size)
        # Interpolate to find the depth at the cell's position
        self.depth = self.base_depth + self.delta_z - (2 * self.delta_z * diagonal_ratio)
        
        return self.depth

    def plot_projection_images(self):
        """
        Plot projection images from ops data.

        Parameters:
            ops (dict): Dictionary containing ops data.
            roi_loc (np.ndarray): ROI locations.
            ind_cells (np.ndarray): Indices of cells.
            num_roi (int): Number of ROIs.
            directory_path (str): Directory path for saving the figure.
            filename (str): Filename for saving the figure.
            file_format (str): The file format for saving the figure ('svg', 'tif', 'png', 'jpg', 'jpeg' etc.).

        """
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        titles = ["Registered, Max Projection", "Registered, Max Projection with ROI#",
                "Mean registered", "High-pass filtered Mean registered image"]
        keys = ['max_proj', 'max_proj', 'meanImg', 'meanImgE']
        for ax, key, title in zip(axs.ravel(), keys, titles):
            vmin_val, vmax_val = (0, 255) if title != "High-pass filtered Mean registered image" else (None, None)
            ax.imshow(self.ops[key], vmin=vmin_val, vmax=vmax_val, cmap='gray')
            ax.set_title(title)
            ax.axis('off')

        for cell in range(self.num_roi):
            axs[0, 1].text(self.roi_loc[cell, 1] + 5, self.roi_loc[cell, 0], self.ind_cells[0][cell], fontsize=11, color='magenta')

        supported_formats = {'svg', 'tif', 'png', 'jpg', 'jpeg'}
        if self.file_format not in supported_formats:
            raise ValueError(f"Unsupported file format: '{self.file_format}'. Supported formats: {supported_formats}")

        figure_dir = os.path.join(self.directory_path, 'Figure')
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
        fig.savefig(os.path.join(figure_dir, f'{self.filename[1:]}_FOV.{self.file_format}'), dpi=300, transparent=True)
    # ---------------------------[SUPPORT FUNCTION FOR read_Suite2p]---------------------------------------------------

    # ---------------------------[SUPPORT FUNCTION FOR ∆F/F0 calculation]----------------------------------------------
    def moving_percentile(self, data: np.array, window_size: int, percentile: int) -> np.array:
        """
        Calculate moving percentile for 1D array.

        Parameters:
            data (np.array): 1D array.
            window_size (int): Window size.
            percentile (int): Percentile value.
        
        Returns:
            np.array: Moving percentile array.
        """
        # Pad data at the start and end to handle the window size at the edges
        padded_data = np.pad(data, (window_size // 2, window_size - 1 - window_size // 2), mode='reflect')
        result = np.zeros(len(data))
        # Calculate percentile for each window
        for i in range(len(data)):
            window = padded_data[i:i + window_size]
            result[i] = np.percentile(window, percentile)
        return result

    def standard_dF_F0_one_cell(self, data: np.array, window_size: int, percentile: int):
        '''
        Raw traces extracted by Suite2p were further processed as follows: 
        A baseline fluorescence estimate F0 was computed as the 30th percentile in a 150-s moving window. 
        ∆F/F0 traces were computed by subtracting and dividing the raw trace by the baseline. 
        ∆F/F0 traces were standardized by subtracting the median and dividing by the standard deviation.

        Parameters:
            data (np.array): 1D array.
            window_size (int): Window size.
            percentile (int): Percentile value. 

        Returns:
            tuple: Standardized dF/F0, baseline, and dF/F0 for individual cells.
        '''
        baseline = self.moving_percentile(data, window_size, percentile)
        dF_F0 = (data - baseline) / baseline
        standardized_dF_F0 = (dF_F0 - np.median(dF_F0)) / np.std(dF_F0)
        return standardized_dF_F0, baseline, dF_F0

    def standard_dF_F0(self, window_size: int = 150, percentile: int = 30, plot: bool = True) -> tuple:
        '''
        Process all cells with function "standard_dF_F0_one_cell" and plot figure to save.

        Parameters:
            F (np.ndarray): 2D array.
            window_size (int): Window size (in sec), default 150s. 
                [Daniel J Millman, 2020 Elife use 180s for GCaMP6f for exc, SST and VIP; Noah L. Pettit, 2022 Nature use 60s for jRGECO1a for excitatory neurons]
                Tested 60s: not reflecting raw traces well; 120s - 180s all good, thus picked 150s for GCaMP7s and jGCaMP8m.
            percentile (int): Percentile value, default 30. 
            plot (bool): Flag to indicate whether to plot figure or not. Default is True.

        Returns:
            tuple: Standardized dF/F0, baseline, and dF/F0 for all cells.
        '''

        window_size_frame = window_size * self.FRAME_RATE

        baseline, standardized_dF_F0, dF_F0 = np.zeros_like(self.F), np.zeros_like(self.F), np.zeros_like(self.F)
        for i in range(self.num_roi):
            standardized_dF_F0[i], baseline[i], dF_F0[i] = self.standard_dF_F0_one_cell(self.F[i], window_size_frame, percentile)

        if plot:
            # plot figure
            # check F and whether standardized_dF_F0 looks good for each cell
            x_axis = np.arange(self.F.shape[1])/self.FRAME_RATE
            fig, axs = plt.subplots(self.num_roi, 4, figsize=(25, self.num_roi))
            for i in range(self.num_roi):
                axs[i,0].plot(x_axis, self.F[i])
                axs[i,0].set_ylabel(f'#{i} \n ROI {self.ind_cells[0][i]}')
                axs[i,1].plot(x_axis, standardized_dF_F0[i])
                axs[i,2].plot(x_axis, dF_F0[i])
                axs[i,3].plot(x_axis, baseline[i])
            axs[0,0].set_title('raw F')
            axs[0,1].set_title('standardized_dF_F0')
            axs[0,2].set_title('dF_F0')
            axs[0,3].set_title('baseline F0')
            for j in range(4):
                axs[i,j].set_xlabel('Time (s)')
            plt.tight_layout()

            # Construct the full path for the figure directory
            figure_dir = os.path.join(self.directory_path, 'Figure')
            # Check if the directory exists, and create it if it doesn't
            if not os.path.exists(figure_dir):
                os.makedirs(figure_dir)
            fig.savefig(os.path.join(figure_dir, f'{self.filename[1:]}_signal.{self.file_format}'), dpi=300, transparent=True)

        return standardized_dF_F0, baseline, dF_F0
    # ---------------------------[SUPPORT FUNCTION FOR ∆F/F0 calculation]----------------------------------------------

class SpeedDataHandler:
    '''
    A class for processing speed data obtained from ThorSync output. It is designed to bin 
    speed data to match the time resolution of imaging data (typically 30 Hz). This class 
    handles the reading, smoothing, and transformation of speed data to a consistent format 
    for further analysis.

    Attributes:
        FRAME_RATE (int): The imaging frame rate in Hertz, set to 30 Hz.
        DATAPOINTS_PER_FRAME (int): Number of speed datapoints per imaging frame, set to 1000.

    Methods:
        __init__(...): Initializes a new instance of the class with the specified parameters.
        binning_speed(): Bins and transforms the speed data to match the imaging frame rate.
    '''
    # PARAMETERS
    FRAME_RATE = 30  # Imaging frame rate, Hz
    DATAPOINTS_PER_FRAME = 1000 # Speed datapoints per imaging frame

    def __init__(self, directory_path: str, speed_filename: str,  
                add_one: bool = True, file_format: str = 'tif'):
        """
        Initialize the SpeedDataHandler with specific parameters for speed data processing.

        Parameters:
            directory_path (str): Path to the directory containing speed data files.
            speed_filename (str): Name of the speed data file to process.
            add_one (bool): A flag indicating whether to add an additional frame at the start 
                            for datasets with one frame less. Defaults to True.
            file_format (str): File format for saving plots. Defaults to 'tif'.
        """
                 
        self.directory_path = directory_path
        self.speed_filename = speed_filename
        self.add_one = add_one
        self.file_format = file_format

    # ---------------------------FUNCTION FOR extract speed data from wheel--------------------------------------------
    def binning_speed(self) -> np.array:
        """
        Bins the speed data to match the time resolution of the imaging data (30 Hz). The method 
        involves transforming the raw speed data into a smoothed speed measurement in cm/s. An 
        additional frame can be added at the start if necessary.

        Parameters:
            add_one (bool): Optional flag to indicate whether to add an additional frame at the start,
                            defaults to the class instance value.

        Returns:
            np.array: A 1D numpy array representing the binned speed data in cm/s.

        Raises:
            ValueError: If 'add_one' is not a boolean value.
        """
        # Validation
        if not isinstance(self.add_one, bool):
            raise ValueError("add_one must be a boolean value.")
        
        speed_data = np.load(self.directory_path + self.speed_filename) # convoled/smoothed speed data

        # Data Preparation
        speed_data_smooth_data = (np.hstack((speed_data, speed_data[0])) if self.add_one else np.hstack((speed_data)))
        
        # Binning and Transformation
        binned_speed_raw = np.average(speed_data_smooth_data.reshape(-1, int(self.DATAPOINTS_PER_FRAME)), axis=1).flatten()
        binned_speed = (binned_speed_raw-1.6604)* 287.22 - 0.24 # measeured from wheel, speedUnit = 286.3444398115298

        # Output Shape Logging
        print(f"shape of 'binned speed'= {binned_speed.shape}")
        
        fig = plt.figure(figsize=(10, 2))
        plt.plot((np.arange(binned_speed.shape[0])/self.FRAME_RATE), binned_speed)
        plt.xlabel('Time(s)')
        plt.ylabel('Speed (cm/s)')
        plt.tight_layout()

        # Construct the full path for the figure directory
        figure_dir = os.path.join(self.directory_path, 'Figure')
        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
        fig.savefig(os.path.join(figure_dir, f'{self.speed_filename[1:-4]}.{self.file_format}'), dpi=300, transparent=True)

        return binned_speed
    # ---------------------------FUNCTION FOR extract speed data from wheel--------------------------------------------

class BehaviorDataHandler:
    """
    A class for processing behavior data obtained from Facemap output. This class is designed to 
    read and analyze video data, focusing on pupil dynamics and blink detection. It includes 
    functionalities for smoothing data, detecting blinks, interpolating pupil data during blinks,
    and generating plots for visual analysis.

    Attributes:
        VIDEO_FRAMERATE (float): Frame rate of the video data.
        [Other attributes like window_size, velocity_threshold, etc., can be described here.]

    Methods:
        __init__(...): Initializes a new instance of the class.
        read_video(): Reads and processes video data from Facemap output.
        plot_video(): Generates plots based on the processed video data.
    """
    def __init__(self, directory_path: str, 
                 behavior_filename: str, 
                 VIDEO_FRAMERATE: float,
                 time_from_start: float = 2,
                 window_size: int = 19,
                 velocity_threshold: float = -0.15,
                 zero_threshold: float = 0.02,
                 deviation_onset: int = 10,
                 deviation_offset: int = 15,
                 merge_threshold: int = 10,
                 y_lim: list = None,
                 file_format: str = 'tif'):
        """
        Initialize the BehaviorDataHandler with specific parameters for behavior data processing.

        Parameters:
            directory_path (str): Path to the directory containing behavior data files.
            behavior_filename (str): Name of the behavior data file.
            VIDEO_FRAMERATE (float): Frame rate of the video data.
            window_size (int): Size of the window for smoothing operations.
            velocity_threshold (float): Threshold value for detecting velocity changes in blinks.
            zero_threshold (float): Threshold for considering values as zero in blink detection.
            deviation_onset (int): Number of frames before the blink onset to consider in analysis.
            deviation_offset (int): Number of frames after the blink offset to consider.
            merge_threshold (int): Threshold for merging close blinks.
            y_lim (list): Y-axis limits for plotting. Defaults to None.
            file_format (str): File format for saving plots. Defaults to 'tif'.
        """
        self.directory_path = directory_path
        self.behavior_filename = behavior_filename
        self.video_framerate = VIDEO_FRAMERATE
        self.time_from_start = time_from_start

        self.window_size = window_size
        self.velocity_threshold = velocity_threshold
        self.zero_threshold =zero_threshold
        self.deviation_onset = deviation_onset 
        self.deviation_offset = deviation_offset
        self.merge_threshold = merge_threshold
        self.y_lim = y_lim
        self.file_format = file_format
    # ---------------------------FUNCTION FOR extract behavior data from Facemap---------------------------------------
    def read_video(self) -> tuple:
        """
        Read video data from Facemap output. Extract required data, remove the first 2s and the last frame.
        Compute z-scores. Handle blinks in the pupil data.

        Parameters:
            directory_path (str): Folder of the video data file.
            behavior_filename (str): Filename of the video data file.
            video_frame_rate (float): The frame rate of the video.
            include_blink_nan (bool): If True, replace blink frames with NaNs in pupil data. Default is True.
            y_lim (list): The y-axis limits for the velocity plot. Default is None.
            
        Returns:
            tuple: Extracted video data components and computed values. This includes
                motion_SVD: list of motion SVDs - each is nframes x components (500)
                video_running: list of running outputs - each is nframes x 2, for X and Y fourier space motion on each frame
                merged_blinks: list of blink outputs, after merging - each is [#blinks, 2 (onset and offset)]
                timestamps: For each blink, four equally spaced time points are required. t2 is the blink onset; t3 is the blink offset; t1=t2-t3+t2; t4=t3-t2+t3
                blink_zscored: list of blink outputs, after zscored - each is nframes
                blink_zscored_smooth_velocity: velocity of smoothed blink z-score data - each is nframes
                pupil_area_smooth: raw from facemap - each is nframes
                pupil_zscored: list of pupil area_smooth outputs, after zscored - each is nframes
                pupil_nan: list of pupil_zscored outputs, after removing blinks with NaN - each is nframes
                pupil_nan_smooth: smoothed pupil_nan - each is nframes
                pupil_interpolated: list of pupil_nan outputs, after interpolation - each is nframes (Based on these timestamps and the associated pupil sizes (from the original, unsmoothed signal), a cubic-spline fit is generated. The original signal between t2 and t3 is replaced by the cubic spline. Thus, the signal is left unchanged, except for the blink period)
                pupil_interpolated_smooth: Smoothed pupil_interpolated - each is nframes
                pupil_com: center of mass in pixel, substracted from mean - each is nframes x 2 (Y, X)  
                pupil_com_nan (after replacing blink with NaN) - each is nframes x 2
                pupil_com_nan_smooth (smoothed pupil_com_nan) - each is nframes x 2
                pupil_com_interpolated (with cubic-spline interpolation) - each is nframes x 2
                pupil_com_interpolated_smooth(Smoothed pupil_com_interpolated)pupil_com_interpolated: list of pupil_com outputs, after interpolation - each is nframes x 2
    
        Raises:
            FileNotFoundError: If the file is not found.
            ValueError: If the file format is not as expected.
        """
        if not isinstance(self.directory_path + self.behavior_filename, str) or not os.path.exists(self.directory_path + self.behavior_filename):
            raise FileNotFoundError(f"File not found or invalid path: {self.directory_path + self.behavior_filename}")

        try:
            video_data = np.load(self.directory_path + self.behavior_filename, allow_pickle=True).item()
            if not isinstance(video_data, dict):
                raise ValueError("Loaded video data is not in the expected dictionary format.")
        except Exception as e:  # Catching a broader exception range during loading
            raise ValueError(f"Error loading video file: {e}")
        
        # Extract required data, remove the first 2s and the last frame
        motion_SVD = self.remove_artifact(video_data['motSVD'][1:][0]) # first is "multivideo SVD" (empty if not computed)
        video_running = self.remove_artifact(video_data['running'][0])
        blink_data = self.remove_artifact(video_data['blink'][0])
        pupil_area_smooth = self.remove_artifact(video_data['pupil'][0]['area_smooth']) # pupil - dict_keys(['area', 'com'(center-of-mass), 'axdir', 'axlen', 'area_smooth', 'com_smooth'])
        pupil_com = video_data['pupil'][0]['com'] # Y, X
        pupil_com = pupil_com - np.mean(pupil_com[0:20,:], axis=0)
        pupil_com = self.remove_artifact(pupil_com)
        # Compute z-scores
        blink_zscored = zscore(blink_data, ddof=1, nan_policy= 'omit')
        pupil_zscored = zscore(pupil_area_smooth, ddof=1, nan_policy= 'omit') # x/y movement of pupil is not controled because the stimuli is full screen
        self.length_of_data = len(pupil_zscored)



        # Handle blinks in the pupil data
        # self.blinks, self.pupil_area_with_nan, self.pupil_area_interpolated, self.pupil_com_with_nan, self.interpolated_pupil_com = self.handle_blinks()
        [merged_blinks, timestamps, 
        blink_zscored_smooth_velocity, 
        pupil_nan, pupil_nan_smooth, 
        pupil_interpolated, pupil_interpolated_smooth,
        pupil_com_nan, pupil_com_nan_smooth,
        pupil_com_interpolated, pupil_com_interpolated_smooth] = self.interpolarate_pupil(blink_zscored, pupil_zscored, pupil_com)

        self.motion_SVD = motion_SVD
        self.video_running = video_running
        self.merged_blinks = merged_blinks
        self.timestamps = timestamps
        self.blink_zscored = blink_zscored
        self.blink_zscored_smooth_velocity = blink_zscored_smooth_velocity
        self.pupil_area_smooth = pupil_area_smooth
        self.pupil_zscored = pupil_zscored
        self.pupil_nan = pupil_nan
        self.pupil_nan_smooth = pupil_nan_smooth
        self.pupil_interpolated = pupil_interpolated
        self.pupil_interpolated_smooth = pupil_interpolated_smooth
        self.pupil_com = pupil_com
        self.pupil_com_nan = pupil_com_nan
        self.pupil_com_nan_smooth = pupil_com_nan_smooth
        self.pupil_com_interpolated = pupil_com_interpolated
        self.pupil_com_interpolated_smooth = pupil_com_interpolated_smooth

        return (motion_SVD, 
            video_running, 
            merged_blinks, timestamps, 
            blink_zscored, blink_zscored_smooth_velocity,
            pupil_area_smooth, pupil_zscored, 
            pupil_nan, pupil_nan_smooth, 
            pupil_interpolated, pupil_interpolated_smooth,
            pupil_com, 
            pupil_com_nan, pupil_com_nan_smooth, 
            pupil_com_interpolated, pupil_com_interpolated_smooth)

    def interpolarate_pupil(self, blink_data_zscored: np.array, pupil_area_zscored: np.array, pupil_com: np.ndarray) -> tuple: 
        '''
        Interpolates pupil data during blinks using cubic spline interpolation. This method smooths the 
        blink data, detects and merges blinks, computes timestamps for each blink, and nans or interpolates the 
        pupil area and center of mass data for these periods. [Mathôt 2013]

        Parameters:
            blink_data_zscored (np.array): Z-scored blink data.
            pupil_area_zscored (np.array): Z-scored pupil area data.
            pupil_com (np.array): Pupil center of mass data.

        Returns:
            tuple: Contains various processed data elements including:
                - merged_blinks: Merged blink intervals.
                - timestamps: Timestamps related to blink events.
                - blink_zscored_smooth_velocity: Smoothed velocity of z-scored blink data.
                - pupil_nan: Pupil area data with NaNs inserted during blinks.
                - pupil_nan_smooth: Smoothed version of pupil_nan.
                - pupil_interpolated: Pupil area data interpolated during blinks.
                - pupil_interpolated_smooth: Smoothed version of pupil_interpolated.
                - pupil_com_nan: Pupil center of mass data with NaNs during blinks.
                - pupil_com_nan_smooth: Smoothed version of pupil_com_nan.
                - pupil_com_interpolated: Pupil center of mass data interpolated during blinks.
                - pupil_com_interpolated_smooth: Smoothed version of pupil_com_interpolated.

        Process:
            1. Smooth blink data and calculate its velocity.
            2. Detect and merge blinks, adding deviation periods.
            3. Extract timestamps and pupil data for each blink.
            4. Interpolate pupil area and center of mass data during blinks.
            5. Insert NaNs and smoothed data for both pupil area and center of mass during blinks.
        '''

        # get velocity of blinks after smoothing blink_zscored
        blink_zscored_smooth = self.smooth_window(blink_data_zscored, window_type='hanning')
        blink_zscored_smooth_velocity = np.r_[0, np.diff(blink_zscored_smooth)]
        # get blinks, merge the nearby ones and add deviation period -> get onset and offset of blinks
        merged_blinks = self.detect_blinks(blink_zscored_smooth_velocity)
        # get four timestamps on each blink
        timestamps = self.blink_timestamp(merged_blinks)
        # get raw four pupil datapoints for each blink timestamp
        pupil_timestamps = self.extract_pupil_from_timestamp(pupil_area_zscored, timestamps)
        # interpolarate each blink between the blink onset and offset
        pupil_times_interpolation = self.cubic_spline_interpolation(timestamps, pupil_timestamps)

        # replace pupil data with blink to nan or interpolarated pupil data
        pupil_interpolated = np.copy(pupil_area_zscored)
        pupil_nan = np.copy(pupil_area_zscored)
        for i, blink in enumerate(merged_blinks):
            pupil_interpolated[blink[0]:blink[1]] = pupil_times_interpolation[i]
            pupil_nan[blink[0]:blink[1]] = np.nan
        # get smoothed nan or interpolarated pupil data
        pupil_interpolated_smooth = self.smooth_window(pupil_interpolated, window_type='hanning')
        pupil_nan_smooth = self.smooth_window(pupil_nan, window_type='hanning')
            
        # do the same for pupil_com
        # get raw four pupil datapoints for each blink timestamp and interpolarate each blink between the blink onset and offset
        pupil_com_timestamps_Y = self.extract_pupil_from_timestamp(pupil_com[:,0], timestamps)
        pupil_com_times_Y_interpolation = self.cubic_spline_interpolation(timestamps, pupil_com_timestamps_Y)
        pupil_com_timestamps_X = self.extract_pupil_from_timestamp(pupil_com[:,1], timestamps)
        pupil_com_times_X_interpolation = self.cubic_spline_interpolation(timestamps, pupil_com_timestamps_X)

        pupil_com_nan = np.copy(pupil_com)
        pupil_com_interpolated = np.copy(pupil_com)
        for i, blink in enumerate(merged_blinks):
            pupil_com_interpolated[blink[0]:blink[1], 0] = pupil_com_times_Y_interpolation[i]
            pupil_com_interpolated[blink[0]:blink[1], 1] = pupil_com_times_X_interpolation[i]
            pupil_com_nan[blink[0]:blink[1], :] = np.nan
        
        # get smoothed nan or interpolarated pupil data
        pupil_com_interpolated_smooth = np.copy(pupil_com)
        pupil_com_interpolated_smooth[:,0] = self.smooth_window(pupil_com_interpolated[:, 0], window_type='hanning')
        pupil_com_interpolated_smooth[:,1] = self.smooth_window(pupil_com_interpolated[:, 1], window_type='hanning')
            
        pupil_com_nan_smooth = np.copy(pupil_com)
        pupil_com_nan_smooth[:,0] = self.smooth_window(pupil_com_nan[:,0], window_type='hanning')
        pupil_com_nan_smooth[:,1] = self.smooth_window(pupil_com_nan[:,1], window_type='hanning')
            
        return (merged_blinks, timestamps, 
        blink_zscored_smooth_velocity, 
        pupil_nan, pupil_nan_smooth, 
        pupil_interpolated, pupil_interpolated_smooth,
        pupil_com_nan, pupil_com_nan_smooth,
        pupil_com_interpolated, pupil_com_interpolated_smooth)
                   
    # ---------------------------[SUPPORT FUNCTION FOR read_video]---------------------------------------------------
    def remove_artifact(self, signal: np.ndarray) -> np.array:
        '''
        Remove the artifact at the beginning of the recording and the last frame before stop; set those frames to nan.

        Parameters:
            signal (np.ndarray): The signal to remove artifact from.
            time_from_start (float): The time from the start to remove artifact from, in seconds. Default is 2s.
        
        Returns:
            np.array: The signal with artifact replaced with nan.
        '''
        signal[:int(self.video_framerate*self.time_from_start)] = np.nan
        # remove the last frame
        signal[-1] = np.nan
        return signal    
    
    def smooth_window(self, signal: np.array, window_type: str='hanning') -> np.array:
        '''
        Smooth a signal using a window of the specified size.

        Parameters:
            signal (np.array): The signal to smooth.
            window_type (str): The type of window to use. Default is 'hanning'.

        Returns:
            np.array: The smoothed signal.
        '''
        if window_type == 'hanning':
            window = hann(self.window_size)
        else:
            raise ValueError("Window type not supported.")
        return convolve(signal, window/window.sum(), mode='same')

    def detect_blinks(self, data: np.array) -> list:
        """
        Detects and processes blink events in the provided data array. This method identifies 
        individual blinks, adjusts them by adding a deviation period before the onset and after 
        the offset, and merges nearby blinks based on a predefined threshold.

        Parameters:
            data (np.array): The array containing blink data.

        Returns:
            list: A list of tuples, each tuple representing the start and end indices of a 
                merged blink event in the data array.

        Process:
            1. Detect individual blinks based on transitions in the data.
            2. Adjust each detected blink by adding a deviation period at its onset and offset.
            3. Merge blinks that are close to each other based on the 'merge_threshold'.

        Notes:
            - A blink is considered to start when the data changes from non-negative to negative 
            and to end when it goes from positive to non-positive.
            - Blinks that are closer to each other than the 'merge_threshold' are considered 
            as part of a single blink event.
        """
        blinks = []
        in_blink = False

        for i in range(1, len(data)):
            # Check for onset of blink (0 to negative)
            if data[i-1] >= - self.zero_threshold and data[i] < 0 and not in_blink:
                blink_start = i
                in_blink = True

            # Check for offset of blink (positive to 0)
            elif data[i-1] > 0 and data[i] <= self.zero_threshold and in_blink:
                blink_end = i
                in_blink = False

                # Check if the velocity threshold is met
                if any(vel < self.velocity_threshold for vel in data[blink_start:blink_end]):
                    blinks.append((blink_start, blink_end))
        
        # Add a deviation period before the onset and after the offset
        adjusted_blinks = []
        for blink in blinks:
            adjusted_start = max(blink[0] - self.deviation_onset, 0)  # Ensure it doesn't go below 0
            adjusted_end = min(blink[1] + self.deviation_offset, len(data))  # Ensure it doesn't exceed data length
            adjusted_blinks.append((adjusted_start, adjusted_end))    
        
        # Merge nearby blinks
        merged_blinks = []
        i = 0
        while i < len(adjusted_blinks):
            current_blink = adjusted_blinks[i]

            # Check for subsequent blinks to merge
            while i + 1 < len(adjusted_blinks) and adjusted_blinks[i + 1][0] - current_blink[1] <= self.merge_threshold:
                # Extend the current blink to the end of the next blink
                current_blink = (current_blink[0], adjusted_blinks[i + 1][1])
                i += 1  # Move to the next blink for possible further merging
            merged_blinks.append(current_blink)
            # Skip to the blink after the last one included in the merged blink
            i += 1
        return merged_blinks

    def blink_timestamp(self, merged_blinks: list) -> np.array:
        '''
        Modified from the approach in Mathôt 2013. A simple way to reconstruct pupil size during eye blinks.
        Four equally spaced time points are required. t2 is the blink onset; t3 is the blink offset; t1=t2-t3+t2; t4=t3-t2+t3

        Parameters:
            merged_blinks (list): A list of tuples, each tuple representing the start and end indices of a 
                merged blink event in the data array.
        
        Returns:
            np.array: A 2D array of timestamps for each blink, each row representing a blink and each column 
                representing a timestamp.
        '''
        timestamps = np.zeros((len(merged_blinks),4))
        for i, blink in enumerate(merged_blinks):
            t2 = int(blink[0]) # t2 = blink[0]
            t3 = int(blink[1]) # t3 = blink[1]
            timestamps[i, 0] = t2 - 1 * (t3 - t2) # t1 = t2-(t3-t2) in Mathôt 2013
            timestamps[i, 1] = t2
            timestamps[i, 2] = t3
            timestamps[i, 3] = min((t3 - t2) * 1 + t3, self.length_of_data-1) # t4 = (t3-t2)+t3 in Mathôt 2013
        return timestamps

    def extract_pupil_from_timestamp(self, data: np.array, timestamps: np.array) -> np.array:
        """
        Extracts specific data points from an array based on provided timestamps. This method is 
        particularly useful for gathering data at critical points before, during, and after each blink, 
        as identified by the timestamps. It is a key step in preparing data for interpolation during 
        blink events.

        Parameters:
            data (np.array): The array from which data points are to be extracted. This could be any 
                            relevant data series, such as pupil area or center of mass data.
            timestamps (np.array): A numpy array containing the timestamps for each event of interest. 
                                Each row in the array represents an event and contains critical 
                                timestamps.

        Returns:
            np.array: A numpy array where each row contains the data points corresponding to the 
                    timestamps of each event.

        Notes:
            - This method is not specific to pupil data and can be applied to any array where event-based 
            data extraction is needed.
            - Ensures that the extracted data at each timestamp is aligned with the events of interest, 
            which is crucial for subsequent analysis or interpolation.
            - If the first timestamp is NaN, the method replaces it with the first available data point.
        """

        data_timestamps = np.zeros_like(timestamps)
        for i, timestamp in enumerate(timestamps):
            timestamp = [int(value) for value in timestamp]
            data_timestamps[i] = data[timestamp]
        if np.isnan(data_timestamps[0,0]):
            timestamps[0,0] = np.ceil(self.video_framerate * self.time_from_start)
            data_timestamps[0,0] = data[int(timestamps[0,0])]
        if np.isnan(data_timestamps[-1,-1]):
            data_timestamps[-1,-1] = data[int(timestamps[-1,-2])]
        return data_timestamps

    def cubic_spline_interpolation(self, timestamps, data_timestamps):
        """
        Performs cubic spline interpolation on a series of data points associated with specific timestamps. 
        This method is useful for reconstructing a smooth curve through these data points, especially in 
        scenarios where continuous data collection is interrupted, like during blinks in eye-tracking data.

        Parameters:
            timestamps (np.array): A numpy array containing timestamps for each event of interest. 
                                    Each row represents an event and contains critical timestamps.
            data_timestamps (np.array): A numpy array where each row contains data points 
                                        corresponding to the timestamps of each event.

        Returns:
            list: A list of numpy arrays, each containing interpolated data values. 
                  The interpolation is done between the second and third timestamps (t2 and t3) for each event.

        Notes:
            - Cubic Spline interpolation is used for its smoothness and ability to fit a curve through 
            a set of data points effectively.
            - The method is particularly useful in contexts where data continuity needs to be maintained 
            despite interruptions in data collection.
            - This approach is often applied in pupilometry to reconstruct pupil size during periods 
            of occlusion (e.g., blinks).
        """
        
        data_times_interpolation = []
        for i, timestamp in enumerate(timestamps):
            cs = CubicSpline(timestamp, data_timestamps[i])
            # Generate a range for plotting or further analysis
            x0 = int(timestamp[1])
            x1 = int(timestamp[2])
            xnew = np.linspace(x0, x1, x1-x0)
            ynew = cs(xnew)
            data_times_interpolation.append(ynew)

        return data_times_interpolation           
    # ---------------------------[SUPPORT FUNCTION FOR read_video]---------------------------------------------------

    def plot_video(self) -> None:
        """
        Generates a series of plots to visualize different components of the video data, 
        particularly focusing on eye movement dynamics and blink events. This method 
        plots motion SVD data, running data, z-scored blink data, velocity of blink z-score, 
        pupil area data (raw, interpolated, and smoothed), and pupil center of mass data.

        The method highlights blink events on each plot for better correlation between 
        different data components and the blink events.

        Notes:
            - Each subplot represents a different aspect of the eye movement or blink data.
            - Blink events are highlighted in gray on each subplot.
            - The method saves the generated plot as an image file in the specified directory.
        """
        fig, axs = plt.subplots(7, 1, figsize=(10, 8))
        for blink in self.merged_blinks:
            for i in range(7):
                axs[i].axvspan(blink[0]/self.video_framerate, blink[1]/self.video_framerate, color='gray', alpha=0.15)

        x_axis = np.arange(self.length_of_data)/self.video_framerate
        axs[0].plot(x_axis, self.motion_SVD[:,0], color='blue', label='axis 0', alpha=0.5)
        axs[0].plot(x_axis, self.motion_SVD[:,1], color='green', label='axis 1', alpha=0.5)
        axs[0].plot(x_axis, self.motion_SVD[:,2], color='red', label='axis 2', alpha=0.5)
        axs[0].legend()
        axs[0].set_title('Video motion_SVD')
        axs[0].set_ylabel('A.U.')
        
        axs[1].plot(x_axis, self.video_running[:,0], color='blue', label='X', alpha=0.5)
        axs[1].plot(x_axis, self.video_running[:,1], color='green', label='Y', alpha=0.5)
        axs[1].legend()
        axs[1].set_title('Video running')
        axs[1].set_ylabel('A.U.')
        
        axs[2].plot(x_axis, self.blink_zscored, linewidth=0.75)
        axs[2].set_title('blink_zscored')
        axs[2].set_ylabel('z-score')
        for timestamp in self.timestamps:
            for i in range(4):
                axs[2].scatter(timestamp[i]/self.video_framerate, self.blink_zscored[int(timestamp[i])], color='orange', s=0.5)

        axs[3].plot(x_axis, self.blink_zscored_smooth_velocity)
        axs[3].set_title('blink_zscored_smooth_velocity')
        axs[3].set_ylim(self.y_lim)
        axs[3].set_ylabel('velocity of\nz-score')
        for timestamp in self.timestamps:
            for i in range(4):
                axs[3].scatter(timestamp[i]/self.video_framerate, self.blink_zscored_smooth_velocity[int(timestamp[i])], color='orange', s=0.5)
        axs[3].axhline(y=self.velocity_threshold, color='green', linestyle='--', alpha=0.5)

        axs[4].plot(x_axis, self.pupil_zscored, alpha=0.5, linewidth=0.45)
        axs[4].plot(x_axis, self.pupil_interpolated, color='blue', alpha=0.5, linewidth=0.35)
        axs[4].plot(x_axis, self.pupil_interpolated_smooth, color='red', alpha=0.5, linewidth=0.25)
        axs[4].set_title('Pupil area (black), interpolated (blue), smoothed (red)')
        axs[4].set_ylabel('z-score')
        for timestamp in self.timestamps:
            for i in range(4):
                axs[4].scatter(timestamp[i]/self.video_framerate, self.pupil_zscored[int(timestamp[i])], color='orange', s=0.5)

        axs[5].plot(x_axis, self.pupil_com[:,0], alpha=0.5, linewidth=0.45)
        axs[5].plot(x_axis, self.pupil_com_interpolated[:,0], color='blue', alpha=0.5, linewidth=0.35)
        axs[5].plot(x_axis, self.pupil_com_interpolated_smooth[:,0], color='red', alpha=0.5, linewidth=0.25)
        axs[5].set_title('Pupil center of mass, Y (black), interpolated (blue), smoothed (red)')
        axs[5].set_ylabel('Position\n(pixel)')

        axs[6].plot(x_axis, self.pupil_com[:,1], alpha=0.5, linewidth=0.45)
        axs[6].plot(x_axis, self.pupil_com_interpolated[:,1], color='blue', alpha=0.5, linewidth=0.35)
        axs[6].plot(x_axis, self.pupil_com_interpolated_smooth[:,1], color='red', alpha=0.5, linewidth=0.25)
        axs[6].set_title('Pupil center of mass, X (black), interpolated (blue), smoothed (red)')
        axs[6].set_ylabel('Position\n(pixel)')
        
        axs[-1].set_xlabel('Time (s)')

        plt.tight_layout()

        figure_dir = os.path.join(self.directory_path, 'Figure')
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
        fig.savefig(os.path.join(figure_dir, f'{self.behavior_filename[1:-8]}video.{self.file_format}'), dpi=1200, transparent=True)
    # ---------------------------FUNCTION FOR extract behavior data from Facemap---------------------------------------

class StimulusDataHandler:
    """
    A class for processing stimulus data. This class is designed to read stimulus data from stumulus type input
    It includes reading stimulus files and extracting stimulus sequences.

    Attributes:
        stimuli_file_path (str): Path to the directory containing stimulus files.
        stimuli_type (str): Type of the stimulus.
        stimuli_type_doc (str): Description of the stimulus type.
        stimuli_file (str): Name of the stimulus file.
        printInfo (bool): Whether to print information about the stimulus data. Defaults to True.
    
    Methods:
        __init__(...): Initializes a new instance of the class.
        read_StimuliType(...): Reads and processes stimulus.
    """

    def __init__(self, 
                 stimuli_file_path: str, 
                 stimuli_type: str, 
                 printInfo: bool = True):
        
        self.stimuli_file_path = stimuli_file_path
        self.stimuli_type = stimuli_type
        self.printInfo = printInfo
    
    def set_stimuli_info(self):
        """
        Set stimulus information based on the provided stimulus type.

        Returns:
            tuple: stimuli_file (name of the stimulus file), stimuli_type_doc (description of the stimulus type)
        """

        # Handling different stimulus types
        if self.stimuli_type == 'A':
            self.stimuli_type_doc = 'spontaneous, dark'
            self.stimuli_file = ''
        elif self.stimuli_type == 'B':
            self.stimuli_type_doc = 'spontaneous, gray'
            self.stimuli_file = ''
        elif 'C' in self.stimuli_type:
            self.stimuli_type_doc = 'orientation tuning'
            self.stimuli_file = r'\Orientation30_10reps_random' + self.stimuli_type[-1] + '.txt'
        elif 'D' in self.stimuli_type:
            self.stimuli_type_doc = 'orientation and contrast tuning'
            self.stimuli_file = r'\Ori_Contrast_15reps_random' + self.stimuli_type[-1] + '.txt'
        elif 'E' in self.stimuli_type:
            if len(self.stimuli_type) > 1:
                self.stimuli_type_doc = 'repetition, 80% contrast, repeat ' + self.stimuli_type[-1]
            else:
                self.stimuli_type_doc = 'repetition, 80% contrast'
            self.stimuli_file = r'\repetition_0_180_5reps.txt'
        elif 'F' in self.stimuli_type:
            if len(self.stimuli_type) > 1:
                self.stimuli_type_doc = 'repetition, 20% contrast, repeat ' + self.stimuli_type[-1]
            else:
                self.stimuli_type_doc = 'repetition, 20% contrast'
            self.stimuli_file = r'\repetition_0_180_5reps.txt'
        return self.stimuli_file, self.stimuli_type_doc 

    def read_StimuliFile(self):
        """
        Read stimuli information from the provided file.

        Returns:
            tuple: stimuli_seq (all the stimuli in the file), stimuli_list (all stimuli types)
        
        Raises:
            FileNotFoundError: If the stimuli file is not found.
            ValueError: If the file format is not as expected.
        """
        try:
            stimuli_seq = np.loadtxt(self.stimuli_file_path + self.stimuli_file, delimiter='\t')
        except FileNotFoundError:
            raise FileNotFoundError(f"Stimuli file not found: {self.stimuli_file_path + self.stimuli_file}")
        except ValueError:
            raise ValueError(f"Error in reading the stimuli file: {self.stimuli_file_path + self.stimuli_file}")
        stimuli_list = np.unique(stimuli_seq, axis=0)
        return stimuli_seq, stimuli_list

    def read_StimuliType(self):
        """
        Read stimuli information from the provided file.

        Returns:
            tuple: stimuli_seq (all the stimuli in the file), stimuli_list (all stimuli types), 
                stimuli_file (name of the stimulus file), stimuli_type_doc (description of the stimulus type)
        """

        self.stimuli_file, self.stimuli_type_doc = self.set_stimuli_info()

        if self.stimuli_type in ['A', 'B']:
            stimuli_seq, stimuli_list = [],[]
        else:
            stimuli_seq, stimuli_list = self.read_StimuliFile()

        if self.printInfo:
            print("Stimuli Information")
            print("stimuli_type:", self.stimuli_type)
            print("stimuli_type_doc:", self.stimuli_type_doc)
            print("stimuli_file:", self.stimuli_file)
            print("stimuli_seq:", stimuli_seq)
            if self.stimuli_type != 'A' and self.stimuli_type != 'B':
                print("shape of 'stimuli_seq':", stimuli_seq.shape)
            print("stimuli_list:", stimuli_list)
            if self.stimuli_type != 'A' and self.stimuli_type != 'B':
                print("shape of 'stimuli_list':", stimuli_list.shape)
        return stimuli_seq, stimuli_list, self.stimuli_file, self.stimuli_type_doc 