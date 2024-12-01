import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import welch, find_peaks, butter, filtfilt

# Load the datasets
t1_b_file_path = '/Users/shantanu/Downloads/ma_lab/T1_B.csv'
t2_a_file_path = '/Users/shantanu/Downloads/ma_lab/T2_A.csv'
t2_b_file_path = '/Users/shantanu/Downloads/ma_lab/T2_B.csv'

t1_b_data = pd.read_csv(t1_b_file_path, skiprows=3)
t2_a_data = pd.read_csv(t2_a_file_path, skiprows=3)
t2_b_data = pd.read_csv(t2_b_file_path, skiprows=3)

# Convert the 'Date' column to datetime format
t1_b_data['Date'] = pd.to_datetime(t1_b_data['Date'])
t2_a_data['Date'] = pd.to_datetime(t2_a_data['Date'])
t2_b_data['Date'] = pd.to_datetime(t2_b_data['Date'])

# Combine X, Y, Z into a single magnitude
t1_b_data['Magnitude'] = (t1_b_data['X (mg)']**2 + t1_b_data['Y (mg)']**2 + t1_b_data['Z (mg)']**2)**0.5
t2_a_data['Magnitude'] = (t2_a_data['X (mg)']**2 + t2_a_data['Y (mg)']**2 + t2_a_data['Z (mg)']**2)**0.5
t2_b_data['Magnitude'] = (t2_b_data['X (mg)']**2 + t2_b_data['Y (mg)']**2 + t2_b_data['Z (mg)']**2)**0.5

# Function to compute PSD
def compute_psd(data: pd.Series, fs: int = 1000):
    f, Pxx = welch(data, fs, nperseg=1024)
    return f, Pxx

# Function to find peaks in PSD
def find_psd_peaks(frequencies: np.ndarray, psd_values: np.ndarray, height: float = None, distance: int = 10):
    peaks, properties = find_peaks(psd_values, height=height, distance=distance)
    peak_frequencies = frequencies[peaks]
    peak_heights = properties['peak_heights']
    return peak_frequencies, peak_heights

# Function to filter the magnitude vector based on frequency ranges
def bandpass_filter(data: pd.Series, lowcut: float, highcut: float, fs: int = 1000, order: int = 4):
    nyquist = 0.5 * fs
    # Normalize the frequencies by the Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Check if the normalized frequencies are valid
    if low <= 0 or high >= 1 or low >= high:
        # Return original data if the frequencies are invalid
        print(f"Skipping filter: invalid frequency range ({lowcut}, {highcut})")
        return data

    # Design bandpass filter
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Compute PSD for each dataset
psd_results = {
    'T1_B': compute_psd(t1_b_data['Magnitude']),
    'T2_A': compute_psd(t2_a_data['Magnitude']),
    'T2_B': compute_psd(t2_b_data['Magnitude'])
}

# Find peaks in the PSD plots
psd_peaks = {}
for key in psd_results:
    f, Pxx = psd_results[key]
    peak_frequencies, peak_heights = find_psd_peaks(f, Pxx, height=1e-4)
    psd_peaks[key] = (peak_frequencies, peak_heights)

# Filter the magnitude vectors based on the peak frequencies
def filter_data_based_on_peaks(data: pd.DataFrame, peak_frequencies: np.ndarray, fs: int = 1000):
    filtered_signals = []
    for peak_freq in peak_frequencies:
        # Define a narrow band around each peak frequency for filtering
        lowcut = peak_freq - 1  # 1 Hz below the peak
        highcut = peak_freq + 1  # 1 Hz above the peak
        filtered_signal = bandpass_filter(data['Magnitude'], lowcut, highcut, fs=fs)
        filtered_signals.append(filtered_signal)
    
    # Sum the filtered signals to reconstruct the overall filtered signal
    return np.sum(filtered_signals, axis=0) if filtered_signals else data['Magnitude']

filtered_magnitudes = {
    'T1_B': filter_data_based_on_peaks(t1_b_data, psd_peaks['T1_B'][0]),
    'T2_A': filter_data_based_on_peaks(t2_a_data, psd_peaks['T2_A'][0]),
    'T2_B': filter_data_based_on_peaks(t2_b_data, psd_peaks['T2_B'][0])
}

# Plot overlayed PSD results with Plotly
def plot_overlayed_psd(psd_results: dict, psd_peaks: dict):
    fig = go.Figure()

    for key in psd_results.keys():
        f, Pxx = psd_results[key]
        fig.add_trace(go.Scatter(x=f, y=Pxx, mode='lines', name=f'{key} PSD'))

        # Add the peaks to the plot
        peak_freqs, peak_heights = psd_peaks[key]
        fig.add_trace(go.Scatter(x=peak_freqs, y=peak_heights, mode='markers', name=f'{key} Peaks'))

    fig.update_layout(
        title='Overlayed PSD Results with Peaks',
        xaxis_title='Frequency [Hz]',
        yaxis_title='PSD [V**2/Hz]',
        yaxis_type='log'
    )

    fig.show()

plot_overlayed_psd(psd_results, psd_peaks)

# Reconstruct the time series and match with timestamps
def reconstruct_time_series_with_timestamps(data: pd.DataFrame, filtered_magnitude: np.ndarray):
    result = data[['Date']].copy()
    result['Filtered Magnitude'] = filtered_magnitude
    return result

# Generate reconstructed time series
reconstructed_time_series = {
    'T1_B': reconstruct_time_series_with_timestamps(t1_b_data, filtered_magnitudes['T1_B']),
    'T2_A': reconstruct_time_series_with_timestamps(t2_a_data, filtered_magnitudes['T2_A']),
    'T2_B': reconstruct_time_series_with_timestamps(t2_b_data, filtered_magnitudes['T2_B'])
}

# Output one of the reconstructed time series
reconstructed_time_series['T1_B'].head()