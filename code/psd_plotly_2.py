import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.fft import fft, ifft, fftfreq

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

# Define function to filter and reconstruct signal
def filter_frequency_band(data: pd.Series, fs: int, band: tuple):
    # Convert the pandas Series to a NumPy array
    data = np.asarray(data)
    
    # Perform FFT
    N = len(data)
    fft_values = fft(data)
    freqs = fftfreq(N, 1/fs)
    
    # Create a mask to zero out frequencies outside the desired band
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    fft_filtered = np.zeros_like(fft_values)
    fft_filtered[band_mask] = fft_values[band_mask]
    
    # Perform inverse FFT to get the time-domain signal from the filtered frequencies
    reconstructed_signal = ifft(fft_filtered)
    
    return np.real(reconstructed_signal)

# Define sampling frequency (fs)
fs = 20  # Adjust this if the sampling rate is different

# Define the frequency bands for each dataset
bands = {
    'T1_B': (1, 5),
    'T2_A': (5, 10),
    'T2_B': (10, 20)
}

# Filter and reconstruct the signals for each dataset
t1_b_filtered = filter_frequency_band(t1_b_data['Magnitude'], fs, bands['T1_B'])
t2_a_filtered = filter_frequency_band(t2_a_data['Magnitude'], fs, bands['T2_A'])
t2_b_filtered = filter_frequency_band(t2_b_data['Magnitude'], fs, bands['T2_B'])

# Function to plot filtered signals using Plotly
def plot_filtered_signal(filtered: np.ndarray, title: str, time_index):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_index, y=filtered, mode='lines', name='Filtered Signal'))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Magnitude',
        legend_title='Signal'
    )
    
    fig.show()

# Plot the results for each dataset
plot_filtered_signal(t1_b_filtered, 'T1_B: Filtered Signal (3.5-4.5 Hz)', t1_b_data['Date'])
plot_filtered_signal(t2_a_filtered, 'T2_A: Filtered Signal (2.5-5 Hz)', t2_a_data['Date'])
plot_filtered_signal(t2_b_filtered, 'T2_B: Filtered Signal (3-6 Hz)', t2_b_data['Date'])
