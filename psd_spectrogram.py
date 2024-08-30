import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import spectrogram

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

# Function to compute and plot spectrogram
def plot_spectrogram(data: pd.Series, title: str, fs: int = 1000, nperseg: int = 256) -> None:
    f, t, Sxx = spectrogram(data, fs, nperseg=nperseg)
    fig = go.Figure(data=go.Heatmap(
        z=10 * np.log10(Sxx),  # Convert to dB scale for better visibility
        x=t,
        y=f,
        zmin=-100,  # Adjust these values based on your data
        zmax=0
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Time [s]',
        yaxis_title='Frequency [Hz]',
        yaxis_type='log'
    )
    fig.show()

# Plot spectrogram for each dataset
plot_spectrogram(t1_b_data['Magnitude'], 'Spectrogram of T1_B Magnitude')
plot_spectrogram(t2_a_data['Magnitude'], 'Spectrogram of T2_A Magnitude')
plot_spectrogram(t2_b_data['Magnitude'], 'Spectrogram of T2_B Magnitude')
