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

# Function to compute Fourier Transform, Amplitude, and Phase
def compute_fft(data, fs=1000):
    N = len(data)
    
    # Convert to numpy array to avoid alignment issues
    data = np.asarray(data)
    
    fft_values = fft(data)
    freqs = fftfreq(N, 1/fs)

    amplitude = np.abs(fft_values)
    phase = np.angle(fft_values)

    return freqs, amplitude, phase

# Function to reconstruct the time-domain signal using Inverse FFT
def reconstruct_signal(amplitude, phase):
    # Combine amplitude and phase to form complex Fourier coefficients
    fft_values = amplitude * np.exp(1j * phase)
    
    # Perform inverse FFT to get back the time-domain signal
    reconstructed_signal = ifft(fft_values)
    
    return np.real(reconstructed_signal)

# Process and plot T1_B
freqs_t1_b, amplitude_t1_b, phase_t1_b = compute_fft(t1_b_data['Magnitude'])
t1_b_data['Reconstructed'] = reconstruct_signal(amplitude_t1_b, phase_t1_b)

fig_t1_b = go.Figure()
fig_t1_b.add_trace(go.Scatter(x=t1_b_data['Date'], y=t1_b_data['Magnitude'], mode='lines', name='Original Magnitude'))
fig_t1_b.add_trace(go.Scatter(x=t1_b_data['Date'], y=t1_b_data['Reconstructed'], mode='lines', name='Reconstructed Signal', line=dict(dash='dash')))
fig_t1_b.update_layout(
    title='Original vs Reconstructed Signal for T1_B',
    xaxis_title='Time',
    yaxis_title='Magnitude',
    legend_title='Signal'
)
fig_t1_b.show()

# Process and plot T2_A
freqs_t2_a, amplitude_t2_a, phase_t2_a = compute_fft(t2_a_data['Magnitude'])
t2_a_data['Reconstructed'] = reconstruct_signal(amplitude_t2_a, phase_t2_a)

fig_t2_a = go.Figure()
fig_t2_a.add_trace(go.Scatter(x=t2_a_data['Date'], y=t2_a_data['Magnitude'], mode='lines', name='Original Magnitude'))
fig_t2_a.add_trace(go.Scatter(x=t2_a_data['Date'], y=t2_a_data['Reconstructed'], mode='lines', name='Reconstructed Signal', line=dict(dash='dash')))
fig_t2_a.update_layout(
    title='Original vs Reconstructed Signal for T2_A',
    xaxis_title='Time',
    yaxis_title='Magnitude',
    legend_title='Signal'
)
fig_t2_a.show()

# Process and plot T2_B
freqs_t2_b, amplitude_t2_b, phase_t2_b = compute_fft(t2_b_data['Magnitude'])
t2_b_data['Reconstructed'] = reconstruct_signal(amplitude_t2_b, phase_t2_b)

fig_t2_b = go.Figure()
fig_t2_b.add_trace(go.Scatter(x=t2_b_data['Date'], y=t2_b_data['Magnitude'], mode='lines', name='Original Magnitude'))
fig_t2_b.add_trace(go.Scatter(x=t2_b_data['Date'], y=t2_b_data['Reconstructed'], mode='lines', name='Reconstructed Signal', line=dict(dash='dash')))
fig_t2_b.update_layout(
    title='Original vs Reconstructed Signal for T2_B',
    xaxis_title='Time',
    yaxis_title='Magnitude',
    legend_title='Signal'
)
fig_t2_b.show()
