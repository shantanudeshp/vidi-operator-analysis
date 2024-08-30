import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import welch

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

# Plot acceleration data with Plotly
def plot_acceleration(data: pd.DataFrame, title: str) -> None:
    fig = px.line(data, x='Date', y=['X (mg)', 'Y (mg)', 'Z (mg)'], title=title)
    fig.show()

# Plot acceleration data for T1_B, T2_A, and T2_B
plot_acceleration(t1_b_data, 'T1_B Acceleration Data')
plot_acceleration(t2_a_data, 'T2_A Acceleration Data')
plot_acceleration(t2_b_data, 'T2_B Acceleration Data')

# Function to compute PSD
def compute_psd(data, fs=1000):
    f, Pxx = welch(data, fs, nperseg=1024)
    return f, Pxx

# Compute PSD for each axis and dataset
psd_results = {
    'T1_B': {
        'X': compute_psd(t1_b_data['X (mg)']),
        'Y': compute_psd(t1_b_data['Y (mg)']),
        'Z': compute_psd(t1_b_data['Z (mg)'])
    },
    'T2_A': {
        'X': compute_psd(t2_a_data['X (mg)']),
        'Y': compute_psd(t2_a_data['Y (mg)']),
        'Z': compute_psd(t2_a_data['Z (mg)'])
    },
    'T2_B': {
        'X': compute_psd(t2_b_data['X (mg)']),
        'Y': compute_psd(t2_b_data['Y (mg)']),
        'Z': compute_psd(t2_b_data['Z (mg)'])
    }
}

# Plot overlayed PSD results with Plotly
def plot_psd(psd_results, axis):
    fig = go.Figure()

    for key in psd_results.keys():
        f, Pxx = psd_results[key][axis]
        fig.add_trace(go.Scatter(x=f, y=Pxx, mode='lines', name=f'{key} - {axis}'))

    fig.update_layout(
        title=f'Overlayed PSD Results - {axis}',
        xaxis_title='Frequency [Hz]',
        yaxis_title='PSD [V**2/Hz]',
        yaxis_type='log'
    )

    fig.show()

# Plot PSD for X, Y, and Z axes
plot_psd(psd_results, 'X')
plot_psd(psd_results, 'Y')
plot_psd(psd_results, 'Z')