import pandas as pd
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

# Combine X, Y, Z into a single magnitude
t1_b_data['Magnitude'] = (t1_b_data['X (mg)']**2 + t1_b_data['Y (mg)']**2 + t1_b_data['Z (mg)']**2)**0.5
t2_a_data['Magnitude'] = (t2_a_data['X (mg)']**2 + t2_a_data['Y (mg)']**2 + t2_a_data['Z (mg)']**2)**0.5
t2_b_data['Magnitude'] = (t2_b_data['X (mg)']**2 + t2_b_data['Y (mg)']**2 + t2_b_data['Z (mg)']**2)**0.5

# Function to compute PSD
def compute_psd(data, fs=1000):
    f, Pxx = welch(data, fs, nperseg=1024)
    return f, Pxx

# Compute PSD for each dataset
psd_results = {
    'T1_B': compute_psd(t1_b_data['Magnitude']),
    'T2_A': compute_psd(t2_a_data['Magnitude']),
    'T2_B': compute_psd(t2_b_data['Magnitude'])
}

# Plot overlayed PSD results with Plotly
def plot_overlayed_psd(psd_results):
    fig = go.Figure()

    for key in psd_results.keys():
        f, Pxx = psd_results[key]
        fig.add_trace(go.Scatter(x=f, y=Pxx, mode='lines', name=key))

    fig.update_layout(
        title='Overlayed PSD Results',
        xaxis_title='Frequency [Hz]',
        yaxis_title='PSD [V**2/Hz]',
        yaxis_type='log'
    )

    fig.show()

plot_overlayed_psd(psd_results)