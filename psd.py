import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch

# Load the datasets
t1_b_file_path = '/Users/shantanu/Downloads/ma_lab/T1_B.csv'
t2_a_file_path = '/Users/shantanu/Downloads/ma_lab/T2_A.csv'
t2_b_file_path = '/Users/shantanu/Downloads/ma_lab/T2_B.csv'

t1_b_data = pd.read_csv(t1_b_file_path, skiprows=3)
t2_a_data = pd.read_csv(t2_a_file_path, skiprows=3)
t2_b_data = pd.read_csv(t2_b_file_path, skiprows=3)

# Display the first few rows of each dataset to understand their structure
print("T1_B Data:")
print(t1_b_data.head())
print("\nT2_A Data:")
print(t2_a_data.head())
print("\nT2_B Data:")
print(t2_b_data.head())

t1_b_data['Date'] = pd.to_datetime(t1_b_data['Date'])
t2_a_data['Date'] = pd.to_datetime(t2_a_data['Date'])
t2_b_data['Date'] = pd.to_datetime(t2_b_data['Date'])

# Function to plot acceleration data
def plot_acceleration(data: pd.DataFrame, title: str) -> None:
    plt.figure(figsize=(14, 7))
    plt.suptitle(title)

    plt.subplot(3, 1, 1)
    plt.plot(data['Date'], data['X (mg)'], label='X (mg)')
    plt.ylabel('X (mg)')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(data['Date'], data['Y (mg)'], label='Y (mg)', color='orange')
    plt.ylabel('Y (mg)')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(data['Date'], data['Z (mg)'], label='Z (mg)', color='green')
    plt.ylabel('Z (mg)')
    plt.legend()

    plt.xlabel('Date')
    plt.show()

# Plot acceleration data for T1_B, T2_A, and T2_B
plot_acceleration(t1_b_data, 'T1_B Acceleration Data')
plot_acceleration(t2_a_data, 'T2_A Acceleration Data')
plot_acceleration(t2_b_data, 'T2_B Acceleration Data')

def compute_psd(data, fs=1000):
    f, Pxx = welch(data, fs, nperseg=1024)
    return f, Pxx

# Compute and plot PSD for each dataset
def plot_psd(data: pd.DataFrame, axis: str, title: str) -> None:
    f, Pxx = compute_psd(data[axis])
    plt.figure(figsize=(10, 6))
    plt.semilogy(f, Pxx)
    plt.title(f'PSD of {title} - {axis}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()

# Example for T1_B X-axis
plot_psd(t1_b_data, 'X (mg)', 'T1_B Acceleration Data')
plot_psd(t2_a_data, 'X (mg)', 'T2_A Acceleration Data')
plot_psd(t2_b_data, 'X (mg)', 'T2_B Acceleration Data')
