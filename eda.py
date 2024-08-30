import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

# Load the T1_B dataset
t1_b_file_path = '/Users/shantanu/Downloads/ma_lab/T1_B.csv'
t1_b_data = pd.read_csv(t1_b_file_path, skiprows=3)

# Convert the 'Date' column to datetime format
t1_b_data['Date'] = pd.to_datetime(t1_b_data['Date'])

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

# Plot acceleration data for T1_B
plot_acceleration(t1_b_data, 'T1_B Acceleration Data')