import os
import pandas as pd
import numpy as np
from skfda.representation.grid import FDataGrid

# Define the paths to the folders containing the CSV files for T1 and T2
folder_path_t1 = '/Users/shantanu/Downloads/ma_lab/tool/t1_selected'
folder_path_t2 = '/Users/shantanu/Downloads/ma_lab/tool/t2_selected'

# Function to process files and create FDataGrid with labels
def create_fd_grid(folder_path):
    # List all the CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Load and process each CSV file into a list of magnitudes
    magnitude_vectors = []
    timestamps = []
    labels = []

    for file in csv_files:
        # Determine if the task was left or right handed based on the filename
        if 'L' in file:
            hand_label = 'Left Handed'
        elif 'R' in file:
            hand_label = 'Right Handed'
        else:
            continue

        labels.append(hand_label)
        
        # Load the data and compute the magnitude
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path, skiprows=4)
        
        # Dynamically select the column names for X, Y, Z
        x_col = next((col for col in data.columns if 'X' in col), None)
        y_col = next((col for col in data.columns if 'Y' in col), None)
        z_col = next((col for col in data.columns if 'Z' in col), None)
        time_col = 'Timestamp' if 'Timestamp' in data.columns else 'Date'
        
        if x_col and y_col and z_col and time_col:
            # Calculate magnitude as the Euclidean distance
            magnitude = np.sqrt(data[x_col]**2 + data[y_col]**2 + data[z_col]**2)
            magnitude_vectors.append(magnitude.values)
            
            # Normalize timestamps to the [0, 1] interval
            time_values = pd.to_datetime(data[time_col])
            time_diff = (time_values - time_values.min()).dt.total_seconds()
            normalized_time = time_diff / time_diff.max()
            timestamps.append(normalized_time.values)

    # Find the minimum length to truncate all vectors to the same length
    min_length = min(len(mag) for mag in magnitude_vectors)
    truncated_magnitudes = [mag[:min_length] for mag in magnitude_vectors]
    truncated_times = [time[:min_length] for time in timestamps]

    # Create an FDataGrid object with normalized time points
    fd = FDataGrid(data_matrix=np.array(truncated_magnitudes), grid_points=np.array(truncated_times[0]))
    
    return fd, labels

# Create FDataGrid objects for T1 and T2
fd_t1, labels_t1 = create_fd_grid(folder_path_t1)
fd_t2, labels_t2 = create_fd_grid(folder_path_t2)

# Combine the labels for both data sets for further analysis
labels_combined = ['T1'] * len(labels_t1) + ['T2'] * len(labels_t2)
fd_combined = fd_t1.concatenate(fd_t2)

# Inspect the FDataGrid and the data
print(f"T1 Data: {fd_t1.data_matrix.shape}, Labels: {set(labels_t1)}")
print(f"T2 Data: {fd_t2.data_matrix.shape}, Labels: {set(labels_t2)}")
print(f"Combined Data: {fd_combined.data_matrix.shape}, Combined Labels: {set(labels_combined)}")

# Return the FDataGrid and labels for further visualization or analysis
fd_t1, labels_t1, fd_t2, labels_t2, fd_combined, labels_combined
