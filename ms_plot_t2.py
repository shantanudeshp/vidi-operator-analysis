import os
import pandas as pd
import numpy as np
import skfda
import matplotlib.pyplot as plt

# Define the correct path to the folder containing the CSV files for T1
folder_path = '/Users/shantanu/Downloads/ma_lab/tool/t1_selected'  # Adjust this path if needed

# List all the CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Initialize lists to hold the magnitude vectors and labels
magnitude_vectors = []
labels = []

# Process each file
for file in csv_files:
    # Determine if the task was left or right handed based on the filename
    if 'L' in file:
        labels.append('Left Handed')
    elif 'R' in file:
        labels.append('Right Handed')

    # Load the data, skipping the initial metadata rows
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path, skiprows=4)

    # Print out the columns to inspect if needed
    print(f"Processing {file} with columns: {data.columns}")

    # Dynamically determine the correct column names
    possible_x_cols = ['X (mg)', 'X (mGa)']
    possible_y_cols = ['Y (mg)', 'Y (mGa)']
    possible_z_cols = ['Z (mg)', 'Z (mGa)']
    
    x_col = next((col for col in possible_x_cols if col in data.columns), None)
    y_col = next((col for col in possible_y_cols if col in data.columns), None)
    z_col = next((col for col in possible_z_cols if col in data.columns), None)
    
    if not x_col or not y_col or not z_col:
        print(f"Skipping file {file} due to missing required columns.")
        continue
    
    # Compute the magnitude vector
    magnitude = (data[x_col]**2 + data[y_col]**2 + data[z_col]**2)**0.5

    # Append to the list of magnitude vectors
    magnitude_vectors.append(magnitude)

# Find the minimum length of the vectors
min_length = min(len(mag) for mag in magnitude_vectors)

# Truncate each magnitude vector to the minimum length
magnitude_vectors = [mag[:min_length] for mag in magnitude_vectors]

# Stack all magnitudes into a single array
magnitudes = np.stack(magnitude_vectors)

# Create the FDataGrid object needed by scikit-fda
fd = skfda.FDataGrid(data_matrix=magnitudes)

# Compute magnitude and shape manually
magnitude = fd.data_matrix.mean(axis=1)
shape = np.linalg.norm(fd.data_matrix - magnitude[:, None], axis=2).mean(axis=1)

# Plot manually with color-coding
colors = ['red' if label == 'Left Handed' else 'blue' for label in labels]

plt.figure(figsize=(10, 6))
plt.scatter(magnitude, shape, c=colors)

plt.title("Magnitude-Shape Plot (MS Plot) for T1L and T1R")
plt.xlabel('Magnitude')
plt.ylabel('Shape')
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Left Handed'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Right Handed')],
           title="Handedness")
plt.show()
