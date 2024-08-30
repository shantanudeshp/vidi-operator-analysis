import os
import pandas as pd
import numpy as np
import skfda
import matplotlib.pyplot as plt

# Define the path to the folder containing the CSV files
folder_path = '/Users/shantanu/Downloads/ma_lab/tool/t2_selected'

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

    # Load the data
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path, skiprows=3)

    # Compute the magnitude vector
    magnitude = (data['X (mg)']**2 + data['Y (mg)']**2 + data['Z (mg)']**2)**0.5

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

plt.title("Magnitude-Shape Plot (MS Plot) for T2L and T2R")
plt.xlabel('Magnitude')
plt.ylabel('Shape')
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Left Handed'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Right Handed')],
           title="Handedness")
plt.show()
