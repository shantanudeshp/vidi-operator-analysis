import os
import pandas as pd
import numpy as np
import skfda
import matplotlib.pyplot as plt

def process_files(folder_path, label_prefix):
    # List all the CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Initialize lists to hold the magnitude vectors and labels
    magnitude_vectors = []
    labels = []

    # Process each file
    for file in csv_files:
        # Determine if the task was left or right handed based on the filename
        if 'L' in file:
            labels.append(f'{label_prefix} Left Handed')
        elif 'R' in file:
            labels.append(f'{label_prefix} Right Handed')

        # Load the data, skipping the initial metadata rows
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path, skiprows=4)

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

    return magnitude_vectors, labels

def process_and_plot_all(folder_path_t1, folder_path_t2, title):
    # Process files for T1 and T2
    t1_magnitudes, t1_labels = process_files(folder_path_t1, 'T1')
    t2_magnitudes, t2_labels = process_files(folder_path_t2, 'T2')

    # Combine magnitudes and labels
    all_magnitudes = t1_magnitudes + t2_magnitudes
    all_labels = t1_labels + t2_labels

    # Find the minimum length of the vectors
    min_length = min(len(mag) for mag in all_magnitudes)

    # Truncate each magnitude vector to the minimum length
    all_magnitudes = [mag[:min_length] for mag in all_magnitudes]

    # Stack all magnitudes into a single array
    magnitudes = np.stack(all_magnitudes)

    # Create the FDataGrid object needed by scikit-fda
    fd = skfda.FDataGrid(data_matrix=magnitudes)

    # Compute magnitude and shape manually
    magnitude = fd.data_matrix.mean(axis=1)
    shape = np.linalg.norm(fd.data_matrix - magnitude[:, None], axis=2).mean(axis=1)

    # Define color coding based on labels
    colors = ['red' if 'Left' in label else 'blue' for label in all_labels]
    markers = ['o' if 'T1' in label else 'x' for label in all_labels]

    # Plot all data
    plt.figure(figsize=(10, 6))

    for i, (mag, sh, color, marker, label) in enumerate(zip(magnitude, shape, colors, markers, all_labels)):
        plt.scatter(mag, sh, color=color, marker=marker, label=label if label not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.title(f"Magnitude-Shape Plot (MS Plot) for {title}")
    plt.xlabel('Magnitude')
    plt.ylabel('Shape')

    # Create a custom legend for color and marker types
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Left Handed'),
                        plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='blue', markersize=10, label='Right Handed'),
                        plt.Line2D([0], [0], marker='o', color='red', markersize=10, label='T1'),
                        plt.Line2D([0], [0], marker='x', color='blue', markersize=10, label='T2')],
               title="Task & Handedness")

    plt.show()

# Define the paths to the folders containing the CSV files for T1 and T2
folder_path_t1 = '/Users/shantanu/Downloads/ma_lab/tool/t1_selected'
folder_path_t2 = '/Users/shantanu/Downloads/ma_lab/tool/t2_selected'

# Process and plot for all
process_and_plot_all(folder_path_t1, folder_path_t2, "T1 and T2 - Left and Right Handed")