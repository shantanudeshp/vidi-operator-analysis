import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skfda.representation.grid import FDataGrid
from skfda.exploratory.depth.multivariate import SimplicialDepth
from skfda.exploratory.visualization import MagnitudeShapePlot

# Define the paths to the folders containing the CSV files for T1 and T2
folder_path_t1 = '/Users/shantanu/ma_lab/tool/t1_selected'
folder_path_t2 = '/Users/shantanu/ma_lab/tool/t2_selected'

# Function to process files and create FDataGrid
def create_fd_grid(folder_path):
    # List all the CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Load and process each CSV file into a list of magnitudes
    magnitude_vectors = []
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

        if x_col and y_col and z_col:
            # Calculate magnitude as the Euclidean distance
            magnitude = np.sqrt(data[x_col]**2 + data[y_col]**2 + data[z_col]**2)
            magnitude_vectors.append(magnitude.values)

    # Find the minimum length to truncate all vectors to the same length
    min_length = min(len(mag) for mag in magnitude_vectors)
    magnitude_vectors = [mag[:min_length] for mag in magnitude_vectors]

    # Create an FDataGrid object
    fd = FDataGrid(data_matrix=np.array(magnitude_vectors))
    return fd, labels

# Create FDataGrid objects for T1 and T2
fd_t1, labels_t1 = create_fd_grid(folder_path_t1)
fd_t2, labels_t2 = create_fd_grid(folder_path_t2)

# Define colors for handedness
color_map = {'Left Handed': 'blue', 'Right Handed': 'red'}
colors_t1 = [color_map[label] for label in labels_t1]
colors_t2 = [color_map[label] for label in labels_t2]

# Create and plot the MS-Plot for T1
msplot_t1 = MagnitudeShapePlot(fd_t1, multivariate_depth=SimplicialDepth())
msplot_t1.color = 0.3
msplot_t1.outliercol = 0.7
msplot_t1.plot()

# Overlay the scatter points with colors for handedness
for point, color in zip(msplot_t1.points, colors_t1):
    plt.scatter(point[0], point[1], color=color, label='Left Handed' if color == 'blue' else 'Right Handed')

# Remove duplicate labels in the legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), title="Handedness")
plt.title("MS-Plot for T1")
plt.xlabel("Magnitude Outlyingness")
plt.ylabel("Shape Outlyingness")
plt.show()

# Create and plot the MS-Plot for T2
msplot_t2 = MagnitudeShapePlot(fd_t2, multivariate_depth=SimplicialDepth())
msplot_t2.color = 0.3
msplot_t2.outliercol = 0.7
msplot_t2.plot()

# Overlay the scatter points with colors for handedness
for point, color in zip(msplot_t2.points, colors_t2):
    plt.scatter(point[0], point[1], color=color, label='Left Handed' if color == 'blue' else 'Right Handed')

# Remove duplicate labels in the legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), title="Handedness")
plt.title("MS-Plot for T2")
plt.xlabel("Magnitude Outlyingness")
plt.ylabel("Shape Outlyingness")
plt.show()
