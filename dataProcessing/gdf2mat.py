import os
import scipy.io as sio
from biosig import BioSig

# Set the directory path
directory = '/home/artinx/workspace/COURSE/datasets/BCICIV_2a_gdf/'
output = '/home/artinx/workspace/COURSE/datasets/BCICIV_2a_mat/'
# Get all GDF files in the directory
gdf_files = [f for f in os.listdir(directory) if f.endswith('.gdf')]

# Loop through each GDF file and convert it to .mat
for gdf_file in gdf_files:
    # Construct the full file path
    gdf_file_path = os.path.join(directory, gdf_file)
    
    # Load the GDF file using BioSig
    signal, header = BioSig(gdf_file_path).load()
    
    # Create a dictionary for saving the data to .mat
    mat_data = {
        'signal': signal,
        'header': header
    }
    
    # Create a .mat file path by replacing the .gdf extension with .mat
    mat_file_path = os.path.join(output, gdf_file.replace('.gdf', '.mat'))
    
    # Save the data to a .mat file
    sio.savemat(mat_file_path, mat_data)
    print(f"Converted {gdf_file} to {mat_file_path}")

print("All files have been converted.")