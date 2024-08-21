import os
import zipfile

# Specify the folder containing zip files
folder_path = r"RAVDESS/"

# Get a list of all files in the folder
file_list = os.listdir(folder_path)

# Iterate over each file in the folder
for file_name in file_list:
    # Check if the file is a zip file
    if file_name.endswith(".zip"):
        # Create a full file path
        file_path = os.path.join(folder_path, file_name)
        
        # Create a directory to extract the zip file contents
        extract_dir = os.path.join(folder_path, os.path.splitext(file_name)[0])
        os.makedirs(extract_dir, exist_ok=True)
        
        # Extract the contents of the zip file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Delete the zip file
        os.remove(file_path)
