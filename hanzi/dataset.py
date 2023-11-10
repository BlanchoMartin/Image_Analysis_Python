import os
import shutil
import random
from pathlib import Path

def delete_half_images(folder_path):
    for root, dirs, files in os.walk(folder_path):
        # Ensure that the list of files is not empty
        if files:
            # Calculate the number of files to keep
            num_files_to_keep = len(files) // 2

            # Randomly select half of the files to delete
            files_to_delete = random.sample(files, len(files) - num_files_to_keep)

            # Delete the selected files
            for file_to_delete in files_to_delete:
                file_path = os.path.join(root, file_to_delete)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

if __name__ == "__main__":
    # Specify the root folder path
    root_folder_path = "./Train"

    # Call the function to delete half of the images
    delete_half_images(root_folder_path)



def keep_numeric_subdirectories(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for directory in dirs[:]:
            if not directory.isdigit() or int(directory) > 9:
                # Remove subdirectory and its contents
                subdirectory_path = os.path.join(root, directory)
                shutil.rmtree(subdirectory_path)
                print(f"Removed: {subdirectory_path}")

if __name__ == "__main__":
    # Specify the root folder path
    root_folder_path = "./Test"

    # Call the function to remove subdirectories
    keep_numeric_subdirectories(root_folder_path)
