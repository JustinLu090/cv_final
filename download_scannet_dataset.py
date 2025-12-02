import os
import requests
import zipfile
import io

# Define the list of files to download
files_to_download = [
    "http://kaldir.vc.in.tum.de/scannet/v2/tasks/scannet_frames_25k.zip",
    "http://kaldir.vc.in.tum.de/scannet/v2/tasks/scannet_frames_test.zip"
]

# Define the target directory for extraction
target_directory = "./scannet_data"

def download_and_extract(url, extract_path):
    """Downloads a ZIP file from a specified URL and extracts it to the given path."""
    
    # Extract the file name from the URL
    file_name = os.path.basename(url)
    print(f"--- Processing file: {file_name} ---")

    try:
        # 1. Download the file
        print(f"Downloading {file_name}...")
        # Set stream=True to download large files in chunks
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if the HTTP request was successful

        # Use io.BytesIO to treat the downloaded content as an in-memory file object
        # This avoids writing the large ZIP file to disk twice (once for download, once for extraction)
        z = zipfile.ZipFile(io.BytesIO(response.content))
        print("Download complete.")
        
        # 2. Extract the file
        print(f"Extracting {file_name} to {os.path.abspath(extract_path)}...")
        z.extractall(extract_path)
        print("Extraction complete.")

    except requests.exceptions.RequestException as e:
        print(f"Download failed or network error: {e}")
    except zipfile.BadZipFile:
        print(f"Error: {file_name} is not a valid ZIP file.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    print("-" * (len(file_name) + 16) + "\n")


if __name__ == "__main__":
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
        print(f"Created directory: {os.path.abspath(target_directory)}")

    for file_url in files_to_download:
        download_and_extract(file_url, target_directory)

    print("All files processed!")