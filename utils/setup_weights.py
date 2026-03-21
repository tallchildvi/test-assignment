import os
import zipfile
import gdown

def setup_weights():

    file_id = '11RTiStDJ0UZ_MLY5l2tB2gPDiozTcKvj'
    url = f'https://drive.google.com/uc?id={file_id}'

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    
    output_zip = os.path.join(project_root, 'weights.zip')
    extract_to = project_root

    print("Starting weights setup...")

    if not os.path.exists(output_zip):
        try:
            gdown.download(url, output_zip, quiet=False)
        except Exception as e:
            print(f"Error downloading file: {e}")
            return
    else:
        print("weights.zip already exists.")

    try:
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
    except Exception as e:
        print(f"Error during extraction: {e}")
        return

    if os.path.exists(output_zip):
        os.remove(output_zip)
        print("Temporary ZIP file removed.")

    print("\nWeights is ready")

if __name__ == "__main__":
    setup_weights()