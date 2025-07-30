import os
import shutil
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from dotenv import load_dotenv

load_dotenv()

def save_and_push_best_model(best_model_path: str, output_dir: str = "outputs/checkpoints/best_model"):
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    if not folder_id:
        raise ValueError("GOOGLE_DRIVE_FOLDER_ID is not set in .env file")

    os.makedirs(output_dir, exist_ok=True)
    dst_path = os.path.join(output_dir, os.path.basename(best_model_path))
    shutil.copy(best_model_path, dst_path)
    print(f"[INFO] Best model saved to: {dst_path}")

    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    file_metadata = {'title': os.path.basename(dst_path), 'parents': [{'id': folder_id}]}
    gfile = drive.CreateFile(file_metadata)
    gfile.SetContentFile(dst_path)
    gfile.Upload()

    print(f"[INFO] Best model uploaded to Google Drive successfully: {gfile['alternateLink']}")
