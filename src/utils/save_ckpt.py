import os
import shutil


def save_and_push_best_model(
    best_model_path: str,
    output_dir: str = "/content/drive/MyDrive/ent_models/best_models",
):
    """
    Lưu model tốt nhất vào thư mục Google Drive đã mount.
    Args:
        best_model_path (str): Đường dẫn tới file model tốt nhất.
        output_dir (str): Thư mục trong Google Drive để lưu model.
    """
    # Tạo thư mục output trong Google Drive nếu chưa có
    os.makedirs(output_dir, exist_ok=True)

    # Copy file model vào Drive
    dst_path = os.path.join(output_dir, os.path.basename(best_model_path))
    shutil.copy(best_model_path, dst_path)

    print(f"[INFO] Best model copied to Google Drive: {dst_path}")
