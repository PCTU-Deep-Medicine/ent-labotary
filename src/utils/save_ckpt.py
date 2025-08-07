import os
import shutil


def save_and_push_best_model(
    best_model_path: str,
    output_dir: str = None,
):
    """
    Lưu model tốt nhất vào thư mục /outputs/best_models (đường dẫn tuyệt đối từ project root).

    Args:
        best_model_path (str): Đường dẫn tới file model tốt nhất.
        output_dir (str): Nếu None, mặc định sẽ là '<PROJECT_ROOT>/outputs/best_models'.
    """
    if output_dir is None:
        # Lấy đường dẫn tuyệt đối đến thư mục project gốc
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../..")
        )
        output_dir = os.path.join(
            project_root, "ent-labotary", "outputs", "best_models"
        )

    os.makedirs(output_dir, exist_ok=True)

    dst_path = os.path.join(output_dir, os.path.basename(best_model_path))
    shutil.copy(best_model_path, dst_path)

    print(f"[INFO] Best model copied to: {dst_path}")
