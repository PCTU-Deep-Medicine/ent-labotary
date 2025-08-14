from huggingface_hub import upload_folder


def upload_checkpoint(repo_id: str, folder_path: str):
    upload_folder(
        repo_id=repo_id,
        folder_path=folder_path,
        repo_type="model",  # thường là "model",
        ignore_patterns=["checkpoints/**"],
    )
