from huggingface_hub import create_repo, upload_folder


def save_and_push_best_model(
    model_dir: str,
):
    """
    Upload the model to HuggingFace
    """

    # replace .env by "hf_wy............."
    model_name = model_dir.split("/")[-1]

    # FIX: Must setup repo_id before running this script
    repo_id = ""

    create_repo(repo_id, exist_ok=True)
    upload_folder(
        repo_id=repo_id, folder_path=model_dir, commit_message=f"uploading {model_name}"
    )
