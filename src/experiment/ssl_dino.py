import os
import sys

import lightly_train
import timm

# import torch.nn as nn

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # noqa: E402

from utils.upload_ckpt import upload_checkpoint  # noqa: E402

if __name__ == "__main__":
    model = timm.create_model(
        "resnet50",
        pretrained=True,
        # dynamic_img_size=True,
    )  # Load the model.

    # if isinstance(getattr(model, "global_pool", None), str):
    #     model.global_pool = nn.AdaptiveAvgPool2d(1)  # giờ _pool sẽ callable
    lightly_train.train(
        out="outputs/ssl_dino2/resnet50",  # Output directory.
        data="data/12endo/train",  # Directory with images.
        model=model,  # Pass the TIMM model.
        method="dino",  # Use DINO method.
        epochs=300,
        batch_size=64,
        transform_args={
            "image_size": (224, 224),
            # "local_view": {"num_views": 0},  # <-- TẮT LOCAL CROPS
        },
        # loggers={"wandb": {"project": "ent-endoscopy-ssl"}},
        num_workers=64,
        # resume_interrupted=True,
        overwrite=True,  # Overwrite existing outputs.
    )

    lightly_train.export(
        out="outputs/ssl_dino2/resnet50/resnet50_dino.pt",
        checkpoint="outputs/ssl_dino2/resnet50/checkpoints/last.ckpt",
        part="model",
        format="torch_state_dict",
    )

    upload_checkpoint(repo_id="coung21/ent", folder_path="outputs")
