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
        "timm/efficientnet_b4.ra2_in1k",
        pretrained=True,
        # dynamic_img_size=True,
    )  # Load the model.

    # if isinstance(getattr(model, "global_pool", None), str):
    #     model.global_pool = nn.AdaptiveAvgPool2d(1)  # giờ _pool sẽ callable
    lightly_train.train(
        out="outputs/ssl_distil/efficientnet",  # Output directory.
        data="data/12endo/train",  # Directory with images.
        model="timm/efficientnet_b4.ra2_in1k",  # Pass theH", "d TIMM model.
        method="distillation",  # Use DINO method.
        epochs=300,
        batch_size=128,
        transform_args={
            "image_size": (224, 224),
            # "local_view": {"num_views": 0},  # <-- TẮT LOCAL CROPS
        },
        method_args={
            "teacher": "dinov3/vitb16",
            # Replace with your own url
            "teacher_url": 'https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoicGhjdzV5anNzeG96OG1vdHdqenc5dGxqIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU4MzYxODh9fX1dfQ__&Signature=QqMep2UH3Bgdo3tWYNX7hA9jKn4eqZkc-C1HgjFPWz9%7EgxB14IhnGZpcS-SynMPjn1R4gRX-YngpOJIW46FKvcdjKE38v7Tp2CDbm1DKW8G3ykjper3XkXKTtjnpQFlQnftSV8bNSeNrjHc1aCK1j2m4hy9jN0w1Y5vvcnLXNhvGz%7E4E43ce0hCMY-cfJC%7EYXTawGrWnAQto9oThOxyQwJaQfMUE5XY7WWRCHlN-0v2hMbmR5EleGwdWkHpM3kH6SqF6SyLGjxjSv2XF1L3x2jrulAhS%7EbdLH-Ef7s3-tMyI0VLTbPyqEpDEaO9L1%7EePDYQZwnLxJZsBCvtRU8sQpQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=752086624458100',  # noqa: E501
        },
        loggers={"wandb": {"project": "ent-endoscopy-ssl"}},
        num_workers=64,
        resume_interrupted=True,
        overwrite=True,  # Overwrite existing outputs.
    )

    lightly_train.export(
        out="outputs/ssl_distil/efficientnet/efficientnet_distil.pth",
        checkpoint="outputs/ssl_distil/efficientnet/checkpoints/last.ckpt",
        part="model",
        format="torch_state_dict",
    )

    upload_checkpoint(repo_id="coung21/ent", folder_path="outputs")
