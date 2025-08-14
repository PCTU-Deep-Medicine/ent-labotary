import lightly_train
import timm

if __name__ == "__main__":
    model = timm.create_model(
        "vit_small_patch16_224.augreg_in21k", pretrained=True, dynamic_img_size=True
    )  # Load the model.
    lightly_train.train(
        out="outputs/ssl_dino/vit16s",  # Output directory.
        data="data/kyucapsule",  # Directory with images.
        model=model,  # Pass the TIMM model.
        method="dino",  # Use DINO method.
        epochs=300,
        batch_size=32,
        # model_args={"dynamic_img_size": True},
        transform_args={
            "image_size": (224, 224),
        },
        # loggers={"wandb": {"project": "ent-endoscopy-ssl"}},
        overwrite=True,  # Overwrite existing outputs.
    )

    lightly_train.export(
        out="outputs/ssl_dino/vit16s/vit_small_patch16_224_dino.pt",
        checkpoint="outputs/ssl_dino/vit16s/checkpoints/last.ckpt",
        part="model",
        format="torch_state_dict",
    )
