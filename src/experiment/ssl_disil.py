import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="outputs/ssl_disil",
        data="data/kyucapsule",
        model="timm/resnet50",
        method="distillation",
        epochs=300,
        batch_size=32,
        transform_args={
            "image_size": (224, 224),
        },
        # method_args={
        #     "teacher": "dinov2/vits14"
        # }
    )

    lightly_train.export(
        out="outputs/ssl_disil/resnet50_distil.pt",
        checkpoint="outputs/ssl_disil/checkpoints/last.ckpt",
        part="model",
        format="torch_state_dict",
    )
