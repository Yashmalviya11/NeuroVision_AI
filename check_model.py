import timm

model = timm.create_model(
    "vit_small_patch16_224",
    pretrained=False,
    num_classes=4
)

print(model)