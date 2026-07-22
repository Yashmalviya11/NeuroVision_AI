import torch
import timm
from PIL import Image
from torchvision import transforms


class Predictor:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.classes = [
            "MildDemented",
            "ModerateDemented",
            "NonDemented",
            "VeryMildDemented"
        ]

        self.model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=4
        )

        self.model.load_state_dict(
            torch.load(
                "models/best_model.pth",
                map_location=self.device
            )
        )

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

    def predict(self, image):

        image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0)

        tensor = tensor.to(self.device)

        with torch.no_grad():

            output = self.model(tensor)

            probabilities = torch.softmax(output, dim=1)

            confidence, predicted = torch.max(
                probabilities,
                dim=1
            )

        return {
            "prediction": self.classes[predicted.item()],
            "confidence": confidence.item()*100,
            "probabilities": probabilities.squeeze().cpu().numpy()
        }