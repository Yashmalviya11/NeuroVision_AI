import numpy as np
import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class ViTGradCAM:

    def __init__(self, predictor):

        self.predictor = predictor
        self.model = predictor.model
        self.device = predictor.device

        target_layers = [self.model.blocks[-1].norm1]

        def reshape_transform(tensor):

            result = tensor[:, 1:, :]

            result = result.reshape(
                tensor.size(0),
                14,
                14,
                tensor.size(2)
            )

            result = result.permute(0, 3, 1, 2)

            return result

        self.cam = GradCAM(
            model=self.model,
            target_layers=target_layers,
            reshape_transform=reshape_transform
        )

    def generate(self, image):

        image = image.convert("RGB")

        input_tensor = (
            self.predictor.transform(image)
            .unsqueeze(0)
            .to(self.device)
        )

        grayscale_cam = self.cam(
            input_tensor=input_tensor
        )[0]

        rgb_img = (
            np.array(image.resize((224, 224)))
            .astype(np.float32)
            / 255.0
        )

        # Grad-CAM Overlay
        visualization = show_cam_on_image(
            rgb_img,
            grayscale_cam,
            use_rgb=True
        )

        # --------------------------
        # Highlight Important Region
        # --------------------------

        heatmap = (grayscale_cam * 255).astype(np.uint8)

        _, thresh = cv2.threshold(
            heatmap,
            180,
            255,
            cv2.THRESH_BINARY
        )

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        highlighted = np.array(
            image.resize((224, 224))
        ).copy()

        cv2.drawContours(
            highlighted,
            contours,
            -1,
            (255, 255, 0),   # Yellow
            2
        )

        return visualization, highlighted