import numpy as np
import cv2
import torch

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class ViTGradCAM:

    # ---- Settings you can tweak ----
    USE_GRADCAM_PLUS_PLUS = False   # set True to try Grad-CAM++ (often cleaner on ViTs)
    TARGET_BLOCK = -2               # try -1 if -2 looks worse
    BLUR_SIGMA = 1.5                # more = smoother map
    # --------------------------------

    def __init__(self, predictor):

        self.predictor = predictor
        self.model = predictor.model
        self.device = predictor.device

        self.model.eval()

        # Make sure gradients can flow
        for p in self.model.parameters():
            p.requires_grad_(True)

        target_layers = [self.model.blocks[self.TARGET_BLOCK].norm1]

        def reshape_transform(tensor):
            # Keep only patch tokens (drops CLS / extra tokens)
            result = tensor[:, -196:, :]
            result = result.reshape(tensor.size(0), 14, 14, tensor.size(2))
            result = result.permute(0, 3, 1, 2)
            return result

        cam_class = GradCAMPlusPlus if self.USE_GRADCAM_PLUS_PLUS else GradCAM

        self.cam = cam_class(
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

        # Force gradients ON, even if the caller used no_grad
        with torch.enable_grad():

            input_tensor.requires_grad_(True)
            self.model.zero_grad()

            outputs = self.model(input_tensor)
            predicted_class = int(torch.argmax(outputs, dim=1).item())

            targets = [ClassifierOutputTarget(predicted_class)]

            grayscale_cam = self.cam(
                input_tensor=input_tensor,
                targets=targets
            )[0]

        grayscale_cam = np.nan_to_num(
            grayscale_cam, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)

        print("GradCAM min/max/mean:",
              grayscale_cam.min(), grayscale_cam.max(), grayscale_cam.mean())

        # Fallback: if CAM is empty, use input-gradient saliency
        if grayscale_cam.max() < 1e-8:
            print("CAM empty -> falling back to input-gradient saliency")
            x = input_tensor.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                out = self.model(x)
                out[0, predicted_class].backward()
            sal = x.grad.abs().max(dim=1)[0][0].cpu().numpy()
            grayscale_cam = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), 3)

        # Smooth first (removes the dotted 14x14 grid look), then upsample
        grayscale_cam = cv2.GaussianBlur(
            grayscale_cam, (0, 0), self.BLUR_SIGMA
        )
        grayscale_cam = cv2.resize(
            grayscale_cam, (224, 224), interpolation=cv2.INTER_CUBIC
        )

        # Mask out the black background so highlights stay on the brain
        gray = np.array(image.convert("L").resize((224, 224))).astype(np.float32) / 255.0
        mask = (gray > 0.05).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (0, 0), 3)
        grayscale_cam = grayscale_cam * mask

        # Normalize to 0-1
        grayscale_cam = np.clip(grayscale_cam, 0, None)
        grayscale_cam = grayscale_cam - grayscale_cam.min()
        grayscale_cam = grayscale_cam / (grayscale_cam.max() + 1e-8)

        # Original MRI
        rgb_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

        # Standalone heatmap
        heatmap = cv2.applyColorMap(
            np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # MRI with Grad-CAM overlay
        highlighted = show_cam_on_image(
            rgb_img, grayscale_cam, use_rgb=True, image_weight=0.5
        )

        return heatmap, highlighted