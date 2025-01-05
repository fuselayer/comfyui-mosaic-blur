import torch
from PIL import Image, ImageFilter
import numpy as np
import cv2

import folder_paths

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def mosaic_blur_pillow(image: Image.Image, block_size: int) -> Image.Image:
    """Applies a mosaic blur to an image using Pillow."""
    # Convert to RGBA only if not already an RGBA image
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    width, height = image.size
    new_image = Image.new("RGBA", (width, height))  # Create new image in RGBA mode

    for x in range(0, width, block_size):
        for y in range(0, height, block_size):
            box = (x, y, x + block_size, y + block_size)
            region = image.crop(box)
            # Resize with NEAREST to avoid creating new colors
            color = region.resize((1, 1), resample=Image.Resampling.NEAREST).getpixel((0, 0))
            new_image.paste(color, box)

    return new_image

def mosaic_blur_cv2(image: np.ndarray, block_size: int) -> np.ndarray:
    """Applies a mosaic blur to an image using OpenCV."""
    (h, w) = image.shape[:2]

    # Handle alpha channel for cv2
    has_alpha = image.shape[2] == 4
    if has_alpha:
        alpha_channel = image[:, :, 3]
        image = image[:, :, :3]

    image = cv2.resize(image, (w // block_size, h // block_size), interpolation=cv2.INTER_NEAREST)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)

    # Restore alpha channel
    if has_alpha:
        image = np.dstack((image, alpha_channel))

    return image

class ImageMosaic:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "method": (["pillow", "cv2"],),
                "block_size": ("INT", {"default": 10, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Image/Blur"
    FUNCTION = "image_mosaic"

    def image_mosaic(self, images, method, block_size):
        mosaic_images = []
        for image in images:
            if method == "pillow":
                img = tensor2pil(image)
                # Convert to RGBA only if the image has an alpha channel
                if img.mode != "RGBA" and image.shape[-1] == 4:
                    img = img.convert("RGBA")
                img = mosaic_blur_pillow(img, block_size)
                img = pil2tensor(img)
            else:  # method == "cv2"
                # Convert to PIL Image, preserving potential alpha channel
                img_pil = tensor2pil(image)
                # Split into RGB and alpha channels
                if img_pil.mode == "RGBA":
                    alpha_channel = img_pil.getchannel('A')
                    img_rgb = img_pil.convert('RGB')
                    img_np = np.array(img_rgb)
                else:
                    img_np = np.array(img_pil)

                img_np = mosaic_blur_cv2(img_np, block_size)

                # Convert back to PIL Image
                if img_pil.mode == "RGBA":
                    img_pil = Image.fromarray(img_np, 'RGB')
                    img_pil.putalpha(alpha_channel)  # Re-attach alpha channel
                else:
                    img_pil = Image.fromarray(img_np)

                img = pil2tensor(img_pil)

            mosaic_images.append(img)

        return (torch.cat(mosaic_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "ImageMosaic": ImageMosaic,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageMosaic": "Image Mosaic",
}