import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import os

from msrcrmejorado import apply_msrcr

class UnderwaterDataset(Dataset):
    def __init__(self, raw_dir, reference_dir, transform=None):
        self.raw_dir = raw_dir
        self.reference_dir = reference_dir
        self.transform = transform
        self.raw_list = os.listdir(raw_dir)
        self.reference_list = os.listdir(reference_dir)

    def __len__(self):
        return len(self.raw_list)

    def __getitem__(self, idx):
        raw_name = os.path.join(self.raw_dir, self.raw_list[idx])
        reference_name = os.path.join(self.reference_dir, self.reference_list[idx])

        raw_image = cv2.imread(raw_name)
        reference_image = cv2.imread(reference_name)

        if raw_image is None or reference_image is None:
            raise FileNotFoundError(f"Image file '{raw_name}' or '{reference_name}' not found.")

        # Procesar imágenes sin convertir a RGB, manteniendo todo en BGR
        he_image = cv2.equalizeHist(cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY))
        he_image = cv2.cvtColor(he_image, cv2.COLOR_GRAY2BGR)  # Mantener BGR

        gc_image = cv2.pow(raw_image / 255.0, 0.5) * 255.0
        gc_image = gc_image.astype(np.uint8)  # Mantener BGR

        wb_image = self.simple_white_balance(raw_image)
        msrcr_image = apply_msrcr(raw_image)  # MSRCR aplicado en BGR

        # Redimensionar las imágenes
        raw_image = cv2.resize(raw_image, (256, 256))
        he_image = cv2.resize(he_image, (256, 256))
        gc_image = cv2.resize(gc_image, (256, 256))
        wb_image = cv2.resize(wb_image, (256, 256))
        msrcr_image = cv2.resize(msrcr_image, (256, 256))
        reference_image = cv2.resize(reference_image, (256, 256))

        # Convertir las imágenes a formato PIL en BGR
        raw_image = Image.fromarray(raw_image)
        he_image = Image.fromarray(he_image)
        gc_image = Image.fromarray(gc_image)
        wb_image = Image.fromarray(wb_image)
        msrcr_image = Image.fromarray(msrcr_image)
        reference_image = Image.fromarray(reference_image)

        if self.transform:
            raw_image = self.transform(raw_image)
            he_image = self.transform(he_image)
            gc_image = self.transform(gc_image)
            wb_image = self.transform(wb_image)
            msrcr_image = self.transform(msrcr_image)
            reference_image = self.transform(reference_image)

        return raw_image, he_image, gc_image, wb_image, msrcr_image, reference_image

    def simple_white_balance(self, img):
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.2)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.2)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result

# Definir las transformaciones que se aplicarán a cada imagen
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Redimensionar la imagen a 256x256 píxeles
    transforms.ToTensor(),  # Convertir la imagen a un tensor
])
