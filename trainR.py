import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import UnderwaterDataset
from ImprovedWaterNet import FullAdaptedWaterNet
import os
from torchvision import transforms
import numpy as np
from skimage.color import rgb2lab, deltaE_cie76

# Configuración del dispositivo (CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicialización del modelo y configuración del optimizador
model = FullAdaptedWaterNet().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

criterion = nn.MSELoss()

# Definición de las transformaciones a aplicar a las imágenes
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Carga del conjunto de datos
raw_dir = './data/raw'
reference_dir = './data/reference'

print("Loading dataset...")
train_dataset = UnderwaterDataset(raw_dir=raw_dir, reference_dir=reference_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
print("Dataset loaded.")

# Función para calcular el ΔE
def calculate_delta_e(output, reference):
    output_lab = rgb2lab(output)
    reference_lab = rgb2lab(reference)
    delta_e = deltaE_cie76(output_lab, reference_lab)
    return np.mean(delta_e)

# Ciclo de entrenamiento
num_epochs = 20
total_steps = len(train_loader) * num_epochs

print("Starting training...")
step = 0
for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    for i, (raw_images, he_images, gc_images, wb_images, msrcr_images, reference_images) in enumerate(train_loader):
        raw_images, he_images, gc_images, wb_images, msrcr_images, reference_images = \
            raw_images.to(device), he_images.to(device), gc_images.to(device), wb_images.to(device), msrcr_images.to(device), reference_images.to(device)

        optimizer.zero_grad()  # Reiniciar los gradientes

        # Forward pass: pasar las imágenes a través del modelo
        outputs = model(raw_images, he_images, gc_images, wb_images, msrcr_images)

        # Convertir los tensores a numpy arrays para calcular ΔE
        outputs_np = outputs.detach().cpu().permute(0, 2, 3, 1).numpy()
        reference_np = reference_images.detach().cpu().permute(0, 2, 3, 1).numpy()

        # Calcular la métrica ΔE para cada imagen del lote
        delta_e_batch = []
        for output_img, ref_img in zip(outputs_np, reference_np):
            delta_e = calculate_delta_e(output_img, ref_img)
            delta_e_batch.append(delta_e)
        mean_delta_e = np.mean(delta_e_batch)
        print(f'Delta E: {mean_delta_e:.4f}')

        # Calcular la pérdida
        loss = criterion(outputs, reference_images)
        loss.backward()  # Backward pass: calcular los gradientes
        optimizer.step()  # Actualizar los parámetros del modelo

        step += 1
        progress = (step / total_steps) * 100
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, Progress: {progress:.2f}%')

print("Saving model...")
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, 'model.pth')

torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
