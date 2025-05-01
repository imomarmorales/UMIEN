import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from msrcrmejorado import apply_msrcr  # Asegúrate de tener implementado MSRCR
from ImprovedWaterNet import FullAdaptedWaterNet
import time

# Cargar el modelo preentrenado
print("Cargando el modelo...")
model = FullAdaptedWaterNet()  # Asegúrate de cargar el modelo entrenado en tu caso
model.load_state_dict(torch.load("D:/Documentos/CNN/Output/improved_waternetreal9.pth", map_location=torch.device('cpu')))
model.eval()  # Pon el modelo en modo de evaluación
print("Modelo cargado correctamente.")

# Definir las transformaciones que se aplicarán a la imagen
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Cambia el tamaño a lo que espera tu modelo
    transforms.ToTensor(),  # Convertir a tensor
])

# Función para aplicar los filtros y preprocesar la imagen
def preprocess_image_with_filters(image_path):
    print(f"Preprocesando la imagen: {image_path}")
    start_time = time.time()

    # Leer la imagen
    frame = cv2.imread(image_path)

    # Aplicar filtros tal como en el dataset
    he_image = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    he_image = cv2.cvtColor(he_image, cv2.COLOR_GRAY2BGR)

    gc_image = cv2.pow(frame / 255.0, 0.5) * 255.0
    gc_image = gc_image.astype(np.uint8)

    wb_image = simple_white_balance(frame)
    msrcr_image = apply_msrcr(frame)

    # Redimensionar las imágenes
    frame_resized = cv2.resize(frame, (256, 256))
    he_resized = cv2.resize(he_image, (256, 256))
    gc_resized = cv2.resize(gc_image, (256, 256))
    wb_resized = cv2.resize(wb_image, (256, 256))
    msrcr_resized = cv2.resize(msrcr_image, (256, 256))

    # Mostrar el tiempo transcurrido para el preprocesamiento
    preprocess_time = time.time() - start_time
    print(f"Preprocesamiento completado en {preprocess_time:.2f} segundos.")

    # Convertir las imágenes a formato PIL para pasarlas por el modelo
    frame_resized = Image.fromarray(frame_resized)
    he_resized = Image.fromarray(he_resized)
    gc_resized = Image.fromarray(gc_resized)
    wb_resized = Image.fromarray(wb_resized)
    msrcr_resized = Image.fromarray(msrcr_resized)

    return frame_resized, he_resized, gc_resized, wb_resized, msrcr_resized, frame

def simple_white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.2)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.2)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

# Preprocesar una imagen de ejemplo
image_path = 'D:/Documentos/CNN/entradas/6.jpg'  # Ruta a la imagen que quieres procesar
frame_resized, he_resized, gc_resized, wb_resized, msrcr_resized, frame = preprocess_image_with_filters(image_path)

# Aplicar las transformaciones necesarias para pasarlas al modelo
frame_resized = transform(frame_resized).unsqueeze(0)
he_resized = transform(he_resized).unsqueeze(0)
gc_resized = transform(gc_resized).unsqueeze(0)
wb_resized = transform(wb_resized).unsqueeze(0)
msrcr_resized = transform(msrcr_resized).unsqueeze(0)

# Realizar la predicción usando el modelo
print("Realizando inferencia con el modelo...")
start_time = time.time()

with torch.no_grad():
    output_frame = model(frame_resized, he_resized, gc_resized, wb_resized, msrcr_resized)

inference_time = time.time() - start_time
print(f"Inferencia completada en {inference_time:.2f} segundos.")

# Convertir el tensor de salida en una imagen OpenCV
output_frame = output_frame.squeeze().permute(1, 2, 0).numpy()  # Remover el batch y reordenar
output_frame = (output_frame * 255).astype(np.uint8)  # Volver a escala 0-255
output_frame = cv2.resize(output_frame, (frame.shape[1], frame.shape[0]))  # Redimensionar a la original

# Guardar la imagen procesada
output_image_path = "D:/Documentos/CNN/resultados/6s.png"
cv2.imwrite(output_image_path, output_frame)
print(f"Imagen procesada guardada en: {output_image_path}")

# Mostrar la imagen procesada
cv2.imshow('Processed Image', output_frame)
cv2.waitKey(0)  # Espera a que se cierre la ventana
cv2.destroyAllWindows()
