import cv2
import numpy as np


def single_scale_retinex(img, sigma):
    return np.log10(img + 1.0) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma) + 1.0)


def multi_scale_retinex(img, sigmas, weights):
    retinex = np.zeros_like(img)
    for sigma, weight in zip(sigmas, weights):
        retinex += weight * single_scale_retinex(img, sigma)
    return retinex


def color_restoration(img, alpha, g):
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_restored = g * (np.log10(alpha * img + 1) - np.log10(img_sum + 1))
    return color_restored


def apply_adaptive_histogram_equalization(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(img_lab))  # Convertir la tupla a una lista

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])

    img_lab = cv2.merge(lab_planes)
    img_output = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    return img_output


def MSRCR(img, sigmas=[15, 80, 250], weights=[1 / 3, 1 / 3, 1 / 3], alpha=128, g=0.5, dynamic=3):
    img = img.astype(np.float32) / 255.0
    retinex = multi_scale_retinex(img, sigmas, weights)
    color_restored = color_restoration(img, alpha, g)
    msrcr = retinex * color_restored

    # Calcular min y max
    mean = np.mean(msrcr)
    std = np.std(msrcr)
    min_val = mean - dynamic * std
    max_val = mean + dynamic * std

    # Aplicar la función de mapeo
    msrcr = np.clip((msrcr - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)

    # Aplicar ecualización adaptativa del histograma
    final_image = apply_adaptive_histogram_equalization(msrcr)

    return final_image


def apply_msrcr(image, sigmas=[15, 80, 250], weights=[0.1, 0.6, 0.3], alpha=128, beta=1.0, gain=1.0):
    """
    Aplica el filtro MSRCR mejorado a la imagen.

    :param image: Imagen de entrada en formato BGR.
    :param sigmas: Lista de tamaños de escalas para el filtro.
    :param weights: Lista de pesos para cada escala.
    :param alpha: Parámetro de ajuste para el término logarítmico.
    :param beta: Parámetro de ajuste para el término logarítmico.
    :param gain: Ganancia aplicada a la imagen final.
    :return: Imagen procesada con MSRCR.
    """
    # Convertir la imagen a float32
    img = image.astype(np.float32) / 255.0

    # Calcular el log de la imagen
    img_log = np.log1p(img * alpha)

    # Crear la imagen de salida inicial
    img_msrcr = np.zeros_like(img_log)

    # Aplicar el filtro Retinex Multiescala
    for sigma, w in zip(sigmas, weights):
        kernel = cv2.getGaussianKernel(ksize=sigma, sigma=sigma / 3)
        kernel = kernel * kernel.T
        img_blur = cv2.filter2D(img_log, -1, kernel)
        img_msrcr += w * (img_log - img_blur)

    # Restauración del color
    img_msrcr = gain * (img_msrcr - np.log1p(beta * img))
    img_msrcr = np.clip(img_msrcr, 0, 1)

    # Convertir de nuevo a uint8
    img_msrcr = (img_msrcr * 255).astype(np.uint8)

    # Aplicar CLAHE en la imagen final
    img_lab = cv2.cvtColor(img_msrcr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_lab = cv2.merge((l, a, b))
    img_msrcr = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

    return img_msrcr


# Ejemplo de uso
input_image_path = 'D:/Documentos/CNN/pruebas/5.png'
output_image_path = 'D:/Documentos/CNN/salidas/sfi26xd.png'
image = cv2.imread(input_image_path)
image_msrcr = MSRCR(image)
cv2.imwrite(output_image_path, image_msrcr)
