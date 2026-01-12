"""
Utilidades para el preprocesamiento de imágenes termográficas.

Este módulo contiene las funciones necesarias para preparar las imágenes
antes de la extracción de características y clasificación.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional
import io


# Constantes de normalización ImageNet (requeridas para transfer learning)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization).

    CLAHE mejora el contraste local sin saturar la imagen, lo cual es
    especialmente útil para imágenes termográficas con diferentes
    paletas de colores.

    Args:
        image: Imagen de entrada (BGR o escala de grises)
        clip_limit: Umbral para limitación de contraste
        tile_grid_size: Tamaño de la cuadrícula para ecualización

    Returns:
        Imagen con CLAHE aplicado
    """
    # Convertir a espacio de color LAB si es imagen a color
    if len(image.shape) == 3 and image.shape[2] == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Aplicar CLAHE al canal L
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_clahe = clahe.apply(l_channel)

        # Combinar canales
        lab_clahe = cv2.merge([l_clahe, a_channel, b_channel])
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    else:
        # Imagen en escala de grises
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        result = clahe.apply(image)

    return result


def preprocess_image_from_bytes(
    image_bytes: bytes,
    target_size: Tuple[int, int] = (224, 224),
    apply_clahe_norm: bool = True,
    normalize_imagenet: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pipeline completo de preprocesamiento para imágenes termográficas.

    Pasos del pipeline:
    1. Cargar imagen desde bytes
    2. Aplicar CLAHE para normalización de contraste
    3. Redimensionar al tamaño objetivo
    4. Convertir a RGB (3 canales)
    5. Normalizar al rango [0, 1]
    6. Opcionalmente aplicar normalización ImageNet

    Args:
        image_bytes: Bytes de la imagen cargada
        target_size: Tamaño de salida (alto, ancho)
        apply_clahe_norm: Si aplicar normalización CLAHE
        normalize_imagenet: Si aplicar normalización media/std de ImageNet

    Returns:
        Tupla de (imagen_preprocesada, imagen_original_rgb, imagen_clahe_rgb)
    """
    # Cargar imagen desde bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("No se pudo cargar la imagen")

    # Guardar imagen original en RGB para visualización
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Aplicar CLAHE
    if apply_clahe_norm:
        image_clahe = apply_clahe(image)
    else:
        image_clahe = image.copy()

    # Guardar imagen con CLAHE en RGB para visualización
    clahe_rgb = cv2.cvtColor(image_clahe, cv2.COLOR_BGR2RGB)

    # Redimensionar al tamaño objetivo
    image_resized = cv2.resize(image_clahe, target_size, interpolation=cv2.INTER_LINEAR)

    # Convertir BGR a RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # Normalizar a [0, 1]
    image_normalized = image_rgb.astype(np.float32) / 255.0

    # Aplicar normalización ImageNet si se solicita
    if normalize_imagenet:
        image_normalized = (image_normalized - IMAGENET_MEAN) / IMAGENET_STD

    return image_normalized, original_rgb, clahe_rgb


def denormalize_for_display(image: np.ndarray) -> np.ndarray:
    """
    Desnormaliza una imagen para visualización.

    Args:
        image: Imagen normalizada con ImageNet

    Returns:
        Imagen en rango [0, 255] para visualización
    """
    # Revertir normalización ImageNet
    image = image * IMAGENET_STD + IMAGENET_MEAN

    # Clip y convertir a uint8
    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    return image
