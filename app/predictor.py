"""
Módulo de predicción para clasificación de imágenes termográficas.

Este módulo contiene la clase ThermographyPredictor que encapsula
toda la lógica de extracción de características y clasificación.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import joblib
from pathlib import Path
from typing import Tuple, Dict, Optional

from .utils import preprocess_image_from_bytes, IMAGENET_MEAN, IMAGENET_STD


class ThermographyPredictor:
    """
    Predictor para clasificación de imágenes termográficas de mama.

    Utiliza ResNet50 pre-entrenado para extracción de características
    y un clasificador SVM entrenado para la predicción final.
    """

    # Dimensión de características de ResNet50
    FEATURE_DIM = 2048

    # Etiquetas de clase
    CLASS_LABELS = {0: "Saludable", 1: "Anomalía Detectada"}

    def __init__(self, model_path: str):
        """
        Inicializa el predictor cargando el modelo SVM y el extractor de características.

        Args:
            model_path: Ruta al archivo .joblib del modelo SVM entrenado
        """
        self.model_path = Path(model_path)
        self.device = self._get_device()
        self.feature_extractor = self._load_feature_extractor()
        self.classifier = self._load_classifier()

    def _get_device(self) -> torch.device:
        """
        Obtiene el mejor dispositivo disponible para computación.

        Prioridad: MPS (Apple Silicon) > CUDA > CPU
        """
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _load_feature_extractor(self) -> nn.Module:
        """
        Carga ResNet50 pre-entrenado como extractor de características.
        """
        # Cargar ResNet50 con pesos de ImageNet
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        base_model = models.resnet50(weights=weights)

        # Remover la capa FC final para obtener características
        modules = list(base_model.children())[:-1]
        model = nn.Sequential(*modules, nn.Flatten())

        # Mover al dispositivo y poner en modo evaluación
        model = model.to(self.device)
        model.eval()

        return model

    def _load_classifier(self):
        """
        Carga el clasificador SVM entrenado desde archivo.

        El modelo se guarda como un diccionario con las claves:
        - 'model': El pipeline sklearn (StandardScaler + SVM)
        - 'metadata': Información adicional del modelo
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en: {self.model_path}"
            )

        saved_data = joblib.load(self.model_path)

        # El modelo se guarda como diccionario con 'model' y 'metadata'
        if isinstance(saved_data, dict) and 'model' in saved_data:
            return saved_data['model']
        else:
            # Compatibilidad: si se guardó directamente el modelo
            return saved_data

    def _preprocess_for_cnn(self, image: np.ndarray) -> torch.Tensor:
        """
        Prepara la imagen para entrada a la CNN.

        Args:
            image: Imagen preprocesada (H, W, 3)

        Returns:
            Tensor listo para CNN (1, 3, H, W)
        """
        # Expandir dimensión del batch
        image = np.expand_dims(image, axis=0)

        # Convertir de (N, H, W, C) a (N, C, H, W)
        image = np.transpose(image, (0, 3, 1, 2))

        # Convertir a tensor
        tensor = torch.from_numpy(image).float()

        return tensor

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extrae características de una imagen usando ResNet50.

        Args:
            image: Imagen preprocesada y normalizada (H, W, 3)

        Returns:
            Vector de características (2048,)
        """
        # Preparar imagen para CNN
        tensor = self._preprocess_for_cnn(image)
        tensor = tensor.to(self.device)

        # Extraer características
        with torch.no_grad():
            features = self.feature_extractor(tensor)

        # Mover a CPU y convertir a numpy
        features = features.cpu().numpy()

        return features.flatten()

    def predict(self, image_bytes: bytes) -> Dict:
        """
        Realiza la predicción completa sobre una imagen.

        Args:
            image_bytes: Bytes de la imagen a clasificar

        Returns:
            Diccionario con:
                - prediction: 0 (saludable) o 1 (anomalía)
                - label: Etiqueta textual
                - confidence: Confianza de la predicción (0-1)
                - probabilities: Probabilidades para cada clase
                - original_image: Imagen original en RGB
                - clahe_image: Imagen con CLAHE aplicado
        """
        # Preprocesar imagen
        preprocessed, original_rgb, clahe_rgb = preprocess_image_from_bytes(
            image_bytes,
            target_size=(224, 224),
            apply_clahe_norm=True,
            normalize_imagenet=True
        )

        # Extraer características
        features = self.extract_features(preprocessed)
        features = features.reshape(1, -1)

        # Clasificar
        prediction = self.classifier.predict(features)[0]

        # Obtener probabilidades si el clasificador las soporta
        if hasattr(self.classifier, 'predict_proba'):
            probabilities = self.classifier.predict_proba(features)[0]
            confidence = probabilities[prediction]
        else:
            # Si no hay probabilidades, usar la decisión del clasificador
            decision = self.classifier.decision_function(features)[0]
            # Convertir decisión a pseudo-probabilidad usando sigmoide
            prob_positive = 1 / (1 + np.exp(-decision))
            probabilities = np.array([1 - prob_positive, prob_positive])
            confidence = probabilities[prediction]

        return {
            'prediction': int(prediction),
            'label': self.CLASS_LABELS[prediction],
            'confidence': float(confidence),
            'probabilities': {
                'saludable': float(probabilities[0]),
                'anomalia': float(probabilities[1])
            },
            'original_image': original_rgb,
            'clahe_image': clahe_rgb
        }

    def get_model_info(self) -> Dict:
        """
        Obtiene información sobre el modelo cargado.

        Returns:
            Diccionario con información del modelo
        """
        return {
            'extractor': 'ResNet50 (ImageNet)',
            'classifier': 'SVM (RBF Kernel)',
            'feature_dim': self.FEATURE_DIM,
            'device': str(self.device),
            'model_path': str(self.model_path)
        }
