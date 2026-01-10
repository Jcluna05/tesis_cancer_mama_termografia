# Evaluación de Modelos de Redes Neuronales sobre Imágenes Termográficas para la Detección del Cáncer de Mama

## Tesis de Maestría en Inteligencia Artificial Aplicada

Este proyecto implementa un sistema de clasificación automática de imágenes termográficas de mama para la detección temprana del cáncer, utilizando técnicas de transfer learning con redes neuronales convolucionales (CNNs) preentrenadas.

## Descripción del Proyecto

### Problema
Clasificación binaria de imágenes termográficas de mama:
- **Healthy**: Pacientes sanas
- **Sick**: Pacientes con anomalías térmicas (posible indicador de cáncer)

### Metodología
Feature Extraction con CNNs preentrenadas + Clasificador SVM:
- **Modelo 1**: EfficientNet-B0 (extractor) + SVM (clasificador)
- **Modelo 2**: ResNet50 (extractor) + SVM (clasificador)

Este enfoque de transfer learning es especialmente adecuado para datasets pequeños, ya que aprovecha las representaciones aprendidas de ImageNet.

### Dataset
- **Fuente**: DMR-IR del Visual Lab UFF (Universidad Federal Fluminense, Brasil)
- **Total**: 272 imágenes
  - Healthy: 177 imágenes
  - Sick: 95 imágenes

## Estructura del Proyecto

```
tesis_cancer_mama_termografia/
├── data/
│   ├── raw/                    # Dataset original (DMR-IR)
│   └── processed/              # Datos preprocesados
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Preprocesamiento con CLAHE
│   ├── feature_extraction.py   # Extracción de características con CNNs
│   ├── models.py               # Entrenamiento de clasificadores SVM
│   ├── evaluation.py           # Métricas y evaluación
│   └── visualization.py        # Generación de figuras
├── notebooks/
│   └── experimentos.ipynb      # Notebook principal de experimentos
├── results/
│   ├── figures/                # Figuras para la tesis (300 DPI)
│   ├── models/                 # Modelos entrenados (.joblib)
│   ├── metrics/                # Métricas en JSON
│   └── features/               # Features extraídas (.npy)
├── requirements.txt
└── README.md
```

## Requisitos

### Hardware Recomendado
- macOS con Apple Silicon (M1/M2/M3/M4) para aceleración MPS
- O sistema con GPU NVIDIA para aceleración CUDA
- Mínimo 8GB RAM (16GB recomendado)

### Software
- Python 3.10+
- PyTorch 2.0+

### Instalación

1. Clonar el repositorio:
```bash
git clone <url-del-repositorio>
cd tesis_cancer_mama_termografia
```

2. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Verificar instalación:
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"MPS disponible: {torch.backends.mps.is_available()}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
```

## Uso

### Preparar Dataset

1. Obtener el dataset DMR-IR del Visual Lab UFF
2. Organizar las imágenes en la siguiente estructura:
```
data/raw/DMR-IR/
├── healthy/
│   ├── imagen1.png
│   └── ...
└── sick/
    ├── imagen1.png
    └── ...
```

### Ejecutar Experimentos

1. Abrir Jupyter Notebook:
```bash
jupyter notebook notebooks/experimentos.ipynb
```

2. Ejecutar las celdas en orden

### Pipeline de Procesamiento

El notebook ejecuta automáticamente:

1. **Preprocesamiento**:
   - Normalización CLAHE para manejo de diferentes paletas de color
   - Redimensionado a 224x224
   - Normalización ImageNet

2. **División del Dataset**:
   - 70% entrenamiento
   - 15% validación
   - 15% test
   - División estratificada para mantener balance de clases

3. **Data Augmentation** (solo en entrenamiento):
   - Flip horizontal
   - Rotación (±15°)
   - Ajustes de brillo/contraste
   - Ruido gaussiano

4. **Extracción de Características**:
   - EfficientNet-B0: 1,280 features
   - ResNet50: 2,048 features

5. **Entrenamiento de SVM**:
   - GridSearchCV para optimización de hiperparámetros
   - Cross-validation de 5 folds
   - class_weight='balanced' para manejo de desbalance

6. **Evaluación**:
   - Accuracy, Precision, Recall, Specificity, F1-Score, AUC-ROC
   - Matrices de confusión
   - Curvas ROC

## Métricas de Evaluación

| Métrica | Descripción | Importancia Clínica |
|---------|-------------|---------------------|
| Accuracy | Precisión general | Rendimiento global |
| Precision | TP / (TP + FP) | Reducir falsos positivos |
| Recall (Sensitivity) | TP / (TP + FN) | **Crítico**: No perder casos enfermos |
| Specificity | TN / (TN + FP) | Identificar correctamente sanos |
| F1-Score | Media armónica | Balance precision-recall |
| AUC-ROC | Área bajo curva ROC | Capacidad discriminativa |

## Resultados

Los resultados se guardan automáticamente en:

- **Figuras**: `results/figures/` (PNG y PDF, 300 DPI)
- **Modelos**: `results/models/` (.joblib)
- **Métricas**: `results/metrics/comparison_results.json`
- **Features**: `results/features/` (.npy)

### Figuras Generadas

1. `confusion_matrix_efficientnet_b0_svm.png` - Matriz de confusión EfficientNet
2. `confusion_matrix_resnet50_svm.png` - Matriz de confusión ResNet
3. `roc_curves_comparison.png` - Curvas ROC comparativas
4. `metrics_comparison.png` - Comparación de métricas
5. `feature_distribution_*.png` - Visualización t-SNE de features
6. `preprocessing_examples.png` - Ejemplos de preprocesamiento
7. `class_distribution_*.png` - Distribución de clases

## Módulos

### `src/preprocessing.py`
- `preprocess_image()`: Pipeline completo de preprocesamiento
- `apply_clahe()`: Normalización de contraste adaptativa
- `load_dataset()`: Carga del dataset completo
- `get_data_splits()`: División estratificada train/val/test

### `src/feature_extraction.py`
- `get_feature_extractor()`: Carga CNN preentrenada
- `extract_features()`: Extracción de características
- `extract_features_with_augmentation()`: Extracción con augmentation

### `src/models.py`
- `train_svm_classifier()`: Entrenamiento con GridSearchCV
- `create_pipeline()`: Pipeline StandardScaler + SVM
- `save_model()` / `load_model()`: Persistencia de modelos

### `src/evaluation.py`
- `evaluate_model()`: Evaluación completa
- `calculate_metrics()`: Cálculo de todas las métricas
- `compare_models()`: Tabla comparativa
- `get_roc_data()`: Datos para curvas ROC

### `src/visualization.py`
- `plot_confusion_matrix()`: Matriz de confusión
- `plot_roc_curves()`: Curvas ROC
- `plot_metrics_comparison()`: Gráfico de barras
- `plot_feature_distribution()`: Visualización t-SNE/PCA
- `create_latex_table()`: Tabla para LaTeX

## Consideraciones Importantes

### Evitar Data Leakage
- La división del dataset se realiza **ANTES** de cualquier augmentation
- El augmentation solo se aplica al conjunto de entrenamiento
- Las features de validación y test se extraen sin augmentation

### Manejo de Desbalance
- Se utiliza `class_weight='balanced'` en SVM
- División estratificada mantiene proporciones
- F1-Score como métrica de optimización

### Reproducibilidad
- Seed fijo: `random_state=42`
- Features guardadas para reproducir resultados
- Modelos serializados con joblib

## Cita

Si utilizas este código, por favor cita:

```bibtex
@mastersthesis{,
    title={Evaluación de Modelos de Redes Neuronales sobre Imágenes Termográficas para la Detección del Cáncer de Mama},
    author={},
    year={2026},
    school={Universidad}
}
```

## Licencia

Este proyecto es parte de una tesis de maestría. Contactar al autor para permisos de uso.

## Referencias

- Dataset DMR-IR: Visual Lab - Universidade Federal Fluminense
- EfficientNet: Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
- ResNet: He, K., et al. (2016). Deep Residual Learning for Image Recognition
