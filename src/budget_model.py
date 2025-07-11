"""
Smart Budget AI - Modelo de Red Neuronal
Implementación del modelo de deep learning para predicción de presupuestos
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import joblib
import os
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BudgetNeuralNetwork:
    """
    Red neuronal para predicción de presupuestos personalizados
    """
    
    def __init__(self, input_dim=None, hidden_layers=[128, 64, 32], dropout_rate=0.3):
        """
        Inicializa la red neuronal
        
        Args:
            input_dim (int): Dimensión de entrada
            hidden_layers (list): Lista con el número de neuronas por capa oculta
            dropout_rate (float): Tasa de dropout para regularización
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        self.is_trained = False
        
    def build_model(self, input_dim):
        """
        Construye la arquitectura de la red neuronal
        
        Args:
            input_dim (int): Dimensión de entrada
        """
        self.input_dim = input_dim
        
        # Definir la arquitectura
        model = keras.Sequential([
            # Capa de entrada
            layers.Dense(self.hidden_layers[0], 
                        activation='relu', 
                        input_shape=(input_dim,),
                        name='input_layer'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            
            # Capas ocultas
            layers.Dense(self.hidden_layers[1], 
                        activation='relu',
                        name='hidden_layer_1'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            
            layers.Dense(self.hidden_layers[2], 
                        activation='relu',
                        name='hidden_layer_2'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate / 2),
            
            # Capa adicional para mejor capacidad de aprendizaje
            layers.Dense(16, 
                        activation='relu',
                        name='hidden_layer_3'),
            layers.Dropout(0.1),
            
            # Capa de salida
            layers.Dense(1, 
                        activation='linear',
                        name='output_layer')
        ])
        
        # Compilar el modelo
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        self.model = model
        
        logger.info("Modelo construido exitosamente")
        logger.info(f"Arquitectura: {[input_dim] + self.hidden_layers + [1]}")
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, verbose=1):
        """
        Entrena la red neuronal
        
        Args:
            X_train (np.array): Datos de entrenamiento
            y_train (np.array): Etiquetas de entrenamiento
            X_val (np.array): Datos de validación
            y_val (np.array): Etiquetas de validación
            epochs (int): Número de épocas
            batch_size (int): Tamaño del batch
            verbose (int): Nivel de verbosidad
            
        Returns:
            keras.callbacks.History: Historia del entrenamiento
        """
        if self.model is None:
            self.build_model(X_train.shape[1])
        
        logger.info("Iniciando entrenamiento del modelo...")
        
        # Callbacks para mejorar el entrenamiento
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Preparar datos de validación
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Entrenar el modelo
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        logger.info("Entrenamiento completado")
        
        return self.history
    
    def predict(self, X):
        """
        Realiza predicciones con el modelo entrenado
        
        Args:
            X (np.array): Datos de entrada
            
        Returns:
            np.array: Predicciones
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def evaluate(self, X_test, y_test):
        """
        Evalúa el modelo en datos de prueba
        
        Args:
            X_test (np.array): Datos de prueba
            y_test (np.array): Etiquetas verdaderas
            
        Returns:
            dict: Métricas de evaluación
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de evaluar")
        
        # Predicciones
        y_pred = self.predict(X_test)
        
        # Calcular métricas
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # R² score
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2_score': r2
        }
        
        logger.info("Métricas de evaluación:")
        for metric, value in metrics.items():
            logger.info(f"{metric.upper()}: {value:.4f}")
        
        return metrics
    
    def save_model(self, filepath):
        """
        Guarda el modelo entrenado
        
        Args:
            filepath (str): Ruta donde guardar el modelo
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de guardarlo")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar modelo
        self.model.save(filepath)
        
        # Guardar metadatos
        metadata = {
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'is_trained': self.is_trained,
            'save_date': datetime.now().isoformat()
        }
        
        metadata_path = filepath.replace('.h5', '_metadata.joblib')
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Modelo guardado en: {filepath}")
    
    def load_model(self, filepath):
        """
        Carga un modelo previamente entrenado
        
        Args:
            filepath (str): Ruta del modelo guardado
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró el modelo en: {filepath}")
        
        # Cargar modelo
        self.model = keras.models.load_model(filepath)
        
        # Cargar metadatos
        metadata_path = filepath.replace('.h5', '_metadata.joblib')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.input_dim = metadata.get('input_dim')
            self.hidden_layers = metadata.get('hidden_layers', [128, 64, 32])
            self.dropout_rate = metadata.get('dropout_rate', 0.3)
            self.is_trained = metadata.get('is_trained', True)
        
        logger.info(f"Modelo cargado desde: {filepath}")
    
    def get_model_summary(self):
        """
        Obtiene un resumen del modelo
        
        Returns:
            str: Resumen del modelo
        """
        if self.model is None:
            return "Modelo no construido"
        
        return self.model.summary()

class BudgetPredictor:
    """
    Clase wrapper para facilitar el uso del modelo de predicción de presupuestos
    """
    
    def __init__(self, model_path=None):
        """
        Inicializa el predictor
        
        Args:
            model_path (str): Ruta del modelo pre-entrenado
        """
        self.model = BudgetNeuralNetwork()
        self.data_processor = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def train_model(self, data_processor, df, validation_split=0.2):
        """
        Entrena el modelo completo
        
        Args:
            data_processor: Instancia de DataProcessor
            df (pd.DataFrame): Datos de entrenamiento
            validation_split (float): Proporción para validación
        """
        self.data_processor = data_processor
        
        # Preparar datos
        X_train, X_test, y_train, y_test = data_processor.prepare_training_data(df)
        
        # Dividir entrenamiento en train/val
        val_size = int(len(X_train) * validation_split)
        X_val = X_train[:val_size]
        y_val = y_train[:val_size]
        X_train = X_train[val_size:]
        y_train = y_train[val_size:]
        
        # Entrenar modelo
        self.model.train(X_train, y_train, X_val, y_val)
        
        # Evaluar en test
        metrics = self.model.evaluate(X_test, y_test)
        
        return metrics
    
    def predict_budget(self, user_data):
        """
        Predice el presupuesto para un usuario
        
        Args:
            user_data (dict): Datos del usuario
            
        Returns:
            float: Presupuesto recomendado
        """
        if not self.model.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Convertir a DataFrame
        df = pd.DataFrame([user_data])
        
        # Procesar datos
        X = self.data_processor.transform_new_data(df)
        
        # Predecir
        prediction = self.model.predict(X)
        
        return float(prediction[0])
    
    def save_model(self, filepath):
        """Guarda el modelo y el procesador de datos"""
        self.model.save_model(filepath)
        
        # Guardar también el data processor
        processor_path = filepath.replace('.h5', '_processor.joblib')
        joblib.dump(self.data_processor, processor_path)
    
    def load_model(self, filepath):
        """Carga el modelo y el procesador de datos"""
        self.model.load_model(filepath)
        
        # Cargar también el data processor
        processor_path = filepath.replace('.h5', '_processor.joblib')
        if os.path.exists(processor_path):
            self.data_processor = joblib.load(processor_path)

if __name__ == "__main__":
    # Ejemplo de uso
    from data_processor import DataProcessor
    
    # Crear datos de ejemplo
    processor = DataProcessor()
    df = processor.generate_sample_data(1000)
    
    # Crear y entrenar modelo
    predictor = BudgetPredictor()
    metrics = predictor.train_model(processor, df)
    
    print("Métricas del modelo:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Ejemplo de predicción
    user_example = {
        'ingresos_mensuales': 3500,
        'gastos_fijos': 1800,
        'gastos_variables': 900,
        'ahorros_actuales': 8000,
        'edad': 28,
        'dependientes': 1,
        'educacion': 'Universidad',
        'estado_civil': 'Casado',
        'tipo_empleo': 'Tiempo_completo',
        'experiencia_laboral': 5,
        'tiene_deudas': 1,
        'monto_deudas': 5000,
        'score_crediticio': 720,
        'gastos_entretenimiento': 250,
        'gastos_salud': 180,
        'gastos_transporte': 350,
        'categoria_riesgo': 'Medio'
    }
    
    budget_prediction = predictor.predict_budget(user_example)
    print(f"\nPresupuesto recomendado: ${budget_prediction:.2f}")
