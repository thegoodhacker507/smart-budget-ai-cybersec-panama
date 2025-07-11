"""
Smart Budget AI - Procesador de Datos
Módulo para el procesamiento y feature engineering de datos financieros
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Clase para procesar datos financieros y preparar features para el modelo
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        
    def generate_sample_data(self, n_samples=1000):
        """
        Genera datos de muestra para entrenamiento del modelo
        
        Args:
            n_samples (int): Número de muestras a generar
            
        Returns:
            pd.DataFrame: DataFrame con datos financieros sintéticos
        """
        logger.info(f"Generando {n_samples} muestras de datos sintéticos...")
        
        np.random.seed(42)
        
        # Generar datos base
        data = {
            'ingresos_mensuales': np.random.normal(3000, 1000, n_samples),
            'gastos_fijos': np.random.normal(1500, 500, n_samples),
            'gastos_variables': np.random.normal(800, 300, n_samples),
            'ahorros_actuales': np.random.normal(5000, 3000, n_samples),
            'edad': np.random.randint(18, 65, n_samples),
            'dependientes': np.random.randint(0, 5, n_samples),
            'educacion': np.random.choice(['Secundaria', 'Universidad', 'Posgrado'], n_samples),
            'estado_civil': np.random.choice(['Soltero', 'Casado', 'Divorciado'], n_samples),
            'tipo_empleo': np.random.choice(['Tiempo_completo', 'Medio_tiempo', 'Freelance'], n_samples),
            'experiencia_laboral': np.random.randint(0, 40, n_samples),
            'tiene_deudas': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'monto_deudas': np.random.exponential(2000, n_samples),
            'score_crediticio': np.random.randint(300, 850, n_samples),
            'gastos_entretenimiento': np.random.normal(200, 100, n_samples),
            'gastos_salud': np.random.normal(150, 75, n_samples),
            'gastos_transporte': np.random.normal(300, 150, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Asegurar valores positivos donde corresponde
        df['ingresos_mensuales'] = np.abs(df['ingresos_mensuales'])
        df['gastos_fijos'] = np.abs(df['gastos_fijos'])
        df['gastos_variables'] = np.abs(df['gastos_variables'])
        df['ahorros_actuales'] = np.abs(df['ahorros_actuales'])
        df['monto_deudas'] = np.where(df['tiene_deudas'] == 0, 0, np.abs(df['monto_deudas']))
        
        # Crear features derivados
        df['ratio_gastos_ingresos'] = (df['gastos_fijos'] + df['gastos_variables']) / df['ingresos_mensuales']
        df['capacidad_ahorro'] = df['ingresos_mensuales'] - df['gastos_fijos'] - df['gastos_variables']
        df['ratio_deuda_ingresos'] = df['monto_deudas'] / (df['ingresos_mensuales'] * 12)
        
        # Crear variable objetivo (presupuesto recomendado)
        df['presupuesto_recomendado'] = self._calculate_budget_target(df)
        
        # Crear categorías de riesgo financiero
        df['categoria_riesgo'] = self._calculate_risk_category(df)
        
        logger.info("Datos sintéticos generados exitosamente")
        return df
    
    def _calculate_budget_target(self, df):
        """Calcula el presupuesto objetivo basado en reglas financieras"""
        # Regla 50/30/20: 50% necesidades, 30% deseos, 20% ahorros
        presupuesto = df['ingresos_mensuales'] * 0.8  # Base conservadora
        
        # Ajustes basados en perfil
        presupuesto += df['dependientes'] * 200  # Más gastos por dependientes
        presupuesto -= df['monto_deudas'] / 12 * 0.5  # Reducir por deudas
        presupuesto += (df['score_crediticio'] - 600) / 250 * 100  # Ajuste por score
        
        return np.maximum(presupuesto, df['ingresos_mensuales'] * 0.3)  # Mínimo 30% de ingresos
    
    def _calculate_risk_category(self, df):
        """Calcula categoría de riesgo financiero"""
        risk_score = 0
        
        # Factores de riesgo
        risk_score += np.where(df['ratio_gastos_ingresos'] > 0.8, 2, 0)
        risk_score += np.where(df['ratio_deuda_ingresos'] > 0.3, 2, 0)
        risk_score += np.where(df['ahorros_actuales'] < df['ingresos_mensuales'], 1, 0)
        risk_score += np.where(df['score_crediticio'] < 600, 1, 0)
        
        # Categorizar
        categories = []
        for score in risk_score:
            if score <= 1:
                categories.append('Bajo')
            elif score <= 3:
                categories.append('Medio')
            else:
                categories.append('Alto')
                
        return categories
    
    def preprocess_data(self, df):
        """
        Preprocesa los datos para entrenamiento del modelo
        
        Args:
            df (pd.DataFrame): DataFrame con datos crudos
            
        Returns:
            tuple: (X_processed, y, feature_names)
        """
        logger.info("Iniciando preprocesamiento de datos...")
        
        df_processed = df.copy()
        
        # Manejar valores faltantes
        df_processed = self._handle_missing_values(df_processed)
        
        # Codificar variables categóricas
        categorical_columns = ['educacion', 'estado_civil', 'tipo_empleo', 'categoria_riesgo']
        df_processed = self._encode_categorical_variables(df_processed, categorical_columns)
        
        # Crear features adicionales
        df_processed = self._create_additional_features(df_processed)
        
        # Seleccionar features para el modelo
        feature_columns = [col for col in df_processed.columns 
                          if col not in ['presupuesto_recomendado']]
        
        X = df_processed[feature_columns]
        y = df_processed['presupuesto_recomendado']
        
        # Escalar features numéricas
        X_scaled = self._scale_features(X)
        
        self.feature_columns = feature_columns
        
        logger.info(f"Preprocesamiento completado. Features: {len(feature_columns)}")
        return X_scaled, y, feature_columns
    
    def _handle_missing_values(self, df):
        """Maneja valores faltantes en el dataset"""
        # Rellenar valores numéricos con la mediana
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col].fillna(df[col].median(), inplace=True)
        
        # Rellenar valores categóricos con la moda
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def _encode_categorical_variables(self, df, categorical_columns):
        """Codifica variables categóricas usando Label Encoding"""
        for col in categorical_columns:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def _create_additional_features(self, df):
        """Crea features adicionales derivados"""
        # Ratios financieros
        df['ratio_ahorros_ingresos'] = df['ahorros_actuales'] / (df['ingresos_mensuales'] * 12)
        df['gastos_totales'] = df['gastos_fijos'] + df['gastos_variables']
        df['ingresos_disponibles'] = df['ingresos_mensuales'] - df['gastos_totales']
        
        # Features demográficos
        df['ratio_dependientes_edad'] = df['dependientes'] / (df['edad'] + 1)
        df['experiencia_normalizada'] = df['experiencia_laboral'] / df['edad']
        
        # Features de comportamiento financiero
        df['eficiencia_gastos'] = df['gastos_totales'] / df['ingresos_mensuales']
        df['potencial_ahorro'] = np.maximum(0, df['ingresos_disponibles'])
        
        return df
    
    def _scale_features(self, X):
        """Escala las features numéricas"""
        return self.scaler.fit_transform(X)
    
    def prepare_training_data(self, df, test_size=0.2, random_state=42):
        """
        Prepara los datos para entrenamiento dividiendo en train/test
        
        Args:
            df (pd.DataFrame): DataFrame procesado
            test_size (float): Proporción de datos para test
            random_state (int): Semilla para reproducibilidad
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X, y, feature_names = self.preprocess_data(df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Datos divididos: Train={len(X_train)}, Test={len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def transform_new_data(self, df):
        """
        Transforma nuevos datos usando los encoders y scalers entrenados
        
        Args:
            df (pd.DataFrame): Nuevos datos a transformar
            
        Returns:
            np.array: Datos transformados listos para predicción
        """
        df_processed = df.copy()
        
        # Aplicar el mismo preprocesamiento
        df_processed = self._handle_missing_values(df_processed)
        
        # Codificar variables categóricas con encoders entrenados
        categorical_columns = ['educacion', 'estado_civil', 'tipo_empleo', 'categoria_riesgo']
        for col in categorical_columns:
            if col in df_processed.columns and col in self.label_encoders:
                df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
        
        # Crear features adicionales
        df_processed = self._create_additional_features(df_processed)
        
        # Seleccionar solo las features usadas en entrenamiento
        if self.feature_columns:
            df_processed = df_processed[self.feature_columns]
        
        # Escalar con el scaler entrenado
        X_scaled = self.scaler.transform(df_processed)
        
        return X_scaled

if __name__ == "__main__":
    # Ejemplo de uso
    processor = DataProcessor()
    
    # Generar datos de muestra
    df = processor.generate_sample_data(1000)
    print("Datos generados:")
    print(df.head())
    print(f"\nForma del dataset: {df.shape}")
    print(f"\nEstadísticas descriptivas:")
    print(df.describe())
    
    # Preparar datos para entrenamiento
    X_train, X_test, y_train, y_test = processor.prepare_training_data(df)
    print(f"\nDatos preparados para entrenamiento:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
