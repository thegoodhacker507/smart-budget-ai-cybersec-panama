"""
Smart Budget AI - Script de Entrenamiento
Script principal para entrenar el modelo de predicción de presupuestos
"""

import os
import sys
import argparse
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Agregar el directorio src al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import DataProcessor
from budget_model import BudgetNeuralNetwork, BudgetPredictor
from financial_advisor import FinancialAdvisor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Crea los directorios necesarios para el proyecto"""
    directories = [
        '../models',
        '../data',
        '../logs',
        '../plots'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directorio creado/verificado: {directory}")

def generate_training_data(n_samples=5000, save_path=None):
    """
    Genera datos de entrenamiento sintéticos
    
    Args:
        n_samples (int): Número de muestras a generar
        save_path (str): Ruta para guardar los datos
        
    Returns:
        pd.DataFrame: Datos generados
    """
    logger.info(f"Generando {n_samples} muestras de datos de entrenamiento...")
    
    processor = DataProcessor()
    df = processor.generate_sample_data(n_samples)
    
    if save_path:
        df.to_csv(save_path, index=False)
        logger.info(f"Datos guardados en: {save_path}")
    
    return df

def train_model(df, model_save_path, validation_split=0.2, epochs=100):
    """
    Entrena el modelo de predicción de presupuestos
    
    Args:
        df (pd.DataFrame): Datos de entrenamiento
        model_save_path (str): Ruta para guardar el modelo
        validation_split (float): Proporción para validación
        epochs (int): Número de épocas de entrenamiento
        
    Returns:
        tuple: (predictor, metrics, history)
    """
    logger.info("Iniciando entrenamiento del modelo...")
    
    # Crear procesador de datos
    processor = DataProcessor()
    
    # Crear predictor
    predictor = BudgetPredictor()
    
    # Entrenar modelo
    metrics = predictor.train_model(processor, df, validation_split)
    
    # Guardar modelo
    predictor.save_model(model_save_path)
    logger.info(f"Modelo guardado en: {model_save_path}")
    
    return predictor, metrics, predictor.model.history

def evaluate_model(predictor, df_test):
    """
    Evalúa el modelo en datos de prueba
    
    Args:
        predictor: Modelo entrenado
        df_test (pd.DataFrame): Datos de prueba
        
    Returns:
        dict: Métricas de evaluación
    """
    logger.info("Evaluando modelo en datos de prueba...")
    
    # Preparar datos de prueba
    X_test, y_test, _ = predictor.data_processor.preprocess_data(df_test)
    
    # Evaluar modelo
    metrics = predictor.model.evaluate(X_test, y_test)
    
    # Hacer predicciones para análisis adicional
    y_pred = predictor.model.predict(X_test)
    
    # Calcular métricas adicionales
    residuals = y_test - y_pred.flatten()
    
    additional_metrics = {
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals),
        'max_error': np.max(np.abs(residuals)),
        'q95_error': np.percentile(np.abs(residuals), 95)
    }
    
    metrics.update(additional_metrics)
    
    logger.info("Métricas de evaluación:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    return metrics

def create_visualizations(history, metrics, save_dir='../plots'):
    """
    Crea visualizaciones del entrenamiento y resultados
    
    Args:
        history: Historia del entrenamiento
        metrics (dict): Métricas del modelo
        save_dir (str): Directorio para guardar gráficos
    """
    logger.info("Creando visualizaciones...")
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Historia del entrenamiento
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Smart Budget AI - Entrenamiento del Modelo', fontsize=16, fontweight='bold')
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Entrenamiento', linewidth=2)
    if 'val_loss' in history.history:
        axes[0, 0].plot(history.history['val_loss'], label='Validación', linewidth=2)
    axes[0, 0].set_title('Pérdida (MSE)', fontweight='bold')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE
    axes[0, 1].plot(history.history['mae'], label='Entrenamiento', linewidth=2)
    if 'val_mae' in history.history:
        axes[0, 1].plot(history.history['val_mae'], label='Validación', linewidth=2)
    axes[0, 1].set_title('Error Absoluto Medio', fontweight='bold')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # MAPE
    axes[1, 0].plot(history.history['mape'], label='Entrenamiento', linewidth=2)
    if 'val_mape' in history.history:
        axes[1, 0].plot(history.history['val_mape'], label='Validación', linewidth=2)
    axes[1, 0].set_title('Error Porcentual Absoluto Medio', fontweight='bold')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('MAPE (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Métricas finales
    metrics_text = f"""Métricas Finales:
    MSE: {metrics.get('mse', 0):.2f}
    MAE: {metrics.get('mae', 0):.2f}
    RMSE: {metrics.get('rmse', 0):.2f}
    MAPE: {metrics.get('mape', 0):.2f}%
    R²: {metrics.get('r2_score', 0):.4f}"""
    
    axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    axes[1, 1].set_title('Métricas de Rendimiento', fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Gráfico de métricas
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    metric_names = ['MSE', 'MAE', 'RMSE', 'MAPE', 'R²']
    metric_values = [
        metrics.get('mse', 0),
        metrics.get('mae', 0),
        metrics.get('rmse', 0),
        metrics.get('mape', 0),
        metrics.get('r2_score', 0)
    ]
    
    bars = ax.bar(metric_names, metric_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    ax.set_title('Métricas de Rendimiento del Modelo', fontsize=14, fontweight='bold')
    ax.set_ylabel('Valor')
    
    # Agregar valores en las barras
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualizaciones guardadas en: {save_dir}")

def test_predictions(predictor, n_tests=5):
    """
    Prueba el modelo con casos de ejemplo
    
    Args:
        predictor: Modelo entrenado
        n_tests (int): Número de casos de prueba
    """
    logger.info("Probando predicciones con casos de ejemplo...")
    
    # Casos de prueba diversos
    test_cases = [
        {
            'nombre': 'Joven profesional',
            'ingresos_mensuales': 2500,
            'gastos_fijos': 1200,
            'gastos_variables': 600,
            'ahorros_actuales': 3000,
            'edad': 25,
            'dependientes': 0,
            'educacion': 'Universidad',
            'estado_civil': 'Soltero',
            'tipo_empleo': 'Tiempo_completo',
            'experiencia_laboral': 3,
            'tiene_deudas': 1,
            'monto_deudas': 8000,
            'score_crediticio': 680,
            'gastos_entretenimiento': 200,
            'gastos_salud': 100,
            'gastos_transporte': 300,
            'categoria_riesgo': 'Medio'
        },
        {
            'nombre': 'Familia establecida',
            'ingresos_mensuales': 4500,
            'gastos_fijos': 2200,
            'gastos_variables': 1200,
            'ahorros_actuales': 15000,
            'edad': 35,
            'dependientes': 2,
            'educacion': 'Universidad',
            'estado_civil': 'Casado',
            'tipo_empleo': 'Tiempo_completo',
            'experiencia_laboral': 12,
            'tiene_deudas': 1,
            'monto_deudas': 25000,
            'score_crediticio': 750,
            'gastos_entretenimiento': 300,
            'gastos_salud': 250,
            'gastos_transporte': 450,
            'categoria_riesgo': 'Bajo'
        },
        {
            'nombre': 'Freelancer',
            'ingresos_mensuales': 3000,
            'gastos_fijos': 1000,
            'gastos_variables': 800,
            'ahorros_actuales': 5000,
            'edad': 30,
            'dependientes': 0,
            'educacion': 'Posgrado',
            'estado_civil': 'Soltero',
            'tipo_empleo': 'Freelance',
            'experiencia_laboral': 8,
            'tiene_deudas': 0,
            'monto_deudas': 0,
            'score_crediticio': 720,
            'gastos_entretenimiento': 250,
            'gastos_salud': 150,
            'gastos_transporte': 200,
            'categoria_riesgo': 'Bajo'
        }
    ]
    
    print("\n" + "="*60)
    print("PRUEBAS DE PREDICCIÓN")
    print("="*60)
    
    for i, case in enumerate(test_cases[:n_tests]):
        try:
            # Hacer predicción
            prediction = predictor.predict_budget(case)
            
            # Calcular algunos ratios para contexto
            current_expenses = case['gastos_fijos'] + case['gastos_variables']
            savings_potential = case['ingresos_mensuales'] - current_expenses
            
            print(f"\nCaso {i+1}: {case['nombre']}")
            print(f"Ingresos mensuales: ${case['ingresos_mensuales']:,}")
            print(f"Gastos actuales: ${current_expenses:,}")
            print(f"Capacidad de ahorro actual: ${savings_potential:,}")
            print(f"PRESUPUESTO RECOMENDADO: ${prediction:,.2f}")
            print(f"Diferencia con gastos actuales: ${prediction - current_expenses:+,.2f}")
            
        except Exception as e:
            logger.error(f"Error en predicción para caso {i+1}: {e}")

def main():
    """Función principal del script de entrenamiento"""
    parser = argparse.ArgumentParser(description='Entrenar modelo Smart Budget AI')
    parser.add_argument('--samples', type=int, default=5000, 
                       help='Número de muestras para entrenamiento')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Número de épocas de entrenamiento')
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help='Proporción de datos para validación')
    parser.add_argument('--model-name', type=str, default='smart_budget_model',
                       help='Nombre del modelo a guardar')
    parser.add_argument('--skip-training', action='store_true',
                       help='Saltar entrenamiento y solo hacer evaluación')
    
    args = parser.parse_args()
    
    logger.info("Iniciando Smart Budget AI - Entrenamiento")
    logger.info(f"Parámetros: samples={args.samples}, epochs={args.epochs}")
    
    # Configurar directorios
    setup_directories()
    
    # Rutas de archivos
    data_path = f'../data/training_data_{args.samples}.csv'
    model_path = f'../models/{args.model_name}.h5'
    
    try:
        if not args.skip_training:
            # Generar datos de entrenamiento
            df = generate_training_data(args.samples, data_path)
            
            # Entrenar modelo
            predictor, metrics, history = train_model(
                df, model_path, args.validation_split, args.epochs
            )
            
            # Crear visualizaciones
            create_visualizations(history, metrics)
            
        else:
            # Cargar modelo existente
            logger.info(f"Cargando modelo existente: {model_path}")
            predictor = BudgetPredictor(model_path)
            
            # Cargar datos para evaluación
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
            else:
                df = generate_training_data(1000)  # Datos pequeños para evaluación
            
            # Evaluar modelo
            metrics = evaluate_model(predictor, df.sample(200))
        
        # Probar predicciones
        test_predictions(predictor)
        
        # Crear análisis financiero de ejemplo
        advisor = FinancialAdvisor()
        example_user = {
            'ingresos_mensuales': 3500,
            'gastos_fijos': 1800,
            'gastos_variables': 900,
            'ahorros_actuales': 8000,
            'edad': 28,
            'dependientes': 1,
            'monto_deudas': 5000,
            'score_crediticio': 720,
            'gastos_entretenimiento': 250,
            'gastos_transporte': 350
        }
        
        analysis = advisor.analyze_financial_health(example_user)
        
        print("\n" + "="*60)
        print("EJEMPLO DE ANÁLISIS FINANCIERO")
        print("="*60)
        print(f"Score de salud financiera: {analysis['overall_score']}/100")
        print(f"Nivel de riesgo: {analysis['risk_level']}")
        print("\nRecomendaciones principales:")
        for rec in analysis['recommendations'][:3]:
            print(f"• {rec}")
        
        logger.info("Entrenamiento completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}")
        raise

if __name__ == "__main__":
    main()
