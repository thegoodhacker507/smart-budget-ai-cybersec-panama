"""
Smart Budget AI - Aplicación Web Flask
Interfaz web para el sistema de recomendaciones de presupuesto inteligente
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import sys
import logging
import traceback
import json
from datetime import datetime

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from data_processor import DataProcessor
    from budget_model import BudgetPredictor
    from financial_advisor import FinancialAdvisor
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Asegúrate de que los archivos estén en el directorio src/")

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación Flask
app = Flask(__name__)
app.secret_key = 'smart_budget_ai_secret_key_2024'

# Variables globales para los modelos
predictor = None
advisor = None
data_processor = None

def initialize_models():
    """Inicializa los modelos y componentes del sistema"""
    global predictor, advisor, data_processor
    
    try:
        logger.info("Inicializando modelos...")
        
        # Inicializar asesor financiero
        advisor = FinancialAdvisor()
        
        # Inicializar procesador de datos
        data_processor = DataProcessor()
        
        # Intentar cargar modelo pre-entrenado
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'smart_budget_model.h5')
        
        if os.path.exists(model_path):
            logger.info(f"Cargando modelo desde: {model_path}")
            predictor = BudgetPredictor(model_path)
        else:
            logger.warning("Modelo no encontrado. Entrenando modelo básico...")
            # Crear y entrenar un modelo básico
            predictor = BudgetPredictor()
            df = data_processor.generate_sample_data(1000)
            predictor.train_model(data_processor, df)
            
            # Guardar modelo
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            predictor.save_model(model_path)
            logger.info("Modelo básico entrenado y guardado")
        
        logger.info("Modelos inicializados correctamente")
        return True
        
    except Exception as e:
        logger.error(f"Error inicializando modelos: {e}")
        logger.error(traceback.format_exc())
        return False

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_budget():
    """Endpoint para análisis de presupuesto"""
    try:
        # Obtener datos del formulario
        user_data = {
            'ingresos_mensuales': float(request.form.get('ingresos_mensuales', 0)),
            'gastos_fijos': float(request.form.get('gastos_fijos', 0)),
            'gastos_variables': float(request.form.get('gastos_variables', 0)),
            'ahorros_actuales': float(request.form.get('ahorros_actuales', 0)),
            'edad': int(request.form.get('edad', 25)),
            'dependientes': int(request.form.get('dependientes', 0)),
            'educacion': request.form.get('educacion', 'Universidad'),
            'estado_civil': request.form.get('estado_civil', 'Soltero'),
            'tipo_empleo': request.form.get('tipo_empleo', 'Tiempo_completo'),
            'experiencia_laboral': int(request.form.get('experiencia_laboral', 1)),
            'tiene_deudas': int(request.form.get('tiene_deudas', 0)),
            'monto_deudas': float(request.form.get('monto_deudas', 0)),
            'score_crediticio': int(request.form.get('score_crediticio', 650)),
            'gastos_entretenimiento': float(request.form.get('gastos_entretenimiento', 0)),
            'gastos_salud': float(request.form.get('gastos_salud', 0)),
            'gastos_transporte': float(request.form.get('gastos_transporte', 0)),
        }
        
        # Validar datos básicos
        if user_data['ingresos_mensuales'] <= 0:
            return jsonify({'error': 'Los ingresos mensuales deben ser mayores a 0'}), 400
        
        # Calcular categoría de riesgo
        total_gastos = user_data['gastos_fijos'] + user_data['gastos_variables']
        ratio_gastos = total_gastos / user_data['ingresos_mensuales']
        
        if ratio_gastos > 0.8:
            user_data['categoria_riesgo'] = 'Alto'
        elif ratio_gastos > 0.6:
            user_data['categoria_riesgo'] = 'Medio'
        else:
            user_data['categoria_riesgo'] = 'Bajo'
        
        # Realizar análisis financiero
        financial_analysis = advisor.analyze_financial_health(user_data)
        
        # Predecir presupuesto recomendado
        try:
            recommended_budget = predictor.predict_budget(user_data)
        except Exception as e:
            logger.warning(f"Error en predicción, usando cálculo básico: {e}")
            # Cálculo básico como fallback
            recommended_budget = user_data['ingresos_mensuales'] * 0.8
        
        # Crear plan de presupuesto
        budget_plan = advisor.create_budget_plan(user_data, recommended_budget)
        
        # Generar metas financieras
        financial_goals = advisor.generate_financial_goals(user_data, financial_analysis)
        
        # Preparar respuesta
        response = {
            'success': True,
            'user_data': user_data,
            'recommended_budget': round(recommended_budget, 2),
            'financial_analysis': financial_analysis,
            'budget_plan': budget_plan,
            'financial_goals': financial_goals,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error en análisis: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Error en el análisis: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    """Endpoint para verificar el estado de la aplicación"""
    try:
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'models_loaded': {
                'predictor': predictor is not None and predictor.model.is_trained,
                'advisor': advisor is not None,
                'data_processor': data_processor is not None
            }
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/api/example')
def get_example_data():
    """Endpoint para obtener datos de ejemplo"""
    examples = [
        {
            'name': 'Joven Profesional',
            'data': {
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
                'gastos_transporte': 300
            }
        },
        {
            'name': 'Familia Establecida',
            'data': {
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
                'gastos_transporte': 450
            }
        }
    ]
    
    return jsonify(examples)

@app.errorhandler(404)
def not_found(error):
    """Manejo de errores 404"""
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Manejo de errores 500"""
    logger.error(f"Error interno: {error}")
    return jsonify({'error': 'Error interno del servidor'}), 500

if __name__ == '__main__':
    # Inicializar modelos
    if not initialize_models():
        logger.error("No se pudieron inicializar los modelos. La aplicación puede no funcionar correctamente.")
    
    # Configurar y ejecutar la aplicación
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Iniciando Smart Budget AI en puerto {port}")
    logger.info(f"Modo debug: {debug}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )
