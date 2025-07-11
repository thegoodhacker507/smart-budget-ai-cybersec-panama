"""
Smart Budget AI - Asesor Financiero
Motor de análisis financiero y generación de recomendaciones personalizadas
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Any

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialAdvisor:
    """
    Clase principal para análisis financiero y generación de recomendaciones
    """
    
    def __init__(self):
        """Inicializa el asesor financiero"""
        self.risk_thresholds = {
            'debt_to_income': 0.36,  # Ratio deuda/ingresos máximo recomendado
            'expense_to_income': 0.80,  # Ratio gastos/ingresos máximo
            'emergency_fund_months': 6,  # Meses de gastos en fondo de emergencia
            'savings_rate_min': 0.20,  # Tasa mínima de ahorro recomendada
        }
        
        self.budget_rules = {
            'housing': 0.30,  # Máximo 30% en vivienda
            'transportation': 0.15,  # Máximo 15% en transporte
            'food': 0.12,  # Máximo 12% en alimentación
            'utilities': 0.08,  # Máximo 8% en servicios
            'entertainment': 0.05,  # Máximo 5% en entretenimiento
            'savings': 0.20,  # Mínimo 20% en ahorros
            'other': 0.10  # 10% para otros gastos
        }
    
    def analyze_financial_health(self, user_data: Dict) -> Dict:
        """
        Analiza la salud financiera del usuario
        
        Args:
            user_data (Dict): Datos financieros del usuario
            
        Returns:
            Dict: Análisis completo de salud financiera
        """
        logger.info("Iniciando análisis de salud financiera...")
        
        analysis = {
            'overall_score': 0,
            'risk_level': 'Bajo',
            'strengths': [],
            'weaknesses': [],
            'ratios': {},
            'recommendations': [],
            'emergency_fund_status': {},
            'debt_analysis': {},
            'savings_analysis': {}
        }
        
        # Calcular ratios financieros clave
        analysis['ratios'] = self._calculate_financial_ratios(user_data)
        
        # Analizar nivel de riesgo
        analysis['risk_level'], risk_score = self._assess_risk_level(user_data, analysis['ratios'])
        
        # Calcular score general (0-100)
        analysis['overall_score'] = self._calculate_overall_score(user_data, analysis['ratios'])
        
        # Analizar fondo de emergencia
        analysis['emergency_fund_status'] = self._analyze_emergency_fund(user_data)
        
        # Analizar deudas
        analysis['debt_analysis'] = self._analyze_debt_situation(user_data)
        
        # Analizar ahorros
        analysis['savings_analysis'] = self._analyze_savings_pattern(user_data)
        
        # Identificar fortalezas y debilidades
        analysis['strengths'], analysis['weaknesses'] = self._identify_strengths_weaknesses(
            user_data, analysis['ratios']
        )
        
        # Generar recomendaciones generales
        analysis['recommendations'] = self._generate_general_recommendations(
            user_data, analysis
        )
        
        logger.info(f"Análisis completado. Score: {analysis['overall_score']}/100")
        return analysis
    
    def _calculate_financial_ratios(self, user_data: Dict) -> Dict:
        """Calcula ratios financieros importantes"""
        ratios = {}
        
        monthly_income = user_data.get('ingresos_mensuales', 0)
        fixed_expenses = user_data.get('gastos_fijos', 0)
        variable_expenses = user_data.get('gastos_variables', 0)
        total_expenses = fixed_expenses + variable_expenses
        debt_amount = user_data.get('monto_deudas', 0)
        savings = user_data.get('ahorros_actuales', 0)
        
        # Ratio gastos/ingresos
        ratios['expense_to_income'] = total_expenses / monthly_income if monthly_income > 0 else 0
        
        # Ratio deuda/ingresos (anual)
        annual_income = monthly_income * 12
        ratios['debt_to_income'] = debt_amount / annual_income if annual_income > 0 else 0
        
        # Ratio ahorros/ingresos
        ratios['savings_to_income'] = savings / annual_income if annual_income > 0 else 0
        
        # Capacidad de ahorro mensual
        monthly_surplus = monthly_income - total_expenses
        ratios['monthly_savings_rate'] = monthly_surplus / monthly_income if monthly_income > 0 else 0
        
        # Meses de gastos cubiertos por ahorros
        ratios['emergency_fund_months'] = savings / total_expenses if total_expenses > 0 else 0
        
        # Ratio gastos fijos vs variables
        ratios['fixed_to_variable_expenses'] = fixed_expenses / variable_expenses if variable_expenses > 0 else 0
        
        return ratios
    
    def _assess_risk_level(self, user_data: Dict, ratios: Dict) -> Tuple[str, int]:
        """Evalúa el nivel de riesgo financiero"""
        risk_score = 0
        
        # Factores de riesgo
        if ratios['debt_to_income'] > self.risk_thresholds['debt_to_income']:
            risk_score += 3
        elif ratios['debt_to_income'] > 0.25:
            risk_score += 1
        
        if ratios['expense_to_income'] > self.risk_thresholds['expense_to_income']:
            risk_score += 3
        elif ratios['expense_to_income'] > 0.70:
            risk_score += 1
        
        if ratios['emergency_fund_months'] < 3:
            risk_score += 2
        elif ratios['emergency_fund_months'] < self.risk_thresholds['emergency_fund_months']:
            risk_score += 1
        
        if ratios['monthly_savings_rate'] < 0:
            risk_score += 3
        elif ratios['monthly_savings_rate'] < 0.10:
            risk_score += 1
        
        # Factores demográficos
        age = user_data.get('edad', 30)
        dependents = user_data.get('dependientes', 0)
        employment_type = user_data.get('tipo_empleo', 'Tiempo_completo')
        
        if age > 50 and ratios['savings_to_income'] < 0.5:
            risk_score += 1
        
        if dependents > 2:
            risk_score += 1
        
        if employment_type in ['Medio_tiempo', 'Freelance']:
            risk_score += 1
        
        # Determinar nivel de riesgo
        if risk_score <= 2:
            return 'Bajo', risk_score
        elif risk_score <= 5:
            return 'Medio', risk_score
        else:
            return 'Alto', risk_score
    
    def _calculate_overall_score(self, user_data: Dict, ratios: Dict) -> int:
        """Calcula un score general de salud financiera (0-100)"""
        score = 100
        
        # Penalizaciones por ratios problemáticos
        if ratios['debt_to_income'] > 0.36:
            score -= 20
        elif ratios['debt_to_income'] > 0.25:
            score -= 10
        
        if ratios['expense_to_income'] > 0.80:
            score -= 15
        elif ratios['expense_to_income'] > 0.70:
            score -= 8
        
        if ratios['emergency_fund_months'] < 3:
            score -= 15
        elif ratios['emergency_fund_months'] < 6:
            score -= 8
        
        if ratios['monthly_savings_rate'] < 0:
            score -= 20
        elif ratios['monthly_savings_rate'] < 0.10:
            score -= 10
        
        # Bonificaciones por buenas prácticas
        if ratios['monthly_savings_rate'] > 0.20:
            score += 5
        
        if ratios['emergency_fund_months'] > 6:
            score += 5
        
        credit_score = user_data.get('score_crediticio', 600)
        if credit_score > 750:
            score += 5
        elif credit_score < 600:
            score -= 10
        
        return max(0, min(100, score))
    
    def _analyze_emergency_fund(self, user_data: Dict) -> Dict:
        """Analiza el estado del fondo de emergencia"""
        savings = user_data.get('ahorros_actuales', 0)
        monthly_expenses = user_data.get('gastos_fijos', 0) + user_data.get('gastos_variables', 0)
        
        months_covered = savings / monthly_expenses if monthly_expenses > 0 else 0
        recommended_amount = monthly_expenses * 6
        deficit = max(0, recommended_amount - savings)
        
        status = 'Excelente' if months_covered >= 6 else \
                'Bueno' if months_covered >= 3 else \
                'Insuficiente' if months_covered >= 1 else 'Crítico'
        
        return {
            'current_amount': savings,
            'months_covered': months_covered,
            'recommended_amount': recommended_amount,
            'deficit': deficit,
            'status': status
        }
    
    def _analyze_debt_situation(self, user_data: Dict) -> Dict:
        """Analiza la situación de deudas"""
        debt_amount = user_data.get('monto_deudas', 0)
        monthly_income = user_data.get('ingresos_mensuales', 0)
        annual_income = monthly_income * 12
        
        debt_to_income = debt_amount / annual_income if annual_income > 0 else 0
        
        # Estimación de pago mensual (asumiendo 5% interés anual, 5 años)
        if debt_amount > 0:
            monthly_rate = 0.05 / 12
            num_payments = 60  # 5 años
            if monthly_rate > 0:
                monthly_payment = debt_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / \
                                ((1 + monthly_rate)**num_payments - 1)
            else:
                monthly_payment = debt_amount / num_payments
        else:
            monthly_payment = 0
        
        payment_to_income = monthly_payment / monthly_income if monthly_income > 0 else 0
        
        status = 'Sin deudas' if debt_amount == 0 else \
                'Manejable' if debt_to_income <= 0.25 else \
                'Preocupante' if debt_to_income <= 0.40 else 'Crítico'
        
        return {
            'total_debt': debt_amount,
            'debt_to_income_ratio': debt_to_income,
            'estimated_monthly_payment': monthly_payment,
            'payment_to_income_ratio': payment_to_income,
            'status': status
        }
    
    def _analyze_savings_pattern(self, user_data: Dict) -> Dict:
        """Analiza el patrón de ahorros"""
        monthly_income = user_data.get('ingresos_mensuales', 0)
        total_expenses = user_data.get('gastos_fijos', 0) + user_data.get('gastos_variables', 0)
        current_savings = user_data.get('ahorros_actuales', 0)
        
        monthly_surplus = monthly_income - total_expenses
        savings_rate = monthly_surplus / monthly_income if monthly_income > 0 else 0
        
        # Proyección de ahorros anuales
        annual_savings_potential = monthly_surplus * 12
        
        status = 'Excelente' if savings_rate >= 0.20 else \
                'Bueno' if savings_rate >= 0.10 else \
                'Insuficiente' if savings_rate >= 0 else 'Déficit'
        
        return {
            'current_savings': current_savings,
            'monthly_surplus': monthly_surplus,
            'savings_rate': savings_rate,
            'annual_savings_potential': annual_savings_potential,
            'status': status
        }
    
    def _identify_strengths_weaknesses(self, user_data: Dict, ratios: Dict) -> Tuple[List[str], List[str]]:
        """Identifica fortalezas y debilidades financieras"""
        strengths = []
        weaknesses = []
        
        # Evaluar fortalezas
        if ratios['monthly_savings_rate'] > 0.15:
            strengths.append("Excelente capacidad de ahorro mensual")
        
        if ratios['debt_to_income'] < 0.20:
            strengths.append("Nivel de deuda muy manejable")
        
        if ratios['emergency_fund_months'] >= 6:
            strengths.append("Fondo de emergencia bien establecido")
        
        if user_data.get('score_crediticio', 600) > 750:
            strengths.append("Excelente historial crediticio")
        
        if ratios['expense_to_income'] < 0.70:
            strengths.append("Gastos bien controlados")
        
        # Evaluar debilidades
        if ratios['monthly_savings_rate'] < 0:
            weaknesses.append("Gastos superan los ingresos")
        
        if ratios['debt_to_income'] > 0.36:
            weaknesses.append("Nivel de deuda muy alto")
        
        if ratios['emergency_fund_months'] < 3:
            weaknesses.append("Fondo de emergencia insuficiente")
        
        if user_data.get('score_crediticio', 600) < 600:
            weaknesses.append("Score crediticio necesita mejora")
        
        if ratios['expense_to_income'] > 0.80:
            weaknesses.append("Gastos demasiado altos")
        
        return strengths, weaknesses
    
    def _generate_general_recommendations(self, user_data: Dict, analysis: Dict) -> List[str]:
        """Genera recomendaciones generales basadas en el análisis"""
        recommendations = []
        ratios = analysis['ratios']
        
        # Recomendaciones por fondo de emergencia
        if analysis['emergency_fund_status']['months_covered'] < 3:
            deficit = analysis['emergency_fund_status']['deficit']
            recommendations.append(
                f"URGENTE: Construir fondo de emergencia. Te faltan ${deficit:,.0f} "
                f"para cubrir 6 meses de gastos."
            )
        
        # Recomendaciones por deudas
        if ratios['debt_to_income'] > 0.36:
            recommendations.append(
                "Reducir deudas es prioritario. Considera consolidación o "
                "estrategia de pago acelerado."
            )
        
        # Recomendaciones por gastos
        if ratios['expense_to_income'] > 0.80:
            recommendations.append(
                "Revisar y reducir gastos. Identifica gastos no esenciales "
                "que puedas eliminar o reducir."
            )
        
        # Recomendaciones por ahorros
        if ratios['monthly_savings_rate'] < 0.10:
            recommendations.append(
                "Aumentar tasa de ahorro. Objetivo mínimo: 10% de ingresos mensuales."
            )
        
        # Recomendaciones por score crediticio
        credit_score = user_data.get('score_crediticio', 600)
        if credit_score < 650:
            recommendations.append(
                "Mejorar score crediticio: pagar puntualmente, reducir utilización "
                "de tarjetas de crédito."
            )
        
        # Recomendaciones por edad
        age = user_data.get('edad', 30)
        if age > 40 and ratios['savings_to_income'] < 0.5:
            recommendations.append(
                "Acelerar ahorros para retiro. A tu edad, deberías tener "
                "al menos 50% de ingresos anuales ahorrados."
            )
        
        return recommendations
    
    def create_budget_plan(self, user_data: Dict, target_budget: float) -> Dict:
        """
        Crea un plan de presupuesto detallado
        
        Args:
            user_data (Dict): Datos del usuario
            target_budget (float): Presupuesto objetivo
            
        Returns:
            Dict: Plan de presupuesto detallado
        """
        logger.info("Creando plan de presupuesto personalizado...")
        
        monthly_income = user_data.get('ingresos_mensuales', 0)
        
        # Distribución recomendada del presupuesto
        budget_distribution = {
            'vivienda': target_budget * self.budget_rules['housing'],
            'transporte': target_budget * self.budget_rules['transportation'],
            'alimentacion': target_budget * self.budget_rules['food'],
            'servicios': target_budget * self.budget_rules['utilities'],
            'entretenimiento': target_budget * self.budget_rules['entertainment'],
            'ahorros': target_budget * self.budget_rules['savings'],
            'otros': target_budget * self.budget_rules['other']
        }
        
        # Ajustar según situación actual
        current_fixed = user_data.get('gastos_fijos', 0)
        current_variable = user_data.get('gastos_variables', 0)
        
        # Calcular diferencias con gastos actuales
        current_total = current_fixed + current_variable
        budget_difference = target_budget - current_total
        
        plan = {
            'target_budget': target_budget,
            'current_spending': current_total,
            'budget_difference': budget_difference,
            'distribution': budget_distribution,
            'recommendations': [],
            'savings_goal': budget_distribution['ahorros'],
            'priority_areas': []
        }
        
        # Generar recomendaciones específicas del presupuesto
        if budget_difference < 0:
            plan['recommendations'].append(
                f"Necesitas reducir gastos en ${abs(budget_difference):,.0f} mensuales"
            )
            plan['priority_areas'] = self._identify_reduction_areas(user_data, budget_distribution)
        else:
            plan['recommendations'].append(
                f"Tienes ${budget_difference:,.0f} adicionales para optimizar tu presupuesto"
            )
        
        # Recomendaciones por categoría
        plan['category_recommendations'] = self._generate_category_recommendations(
            user_data, budget_distribution
        )
        
        return plan
    
    def _identify_reduction_areas(self, user_data: Dict, budget_distribution: Dict) -> List[str]:
        """Identifica áreas donde reducir gastos"""
        areas = []
        
        entertainment_spending = user_data.get('gastos_entretenimiento', 0)
        if entertainment_spending > budget_distribution['entretenimiento']:
            areas.append('entretenimiento')
        
        transport_spending = user_data.get('gastos_transporte', 0)
        if transport_spending > budget_distribution['transporte']:
            areas.append('transporte')
        
        # Agregar otras categorías según datos disponibles
        variable_expenses = user_data.get('gastos_variables', 0)
        if variable_expenses > (budget_distribution['alimentacion'] + 
                               budget_distribution['entretenimiento'] + 
                               budget_distribution['otros']):
            areas.append('gastos_variables')
        
        return areas
    
    def _generate_category_recommendations(self, user_data: Dict, budget_distribution: Dict) -> Dict:
        """Genera recomendaciones específicas por categoría"""
        recommendations = {}
        
        recommendations['vivienda'] = [
            f"Presupuesto recomendado: ${budget_distribution['vivienda']:,.0f}",
            "Incluye renta/hipoteca, seguros, mantenimiento",
            "No debe exceder 30% de ingresos"
        ]
        
        recommendations['transporte'] = [
            f"Presupuesto recomendado: ${budget_distribution['transporte']:,.0f}",
            "Incluye combustible, mantenimiento, seguros",
            "Considera transporte público para ahorrar"
        ]
        
        recommendations['alimentacion'] = [
            f"Presupuesto recomendado: ${budget_distribution['alimentacion']:,.0f}",
            "Planifica comidas, compra inteligente",
            "Limita comidas fuera de casa"
        ]
        
        recommendations['ahorros'] = [
            f"Meta de ahorro: ${budget_distribution['ahorros']:,.0f}",
            "Automatiza tus ahorros",
            "Prioriza fondo de emergencia"
        ]
        
        return recommendations
    
    def generate_financial_goals(self, user_data: Dict, analysis: Dict) -> Dict:
        """
        Genera metas financieras personalizadas
        
        Args:
            user_data (Dict): Datos del usuario
            analysis (Dict): Análisis financiero
            
        Returns:
            Dict: Metas financieras con plazos
        """
        goals = {
            'short_term': [],  # 1-6 meses
            'medium_term': [],  # 6 meses - 2 años
            'long_term': []   # 2+ años
        }
        
        monthly_income = user_data.get('ingresos_mensuales', 0)
        current_savings = user_data.get('ahorros_actuales', 0)
        age = user_data.get('edad', 30)
        
        # Metas a corto plazo
        if analysis['emergency_fund_status']['months_covered'] < 3:
            deficit = analysis['emergency_fund_status']['deficit']
            months_needed = max(1, deficit / (monthly_income * 0.1))
            goals['short_term'].append({
                'goal': 'Fondo de emergencia básico',
                'target_amount': deficit,
                'timeline_months': min(6, months_needed),
                'monthly_required': deficit / min(6, months_needed),
                'priority': 'Alta'
            })
        
        # Metas a mediano plazo
        if analysis['debt_analysis']['total_debt'] > 0:
            debt_amount = analysis['debt_analysis']['total_debt']
            goals['medium_term'].append({
                'goal': 'Eliminar deudas',
                'target_amount': debt_amount,
                'timeline_months': 24,
                'monthly_required': debt_amount / 24,
                'priority': 'Alta'
            })
        
        # Meta de ahorro para vivienda (si es joven y no tiene casa)
        if age < 35:
            house_down_payment = monthly_income * 12 * 3  # 3 años de ingresos
            goals['long_term'].append({
                'goal': 'Enganche para vivienda',
                'target_amount': house_down_payment,
                'timeline_months': 60,
                'monthly_required': house_down_payment / 60,
                'priority': 'Media'
            })
        
        # Meta de retiro
        retirement_target = monthly_income * 12 * 10  # 10 años de ingresos
        years_to_retirement = max(5, 65 - age)
        goals['long_term'].append({
            'goal': 'Fondo de retiro',
            'target_amount': retirement_target,
            'timeline_months': years_to_retirement * 12,
            'monthly_required': retirement_target / (years_to_retirement * 12),
            'priority': 'Media' if age < 40 else 'Alta'
        })
        
        return goals

if __name__ == "__main__":
    # Ejemplo de uso
    advisor = FinancialAdvisor()
    
    # Datos de ejemplo
    user_example = {
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
    
    # Realizar análisis
    analysis = advisor.analyze_financial_health(user_example)
    
    print("=== ANÁLISIS DE SALUD FINANCIERA ===")
    print(f"Score general: {analysis['overall_score']}/100")
    print(f"Nivel de riesgo: {analysis['risk_level']}")
    
    print("\nFortalezas:")
    for strength in analysis['strengths']:
        print(f"✓ {strength}")
    
    print("\nÁreas de mejora:")
    for weakness in analysis['weaknesses']:
        print(f"⚠ {weakness}")
    
    print("\nRecomendaciones:")
    for rec in analysis['recommendations']:
        print(f"• {rec}")
    
    # Crear plan de presupuesto
    budget_plan = advisor.create_budget_plan(user_example, 2800)
    print(f"\n=== PLAN DE PRESUPUESTO ===")
    print(f"Presupuesto objetivo: ${budget_plan['target_budget']:,.0f}")
    print(f"Diferencia con gastos actuales: ${budget_plan['budget_difference']:,.0f}")
    
    # Generar metas financieras
    goals = advisor.generate_financial_goals(user_example, analysis)
    print(f"\n=== METAS FINANCIERAS ===")
    for term, goal_list in goals.items():
        print(f"\n{term.replace('_', ' ').title()}:")
        for goal in goal_list:
            print(f"• {goal['goal']}: ${goal['target_amount']:,.0f} "
                  f"en {goal['timeline_months']} meses")
