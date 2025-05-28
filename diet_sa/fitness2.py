from typing import List, Dict
from colorama import init
from .models import Solution
from .models import FoodItem

init(autoreset=True)

class FitnessEvaluator:
    """
    fitness evaluation class combining comprehensive analysis with cleaner scoring logic.
    """
    
    def __init__(
        self,
        foods: List[FoodItem],
        min_req: Dict[str, float],
        optimal: Dict[str, float],
        weights: Dict[str, float],
        cost_cap: float,
    ):
        self.foods = foods
        self.min_req = min_req
        self.optimal = optimal
        self.weights = weights
        self.cost_cap = cost_cap
        self.food_names = [food.name for food in foods]
        
        # Nutrient direction definitions
        self.directions = {
            'calories': 'min',  # Want to minimize calories
            'fat':      'min',  # Want to minimize fat
            'carbs':    'min',  # Want to minimize carbs
            'protein':  'max',  # Want to maximize protein
            'fiber':    'max',  # Want to maximize fiber
            'calcium':  'max',  # Want to maximize calcium
            'iron':     'max',  # Want to maximize iron
        }

    def score_nutrient(self, val: float, min_val: float, opt_val: float, direction: str, weight: float) -> float:
        """
        Score a single nutrient based on its value, requirements, and direction.
        Uses the cleaner logic from fitness2.
        """
        if direction == 'max':
            # For nutrients we want to maximize (protein, fiber, etc.)
            if val < min_val:
                # Below minimum - heavy penalty
                return -1000 * weight * (min_val - val) / min_val
            elif val <= opt_val:
                # Between minimum and optimal - bonus (closer to optimal is better)
                return 100 * weight * (opt_val - val) / (opt_val - min_val + 1e-6)
            else:
                # Above optimal - light penalty for excess
                return -50 * weight * (val - opt_val) / opt_val

        elif direction == 'min':
            # For nutrients we want to minimize (calories, fat, etc.)
            if val > opt_val:
                # Above optimal - heavy penalty
                return -1000 * weight * (val - opt_val) / opt_val
            elif val >= min_val:
                # Between minimum and optimal - bonus (closer to minimum is better)
                return 100 * weight * (val - min_val) / (opt_val - min_val + 1e-6)
            else:
                # Below minimum - light penalty (too restrictive)
                return -50 * weight * (min_val - val) / min_val

        return 0

    def __call__(self, solution: Solution) -> float:
        """
        Calculate fitness score using the cleaner fitness2 logic.
        Higher scores are better.
        """
        quantities = solution.quantities
        
        # Calculate total nutrients and cost
        nutrient_totals = {nut: 0.0 for nut in self.min_req}
        total_cost = 0.0
        
        for qty, food in zip(quantities, self.foods):
            total_cost += qty * food.price
            for nutrient, per100g in food.nutrients.items():
                if nutrient in nutrient_totals:
                    # Convert per 100g to per kg (multiply by 10)
                    nutrient_totals[nutrient] += per100g * 10 * qty

        # Start with zero fitness
        fitness = 0.0
        
        # 1. Cost penalty - heavy penalty for exceeding budget
        if total_cost > self.cost_cap:
            fitness -= 5000 * (total_cost - self.cost_cap)
        
        # 2. Score each nutrient using the cleaner logic
        for nutrient in self.min_req:
            if nutrient not in nutrient_totals:
                continue
                
            actual = nutrient_totals[nutrient]
            minimum = self.min_req[nutrient]
            optimal_val = self.optimal.get(nutrient, minimum)
            weight = self.weights.get(nutrient, 1.0)
            direction = self.directions.get(nutrient, 'min')
            
            nutrient_score = self.score_nutrient(actual, minimum, optimal_val, direction, weight)
            fitness += nutrient_score
        
        return fitness
        
    def get_detailed_analysis(self, solution: Solution) -> Dict:
        """Generate comprehensive analysis of solution with enhanced reporting."""
        quantities = solution.quantities
        
        # Calculate monthly and daily totals
        monthly_totals = {nut: 0.0 for nut in self.min_req}
        total_cost = 0.0
        individual_costs = []
        
        for qty, food in zip(quantities, self.foods):
            item_cost = qty * food.price
            total_cost += item_cost
            individual_costs.append(item_cost)
            
            for nutrient, per100g in food.nutrients.items():
                if nutrient in monthly_totals:
                    monthly_totals[nutrient] += per100g * 10 * qty
        
        daily_totals = {nut: total / 30 for nut, total in monthly_totals.items()}
        
        # Analyze nutritional status using fitness2-style logic
        nutritional_status = {}
        for nutrient, daily_actual in daily_totals.items():
            daily_min = self.min_req[nutrient] / 30
            daily_opt = self.optimal.get(nutrient, daily_min) / 30
            direction = self.directions.get(nutrient, 'min')
            
            # Status determination based on direction (fitness2 logic)
            if direction == 'max':
                if daily_actual < daily_min:
                    status = "BELOW MINIMUM"
                elif daily_actual <= daily_opt:
                    status = "OPTIMAL RANGE"
                else:
                    status = "ABOVE OPTIMAL"
            else:  # min direction
                if daily_actual > daily_opt:
                    status = "ABOVE OPTIMAL"
                elif daily_actual >= daily_min:
                    status = "OPTIMAL RANGE"
                else:
                    status = "BELOW OPTIMAL"
            
            nutritional_status[nutrient] = {
                "status": status,
                "direction": direction,
                "actual_daily": round(daily_actual, 2),
                "minimum_daily": round(daily_min, 2),
                "optimal_daily": round(daily_opt, 2),
                "percent_of_minimum": round((daily_actual / daily_min) * 100, 1) if daily_min > 0 else 0,
                "percent_of_optimal": round((daily_actual / daily_opt) * 100, 1) if daily_opt > 0 else 0
            }
        
        # Calculate fitness components for debugging
        fitness_breakdown = self._get_fitness_breakdown(solution, monthly_totals, total_cost)
        
        return {
            'total_monthly_cost': round(total_cost, 2),
            'budget_cap': self.cost_cap,
            'budget_utilization_percent': round((total_cost / self.cost_cap) * 100, 1),
            'within_budget': total_cost <= self.cost_cap,
            'budget_status': "WITHIN BUDGET" if total_cost <= self.cost_cap else "OVER BUDGET",
            'monthly_nutrients': {k: round(v, 2) for k, v in monthly_totals.items()},
            'daily_nutrients': {k: round(v, 2) for k, v in daily_totals.items()},
            'nutritional_status': nutritional_status,
            'food_names': self.food_names,
            'food_items': self.food_names,  # For compatibility with fitness2 interface
            'food_weights_kg': [round(q, 3) for q in quantities],
            'food_weights': [round(q, 3) for q in quantities],  # For compatibility
            'food_costs': [round(c, 2) for c in individual_costs],
            'total_food_weight': round(sum(quantities), 2),
            'overall_fitness': round(solution.fitness, 2),
            'fitness_breakdown': fitness_breakdown,
        }
    
    def get_nutrition_report(self, solution) -> Dict:
        """
        Generate a nutrition report compatible with fitness2 interface.
        This is an alias for get_detailed_analysis with simplified output.
        """
        analysis = self.get_detailed_analysis(solution)
        
        # Return simplified format matching fitness2
        return {
            'total_cost': analysis['total_monthly_cost'],
            'budget_cap': analysis['budget_cap'],
            'budget_status': analysis['budget_status'],
            'nutrients_daily': analysis['daily_nutrients'],
            'nutrients_status': {k: v['status'] for k, v in analysis['nutritional_status'].items()},
            'food_items': analysis['food_names'],
            'food_weights': analysis['food_weights'],
            'food_costs': analysis['food_costs'],
        }
    
    def _get_fitness_breakdown(self, solution: Solution, nutrient_totals: Dict, total_cost: float) -> Dict:
        """Break down fitness score into components for analysis."""
        breakdown = {
            'total_fitness': round(solution.fitness, 2),
            'cost_penalty': 0,
            'nutrient_scores': {},
            'nutrient_breakdown': {}
        }
        
        # Cost penalty
        if total_cost > self.cost_cap:
            breakdown['cost_penalty'] = -5000 * (total_cost - self.cost_cap)
        
        # Individual nutrient scores
        for nutrient in self.min_req:
            if nutrient not in nutrient_totals:
                continue
                
            actual = nutrient_totals[nutrient]
            minimum = self.min_req[nutrient]
            optimal_val = self.optimal.get(nutrient, minimum)
            weight = self.weights.get(nutrient, 1.0)
            direction = self.directions.get(nutrient, 'min')
            
            nutrient_score = self.score_nutrient(actual, minimum, optimal_val, direction, weight)
            breakdown['nutrient_scores'][nutrient] = round(nutrient_score, 2)
            
            # Detailed breakdown for this nutrient
            breakdown['nutrient_breakdown'][nutrient] = {
                'actual': round(actual, 2),
                'minimum': round(minimum, 2),
                'optimal': round(optimal_val, 2),
                'direction': direction,
                'weight': weight,
                'score': round(nutrient_score, 2)
            }
        
        return breakdown