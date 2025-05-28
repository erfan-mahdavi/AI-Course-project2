from typing import List, Dict
from colorama import init
from .models import Solution
from .models import FoodItem

init(autoreset=True)

class FitnessEvaluator:
    """
    fitness evaluation class for Simulated Annealing diet optimization.
    
    Uses a clear penalty-based approach with proper directional logic:
    - Heavy penalties for not meeting minimum requirements
    - Medium penalties for exceeding budget
    - Light penalties for deviation from optimal values
    - Proper handling of 'min' vs 'max' nutrient directions
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

    def __call__(self, solution: Solution) -> float:
        """
        Calculate fitness score using directional preferences.
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
                    nutrient_totals[nutrient] += per100g * 10 * qty

        # Start with zero fitness (no penalties)
        fitness = 0.0
        
        # 1. Budget constraint - heavy penalty for exceeding budget
        if total_cost > self.cost_cap:
            # budget_penalty = ((total_cost - self.cost_cap) / self.cost_cap) * 1000
            # fitness -= budget_penalty
            return -10000
        
        # 2. Process each nutrient based on its direction
        for nutrient in self.min_req:
            if nutrient not in nutrient_totals:
                continue
                
            actual = nutrient_totals[nutrient]
            minimum = self.min_req[nutrient]
            optimal_val = self.optimal.get(nutrient, minimum)
            weight = self.weights.get(nutrient, 1.0)
            direction = self.directions.get(nutrient, 'min')
            
            if direction == 'max':
                # For MAX nutrients (protein, fiber, calcium, iron):
                # - Heavy penalty if below minimum requirement
                # - Light bonus if between minimum and optimal (closer to minimum is better)
                # - Small penalty if above optimal (too much is still not ideal)
                
                if actual < minimum:
                    # Critical deficiency - heavy penalty
                    deficiency_penalty = ((minimum - actual) / minimum) * 500 * weight
                    fitness -= deficiency_penalty
                elif actual <= optimal_val:
                    # Good range - bonus for being closer to minimum (more efficient)
                    # Closer to minimum = better, so bonus decreases as we approach optimal
                    distance_from_optimal = optimal_val - actual
                    max_distance = optimal_val - minimum
                    if max_distance > 0:
                        efficiency_bonus = (distance_from_optimal / max_distance) * 500 * weight
                        fitness += efficiency_bonus
                else:
                    # Above optimal - penalty for excess
                    excess_penalty = ((actual - optimal_val) / optimal_val) * 200 * weight
                    fitness -= excess_penalty
                    
            else:  # direction == 'min'
                # For MIN nutrients (calories, fat, carbs):
                # - Heavy penalty if below optimal requirement
                # - Bonus if between minimum and optimal (closer to minimum is better)
                # - Small penalty if above minimum
                
                if actual < optimal_val:
                    # Below optimal - heavy penalty for being too restrictive/unrealistic
                    under_penalty = ((optimal_val - actual) / optimal_val) * 500 * weight
                    fitness -= under_penalty
                elif actual <= minimum:
                    # Between optimal and minimum - bonus for efficiency (closer to minimum is better)
                    distance_from_optimal = actual - optimal_val
                    max_distance = minimum - optimal_val
                    if max_distance > 0:
                        efficiency_bonus = (distance_from_optimal / max_distance) * 500 * weight
                        fitness += efficiency_bonus
                else:
                    # Above minimum - penalty for excess
                    excess_penalty = ((actual - minimum) / minimum) * 200 * weight
                    fitness -= excess_penalty
        
        # 3. Cost efficiency bonus - reward for staying reasonably under budget
        if 3000000.0 <= total_cost <= self.cost_cap:  # 1.5M to 95% of budget
            cost_efficiency = ((self.cost_cap - total_cost) / self.cost_cap) * 100
            fitness += cost_efficiency
        
        return fitness
        
    def get_detailed_analysis(self, solution: Solution) -> Dict:
        """Generate comprehensive analysis of solution."""
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
        
        # Analyze nutritional status with direction-aware logic
        nutritional_status = {}
        for nutrient, daily_actual in daily_totals.items():
            daily_min = self.min_req[nutrient] / 30
            daily_opt = self.optimal.get(nutrient, daily_min) / 30
            direction = self.directions.get(nutrient, 'min')
            
            # Status determination based on direction
            if direction == 'max':
                if daily_actual < daily_min:
                    status = "DEFICIENT"
                elif daily_actual <= daily_opt * 1.1:
                    status = "OPTIMAL"
                else:
                    status = "EXCESSIVE"
            else:  # min direction
                if daily_actual > daily_min:
                    status = "EXCESSIVE"
                elif daily_actual >= daily_opt:
                    status = "OPTIMAL"
                else:
                    status = "TOO_LOW"
            
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
            'monthly_nutrients': {k: round(v, 2) for k, v in monthly_totals.items()},
            'daily_nutrients': {k: round(v, 2) for k, v in daily_totals.items()},
            'nutritional_status': nutritional_status,
            'food_names': self.food_names,
            'food_weights_kg': [round(q, 3) for q in quantities],
            'food_costs': [round(c, 2) for c in individual_costs],
            'total_food_weight': round(sum(quantities), 2),
            'overall_fitness': round(solution.fitness, 2),
            'fitness_breakdown': fitness_breakdown,
        }
    
    def _get_fitness_breakdown(self, solution: Solution, nutrient_totals: Dict, total_cost: float) -> Dict:
        """Break down fitness score into components for analysis."""
        breakdown = {
            'budget_penalty': 0,
            'nutrient_penalties': {},
            'nutrient_bonuses': {},
            'cost_efficiency_bonus': 0,
            'weight_penalty': 0
        }
        
        # Budget penalty
        if total_cost > self.cost_cap:
            breakdown['budget_penalty'] = -((total_cost - self.cost_cap) / self.cost_cap) * 1000
        
        # Cost efficiency bonus
        if 1500000.0 <= total_cost <= self.cost_cap * 0.95:
            breakdown['cost_efficiency_bonus'] = ((self.cost_cap - total_cost) / self.cost_cap) * 100
        
        # Weight penalty
        total_weight = sum(solution.quantities)
        if total_weight < 10:
            breakdown['weight_penalty'] = -(10 - total_weight) * 50
        
        return breakdown