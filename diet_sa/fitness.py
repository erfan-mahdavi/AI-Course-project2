from typing import List, Dict
from colorama import init
from .models import Solution
from .models import FoodItem

init(autoreset=True)

class FitnessEvaluator:
    """
    fitness evaluation class for Simulated Annealing diet optimization.
    
    Uses a clear penalty-based approach:
    - Heavy penalties for not meeting minimum requirements
    - Medium penalties for exceeding budget
    - Light penalties for deviation from optimal values
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
        self.directions = {
            'calories': 'min',
            'fat':      'min',
            'carbs':    'min',
            'protein':  'max',
            'fiber':    'max',
            'calcium':  'max',
            'iron':     'max',
        }

    def __call__(self, solution: Solution) -> float:
        """
        Calculate fitness score using Lili's directional preferences.
        Higher scores are better (minimizing penalties).
        """
        quantities = solution.quantities
        
        # Calculate total nutrients and cost
        nutrient_totals = {nut: 0.0 for nut in self.min_req}
        total_cost = 0.0
        
        for qty, food in zip(quantities, self.foods):
            total_cost += qty * food.price
            for nutrient, per100g in food.nutrients.items():
                nutrient_totals[nutrient] += per100g * 10 * qty

        # Start with zero fitness (no penalties)
        fitness = 0.0
        
        # 1. Budget constraint - can't exceed 4M Toman
        if total_cost > self.cost_cap:
            budget_penalty = (total_cost - self.cost_cap) / (self.cost_cap * 100000)
            fitness -= budget_penalty
        
        # 2. Lili's directional preferences for optimization
        for nutrient in self.min_req:
            actual = nutrient_totals
            if self.directions[nutrient]=='max':
                if actual[nutrient] < self.min_req[nutrient]:
                    excess_calories = self.min_req[nutrient] - actual[nutrient]
                    calorie_penalty = (excess_calories / self.min_req[nutrient]) * 400 * self.weights.get(nutrient,1.0)
                    fitness -= calorie_penalty
                else:
                    excess_calories = actual[nutrient] - self.optimal[nutrient]  
                    if excess_calories < 0:
                        calorie_penalty = (-excess_calories / self.optimal[nutrient]) * 300 * self.weights.get(nutrient,1.0)
                        fitness += calorie_penalty
                    else:
                        calorie_penalty = (excess_calories / self.optimal[nutrient]) * 200 * self.weights.get(nutrient,1.0)
                        fitness -= calorie_penalty
            else:
                if actual[nutrient] > self.min_req[nutrient]:
                    excess_calories = actual[nutrient] - self.min_req[nutrient]
                    calorie_penalty = (excess_calories / self.min_req[nutrient]) * 200 * self.weights.get(nutrient,1.0)
                    fitness -= calorie_penalty
                else:
                    excess_calories = actual[nutrient] - self.optimal[nutrient]  
                    if excess_calories > 0:
                        calorie_penalty = (excess_calories / self.optimal[nutrient]) * 300 * self.weights.get(nutrient,1.0)
                        fitness += calorie_penalty
                    else:
                        calorie_penalty = (-excess_calories / self.optimal[nutrient]) * 400 * self.weights.get(nutrient,1.0)
                        fitness -= calorie_penalty
        
        
        # 3. Cost efficiency bonus - reward for staying under budget
        if total_cost <= self.cost_cap:
            cost_efficiency = ((self.cost_cap - total_cost) / self.cost_cap) * 200
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
                monthly_totals[nutrient] += per100g * 10 * qty
        
        daily_totals = {nut: total / 30 for nut, total in monthly_totals.items()}
        
        # Analyze nutritional status
        nutritional_status = {}
        for nutrient, daily_actual in daily_totals.items():
            daily_min = self.min_req[nutrient] / 30
            daily_opt = self.optimal.get(nutrient, daily_min) / 30
            
            if daily_actual < daily_min:
                status = "DEFICIENT"
            elif daily_actual <= daily_opt * 1.2:  # Within 20% of optimal
                status = "OPTIMAL"
            else:
                status = "EXCESSIVE"
            
            nutritional_status[nutrient] = {
                "status": status,
                "actual_daily": daily_actual,
                "minimum_daily": daily_min,
                "optimal_daily": daily_opt,
                "percent_of_minimum": (daily_actual / daily_min) * 100 if daily_min > 0 else 0,
                "percent_of_optimal": (daily_actual / daily_opt) * 100 if daily_opt > 0 else 0
            }
        
        return {
            'total_monthly_cost': total_cost,
            'budget_cap': self.cost_cap,
            'budget_utilization_percent': (total_cost / self.cost_cap) * 100,
            'within_budget': total_cost <= self.cost_cap,
            'monthly_nutrients': monthly_totals,
            'daily_nutrients': daily_totals,
            'nutritional_status': nutritional_status,
            'food_names': self.food_names,
            'food_weights_kg': quantities,
            'food_costs': individual_costs,
            'total_food_weight': sum(quantities),
            'overall_fitness': solution.fitness,
            'meets_all_minimums': all(status["status"] != "DEFICIENT" for status in nutritional_status.values()),
            'has_excesses': any(status["status"] == "EXCESSIVE" for status in nutritional_status.values()),
        }