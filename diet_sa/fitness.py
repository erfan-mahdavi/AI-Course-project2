from typing import List, Dict
from tabulate import tabulate
from colorama import Fore, init
from .models import Solution
from .models import FoodItem

# Initialize colorama for colored terminal output
init(autoreset=True)

class FitnessEvaluator:
    """
    Fitness evaluation class .
    
    This class implements a sophisticated piecewise nutrient-direction fitness function
    that evaluates SA solutions based on:
      - Base linear penalty for deviations from minimum requirements
      - Directional band rewards for optimal nutrient ranges:
          * 'max' nutrients: reward when min_val ≤ val ≤ opt_val
          * 'min' nutrients: reward when opt_val ≤ val ≤ min_val  
      - Extra penalties for severe violations beyond optimal ranges
      - Heavy budget constraint penalties
    
    The fitness function is designed to guide the SA algorithm toward nutritionally
    balanced and cost-effective food combinations.
    """
    
    def __init__(
        self,
        foods: List[FoodItem],
        min_req: Dict[str, float],
        optimal: Dict[str, float],
        weights: Dict[str, float],
        cost_cap: float,
    ):
        """
        Initialize the fitness evaluator for Simulated Annealing.
        
        Args:
            foods: List of available food items with nutritional data
            min_req: Minimum monthly requirements for each nutrient
            optimal: Optimal monthly targets for each nutrient
            weights: Importance weights for each nutrient in fitness calculation
            cost_cap: Maximum allowed monthly cost in Toman
        """
        self.foods = foods
        self.min_req = min_req
        self.optimal = optimal
        self.weights = weights
        self.cost_cap = cost_cap
        
        # Store food names for better reporting and debugging
        self.food_names = [food.name for food in foods]
        
        # Define nutrient optimization directions
        # 'max': higher values are better (up to optimal) - proteins, vitamins, minerals
        # 'min': lower values are better (down to optimal) - calories, fats, carbs for weight control
        self.directions = {
            'calories': 'min',    # Minimize calories for weight management
            'fat':      'min',    # Minimize fat intake
            'carbs':    'min',    # Minimize carbohydrate intake
            'protein':  'max',    # Maximize protein for muscle health
            'fiber':    'max',    # Maximize fiber for digestive health
            'calcium':  'max',    # Maximize calcium for bone health
            'iron':     'max',    # Maximize iron for blood health
        }

    def score_nutrient(
        self,
        actual_val: float,
        min_val: float,
        opt_val: float,
        direction: str,
        weight: float,
    ) -> float:
        """
        Calculate fitness score for a single nutrient based on its optimization direction.
        
        This method implements the core scoring logic that rewards solutions for:
        1. Meeting minimum nutritional requirements
        2. Staying within optimal ranges for each nutrient type
        3. Avoiding excessive intake that could be harmful
        
        Args:
            actual_val: Actual nutrient value in the current solution
            min_val: Minimum required value for this nutrient
            opt_val: Optimal target value for this nutrient
            direction: Optimization direction ('min', 'max', or 'target')
            weight: Importance weight for this nutrient
            
        Returns:
            Fitness score contribution for this nutrient (higher is better)
        """
        # Prevent division by zero errors
        if min_val <= 0:
            min_val = 0.001
        if opt_val <= 0:
            opt_val = 0.001
            
        # 1) Base penalty for any deviation from minimum requirements
        # This ensures that meeting basic nutritional needs is always prioritized
        base_penalty = -3 * weight * abs(actual_val - min_val) / min_val

        # 2) Directional scoring based on nutrient optimization type
        band_reward = 0.0
        
        if direction == 'max':
            # For nutrients where more is better (protein, fiber, calcium, iron)
            # Optimal range: between minimum and optimal values
            
            if min_val <= actual_val <= opt_val:
                # Perfect range - give substantial reward proportional to how much above minimum
                band_reward = 4 * weight * (actual_val - min_val) / min_val
                
            elif actual_val < min_val:
                # Below minimum - additional penalty for deficiency
                deficiency_penalty = 2 * weight * (actual_val - min_val) / min_val
                band_reward = deficiency_penalty
                
            elif actual_val > opt_val:
                # Above optimal - moderate penalty (excess can sometimes be harmful)
                excess_penalty = -weight * (actual_val - opt_val) / opt_val
                band_reward = excess_penalty
           
        elif direction == 'min':
            # For nutrients where less is better (calories, fat, carbs)
            # Optimal range: between optimal and minimum values (optimal < minimum for these)
            
            if opt_val <= actual_val <= min_val:
                # Perfect range - reward for being closer to optimal (lower) value
                band_reward = 4 * weight * (min_val - actual_val) / min_val
                
            elif actual_val > min_val:
                # Above minimum (too much) - additional penalty for excess
                excess_penalty = -2 * weight * (actual_val - min_val) / min_val
                band_reward = excess_penalty
                
            elif actual_val < opt_val:
                # Below optimal (too little) - penalty for extreme restriction
                restriction_penalty = -weight * (opt_val - actual_val) / opt_val
                band_reward = restriction_penalty
                
        else:  # 'target' direction
            # For nutrients that should hit a specific target value
            # we use quadratic penalty to encourage precision around the target
            deviation = abs(actual_val - opt_val)
            band_reward = -weight * (deviation ** 2) / (opt_val ** 2)

        return base_penalty + band_reward

    def __call__(self, solution: Solution) -> float:
        """
        Calculate the overall fitness score for a Simulated Annealing solution.
        
        This is the main fitness function that the SA algorithm uses to evaluate
        and compare different food combinations. It considers:
        1. Nutritional adequacy across all required nutrients
        2. Budget constraints and cost efficiency
        3. Balance between different nutritional goals
        
        Args:
            solution: SA Solution object containing food quantities
            
        Returns:
            Overall fitness score (higher values indicate better solutions)
        """
        # Extract food quantities from the SA solution
        quantities = solution.quantities
        
        # 1) Calculate total nutrient intake and cost for this solution
        nutrient_totals = {nut: 0.0 for nut in self.min_req}
        total_cost = 0.0
        
        for qty, food in zip(quantities, self.foods):
            # Add to total monthly cost
            total_cost += qty * food.price
            
            # Add to nutrient totals
            # Note: food nutrients are per 100g, multiply by 10 to get per kg
            for nutrient, per100g in food.nutrients.items():
                nutrient_totals[nutrient] += per100g * 10 * qty

        # 2) Apply budget constraint with heavy penalty
        fitness_score = 0.0
        if total_cost > self.cost_cap:
            # Heavy penalty for exceeding budget - makes infeasible solutions very unattractive
            budget_violation = total_cost - self.cost_cap
            budget_penalty = 500 * (budget_violation / self.cost_cap)
            fitness_score -= budget_penalty

        # 3) Evaluate each nutrient individually and sum the scores
        for nutrient, min_requirement in self.min_req.items():
            actual_intake = nutrient_totals[nutrient]
            optimal_target = self.optimal.get(nutrient, min_requirement)
            nutrient_weight = self.weights.get(nutrient, 1.0)
            optimization_direction = self.directions[nutrient]
            
            # Calculate individual nutrient score
            nutrient_score = self.score_nutrient(
                actual_intake, min_requirement, optimal_target, 
                optimization_direction, nutrient_weight
            )
            fitness_score += nutrient_score

        return fitness_score
        
    def get_detailed_analysis(self, solution: Solution) -> Dict:
        """
        Generate a comprehensive analysis of a Simulated Annealing solution.
        
        This method provides detailed insights into how well the solution performs
        across all evaluation criteria, which is useful for:
        - Understanding solution quality
        - Debugging SA algorithm performance
        - Generating reports for users
        
        Args:
            solution: SA Solution object to analyze
            
        Returns:
            Dictionary containing detailed analysis results
        """
        quantities = solution.quantities
        
        # Calculate monthly and daily nutrient totals
        monthly_totals = {nut: 0.0 for nut in self.min_req}
        total_cost = 0.0
        individual_food_costs = []
        individual_food_weights = []
        
        for qty, food in zip(quantities, self.foods):
            item_cost = qty * food.price
            total_cost += item_cost
            individual_food_costs.append(item_cost)
            individual_food_weights.append(qty)
            
            # Calculate nutrient contributions from this food
            for nutrient, per100g in food.nutrients.items():
                # Convert per 100g to per kg (multiply by 10)
                monthly_totals[nutrient] += per100g * 10 * qty
        
        # Convert monthly totals to daily averages for easier interpretation
        daily_totals = {nut: total / 30 for nut, total in monthly_totals.items()}
        
        # Analyze nutritional status for each nutrient
        nutritional_status = {}
        for nutrient, daily_actual in daily_totals.items():
            daily_minimum = self.min_req[nutrient] / 30
            daily_optimal = self.optimal.get(nutrient, self.min_req[nutrient]) / 30
            direction = self.directions[nutrient]
            
            # Determine status based on optimization direction and actual values
            if direction == 'max':
                if daily_actual < daily_minimum:
                    status = "DEFICIENT"
                elif daily_actual <= daily_optimal:
                    status = "OPTIMAL"
                else:
                    status = "EXCESSIVE"
            elif direction == 'min':
                if daily_actual > daily_minimum:
                    status = "EXCESSIVE"
                elif daily_actual >= daily_optimal:
                    status = "OPTIMAL"
                else:
                    status = "DEFICIENT"
            else:  # target
                tolerance = 0.1 * daily_optimal  # 10% tolerance around target
                if abs(daily_actual - daily_optimal) <= tolerance:
                    status = "ON_TARGET"
                else:
                    status = "OFF_TARGET"
            
            nutritional_status[nutrient] = {
                "status": status,
                "actual_daily": daily_actual,
                "minimum_daily": daily_minimum,
                "optimal_daily": daily_optimal,
                "percent_of_minimum": (daily_actual / daily_minimum) * 100 if daily_minimum > 0 else 0,
                "percent_of_optimal": (daily_actual / daily_optimal) * 100 if daily_optimal > 0 else 0
            }
        
        # Calculate fitness score breakdown
        fitness_breakdown = self.get_fitness_breakdown(solution)
        
        return {
            # Cost analysis
            'total_monthly_cost': total_cost,
            'budget_cap': self.cost_cap,
            'budget_utilization_percent': (total_cost / self.cost_cap) * 100,
            'within_budget': total_cost <= self.cost_cap,
            'cost_per_kg': total_cost / sum(individual_food_weights) if sum(individual_food_weights) > 0 else 0,
            
            # Nutritional analysis
            'monthly_nutrients': monthly_totals,
            'daily_nutrients': daily_totals,
            'nutritional_status': nutritional_status,
            
            # Food composition
            'food_names': self.food_names,
            'food_weights_kg': individual_food_weights,
            'food_costs': individual_food_costs,
            'total_food_weight': sum(individual_food_weights),
            
            # Fitness details
            'overall_fitness': solution.fitness,
            'fitness_breakdown': fitness_breakdown,
            
            # Summary flags
            'meets_all_minimums': all(status["status"] != "DEFICIENT" for status in nutritional_status.values()),
            'has_excesses': any(status["status"] == "EXCESSIVE" for status in nutritional_status.values()),
        }
    
    def get_fitness_breakdown(self, solution: Solution) -> Dict:
        """
        Break down the fitness score into individual components for analysis.
        
        This method helps understand how each part of the fitness function
        contributes to the overall score, which is valuable for:
        - Algorithm tuning and debugging
        - Understanding solution trade-offs
        - Identifying areas for improvement
        
        Args:
            solution: SA Solution object to analyze
            
        Returns:
            Dictionary with detailed fitness component breakdown
        """
        quantities = solution.quantities
        
        # Recalculate totals and cost
        nutrient_totals = {nut: 0.0 for nut in self.min_req}
        total_cost = 0.0
        
        for qty, food in zip(quantities, self.foods):
            total_cost += qty * food.price
            for nutrient, per100g in food.nutrients.items():
                nutrient_totals[nutrient] += per100g * 10 * qty

        # Calculate individual component scores
        breakdown = {
            'total_fitness': solution.fitness,
            'budget_component': 0.0,
            'nutrient_components': {},
            'cost_info': {
                'total_cost': total_cost,
                'budget_cap': self.cost_cap,
                'over_budget': total_cost > self.cost_cap
            }
        }
        
        # Budget penalty component
        if total_cost > self.cost_cap:
            budget_violation = total_cost - self.cost_cap
            breakdown['budget_component'] = -500 * (budget_violation / self.cost_cap)
        
        # Individual nutrient score components
        for nutrient, min_requirement in self.min_req.items():
            actual_intake = nutrient_totals[nutrient]
            optimal_target = self.optimal.get(nutrient, min_requirement)
            nutrient_weight = self.weights.get(nutrient, 1.0)
            direction = self.directions[nutrient]
            
            # Calculate this nutrient's contribution to fitness
            nutrient_score = self.score_nutrient(
                actual_intake, min_requirement, optimal_target, direction, nutrient_weight
            )
            
            breakdown['nutrient_components'][nutrient] = {
                'score': nutrient_score,
                'actual_monthly': actual_intake,
                'actual_daily': actual_intake / 30,
                'minimum_required': min_requirement,
                'optimal_target': optimal_target,
                'weight': nutrient_weight,
                'direction': direction
            }
        
        return breakdown
    