from typing import List, Dict
import numpy as np
from tabulate import tabulate
from colorama import Fore, Style, init
from .models import Individual, FoodItem

# Initialize colorama
init(autoreset=True)

class FitnessEvaluator:
    """
    Implements a piecewise nutrient‐direction fitness with:
      - Base linear penalty (real - min_val) for any val ≠ min_val
      - Directional band‐reward for:
          * 'max': reward when min_val ≤ val ≤ opt_val
          * 'min': reward when opt_val ≤ val ≤ min_val
      - Extra linear penalty beyond the band edges
    """
    def __init__(
        self,
        foods: List[FoodItem],
        min_req: Dict[str, float],
        optimal: Dict[str, float],
        weights: Dict[str, float],
        cost_cap: float
    ):
        self.foods    = foods
        self.min_req  = min_req
        self.optimal  = optimal
        self.weights  = weights
        self.cost_cap = cost_cap
        
        # Store food names for better reporting
        self.food_names = [food.name for food in foods]
        
        # Nutrient directions for reference
        self.directions = {
            'calories': 'min',
            'fat':      'min',
            'carbs':    'min',
            'protein':  'max',
            'fiber':    'max',
            'calcium':  'max',
            'iron':     'max'
        }

    def score_nutrient(
        self,
        val: float,
        min_val: float,
        opt_val: float,
        direction: str,
        weight: float
    ) -> float:
        """
        Score a nutrient value based on its direction and constraints
        
        Args:
            val: Actual value of the nutrient
            min_val: Minimum required value
            opt_val: Optimal value
            direction: Direction ('min', 'max', or 'target')
            weight: Weight of this nutrient in the fitness function
            
        Returns:
            Score for this nutrient (higher is better)
        """
        # 1) Base penalty for any deviation from min_val
        base_penalty = -3*weight * abs(val - min_val) / min_val

        # 2) Directional band reward (or extra penalty if outside band)
        band_reward = 0.0
        if direction == 'max':
            if min_val <= val <= opt_val:
                band_reward = 4*weight * (val - min_val) / min_val
            elif val < min_val:
                band_reward = 2*weight * (val - min_val) / min_val
            elif val > opt_val:
                # Add penalty for exceeding optimal for 'max' direction
                band_reward = -weight * (val - opt_val) / opt_val
           
        elif direction == 'min':
            if opt_val <= val <= min_val:
                band_reward = 4*weight * (min_val - val) / min_val
            # If val > min_val: extra punishment (on top of base_penalty)
            elif val > min_val:
                band_reward = -2*weight * (val - min_val) / min_val
            elif val < opt_val:
                # Add penalty for going below optimal for 'min' direction
                band_reward = -weight * (opt_val - val) / opt_val
        else:  # 'target'
            # Quadratic penalty around opt_val if it's not min or max
            band_reward = -weight * ((val - opt_val) ** 2) / opt_val

        return base_penalty + band_reward

    def __call__(self, ind: Individual) -> float:
        """
        Calculate fitness for an individual
        
        Args:
            ind: The individual to evaluate
            
        Returns:
            Fitness score (higher is better)
        """
        # 1) Sum up all nutrient totals and total cost
        totals = {nut: 0.0 for nut in self.min_req}
        cost = 0.0
        for qty, food in zip(ind.chromosome, self.foods):
            cost += qty * food.price
            for nut, per100g in food.nutrients.items():
                # per100g → per‐kg = per100g * 10
                totals[nut] += per100g * 10 * qty

        # 2) heavy budget penalty
        score = 0.0
        if cost > self.cost_cap:
            score -= 500*(cost - self.cost_cap)

        # 3) Nutrient‐by‐nutrient scoring
        for nut, min_val in self.min_req.items():
            val     = totals[nut]
            opt_val = self.optimal.get(nut, min_val)
            w       = self.weights.get(nut, 1.0)
            dirn    = self.directions[nut]
            score  += self.score_nutrient(val, min_val, opt_val, dirn, w)

        return score
        
    def get_nutrition_report(self, ind: Individual) -> Dict:
        """
        Generate a detailed nutrition report for an individual
        
        Args:
            ind: The individual to analyze
            
        Returns:
            Dictionary with nutrition details
        """
        # Calculate total nutrients and cost
        totals_month = {nut: 0.0 for nut in self.min_req}
        cost = 0.0
        food_costs = []
        food_weights = []
        
        for qty, food in zip(ind.chromosome, self.foods):
            item_cost = qty * food.price
            cost += item_cost
            food_costs.append(item_cost)
            food_weights.append(qty)
            
            for nut, per100g in food.nutrients.items():
                # per100g → per‐kg = per100g * 10
                totals_month[nut] += per100g * 10 * qty
        
        # Calculate daily values for easier interpretation
        totals_daily = {nut: val/30 for nut, val in totals_month.items()}
        
        # Check if requirements are met
        status = {}
        for nut, val in totals_daily.items():
            min_val = self.min_req[nut] / 30  # Daily minimum
            opt_val = self.optimal.get(nut, min_val) / 30  # Daily optimal
            dirn = self.directions[nut]
            
            if dirn == 'max':
                if val < min_val:
                    status[nut] = "BELOW MINIMUM"
                elif val <= opt_val:
                    status[nut] = "OPTIMAL RANGE"
                else:
                    status[nut] = "ABOVE OPTIMAL"
            elif dirn == 'min':
                if val > min_val:
                    status[nut] = "ABOVE MINIMUM"
                elif val >= opt_val:
                    status[nut] = "OPTIMAL RANGE"
                else:
                    status[nut] = "BELOW OPTIMAL"
        
        return {
            'total_cost': cost,
            'budget_cap': self.cost_cap,
            'budget_status': "WITHIN BUDGET" if cost <= self.cost_cap else "OVER BUDGET",
            'nutrients_daily': totals_daily,
            'nutrients_status': status,
            'food_items': self.food_names,
            'food_weights': food_weights,
            'food_costs': food_costs
        }