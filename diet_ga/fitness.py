import numpy as np
from typing import List, Dict
from .models import Individual, FoodItem

class FitnessEvaluator:
    """
    Implements a piecewise nutrient‐direction fitness with:
      - Base linear penalty (real - min_val) for any val ≠ min_val
      - Directional band‐reward for:
          * 'max': reward when min_val ≤ val ≤ opt_val
          * 'min': reward when opt_val ≤ val ≤ min_val
      - Extra linear penalty beyond the band edges
      - Quadratic penalty for 'target' around opt_val which we don't have here but can be added for future
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

    def score_nutrient(
        self,
        val: float,
        min_val: float,
        opt_val: float,
        direction: str,
        weight: float
    ) -> float:
        # 1) Base penalty for any deviation from min_val
        base_penalty = -3*weight * abs(val - min_val) / min_val

        # 2) Directional band reward (or extra penalty if outside band)
        band_reward = 0.0
        if direction == 'max':
            if min_val <= val <= opt_val:
                band_reward = 4*weight * (val - min_val) / min_val
            elif val < min_val:
                band_reward = 2*weight * (val - min_val) / min_val
           
        elif direction == 'min':
            if opt_val <= val <= min_val:
                band_reward = 4*weight * (min_val - val) / min_val
            # If val > min_val: extra punishment (on top of base_penalty)
            elif val > min_val:
                band_reward = -2*weight * (val - min_val) / min_val
        else:  # 'target'
            #quadratic penalty around opt_val if it's not min or max 
            print("reached quadratic")
            band_reward = -weight * ((val - opt_val) ** 2) / opt_val

        return base_penalty + band_reward

    def __call__(self, ind: Individual) -> float:
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
        directions = {
            'calories': 'min',
            'fat':      'min',
            'carbs':    'min',
            'protein':  'max',
            'fiber':    'max',
            'calcium':  'max',
            'iron':     'max'
        }

        for nut, min_val in self.min_req.items():
            val     = totals[nut]
            opt_val = self.optimal.get(nut, min_val)
            w       = self.weights.get(nut, 1.0)
            dirn    = directions[nut]
            score  += self.score_nutrient(val, min_val, opt_val, dirn, w)

        return score
