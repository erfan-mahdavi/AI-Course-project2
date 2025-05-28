import random
from typing import List, Callable, Dict
import numpy as np

class FoodItem:
    """
    Represents a single food item with nutritional information and price.
    """
    
    def __init__(self, name: str, nutrients: Dict[str, float], price_per_kg: float):
        self.name = name
        self.nutrients = nutrients  # per 100g
        self.price = price_per_kg   # Toman per kg

    def get_nutrient_density(self, nutrient: str) -> float:
        """Calculate nutrient per unit cost.
            Grams of nutrient per 1000 Toman."""
        if self.price <= 0 or nutrient not in self.nutrients:
            return 0.0
        return (self.nutrients[nutrient] * 1000) / self.price

    def __repr__(self) -> str:
        return f"FoodItem(name='{self.name}', price={self.price})"
    
# ******************************************************************************

class Solution:
    """
    Represents a solution in the Simulated Annealing algorithm.
    Contains food quantities (kg per month) and fitness evaluation.
    """

    MIN_QUANTITY = 0.0
    MAX_QUANTITY = 10.0
    
    def __init__(self, quantities: List[float], evaluator: Callable[['Solution'], float]):
        self.quantities = quantities.copy()
        self.evaluator = evaluator
        self.fitness = evaluator(self)
        self.nutrient = [ 'calories',
                            'fat',
                            'carbs',
                            'protein',
                            'fiber',
                            'calcium',
                            'iron']

        # quantity around minimum value
        self.directions = {
            'calories': 'min',
            'fat':      'min',
            'carbs':    'min',
            'protein':  'max',
            'fiber':    'max',
            'calcium':  'max',
            'iron':     'max',
        }
    
    @classmethod
    def random_solution(cls, num_foods: int, evaluator: Callable, max_qty=None) -> 'Solution':
        """
        Create an intelligent random solution .
        Uses nutritional analysis and cost-efficiency to guide initial selection.
        """
        if max_qty is None:
            max_qty = cls.MAX_QUANTITY
        
        foods = evaluator.foods
        budget_cap = evaluator.cost_cap
        min_req = evaluator.min_req
        
        # step 1: Nutritional Priority Analysis
        nutrient_priorities = cls._analyze_nutrient_priorities(foods, min_req)
        
        # step 2: Balanced Food Selection Strategy
        quantities = cls._strategic_food_selection(
            foods, nutrient_priorities,budget_cap
        )
        
        return cls(quantities, evaluator)
    
    @staticmethod
    def _analyze_nutrient_priorities(foods: List, min_req: Dict[str, float]):
        """
        Analyze which foods are best sources for each nutrient.
        Returns top food sources for each nutrient sorted by efficiency.
        """
        nutrient_sources = {nutrient: [] for nutrient in min_req.keys()}
        
        for i, food in enumerate(foods):
            for nutrient in min_req.keys():
                if nutrient in food.nutrients and food.nutrients[nutrient] > 0:
                    # Calculate nutrient density (nutrient per unit cost)
                    density = food.get_nutrient_density(nutrient)
                    nutrient_sources[nutrient].append((i, food, density))
        
        # Sort each nutrient's sources by efficiency (highest first)
        for nutrient in nutrient_sources:
            nutrient_sources[nutrient] = sorted(nutrient_sources[nutrient],key=lambda x: x[2], reverse=True)
        
        return nutrient_sources
    
    @staticmethod
    def _strategic_food_selection(foods: List, nutrient_priorities: Dict, 
                                  budget_cap: float) -> List[float]:
        """
        Strategic food selection using multiple criteria.
        Ensures nutritional coverage while maintaining cost efficiency.
        """
        quantities = [0.0] * len(foods)
        selected_foods = set()
        
        # Step 1: Select essential high-priority foods for each nutrient
        nutrients = [   
                        'calories',
                        'fat',
                        'carbs',
                        'protein',
                        'fiber',
                        'calcium',
                        'iron'
                    ]
        used_cap = 0
        for nutrient in nutrients:
            if nutrient in nutrient_priorities:
                # Select top 2 sources for each critical nutrient
                top_sources = nutrient_priorities[nutrient][:5]
                for food_idx, food , density in top_sources:
                    if density > 0.1:  # Meaningful density threshold
                        if used_cap<budget_cap:
                            selected_foods.add(food_idx)
                            # Base quantity using gamma distribution
                            base_qty = np.random.gamma(2.5, 2)
                            quantities[food_idx] = max(quantities[food_idx], base_qty)
                            used_cap+=(quantities[food_idx]*food.price)
        
        return quantities
    
    def copy(self) -> 'Solution':
        """Create a deep copy of this solution."""
        return Solution(self.quantities.copy(), self.evaluator)
    
    def recalculate_fitness(self) -> None:
        """Recalculate fitness after modifying quantities."""
        self.fitness = self.evaluator(self)
    
    def _reduce_cost_neighbor(self, foods, step_size):
        """Generate neighbor by reducing expensive foods."""
        new_quantities = self.quantities.copy()
        qty_dict = {col.name:qty for col, qty in zip(foods,new_quantities)}
        costs = [{'food_name':food.name,'food_cost':(qty*food.price)} for food, qty in zip(foods,new_quantities)]
        costs = sorted(costs,key=lambda x: x['food_cost'],reverse=True)
        # expensive = sorted(foods,key=lambda x : x.price,reverse=True)
        for food in costs[:10]:
            name = food['food_name']
            reduction = random.uniform(0, step_size)
            qty_dict[name] = max(0.0, qty_dict[name] - reduction)
        
        new_quantities = [qty for qty in qty_dict.values()]
        return Solution(new_quantities, self.evaluator)

    def perturb_simple(self, step_size: float = 1.5) -> 'Solution':
        """
        Create neighbor by simple random perturbation.
        This is the main neighbor generation method.
        """
        new_quantities = self.quantities.copy()
        
        # Modify 1-3 food items
        num_changes = random.randint(1, 3)
        indices = random.sample(range(len(new_quantities)), num_changes)
        
        for idx in indices:
            # Random change within step_size
            change = random.uniform(-step_size, step_size)
            new_quantities[idx] = max(0.0, new_quantities[idx] + change)
            
            # Small chance to set to zero or random value
            if random.uniform(0,1) < 0.1:
                if random.random() < 0.5:
                    new_quantities[idx] = 0.0
                else:
                    new_quantities[idx] = random.uniform(0.1, 2.0)
        
        return Solution(new_quantities, self.evaluator)
    
    def perturb_focused(self, deficient_nutrients, step_size) -> 'Solution':
        """
        Create neighbor by focusing on a specific nutrient deficiency.
        """
        new_quantities = self.quantities.copy()
        
        for nut in deficient_nutrients:
            if nut[1]=='decrease':
                # Find good sources of target nutrient
                target_nutrient = nut[0]
                foods = self.evaluator.foods
                nutrient_sources = []
                for i, food in enumerate(foods):
                    if target_nutrient in food.nutrients:
                        density = food.get_nutrient_density(target_nutrient)
                        nutrient_sources.append((i, density))
                
                # Sort by nutrient density
                # nutrient_sources = sorted(nutrient_sources,key=lambda x: x[1], reverse=True)
                
                # Decrease 1-2 random foods slightly
                res_idx = [x[0] for x in nutrient_sources]
                for _ in range(random.randint(1, 2)):
                    idx = random.choice(res_idx[:5])
                    decrease = random.uniform(0, step_size)
                    new_quantities[idx] -= decrease

            elif nut[1]=='increase':
                # Find good sources of target nutrient
                target_nutrient = nut[0]
                foods = self.evaluator.foods
                nutrient_sources = []
                for i, food in enumerate(foods):
                    if target_nutrient in food.nutrients:
                        density = food.get_nutrient_density(target_nutrient)
                        nutrient_sources.append((i, density))
                
                # Sort by nutrient density
                # nutrient_sources = sorted(nutrient_sources,key=lambda x: x[1],reverse=True)

                # Increase 1-2 random foods slightly
                res_idx = [x[0] for x in nutrient_sources]
                for _ in range(random.randint(1, 2)):
                    idx = random.choice(res_idx[:5])
                    increase = random.uniform(0, step_size)
                    new_quantities[idx] += increase
        
        return Solution(new_quantities, self.evaluator)
    
    def get_total_cost(self) -> float:
        """Calculate total monthly cost."""
        total_cost = 0.0
        foods = self.evaluator.foods
        for qty, food in zip(self.quantities, foods):
            total_cost += qty * food.price
        return total_cost
    
    def get_nutrient_totals(self) -> Dict[str, float]:
        """Calculate monthly nutrient totals."""
        nutrients = {nut: 0.0 for nut in self.evaluator.min_req}
        foods = self.evaluator.foods
        
        for qty, food in zip(self.quantities, foods):
            for nut, per100g in food.nutrients.items():
                nutrients[nut] += per100g * 10 * qty  #**********
        
        return nutrients
    
    def get_daily_nutrient_totals(self) -> Dict[str, float]:
        """Calculate daily average nutrient intake."""
        monthly_totals = self.get_nutrient_totals()
        return {nut: total / 30 for nut, total in monthly_totals.items()}
    
    def get_deficient_nutrients(self) -> List[str]:
        """Get list of nutrients not satisfy requirements."""
        daily_nutrients = self.get_daily_nutrient_totals()
        min_daily = {nut: req/30 for nut, req in self.evaluator.min_req.items()}
        opt_daily = {nut: req/30 for nut, req in self.evaluator.optimal.items()}
        
        deficient = []
        for nutrient, actual in daily_nutrients.items():
            if self.directions[nutrient] == 'max':
                # For MAX nutrients (protein, fiber, calcium, iron):
                # - Heavy penalty if below minimum requirement
                # - Light bonus if between minimum and optimal (closer to minimum is better)
                # - Small penalty if above optimal (too much is still not ideal)
                
                if actual < min_daily[nutrient]:
                    deficient.append((nutrient,'increase'))
                elif actual <= opt_daily[nutrient]:
                    continue
                else:
                    deficient.append((nutrient,'decrease'))
                    
            else:  # direction == 'min'
                # For MIN nutrients (calories, fat, carbs):
                # - Heavy penalty if below optimal requirement
                # - Bonus if between minimum and optimal (closer to minimum is better)
                # - Small penalty if above minimum
                
                if actual < opt_daily[nutrient]:
                    deficient.append((nutrient,'increase'))
                elif actual <= min_daily[nutrient]:
                    continue
                else:
                    deficient.append((nutrient,'decrease'))
        
        return deficient
    
    def get_detailed_nutritional_analysis(self) -> Dict:
        """Get comprehensive nutritional analysis."""
        return self.evaluator.get_detailed_analysis(self)
    
    def __repr__(self) -> str:
        return f"Solution(fitness={self.fitness:.2f})"
    
    def __lt__(self, other: 'Solution') -> bool:
        return self.fitness < other.fitness