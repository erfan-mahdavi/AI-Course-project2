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
        """Calculate nutrient per unit cost."""
        if self.price <= 0 or nutrient not in self.nutrients:
            return 0.0
        return (self.nutrients[nutrient] / self.price) * 1000
    
    def get_protein_efficiency(self) -> float:
        """Grams of protein per 1000 Toman."""
        return self.get_nutrient_density('protein')

    def __repr__(self) -> str:
        return f"FoodItem(name='{self.name}', price={self.price})"
    
# ******************************************************************************

class Solution:
    """
    Represents a solution in the Simulated Annealing algorithm.
    Contains food quantities (kg per month) and fitness evaluation.
    """

    MIN_QUANTITY = 0.0
    MAX_QUANTITY = 15.0
    
    def __init__(self, quantities: List[float], evaluator: Callable[['Solution'], float]):
        self.quantities = self._validate_quantities(quantities.copy())
        self.evaluator = evaluator
        self.fitness = evaluator(self)
    
    def _validate_quantities(self, quantities: List[float]) -> List[float]:
        """Ensure quantities are within realistic bounds."""
        validated = []
        for qty in quantities:
            bounded_qty = max(self.MIN_QUANTITY, min(self.MAX_QUANTITY, float(qty)))
            validated.append(bounded_qty)
        return validated
    
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
        
        # step 2: Cost-Efficiency Ranking
        cost_efficiency_scores = cls._calculate_cost_efficiency(foods)
        
        # step 3: Balanced Food Selection Strategy
        quantities = cls._strategic_food_selection(
            foods, nutrient_priorities, cost_efficiency_scores, 
            budget_cap, max_qty
        )
        
        # step 4: Budget-Aware Quantity Optimization
        quantities = cls._optimize_quantities_for_budget(
            quantities, foods, budget_cap, min_req
        )
        
        return cls(quantities, evaluator)
    
    @staticmethod
    def _analyze_nutrient_priorities(foods: List, min_req: Dict[str, float]):
        """
        Algorithm 1: Analyze which foods are best sources for each nutrient.
        Returns top food sources for each nutrient sorted by efficiency.
        """
        nutrient_sources = {nutrient: [] for nutrient in min_req.keys()}
        
        for i, food in enumerate(foods):
            for nutrient in min_req.keys():
                if nutrient in food.nutrients and food.nutrients[nutrient] > 0:
                    # Calculate nutrient density (nutrient per unit cost)
                    density = food.get_nutrient_density(nutrient)
                    nutrient_sources[nutrient].append((i, density))
        
        # Sort each nutrient's sources by efficiency (highest first)
        for nutrient in nutrient_sources:
            nutrient_sources[nutrient].sort(key=lambda x: x[1], reverse=True)
        
        return nutrient_sources
    
    @staticmethod
    def _calculate_cost_efficiency(foods: List):
        """
        Algorithm 2: Calculate overall cost efficiency for each food.
        Considers protein density, calorie density, and overall nutritional value.
        """
        efficiency_scores = []
        
        for i, food in enumerate(foods):
            # Multi-factor efficiency calculation
            protein_efficiency = food.get_nutrient_density('protein')
            calorie_efficiency = food.get_nutrient_density('calories')
            
            # Calculate overall nutritional density
            total_nutrients = sum(food.nutrients.values())
            overall_efficiency = (total_nutrients / food.price) * 1000 if food.price > 0 else 0
            
            # Weighted efficiency score
            efficiency = (
                protein_efficiency * 0.4 +      # Protein is crucial
                calorie_efficiency * 0.3 +      # Calories for energy
                overall_efficiency * 0.3        # Overall nutrition
            )
            
            efficiency_scores.append((i, efficiency))
        
        # Sort by efficiency (highest first)
        efficiency_scores.sort(key=lambda x: x[1], reverse=True)
        return efficiency_scores
    
    @staticmethod
    def _strategic_food_selection(foods: List, nutrient_priorities: Dict, 
                                cost_efficiency: List, budget_cap: float, 
                                max_qty: float) -> List[float]:
        """
        Algorithm 3: Strategic food selection using multiple criteria.
        Ensures nutritional coverage while maintaining cost efficiency.
        """
        quantities = [0.0] * len(foods)
        selected_foods = set()
        
        # Step 1: Select essential high-priority foods for each nutrient
        critical_nutrients = ['protein', 'iron', 'calcium', 'fiber']  # Most important
        
        for nutrient in critical_nutrients:
            if nutrient in nutrient_priorities:
                # Select top 2 sources for each critical nutrient
                top_sources = nutrient_priorities[nutrient][:2]
                for food_idx, density in top_sources:
                    if density > 0.1:  # Meaningful density threshold
                        selected_foods.add(food_idx)
                        # Base quantity using gamma distribution
                        base_qty = np.random.gamma(2.0, 1.5)
                        quantities[food_idx] = min(max_qty, max(0.5, base_qty))
        
        # Step 2: Add cost-efficient staple foods
        top_efficient = [idx for idx, score in cost_efficiency[:8]]  # Top 8 efficient foods
        staple_count = 0
        
        for food_idx in top_efficient:
            food = foods[food_idx]
            # Identify staples (high carbs or calories, low cost)
            if (food.nutrients.get('carbs', 0) > 50 or 
                food.nutrients.get('calories', 0) > 300) and food.price < 100000:
                
                selected_foods.add(food_idx)
                # Staples get higher quantities
                staple_qty = np.random.gamma(3.0, 2.0)
                quantities[food_idx] = max(quantities[food_idx], 
                                         min(max_qty, max(1.0, staple_qty)))
                staple_count += 1
                if staple_count >= 3:  # Limit staples
                    break
        
        # Step 3: Add variety foods for balanced nutrition
        remaining_foods = [i for i in range(len(foods)) if i not in selected_foods]
        variety_count = min(len(remaining_foods), len(foods) // 4)  # Add 25% variety
        
        if variety_count > 0:
            # Weighted random selection based on efficiency
            efficiency_weights = []
            for food_idx in remaining_foods:
                # Find efficiency score
                efficiency = 1.0  # Default
                for idx, score in cost_efficiency:
                    if idx == food_idx:
                        efficiency = max(0.1, score)
                        break
                efficiency_weights.append(efficiency)
            
            # Normalize weights
            total_weight = sum(efficiency_weights)
            if total_weight > 0:
                weights = [w/total_weight for w in efficiency_weights]
                
                # Select variety foods using weighted probability
                selected_variety = np.random.choice(
                    remaining_foods, 
                    size=min(variety_count, len(remaining_foods)),
                    replace=False,
                    p=weights
                )
                
                for food_idx in selected_variety:
                    # Variety foods get smaller quantities
                    variety_qty = np.random.gamma(1.0, 0.8)
                    quantities[food_idx] = min(max_qty, max(0.1, variety_qty))
        
        return quantities
    
    @staticmethod
    def _optimize_quantities_for_budget(quantities: List[float], foods: List, 
                                      budget_cap: float, min_req: Dict[str, float]) -> List[float]:
        """
        Algorithm 4: Optimize quantities to fit budget while meeting requirements.
        Uses iterative scaling and smart reduction strategies.
        """
        max_iterations = 10
        
        for iteration in range(max_iterations):
            # Calculate current cost
            total_cost = sum(qty * foods[i].price for i, qty in enumerate(quantities))
            
            if total_cost <= budget_cap:
                break  # Within budget
            
            # Need to reduce cost
            overshoot_ratio = total_cost / budget_cap
            
            if overshoot_ratio > 1.5:  # Significant overshoot
                # Aggressive reduction: target expensive low-efficiency foods
                food_costs = [(i, qty * foods[i].price, foods[i].price) 
                            for i, qty in enumerate(quantities) if qty > 0]
                food_costs.sort(key=lambda x: x[2], reverse=True)  # Sort by price
                
                # Reduce most expensive foods first
                for i, total_cost_item, price in food_costs[:len(food_costs)//3]:
                    if quantities[i] > 0.5:
                        reduction = min(quantities[i] * 0.3, quantities[i] - 0.2)
                        quantities[i] -= reduction
            
            else:  # Moderate overshoot
                # Proportional reduction with smart priorities
                reduction_factor = 0.9 / overshoot_ratio
                
                for i in range(len(quantities)):
                    if quantities[i] > 0:
                        # Reduce less for highly nutritious foods
                        food = foods[i]
                        nutrition_score = (food.nutrients.get('protein', 0) * 2 + 
                                         food.nutrients.get('iron', 0) * 10 +
                                         food.nutrients.get('calcium', 0) * 0.1)
                        
                        # Higher nutrition score = less reduction
                        individual_factor = reduction_factor + (nutrition_score * 0.001)
                        individual_factor = min(0.95, max(0.7, individual_factor))
                        
                        quantities[i] *= individual_factor
                        quantities[i] = max(0.0, quantities[i])
        
        # Final cleanup: ensure minimum quantities for essential foods
        essential_nutrients = ['protein', 'iron']
        for nutrient in essential_nutrients:
            # Find best source of this nutrient that we're using
            best_source_idx = -1
            best_density = 0
            
            for i, qty in enumerate(quantities):
                if qty > 0 and nutrient in foods[i].nutrients:
                    density = foods[i].get_nutrient_density(nutrient)
                    if density > best_density:
                        best_density = density
                        best_source_idx = i
            
            # Ensure minimum quantity for best source
            if best_source_idx >= 0 and quantities[best_source_idx] < 0.5:
                quantities[best_source_idx] = max(0.5, quantities[best_source_idx])
        
        return quantities
    
    def copy(self) -> 'Solution':
        """Create a deep copy of this solution."""
        return Solution(self.quantities.copy(), self.evaluator)
    
    def recalculate_fitness(self) -> None:
        """Recalculate fitness after modifying quantities."""
        self.fitness = self.evaluator(self)
    
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
            if random.random() < 0.1:
                if random.random() < 0.5:
                    new_quantities[idx] = 0.0
                else:
                    new_quantities[idx] = random.uniform(0.1, 3.0)
        
        return Solution(new_quantities, self.evaluator)
    
    def perturb_focused(self, target_nutrient: str = None) -> 'Solution':
        """
        Create neighbor by focusing on a specific nutrient deficiency.
        """
        new_quantities = self.quantities.copy()
        
        if target_nutrient is None:
            # Find most deficient nutrient
            daily_nutrients = self.get_daily_nutrient_totals()
            min_daily = {nut: req/30 for nut, req in self.evaluator.min_req.items()}
            
            max_deficit = 0
            target_nutrient = 'protein'  # default
            for nut, actual in daily_nutrients.items():
                deficit = (min_daily[nut] - actual) / min_daily[nut]
                if deficit > max_deficit:
                    max_deficit = deficit
                    target_nutrient = nut
        
        # Find good sources of target nutrient
        foods = self.evaluator.foods
        nutrient_sources = []
        for i, food in enumerate(foods):
            if target_nutrient in food.nutrients:
                density = food.get_nutrient_density(target_nutrient)
                nutrient_sources.append((i, density))
        
        # Sort by nutrient density
        nutrient_sources.sort(key=lambda x: x[1], reverse=True)
        
        # Increase top 2 sources
        for idx, _ in nutrient_sources[:2]:
            increase = random.uniform(0.2, 1.0)
            new_quantities[idx] += increase
        
        # Decrease 1-2 random foods slightly
        for _ in range(random.randint(1, 2)):
            idx = random.randint(0, len(new_quantities) - 1)
            if new_quantities[idx] > 0.5:
                decrease = random.uniform(0.1, 0.5)
                new_quantities[idx] -= decrease
        
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
                nutrients[nut] += per100g * 10 * qty
        
        return nutrients
    
    def get_daily_nutrient_totals(self) -> Dict[str, float]:
        """Calculate daily average nutrient intake."""
        monthly_totals = self.get_nutrient_totals()
        return {nut: total / 30 for nut, total in monthly_totals.items()}
    
    def get_deficient_nutrients(self) -> List[str]:
        """Get list of nutrients below minimum requirements."""
        daily_nutrients = self.get_daily_nutrient_totals()
        min_daily = {nut: req/30 for nut, req in self.evaluator.min_req.items()}
        
        deficient = []
        for nut, actual in daily_nutrients.items():
            if actual < min_daily[nut]:
                deficient.append(nut)
        
        return deficient
    
    def get_detailed_nutritional_analysis(self) -> Dict:
        """Get comprehensive nutritional analysis."""
        return self.evaluator.get_detailed_analysis(self)
    
    def __repr__(self) -> str:
        return f"Solution(fitness={self.fitness:.2f})"
    
    def __lt__(self, other: 'Solution') -> bool:
        return self.fitness < other.fitness