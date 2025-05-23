"""
Models Module for Diet Optimization - FIXED VERSION

This module contains model classes used in Simulated Annealing
approach to diet optimization. The classes are designed to be flexible and work with
SA optimization method.

Classes:
    FoodItem: Represents a single food item with nutritional data and price
    Solution: Represents a diet solution with enhanced functionality for SA
"""

import random
from typing import List, Callable, Dict, Optional
import numpy as np

class FoodItem:
    """
    Represents a single food item with its nutritional information and price.
    
    This class stores all the necessary information about a food item that
    SA algorithm needs for optimization, including:
    - Nutritional content per 100g
    - Price per kilogram
    - Food name for identification
    """
    
    def __init__(self, name: str, nutrients: Dict[str, float], price_per_kg: float):
        """
        Initialize a food item with nutritional data and pricing.
        
        Args:
            name: Name of the food item (e.g., "Rice", "Chicken Breast")
            nutrients: Dictionary of nutrient content per 100g
                      Expected keys: calories, protein, fat, carbs, fiber, calcium, iron
            price_per_kg: Price in Toman per kilogram
        """
        self.name = name
        self.nutrients = nutrients  # Nutritional values per 100g
        self.price = max(0.01, price_per_kg)   # Ensure positive price
        
        # Validate nutrients
        for key, value in self.nutrients.items():
            if value < 0:
                self.nutrients[key] = 0.0

    def get_nutrient_density(self, nutrient: str) -> float:
        """
        Calculate nutrient density (nutrient per unit cost).
        
        This metric helps identify cost-effective sources of specific nutrients.
        
        Args:
            nutrient: Name of the nutrient (e.g., 'protein', 'calcium')
            
        Returns:
            Nutrient amount per 1000 Toman (for cost comparison)
        """
        if self.price <= 0 or nutrient not in self.nutrients:
            return 0.0
        return (self.nutrients[nutrient] / self.price) * 1000
    
    def get_protein_efficiency(self) -> float:
        """
        Calculate protein per unit cost (protein value metric).
        
        Returns:
            Grams of protein per 1000 Toman
        """
        return self.get_nutrient_density('protein')

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"FoodItem(name='{self.name}', nutrients={self.nutrients}, price={self.price})"

# **************************************************************************************************
class Solution:
    """
    Represents a single solution in the Simulated Annealing algorithm - FIXED VERSION.
    
    A solution consists of food quantities (kg per month) for each available food item.
    This class is designed to work with SA algorithm, providing enhanced
    functionality for solution analysis and manipulation.
    
    IMPROVEMENTS:
    - Realistic quantity bounds and validation
    - Better perturbation strategies with proper bounds checking
    - Simplified and more logical cost-aware adjustments
    - Enhanced local optimization with convergence detection
    - Input validation and error handling
    """
    
    # Class constants for realistic bounds
    MIN_QUANTITY = 0.0          # Minimum quantity (kg per month)
    MAX_QUANTITY = 15.0         # Maximum realistic quantity per food (kg per month)
    MAX_DAILY_QUANTITY = 0.5    # Maximum daily quantity per food (kg per day = 15kg/month)
    
    def __init__(self, quantities: List[float], evaluator: Callable[['Solution'], float]):
        """
        Initialize a solution with given food quantities.
        
        Args:
            quantities: List of food quantities in kg per month for each food item
            evaluator: Fitness evaluation function that takes a Solution and returns fitness score
        """
        # Validate and bound quantities
        self.quantities = self._validate_quantities(quantities.copy())
        self.evaluator = evaluator
        self.fitness = evaluator(self)       # Calculate initial fitness score
    
    def _validate_quantities(self, quantities: List[float]) -> List[float]:
        """
        Validate and bound quantities to realistic ranges.
        
        Args:
            quantities: Raw quantities to validate
            
        Returns:
            Validated and bounded quantities
        """
        validated = []
        for qty in quantities:
            # Ensure non-negative and within realistic bounds
            bounded_qty = max(self.MIN_QUANTITY, min(self.MAX_QUANTITY, float(qty)))
            validated.append(bounded_qty)
        return validated
    
    @classmethod
    def random_solution(cls, num_foods: int, evaluator: Callable, max_qty: Optional[float] = None) -> 'Solution':
        """
        Create a random solution with realistic food quantities - IMPROVED VERSION.
        
        This factory method generates initial solutions for optimization algorithms.
        Uses more realistic distribution and constraints.
        
        Args:
            num_foods: Number of food items available
            evaluator: Fitness evaluation function
            max_qty: Maximum quantity for each food item in kg per month
            
        Returns:
            A new Solution with realistic random food quantities
        """
        if max_qty is None:
            max_qty = cls.MAX_QUANTITY
        
        quantities = []
        
        # Ensure at least some foods are selected (avoid all-zero solutions)
        min_selected_foods = max(3, num_foods // 5)  # At least 20% of foods or minimum 3
        selected_indices = random.sample(range(num_foods), min_selected_foods)
        
        for i in range(num_foods):
            if i in selected_indices:
                # Selected foods get realistic quantities
                # Use gamma distribution for more realistic food quantity distribution
                # Most foods: 0.5-3kg, few foods: higher quantities
                shape = 1.5  # Shape parameter for gamma distribution
                scale = 2.0  # Scale parameter
                qty = np.random.gamma(shape, scale)
                qty = min(max_qty, max(0.1, qty))  # Bound between 0.1 and max_qty
            else:
                # Non-selected foods have a small chance of being included
                if random.random() < 0.2:  # 20% chance
                    qty = random.uniform(0.1, 1.0)  # Small quantity
                else:
                    qty = 0.0
            
            quantities.append(qty)
        
        return cls(quantities, evaluator)
    
    def copy(self) -> 'Solution':
        """
        Create a deep copy of this solution.
        
        This method creates an independent copy that can be modified
        without affecting the original solution.
        
        Returns:
            A new Solution object with identical quantities
        """
        return Solution(self.quantities.copy(), self.evaluator)
    
    def recalculate_fitness(self) -> None:
        """
        Recalculate the fitness score for this solution.
        
        This method should be called whenever the solution's quantities
        have been modified and the fitness needs to be updated.
        """
        self.quantities = self._validate_quantities(self.quantities)
        self.fitness = self.evaluator(self)
    
    def perturb_random(self, step_size: float = 1.5, num_changes: Optional[int] = None) -> 'Solution':
        """
        Create a neighbor solution by randomly perturbing some food quantities - IMPROVED.
        
        This is one of the core methods for generating neighboring solutions
        in the SA algorithm. It applies random changes to a subset of food items.
        
        Args:
            step_size: Maximum change in quantity (kg) for each modified food
            num_changes: Number of food items to modify (random if None)
            
        Returns:
            A new Solution with perturbed quantities
        """
        new_quantities = self.quantities.copy()
        
        # Determine how many food items to modify (more conservative)
        if num_changes is None:
            # Modify 1-3 foods for more focused perturbation
            num_changes = random.randint(1, min(3, len(new_quantities)))
        
        # Select random food items to modify
        indices_to_change = random.sample(range(len(new_quantities)), num_changes)
        
        for idx in indices_to_change:
            current_qty = new_quantities[idx]
            
            # Apply random change within step_size bounds
            change = random.uniform(-step_size, step_size)
            new_qty = current_qty + change
            
            # Occasionally set to a completely new random value (exploration)
            if random.random() < 0.1:  # Reduced from 0.05 to 0.1 for more exploration
                new_qty = random.uniform(0, min(5.0, self.MAX_QUANTITY))
            
            # Ensure bounds are respected
            new_quantities[idx] = max(self.MIN_QUANTITY, min(self.MAX_QUANTITY, new_qty))
        
        return Solution(new_quantities, self.evaluator)
    
    def perturb_focused(self, focus_nutrient: str, step_size: float = 1.5) -> 'Solution':
        """
        Create a neighbor solution by focusing on improving a specific nutrient - IMPROVED.
        
        This perturbation method is more intelligent than random perturbation.
        It identifies foods that are good sources of the target nutrient and
        increases their quantities while potentially decreasing others.
        
        Args:
            focus_nutrient: Name of nutrient to focus on (e.g., 'protein', 'iron')
            step_size: Maximum change in quantity per food item
            
        Returns:
            A new Solution with nutrient-focused modifications
        """
        new_quantities = self.quantities.copy()
        foods = self.evaluator.foods
        
        # Find foods that are good sources of the target nutrient
        nutrient_sources = []
        for i, food in enumerate(foods):
            if focus_nutrient in food.nutrients and food.nutrients[focus_nutrient] > 0:
                nutrient_density = food.get_nutrient_density(focus_nutrient)
                # Also consider absolute nutrient content, not just cost efficiency
                absolute_content = food.nutrients[focus_nutrient]
                # Combined score: 70% efficiency, 30% absolute content
                combined_score = 0.7 * nutrient_density + 0.3 * absolute_content
                nutrient_sources.append((i, combined_score, absolute_content))
        
        if not nutrient_sources:
            # Fallback to random perturbation if no sources found
            return self.perturb_random(step_size)
        
        # Sort by combined score (best sources first)
        nutrient_sources.sort(key=lambda x: x[1], reverse=True)
        
        # Increase quantities of top nutrient sources (more conservative)
        top_sources = nutrient_sources[:2]  # Top 2 sources instead of 3
        for idx, score, content in top_sources:
            if new_quantities[idx] < self.MAX_QUANTITY * 0.8:  # Don't increase if already high
                increase = random.uniform(0.1, step_size)
                new_quantities[idx] = min(self.MAX_QUANTITY, new_quantities[idx] + increase)
        
        # Optionally decrease quantities of poor sources (more conservative)
        if len(nutrient_sources) > 3:
            poor_sources = nutrient_sources[-2:]  # Bottom 2 sources
            for idx, score, content in poor_sources:
                if new_quantities[idx] > 0.5:  # Only decrease if quantity is reasonable
                    decrease = random.uniform(0, min(step_size * 0.5, new_quantities[idx] * 0.3))
                    new_quantities[idx] = max(self.MIN_QUANTITY, new_quantities[idx] - decrease)
        
        return Solution(new_quantities, self.evaluator)
    
    def perturb_cost_aware(self, budget_pressure: float = 0.5, step_size: float = 1.5) -> 'Solution':
        """
        Create a neighbor solution with cost-aware perturbations - SIMPLIFIED AND IMPROVED.
        
        This method considers food prices when making changes, preferring
        to increase cheaper foods and decrease expensive ones when cost
        is a concern.
        
        Args:
            budget_pressure: How much to emphasize cost (0.0 to 1.0)
            step_size: Maximum change in quantity per food item
            
        Returns:
            A new Solution with cost-aware modifications
        """
        new_quantities = self.quantities.copy()
        foods = self.evaluator.foods
        current_cost = self.get_total_cost()
        budget_cap = self.evaluator.cost_cap
        
        # Calculate cost pressure (how close we are to budget limit)
        cost_ratio = current_cost / budget_cap if budget_cap > 0 else 0
        
        # Simplified logic: if over budget or close to budget, reduce expensive foods
        if cost_ratio > 0.8:  # If using more than 80% of budget
            # Focus on reducing expensive foods
            food_costs = [(i, food.price) for i, food in enumerate(foods)]
            # Sort by price (most expensive first)
            food_costs.sort(key=lambda x: x[1], reverse=True)
            
            # Reduce expensive foods that have significant quantities
            expensive_foods = food_costs[:len(foods)//2]  # Top half by price
            for idx, price in expensive_foods:
                if new_quantities[idx] > 0.2:  # Only if has reasonable quantity
                    if random.random() < 0.4:  # 40% chance to reduce
                        decrease = random.uniform(0.1, min(step_size, new_quantities[idx] * 0.4))
                        new_quantities[idx] = max(self.MIN_QUANTITY, new_quantities[idx] - decrease)
            
            # Slightly increase some cheap, nutritious foods
            cheap_foods = food_costs[-len(foods)//3:]  # Bottom third by price
            for idx, price in cheap_foods:
                if new_quantities[idx] < self.MAX_QUANTITY * 0.7:  # If not already high
                    if random.random() < 0.3:  # 30% chance to increase
                        increase = random.uniform(0.1, step_size * 0.7)
                        new_quantities[idx] = min(self.MAX_QUANTITY, new_quantities[idx] + increase)
        
        else:
            # Normal perturbation when cost pressure is low
            return self.perturb_random(step_size)
        
        return Solution(new_quantities, self.evaluator)
    
    def local_optimization_step(self, step_size: float = 0.8) -> 'Solution':
        """
        Perform a local optimization step to fine-tune the solution - IMPROVED.
        
        This method makes small, calculated adjustments to improve the
        solution incrementally. It's useful for fine-tuning near-optimal solutions.
        
        Args:
            step_size: Size of optimization steps
            
        Returns:
            A locally optimized Solution
        """
        new_quantities = self.quantities.copy()
        
        try:
            # Get current nutritional status
            current_analysis = self.get_detailed_nutritional_analysis()
            
            # Identify deficient and excessive nutrients
            deficient_nutrients = []
            excessive_nutrients = []
            
            nutritional_status = current_analysis.get('nutritional_status', {})
            for nutrient, status in nutritional_status.items():
                status_val = status.get('status', 'UNKNOWN') if isinstance(status, dict) else str(status)
                if status_val == 'DEFICIENT':
                    deficient_nutrients.append(nutrient)
                elif status_val == 'EXCESSIVE':
                    excessive_nutrients.append(nutrient)
            
            # Focus on the most critical deficiency
            foods = self.evaluator.foods
            if deficient_nutrients:
                target_nutrient = deficient_nutrients[0]  # Most critical
                
                # Find best sources of this nutrient
                best_sources = []
                for i, food in enumerate(foods):
                    if target_nutrient in food.nutrients and food.nutrients[target_nutrient] > 0:
                        efficiency = food.get_nutrient_density(target_nutrient)
                        content = food.nutrients[target_nutrient]
                        # Balance efficiency and content
                        score = 0.6 * efficiency + 0.4 * content
                        best_sources.append((i, score))
                
                if best_sources:
                    best_sources.sort(key=lambda x: x[1], reverse=True)
                    
                    # Increase top source slightly
                    for idx, score in best_sources[:1]:  # Only top source
                        if new_quantities[idx] < self.MAX_QUANTITY * 0.9:
                            increase = random.uniform(0.1, step_size)
                            new_quantities[idx] = min(self.MAX_QUANTITY, new_quantities[idx] + increase)
                            break  # Only modify one food for local optimization
            
            # Handle excesses more conservatively
            if excessive_nutrients and random.random() < 0.3:  # Only sometimes
                target_nutrient = excessive_nutrients[0]
                
                # Find foods contributing most to this nutrient
                contributors = []
                for i, (qty, food) in enumerate(zip(new_quantities, foods)):
                    if qty > 0.2 and target_nutrient in food.nutrients:
                        contribution = qty * food.nutrients[target_nutrient] * 10  # Monthly contribution
                        contributors.append((i, contribution))
                
                if contributors:
                    contributors.sort(key=lambda x: x[1], reverse=True)
                    
                    # Slightly reduce top contributor
                    idx, contribution = contributors[0]
                    if new_quantities[idx] > 0.3:
                        decrease = random.uniform(0.1, min(step_size * 0.5, new_quantities[idx] * 0.2))
                        new_quantities[idx] = max(self.MIN_QUANTITY, new_quantities[idx] - decrease)
        
        except Exception as e:
            # If analysis fails, fallback to small random perturbation
            return self.perturb_random(step_size * 0.5)
        
        return Solution(new_quantities, self.evaluator)
    
    def get_total_cost(self) -> float:
        """
        Calculate the total monthly cost of this solution.
        
        Returns:
            Total cost in Toman for the monthly food basket
        """
        total_cost = 0.0
        foods = self.evaluator.foods
        
        for qty, food in zip(self.quantities, foods):
            total_cost += qty * food.price
            
        return total_cost
    
    def get_total_weight(self) -> float:
        """
        Calculate the total weight of food in this solution.
        
        Returns:
            Total weight in kg for the monthly food basket
        """
        return sum(self.quantities)
    
    def get_nutrient_totals(self) -> Dict[str, float]:
        """
        Calculate the total monthly nutrient intake for this solution.
        
        Returns:
            Dictionary with nutrient names as keys and monthly totals as values
        """
        nutrients = {nut: 0.0 for nut in self.evaluator.min_req}
        foods = self.evaluator.foods
        
        for qty, food in zip(self.quantities, foods):
            for nut, per100g in food.nutrients.items():
                # Convert per 100g to per kg (multiply by 10)
                nutrients[nut] += per100g * 10 * qty
                
        return nutrients
    
    def get_daily_nutrient_totals(self) -> Dict[str, float]:
        """
        Calculate the daily average nutrient intake for this solution.
        
        Returns:
            Dictionary with nutrient names as keys and daily averages as values
        """
        monthly_totals = self.get_nutrient_totals()
        return {nut: total / 30 for nut, total in monthly_totals.items()}
    
    def get_detailed_nutritional_analysis(self) -> Dict:
        """
        Get a comprehensive nutritional analysis of this solution.
        
        This method provides the same detailed analysis as the fitness evaluator
        but can be called directly on the solution for convenience.
        
        Returns:
            Dictionary with detailed nutritional analysis
        """
        try:
            return self.evaluator.get_detailed_analysis(self)
        except Exception as e:
            # Return basic analysis if detailed analysis fails
            return {
                'total_monthly_cost': self.get_total_cost(),
                'monthly_nutrients': self.get_nutrient_totals(),
                'daily_nutrients': self.get_daily_nutrient_totals(),
                'nutritional_status': {},
                'error': str(e)
            }
    
    def get_deficient_nutrients(self) -> List[str]:
        """
        Get a list of nutrients that are below minimum requirements.
        
        Returns:
            List of nutrient names that are deficient
        """
        daily_nutrients = self.get_daily_nutrient_totals()
        min_daily = {nut: req/30 for nut, req in self.evaluator.min_req.items()}
        directions = getattr(self.evaluator, 'directions', {})
        
        deficient = []
        for nut, actual in daily_nutrients.items():
            minimum = min_daily.get(nut, 0)
            direction = directions.get(nut, 'max')
            
            if direction == 'max' and actual < minimum:
                deficient.append(nut)
        
        return deficient
    
    def is_feasible(self, tolerance: float = 0.05) -> bool:
        """
        Check if the solution is feasible (meets basic constraints).
        
        Args:
            tolerance: Tolerance for constraint violations (5% by default)
            
        Returns:
            True if solution is feasible
        """
        # Check budget constraint
        total_cost = self.get_total_cost()
        budget_cap = self.evaluator.cost_cap
        if total_cost > budget_cap * (1 + tolerance):
            return False
        
        # Check if any critical nutrients are severely deficient
        daily_nutrients = self.get_daily_nutrient_totals()
        min_daily = {nut: req/30 for nut, req in self.evaluator.min_req.items()}
        
        for nut, actual in daily_nutrients.items():
            minimum = min_daily.get(nut, 0)
            if minimum > 0 and actual < minimum * (1 - tolerance):
                return False
        
        return True
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"Solution(quantities={[round(q, 2) for q in self.quantities]}, fitness={self.fitness:.2f})"
    
    def __lt__(self, other: 'Solution') -> bool:
        """Less than comparison based on fitness (for sorting)."""
        return self.fitness < other.fitness
    
    def __le__(self, other: 'Solution') -> bool:
        """Less than or equal comparison based on fitness."""
        return self.fitness <= other.fitness
    
    def __gt__(self, other: 'Solution') -> bool:
        """Greater than comparison based on fitness."""
        return self.fitness > other.fitness
    
    def __ge__(self, other: 'Solution') -> bool:
        """Greater than or equal comparison based on fitness."""
        return self.fitness >= other.fitness
    
    def __eq__(self, other: 'Solution') -> bool:
        """Equality comparison based on fitness."""
        return abs(self.fitness - other.fitness) < 1e-6
    
    def __hash__(self) -> int:
        """Hash function for using solutions in sets/dictionaries."""
        return hash(tuple(round(q, 4) for q in self.quantities))