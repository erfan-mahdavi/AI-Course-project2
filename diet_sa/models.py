"""
Models Module for Diet Optimization

This module contains model classes used in both Genetic Algorithm and Simulated Annealing
approaches to diet optimization. The classes are designed to be flexible and work with
both optimization methods.

Classes:
    FoodItem: Represents a single food item with nutritional data and price
    Solution: Represents a diet solution with enhanced functionality for SA
"""

import random
import math
from typing import List, Callable, Dict
import tabulate
import numpy as np

class FoodItem:
    """
    Represents a single food item with its nutritional information and price.
    
    This class stores all the necessary information about a food item that
    both GA and SA algorithms need for optimization, including:
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
        self.price = price_per_kg   # Price in Toman per kg
    
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
    
    def get_calorie_density(self) -> float:
        """
        Calculate caloric density (calories per kg).
        
        Returns:
            Calories per kilogram of this food
        """
        return self.nutrients.get('calories', 0) * 10  # Convert per 100g to per kg
    
    def get_protein_efficiency(self) -> float:
        """
        Calculate protein per unit cost (protein value metric).
        
        Returns:
            Grams of protein per 1000 Toman
        """
        return self.get_nutrient_density('protein')
    
    def __str__(self) -> str:
        """String representation of the food item."""
        return f"FoodItem({self.name}, {self.price:,}T/kg)"
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"FoodItem(name='{self.name}', nutrients={self.nutrients}, price={self.price})"

# **************************************************************************************************
class Solution:
    """
    Represents a single solution in the Simulated Annealing algorithm.
    
    A solution consists of food quantities (kg per month) for each available food item.
    This class is designed to work with both GA and SA algorithms, providing enhanced
    functionality for solution analysis and manipulation.
    
    Key features:
    - Enhanced perturbation methods for neighbor generation
    - Detailed nutritional analysis capabilities
    - Cost and budget tracking
    - Food diversity metrics
    """
    
    def __init__(self, quantities: List[float], evaluator: Callable[['Solution'], float]):
        """
        Initialize a solution with given food quantities.
        
        Args:
            quantities: List of food quantities in kg per month for each food item
            evaluator: Fitness evaluation function that takes a Solution and returns fitness score
        """
        self.quantities = quantities.copy()  # Monthly quantities (kg) for each food item
        self.evaluator = evaluator           # Fitness evaluation function
        self.fitness = evaluator(self)       # Calculate initial fitness score
    
    @classmethod
    def random_solution(cls, num_foods: int, evaluator: Callable, max_qty: float = 30.0) -> 'Solution':
        """
        Create a random solution with random quantities for each food item.
        
        This factory method generates initial solutions for optimization algorithms.
        It uses reasonable bounds to avoid extreme solutions that would be
        immediately rejected.
        
        Args:
            num_foods: Number of food items available
            evaluator: Fitness evaluation function
            max_qty: Maximum quantity for each food item in kg per month
            
        Returns:
            A new Solution with random food quantities
        """
        # Generate random quantities with bias toward smaller values
        # This creates more realistic initial solutions
        quantities = []
        for _ in range(num_foods):
            # Use exponential distribution to favor smaller quantities
            # Most foods will have small quantities, few will have large quantities
            if random.random() < 0.3:  # 30% chance of zero quantity
                qty = 0.0
            else:
                # Exponential distribution for realistic quantity distribution
                qty = min(max_qty, np.random.exponential(scale=max_qty/4))
            quantities.append(qty)
        
        return cls(quantities, evaluator)
    
    @classmethod
    def from_food_priorities(cls, num_foods: int, evaluator: Callable, 
                           food_priorities: List[float] = None) -> 'Solution':
        """
        Create a solution based on food priority weights.
        
        This method creates more informed initial solutions by considering
        food quality metrics like protein content, nutrient density, etc.
        
        Args:
            num_foods: Number of food items available
            evaluator: Fitness evaluation function
            food_priorities: Optional list of priority weights for each food
            
        Returns:
            A new Solution with priority-weighted quantities
        """
        if food_priorities is None:
            food_priorities = [1.0] * num_foods
        
        quantities = []
        for i in range(num_foods):
            priority = food_priorities[i]
            # Higher priority foods get higher base quantities
            base_qty = priority * random.uniform(1, 10)
            # Add some randomness
            qty = max(0, base_qty + random.uniform(-2, 2))
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
        self.fitness = self.evaluator(self)
    
    def perturb_random(self, step_size: float = 2.0, num_changes: int = None) -> 'Solution':
        """
        Create a neighbor solution by randomly perturbing some food quantities.
        
        This is one of the core methods for generating neighboring solutions
        in the SA algorithm. It applies random changes to a subset of food items.
        
        Args:
            step_size: Maximum change in quantity (kg) for each modified food
            num_changes: Number of food items to modify (random if None)
            
        Returns:
            A new Solution with perturbed quantities
        """
        new_quantities = self.quantities.copy()
        
        # Determine how many food items to modify
        if num_changes is None:
            num_changes = random.randint(1, min(5, len(new_quantities)))
        
        # Select random food items to modify
        indices_to_change = random.sample(range(len(new_quantities)), num_changes)
        
        for idx in indices_to_change:
            # Apply random change within step_size bounds
            change = random.uniform(-step_size, step_size)
            new_quantities[idx] = max(0.0, new_quantities[idx] + change)
            
            # Occasionally set to zero (helps remove unnecessary foods)
            if random.random() < 0.1:
                new_quantities[idx] = 0.0
            
            # Occasionally set to a completely new random value (exploration)
            if random.random() < 0.05:
                new_quantities[idx] = random.uniform(0, 25.0)
        
        return Solution(new_quantities, self.evaluator)
    
    def perturb_focused(self, focus_nutrient: str, step_size: float = 2.0) -> 'Solution':
        """
        Create a neighbor solution by focusing on improving a specific nutrient.
        
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
            if focus_nutrient in food.nutrients:
                nutrient_density = food.get_nutrient_density(focus_nutrient)
                nutrient_sources.append((i, nutrient_density))
        
        # Sort by nutrient density (best sources first)
        nutrient_sources.sort(key=lambda x: x[1], reverse=True)
        
        # Increase quantities of top nutrient sources
        top_sources = nutrient_sources[:3]  # Top 3 sources
        for idx, _ in top_sources:
            increase = random.uniform(0, step_size)
            new_quantities[idx] += increase
        
        # Optionally decrease quantities of poor sources
        if len(nutrient_sources) > 3:
            poor_sources = nutrient_sources[-2:]  # Bottom 2 sources
            for idx, _ in poor_sources:
                if new_quantities[idx] > 0:
                    decrease = random.uniform(0, min(step_size, new_quantities[idx]))
                    new_quantities[idx] -= decrease
        
        return Solution(new_quantities, self.evaluator)
    
    def perturb_cost_aware(self, budget_pressure: float = 0.5, step_size: float = 2.0) -> 'Solution':
        """
        Create a neighbor solution with cost-aware perturbations.
        
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
        cost_ratio = current_cost / budget_cap
        actual_pressure = budget_pressure * min(1.0, cost_ratio)
        
        if actual_pressure > 0.3:  # High cost pressure
            # Favor cheaper foods, reduce expensive ones
            food_costs = [(i, food.price) for i, food in enumerate(foods)]
            food_costs.sort(key=lambda x: x[1])  # Sort by price (cheapest first)
            
            # Increase cheaper foods
            cheap_foods = food_costs[:len(foods)//3]  # Cheapest third
            for idx, _ in cheap_foods:
                if random.random() < 0.4:  # 40% chance to increase
                    increase = random.uniform(0, step_size)
                    new_quantities[idx] += increase
            
            # Decrease expensive foods
            expensive_foods = food_costs[-len(foods)//3:]  # Most expensive third
            for idx, _ in expensive_foods:
                if new_quantities[idx] > 0 and random.random() < 0.6:  # 60% chance to decrease
                    decrease = random.uniform(0, min(step_size, new_quantities[idx]))
                    new_quantities[idx] -= decrease
        else:
            # Normal random perturbation when cost pressure is low
            return self.perturb_random(step_size)
        
        return Solution(new_quantities, self.evaluator)
    
    def local_optimization_step(self, step_size: float = 1.0) -> 'Solution':
        """
        Perform a local optimization step to fine-tune the solution.
        
        This method makes small, calculated adjustments to improve the
        solution incrementally. It's useful for fine-tuning near-optimal solutions.
        
        Args:
            step_size: Size of optimization steps
            
        Returns:
            A locally optimized Solution
        """
        new_quantities = self.quantities.copy()
        
        # Get current nutritional status
        current_analysis = self.get_detailed_nutritional_analysis()
        
        # Identify deficient nutrients
        deficient_nutrients = []
        excessive_nutrients = []
        
        for nutrient, status in current_analysis['nutritional_status'].items():
            if status['status'] == 'DEFICIENT':
                deficient_nutrients.append(nutrient)
            elif status['status'] == 'EXCESSIVE':
                excessive_nutrients.append(nutrient)
        
        # Try to fix deficiencies
        foods = self.evaluator.foods
        for nutrient in deficient_nutrients[:2]:  # Focus on top 2 deficiencies
            # Find best sources of this nutrient
            best_sources = []
            for i, food in enumerate(foods):
                if nutrient in food.nutrients:
                    efficiency = food.get_nutrient_density(nutrient)
                    best_sources.append((i, efficiency))
            
            best_sources.sort(key=lambda x: x[1], reverse=True)
            
            # Increase top sources slightly
            for idx, _ in best_sources[:2]:
                increase = random.uniform(0, step_size)
                new_quantities[idx] += increase
        
        # Try to reduce excesses
        for nutrient in excessive_nutrients[:1]:  # Focus on top excess
            # Find foods high in this nutrient and reduce them
            high_contributors = []
            for i, (qty, food) in enumerate(zip(new_quantities, foods)):
                if qty > 0 and nutrient in food.nutrients:
                    contribution = qty * food.nutrients[nutrient] * 10  # Monthly contribution
                    high_contributors.append((i, contribution))
            
            high_contributors.sort(key=lambda x: x[1], reverse=True)
            
            # Reduce top contributors slightly
            for idx, _ in high_contributors[:2]:
                if new_quantities[idx] > 0:
                    decrease = random.uniform(0, min(step_size, new_quantities[idx]))
                    new_quantities[idx] -= decrease
        
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
    
    def is_within_budget(self) -> bool:
        """
        Check if this solution is within the specified budget.
        
        Returns:
            True if solution cost is within budget, False otherwise
        """
        return self.get_total_cost() <= self.evaluator.cost_cap
    
    def get_budget_utilization(self) -> float:
        """
        Calculate what percentage of the budget this solution uses.
        
        Returns:
            Budget utilization as a percentage (0.0 to 100.0+)
        """
        return (self.get_total_cost() / self.evaluator.cost_cap) * 100
    
    def get_detailed_nutritional_analysis(self) -> Dict:
        """
        Get a comprehensive nutritional analysis of this solution.
        
        This method provides the same detailed analysis as the fitness evaluator
        but can be called directly on the solution for convenience.
        
        Returns:
            Dictionary with detailed nutritional analysis
        """
        return self.evaluator.get_detailed_analysis(self)
    
    def meets_minimum_requirements(self) -> bool:
        """
        Check if this solution meets all minimum nutritional requirements.
        
        Returns:
            True if all minimum requirements are met, False otherwise
        """
        daily_nutrients = self.get_daily_nutrient_totals()
        min_daily = {nut: req/30 for nut, req in self.evaluator.min_req.items()}
        directions = self.evaluator.directions
        
        for nut, actual in daily_nutrients.items():
            minimum = min_daily[nut]
            direction = directions.get(nut, 'max')
            
            # For 'max' nutrients, check if we meet minimum
            if direction == 'max' and actual < minimum:
                return False
        
        return True
    
    def get_deficient_nutrients(self) -> List[str]:
        """
        Get a list of nutrients that are below minimum requirements.
        
        Returns:
            List of nutrient names that are deficient
        """
        daily_nutrients = self.get_daily_nutrient_totals()
        min_daily = {nut: req/30 for nut, req in self.evaluator.min_req.items()}
        directions = self.evaluator.directions
        
        deficient = []
        for nut, actual in daily_nutrients.items():
            minimum = min_daily[nut]
            direction = directions.get(nut, 'max')
            
            if direction == 'max' and actual < minimum:
                deficient.append(nut)
        
        return deficient
    
    def get_food_diversity_score(self) -> float:
        """
        Calculate a diversity score based on how many different foods are used.
        
        Higher diversity is generally better for nutritional completeness
        and meal variety.
        
        Returns:
            Diversity score (0.0 to 1.0, higher is more diverse)
        """
        # Count foods with significant quantities
        significant_foods = sum(1 for qty in self.quantities if qty > 0.1)
        total_foods = len(self.quantities)
        
        if total_foods == 0:
            return 0.0
        
        # Basic diversity metric
        basic_diversity = significant_foods / total_foods
        
        # Calculate Shannon entropy for more sophisticated diversity measure
        total_weight = self.get_total_weight()
        if total_weight <= 0:
            return basic_diversity
        
        entropy = 0.0
        for qty in self.quantities:
            if qty > 0:
                proportion = qty / total_weight
                entropy -= proportion * math.log(proportion)
        
        # Normalize entropy (maximum entropy is log(n) where n is number of foods)
        max_entropy = math.log(total_foods)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Combine basic diversity and entropy
        return (basic_diversity + normalized_entropy) / 2
    
    def format_solution_summary(self) -> str:
        """
        Create a formatted string summary of this solution.
        
        Returns:
            Multi-line string with solution overview
        """
        lines = []
        lines.append("="*60)
        lines.append("SOLUTION SUMMARY")
        lines.append("="*60)
        
        # Basic metrics
        total_cost = self.get_total_cost()
        total_weight = self.get_total_weight()
        budget_utilization = self.get_budget_utilization()
        
        lines.append(f"Fitness Score: {self.fitness:.2f}")
        lines.append(f"Total Cost: {total_cost:,.0f} Toman ({budget_utilization:.1f}% of budget)")
        lines.append(f"Total Weight: {total_weight:.2f} kg")
        lines.append(f"Budget Status: {'WITHIN BUDGET' if self.is_within_budget() else 'OVER BUDGET'}")
        lines.append(f"Food Diversity: {self.get_food_diversity_score():.2f}")
        
        # Nutritional status
        daily_nutrients = self.get_daily_nutrient_totals()
        deficient = self.get_deficient_nutrients()
        
        lines.append(f"\nNutritional Status:")
        lines.append(f"Meets minimums: {'YES' if self.meets_minimum_requirements() else 'NO'}")
        if deficient:
            lines.append(f"Deficient nutrients: {', '.join(deficient)}")
        
        lines.append("="*60)
        return "\n".join(lines)
    
    def format_food_basket_table(self) -> str:
        """
        Format the food basket in a readable table format.
        
        Returns:
            Formatted table string showing significant food items
        """
        foods = self.evaluator.foods
        food_names = [food.name for food in foods]
        
        # Filter significant foods
        significant_foods = []
        for i, qty in enumerate(self.quantities):
            if qty > 0.01:  # More than 10g monthly
                name = food_names[i]
                daily_grams = (qty * 1000) / 30
                monthly_cost = qty * foods[i].price
                significant_foods.append([name, qty, daily_grams, monthly_cost])
        
        if not significant_foods:
            return "No significant food items in this solution."
        
        # Sort by quantity (descending)
        significant_foods.sort(key=lambda x: x[1], reverse=True)
        
        # Create table data
        table_data = []
        for name, qty, daily_g, cost in significant_foods:
            table_data.append([
                name,
                f"{qty:.2f}",
                f"{daily_g:.1f}",
                f"{cost:,.0f}"
            ])
        
        headers = ["Food Item", "Monthly (kg)", "Daily (g)", "Cost (Toman)"]
        return tabulate.tabulate(table_data, headers=headers, tablefmt="grid")
    
    def __str__(self) -> str:
        """Brief string representation of the solution."""
        cost = self.get_total_cost()
        weight = self.get_total_weight()
        return f"Solution(fitness={self.fitness:.2f}, cost={cost:,.0f}T, weight={weight:.1f}kg)"
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"Solution(quantities={self.quantities}, fitness={self.fitness})"
    
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
        return hash(tuple(self.quantities))