from typing import List, Dict
from tabulate import tabulate
from colorama import Fore, init
from .models import Solution
from .models import FoodItem
import math

# Initialize colorama for colored terminal output
init(autoreset=True)

class FitnessEvaluator:
    """
    Fitness evaluation class - FIXED VERSION.
    
    This class implements a logically sound and nutritionally accurate fitness function
    that evaluates SA solutions based on:
      - Nutritionally appropriate scoring for different nutrient types
      - Realistic target ranges and minimum requirements
      - Balanced penalty/reward system with proper scaling
      - Smart budget constraint handling
      - Normalized scoring across different nutrient magnitudes
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
        
        self.directions = {
            'calories': 'target',     # Target range: avoid too high/too low
            'fat':      'target',     # Target range: need adequate but not excessive
            'carbs':    'flexible',   # Flexible range: can vary widely
            'protein':  'min_plus',   # Minimum required, more is generally better
            'fiber':    'min_plus',   # Minimum required, more is beneficial
            'calcium':  'min_plus',   # Minimum required, more is beneficial  
            'iron':     'min_plus',   # Minimum required, more is beneficial (up to limit)
        }
        
        # Realistic target ranges for each nutrient type
        self.target_ranges = self._calculate_target_ranges()
        
        # Normalization factors for different nutrient scales
        self.normalization_factors = self._calculate_normalization_factors()

    def _calculate_target_ranges(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate realistic target ranges for each nutrient based on nutritional science.
        
        Returns:
            Dictionary with target ranges for each nutrient
        """
        ranges = {}
        
        for nutrient in self.min_req:
            min_val = self.min_req[nutrient]
            opt_val = self.optimal.get(nutrient, min_val)
            
            if self.directions[nutrient] == 'target':
                # For target nutrients, create a healthy range
                if nutrient == 'calories':
                    # Calories: tight range around 1800-2000
                    ranges[nutrient] = {
                        'ideal_min': min_val * 0.9,    # 1800 kcal
                        'ideal_max': min_val * 1.0,    # 2000 kcal
                        'acceptable_min': min_val * 0.8, # 1600 kcal
                        'acceptable_max': min_val * 1.15 # 2300 kcal
                    }
                elif nutrient == 'fat':
                    # Fat: 20-35% of calories (45-65g for 2000 kcal diet)
                    ranges[nutrient] = {
                        'ideal_min': min_val * 0.75,   # 45g
                        'ideal_max': min_val * 1.08,   # 65g  
                        'acceptable_min': min_val * 0.5, # 30g
                        'acceptable_max': min_val * 1.33 # 80g
                    }
                elif nutrient == 'carbs':
                    # Carbs: flexible but reasonable range
                    ranges[nutrient] = {
                        'ideal_min': min_val * 0.7,    # 175g
                        'ideal_max': min_val * 1.0,    # 250g
                        'acceptable_min': min_val * 0.5, # 125g
                        'acceptable_max': min_val * 1.2  # 300g
                    }
                    
            elif self.directions[nutrient] == 'flexible':
                # For flexible nutrients, wider acceptable range
                ranges[nutrient] = {
                    'ideal_min': min(min_val, opt_val) * 0.8,
                    'ideal_max': max(min_val, opt_val) * 1.2,
                    'acceptable_min': min(min_val, opt_val) * 0.6,
                    'acceptable_max': max(min_val, opt_val) * 1.4
                }
                
            else:  # 'min_plus' nutrients
                # For min_plus nutrients, minimum required, optimal target, reasonable upper limit
                ranges[nutrient] = {
                    'minimum': min_val,
                    'target': max(min_val, opt_val),
                    'upper_limit': max(min_val, opt_val) * 1.5  # Allow 50% above optimal
                }
        
        return ranges

    def _calculate_normalization_factors(self) -> Dict[str, float]:
        """
        Calculate normalization factors to balance different nutrient scales.
        
        Returns:
            Dictionary with normalization factors for each nutrient
        """
        factors = {}
        
        for nutrient, min_val in self.min_req.items():
            # Base normalization on the typical range of the nutrient
            if nutrient == 'calories':
                factors[nutrient] = 1.0 / 2000  # Normalize to ~1 for 2000 kcal
            elif nutrient in ['protein', 'fat', 'carbs']:
                factors[nutrient] = 1.0 / 100   # Normalize to ~1 for 100g
            elif nutrient == 'fiber':
                factors[nutrient] = 1.0 / 30    # Normalize to ~1 for 30g
            elif nutrient == 'calcium':
                factors[nutrient] = 1.0 / 1.0   # Already in reasonable scale (grams)
            elif nutrient == 'iron':
                factors[nutrient] = 50.0        # Scale up from 0.02g to ~1
            else:
                factors[nutrient] = 1.0 / max(min_val, 1.0)
                
        return factors

    def score_nutrient(
        self,
        actual_val: float,
        min_val: float,
        opt_val: float,
        direction: str,
        weight: float,
        nutrient_name: str,
    ) -> float:
        """
        Calculate fitness score for a single nutrient - FIXED VERSION.
        
        This method implements logically sound scoring that:
        1. Properly handles different nutrient types with appropriate ranges
        2. Avoids perverse incentives from illogical min/optimal relationships
        3. Uses normalized scoring for balanced evaluation
        4. Provides clear, interpretable scoring logic
        
        Args:
            actual_val: Actual nutrient value in the current solution
            min_val: Minimum required value for this nutrient
            opt_val: Optimal target value for this nutrient
            direction: Optimization direction ('target', 'flexible', 'min_plus')
            weight: Importance weight for this nutrient
            nutrient_name: Name of the nutrient for range lookup
            
        Returns:
            Normalized fitness score contribution for this nutrient (higher is better)
        """
        # Prevent division by zero and negative values
        actual_val = max(0.0, actual_val)
        min_val = max(0.001, min_val)
        opt_val = max(0.001, opt_val)
        
        # Get normalization factor
        norm_factor = self.normalization_factors.get(nutrient_name, 1.0)
        
        # Calculate base score based on nutrient direction
        if direction == 'target':
            score = self._score_target_nutrient(actual_val, nutrient_name)
        elif direction == 'flexible':
            score = self._score_flexible_nutrient(actual_val, nutrient_name)
        elif direction == 'min_plus':
            score = self._score_min_plus_nutrient(actual_val, min_val, opt_val)
        else:
            # Fallback to simple target scoring
            score = self._score_target_nutrient(actual_val, nutrient_name)
        
        # Apply weight and normalization
        final_score = score * weight * norm_factor
        
        return final_score

    def _score_target_nutrient(self, actual_val: float, nutrient_name: str) -> float:
        """
        Score nutrients that should hit a target range (calories, fat).
        
        Rewards being in ideal range, penalizes deviation.
        """
        ranges = self.target_ranges.get(nutrient_name, {})
        ideal_min = ranges.get('ideal_min', actual_val)
        ideal_max = ranges.get('ideal_max', actual_val)
        acceptable_min = ranges.get('acceptable_min', ideal_min * 0.8)
        acceptable_max = ranges.get('acceptable_max', ideal_max * 1.2)
        
        if ideal_min <= actual_val <= ideal_max:
            # Perfect range - high reward
            return 10.0
        elif acceptable_min <= actual_val <= acceptable_max:
            # Acceptable range - moderate reward
            if actual_val < ideal_min:
                distance = (ideal_min - actual_val) / (ideal_min - acceptable_min)
            else:
                distance = (actual_val - ideal_max) / (acceptable_max - ideal_max)
            return 10.0 - 5.0 * distance  # Score: 5.0 to 10.0
        else:
            # Outside acceptable range - penalty
            if actual_val < acceptable_min:
                excess = (acceptable_min - actual_val) / acceptable_min
            else:
                excess = (actual_val - acceptable_max) / acceptable_max
            return -2.0 * excess  # Negative score, increasing penalty

    def _score_flexible_nutrient(self, actual_val: float, nutrient_name: str) -> float:
        """
        Score nutrients with flexible ranges (carbs).
        
        More lenient scoring, wider acceptable range.
        """
        ranges = self.target_ranges.get(nutrient_name, {})
        ideal_min = ranges.get('ideal_min', actual_val)
        ideal_max = ranges.get('ideal_max', actual_val)
        acceptable_min = ranges.get('acceptable_min', ideal_min * 0.6)
        acceptable_max = ranges.get('acceptable_max', ideal_max * 1.4)
        
        if ideal_min <= actual_val <= ideal_max:
            # Perfect range
            return 8.0
        elif acceptable_min <= actual_val <= acceptable_max:
            # Acceptable range
            if actual_val < ideal_min:
                distance = (ideal_min - actual_val) / (ideal_min - acceptable_min)
            else:
                distance = (actual_val - ideal_max) / (acceptable_max - ideal_max)
            return 8.0 - 3.0 * distance  # Score: 5.0 to 8.0
        else:
            # Outside acceptable range - gentle penalty
            if actual_val < acceptable_min:
                excess = (acceptable_min - actual_val) / acceptable_min
            else:
                excess = (actual_val - acceptable_max) / acceptable_max
            return -1.0 * excess

    def _score_min_plus_nutrient(self, actual_val: float, min_val: float, opt_val: float) -> float:
        """
        Score nutrients where more is generally better up to a limit (protein, fiber, minerals).
        
        Strong penalty for deficiency, reward for adequacy, diminishing returns above optimal.
        """
        target = max(min_val, opt_val)
        upper_limit = target * 1.5
        
        if actual_val < min_val:
            # Deficiency - strong penalty
            deficit_ratio = actual_val / min_val
            return -15.0 * (1.0 - deficit_ratio)  # -15 to 0
        elif min_val <= actual_val <= target:
            # Good range - increasing reward
            progress = (actual_val - min_val) / (target - min_val) if target > min_val else 1.0
            return 5.0 + 5.0 * progress  # 5.0 to 10.0
        elif target < actual_val <= upper_limit:
            # Above optimal but acceptable - slight reward
            excess_ratio = (actual_val - target) / (upper_limit - target)
            return 10.0 - 2.0 * excess_ratio  # 8.0 to 10.0
        else:
            # Excessive - penalty
            excess_ratio = (actual_val - upper_limit) / upper_limit
            return -3.0 * excess_ratio

    def __call__(self, solution: Solution) -> float:
        """
        Calculate the overall fitness score for a Simulated Annealing solution - IMPROVED.
        
        This is the main fitness function that the SA algorithm uses to evaluate
        and compare different food combinations. 
        
        IMPROVEMENTS:
        - Proper nutrient scoring with logical directions
        - Normalized scoring across different nutrient scales  
        - Smart budget constraint handling with gradual penalties
        - Balanced penalty/reward system
        
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

        # 2) Apply budget constraint with smart penalty system
        fitness_score = 0.0
        
        # Progressive budget penalty
        if total_cost > self.cost_cap:
            # Escalating penalty for budget violations
            budget_violation = total_cost - self.cost_cap
            violation_ratio = budget_violation / self.cost_cap
            
            if violation_ratio <= 0.1:  # Up to 10% over budget
                budget_penalty = 50 * violation_ratio  # Gentle penalty
            elif violation_ratio <= 0.25:  # 10-25% over budget  
                budget_penalty = 5 + 100 * (violation_ratio - 0.1)  # Moderate penalty
            else:  # More than 25% over budget
                budget_penalty = 20 + 200 * (violation_ratio - 0.25)  # Severe penalty
                
            fitness_score -= budget_penalty
        else:
            # Small bonus for being well under budget (encourages cost efficiency)
            cost_efficiency = (self.cost_cap - total_cost) / self.cost_cap
            if cost_efficiency > 0.1:  # More than 10% under budget
                fitness_score += 2.0 * min(cost_efficiency, 0.3)  # Max bonus of 0.6

        # 3) Evaluate each nutrient individually and sum the scores
        for nutrient, min_requirement in self.min_req.items():
            actual_intake = nutrient_totals[nutrient]
            optimal_target = self.optimal.get(nutrient, min_requirement)
            nutrient_weight = self.weights.get(nutrient, 1.0)
            optimization_direction = self.directions[nutrient]
            
            # Calculate individual nutrient score using improved method
            nutrient_score = self.score_nutrient(
                actual_intake, min_requirement, optimal_target, 
                optimization_direction, nutrient_weight, nutrient
            )
            fitness_score += nutrient_score

        return fitness_score
        
    def get_detailed_analysis(self, solution: Solution) -> Dict:
        """
        Generate a comprehensive analysis of a Simulated Annealing solution - IMPROVED.
        
        This method provides detailed insights into how well the solution performs
        across all evaluation criteria with improved accuracy and interpretability.
        
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
        
        # Analyze nutritional status for each nutrient using improved logic
        nutritional_status = {}
        for nutrient, daily_actual in daily_totals.items():
            daily_minimum = self.min_req[nutrient] / 30
            daily_optimal = self.optimal.get(nutrient, self.min_req[nutrient]) / 30
            direction = self.directions[nutrient]
            
            # Determine status based on improved nutrient classification
            status = self._determine_nutrient_status(daily_actual, nutrient, daily_minimum, daily_optimal)
            
            nutritional_status[nutrient] = {
                "status": status,
                "actual_daily": daily_actual,
                "minimum_daily": daily_minimum,
                "optimal_daily": daily_optimal,
                "direction": direction,
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
            
            # Summary flags using improved logic
            'meets_all_minimums': self._check_minimum_requirements(nutritional_status),
            'has_excesses': self._check_excesses(nutritional_status),
            'nutritionally_balanced': self._check_balance(nutritional_status),
        }

    def _determine_nutrient_status(self, actual: float, nutrient: str, minimum: float, optimal: float) -> str:
        """
        Determine nutritional status using improved logic based on nutrient type.
        """
        direction = self.directions[nutrient]
        
        if direction == 'target':
            ranges = self.target_ranges.get(nutrient, {})
            ideal_min = ranges.get('ideal_min', minimum) / 30  # Convert to daily
            ideal_max = ranges.get('ideal_max', optimal) / 30
            acceptable_min = ranges.get('acceptable_min', ideal_min * 0.8) / 30
            acceptable_max = ranges.get('acceptable_max', ideal_max * 1.2) / 30
            
            if ideal_min <= actual <= ideal_max:
                return "OPTIMAL"
            elif acceptable_min <= actual <= acceptable_max:
                return "ACCEPTABLE"
            elif actual < acceptable_min:
                return "LOW"
            else:
                return "HIGH"
                
        elif direction == 'flexible':
            ranges = self.target_ranges.get(nutrient, {})
            acceptable_min = ranges.get('acceptable_min', minimum * 0.6) / 30
            acceptable_max = ranges.get('acceptable_max', optimal * 1.4) / 30
            
            if acceptable_min <= actual <= acceptable_max:
                return "ACCEPTABLE"
            elif actual < acceptable_min:
                return "LOW"
            else:
                return "HIGH"
                
        else:  # 'min_plus'
            target = max(minimum, optimal)
            upper_limit = target * 1.5
            
            if actual < minimum:
                return "DEFICIENT"
            elif minimum <= actual <= target:
                return "GOOD"
            elif target < actual <= upper_limit:
                return "HIGH_GOOD"
            else:
                return "EXCESSIVE"

    def _check_minimum_requirements(self, nutritional_status: Dict) -> bool:
        """Check if all minimum requirements are met."""
        for nutrient, status in nutritional_status.items():
            status_val = status.get('status', '')
            if status_val in ['DEFICIENT', 'LOW']:
                return False
        return True

    def _check_excesses(self, nutritional_status: Dict) -> bool:
        """Check if there are any concerning excesses."""
        for nutrient, status in nutritional_status.items():
            status_val = status.get('status', '')
            if status_val == 'EXCESSIVE':
                return True
        return False

    def _check_balance(self, nutritional_status: Dict) -> bool:
        """Check if the diet is nutritionally balanced."""
        good_statuses = ['OPTIMAL', 'GOOD', 'HIGH_GOOD', 'ACCEPTABLE']
        good_count = sum(1 for status in nutritional_status.values() 
                        if status.get('status', '') in good_statuses)
        return good_count >= len(nutritional_status) * 0.8  # 80% of nutrients in good status

    def get_fitness_breakdown(self, solution: Solution) -> Dict:
        """
        Break down the fitness score into individual components for analysis - IMPROVED.
        
        This method helps understand how each part of the fitness function
        contributes to the overall score with better organization and clarity.
        
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
                'over_budget': total_cost > self.cost_cap,
                'budget_utilization': (total_cost / self.cost_cap) * 100
            }
        }
        
        # Budget component calculation (matching the main fitness function)
        if total_cost > self.cost_cap:
            budget_violation = total_cost - self.cost_cap
            violation_ratio = budget_violation / self.cost_cap
            
            if violation_ratio <= 0.1:
                breakdown['budget_component'] = -50 * violation_ratio
            elif violation_ratio <= 0.25:
                breakdown['budget_component'] = -(5 + 100 * (violation_ratio - 0.1))
            else:
                breakdown['budget_component'] = -(20 + 200 * (violation_ratio - 0.25))
        else:
            # Cost efficiency bonus
            cost_efficiency = (self.cost_cap - total_cost) / self.cost_cap
            if cost_efficiency > 0.1:
                breakdown['budget_component'] = 2.0 * min(cost_efficiency, 0.3)
        
        # Individual nutrient score components
        for nutrient, min_requirement in self.min_req.items():
            actual_intake = nutrient_totals[nutrient]
            optimal_target = self.optimal.get(nutrient, min_requirement)
            nutrient_weight = self.weights.get(nutrient, 1.0)
            direction = self.directions[nutrient]
            
            # Calculate this nutrient's contribution to fitness
            nutrient_score = self.score_nutrient(
                actual_intake, min_requirement, optimal_target, direction, nutrient_weight, nutrient
            )
            
            breakdown['nutrient_components'][nutrient] = {
                'score': nutrient_score,
                'actual_monthly': actual_intake,
                'actual_daily': actual_intake / 30,
                'minimum_required': min_requirement,
                'optimal_target': optimal_target,
                'weight': nutrient_weight,
                'direction': direction,
                'normalized_score': nutrient_score / (nutrient_weight * self.normalization_factors.get(nutrient, 1.0))
            }
        
        # Summary statistics
        nutrient_scores = [comp['score'] for comp in breakdown['nutrient_components'].values()]
        breakdown['summary'] = {
            'total_nutrient_score': sum(nutrient_scores),
            'average_nutrient_score': sum(nutrient_scores) / len(nutrient_scores) if nutrient_scores else 0,
            'positive_nutrients': len([s for s in nutrient_scores if s > 0]),
            'negative_nutrients': len([s for s in nutrient_scores if s < 0]),
            'best_nutrient': max(breakdown['nutrient_components'].items(), 
                               key=lambda x: x[1]['score'])[0] if nutrient_scores else None,
            'worst_nutrient': min(breakdown['nutrient_components'].items(), 
                                key=lambda x: x[1]['score'])[0] if nutrient_scores else None
        }
        
        return breakdown

    def get_nutrition_recommendations(self, solution: Solution) -> List[str]:
        """
        Generate specific nutritional recommendations based on the solution analysis.
        
        Args:
            solution: SA Solution object to analyze
            
        Returns:
            List of actionable nutritional recommendations
        """
        recommendations = []
        analysis = self.get_detailed_analysis(solution)
        nutritional_status = analysis.get('nutritional_status', {})
        
        # Check for deficiencies
        deficient_nutrients = []
        low_nutrients = []
        excessive_nutrients = []
        
        for nutrient, status in nutritional_status.items():
            status_val = status.get('status', '')
            if status_val == 'DEFICIENT':
                deficient_nutrients.append(nutrient)
            elif status_val == 'LOW':
                low_nutrients.append(nutrient)
            elif status_val == 'EXCESSIVE':
                excessive_nutrients.append(nutrient)
        
        # Critical deficiency recommendations
        if deficient_nutrients:
            recommendations.append(f"⚠️  CRITICAL: Address deficiencies in {', '.join(deficient_nutrients)}")
            for nutrient in deficient_nutrients:
                good_sources = self._get_good_nutrient_sources(nutrient)
                if good_sources:
                    recommendations.append(f"   • Increase {nutrient}: Consider more {', '.join(good_sources[:3])}")
        
        # Low nutrient recommendations
        if low_nutrients:
            recommendations.append(f"📈 IMPROVE: Boost intake of {', '.join(low_nutrients)}")
            for nutrient in low_nutrients:
                good_sources = self._get_good_nutrient_sources(nutrient)
                if good_sources:
                    recommendations.append(f"   • For {nutrient}: Add {', '.join(good_sources[:2])}")
        
        # Excess recommendations
        if excessive_nutrients:
            recommendations.append(f"📉 REDUCE: Lower intake of {', '.join(excessive_nutrients)}")
            for nutrient in excessive_nutrients:
                high_sources = self._get_high_nutrient_contributors(nutrient, solution)
                if high_sources:
                    recommendations.append(f"   • Reduce {nutrient}: Less {', '.join(high_sources[:2])}")
        
        # Budget recommendations
        if analysis['total_monthly_cost'] > self.cost_cap:
            over_budget = analysis['total_monthly_cost'] - self.cost_cap
            recommendations.append(f"💰 BUDGET: Over budget by {over_budget:,.0f} Toman ({analysis['budget_utilization_percent']:.1f}%)")
            expensive_foods = self._get_expensive_contributors(solution)
            if expensive_foods:
                recommendations.append(f"   • Consider reducing: {', '.join(expensive_foods[:3])}")
        
        # Positive reinforcement
        good_nutrients = [nut for nut, status in nutritional_status.items() 
                         if status.get('status', '') in ['OPTIMAL', 'GOOD', 'HIGH_GOOD']]
        if good_nutrients:
            recommendations.append(f"✅ GOOD: Well-balanced in {', '.join(good_nutrients)}")
        
        return recommendations if recommendations else ["✅ Diet appears well-balanced overall!"]

    def _get_good_nutrient_sources(self, nutrient: str) -> List[str]:
        """Get food names that are good sources of a specific nutrient."""
        if nutrient not in ['calories', 'protein', 'fat', 'carbs', 'fiber', 'calcium', 'iron']:
            return []
        
        # Find foods with high content of this nutrient
        good_sources = []
        for food in self.foods:
            if nutrient in food.nutrients:
                content = food.nutrients[nutrient]
                efficiency = food.get_nutrient_density(nutrient)
                
                # Criteria for being a "good source"
                if nutrient == 'protein' and content > 15:
                    good_sources.append((food.name, efficiency))
                elif nutrient == 'fiber' and content > 5:
                    good_sources.append((food.name, efficiency))
                elif nutrient == 'calcium' and content > 100:
                    good_sources.append((food.name, efficiency))
                elif nutrient == 'iron' and content > 2:
                    good_sources.append((food.name, efficiency))
                elif nutrient in ['calories', 'fat', 'carbs'] and content > 50:
                    good_sources.append((food.name, efficiency))
        
        # Sort by efficiency and return names
        good_sources.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in good_sources[:5]]

    def _get_high_nutrient_contributors(self, nutrient: str, solution: Solution) -> List[str]:
        """Get foods that contribute most to a specific nutrient in the current solution."""
        contributors = []
        
        for qty, food in zip(solution.quantities, self.foods):
            if qty > 0.1 and nutrient in food.nutrients:
                contribution = qty * food.nutrients[nutrient] * 10  # Monthly contribution
                contributors.append((food.name, contribution))
        
        contributors.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in contributors[:5]]

    def _get_expensive_contributors(self, solution: Solution) -> List[str]:
        """Get foods that contribute most to the total cost."""
        contributors = []
        
        for qty, food in zip(solution.quantities, self.foods):
            if qty > 0.1:
                cost_contribution = qty * food.price
                contributors.append((food.name, cost_contribution))
        
        contributors.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in contributors[:5]]

    def validate_configuration(self) -> List[str]:
        """
        Validate the fitness evaluator configuration and return any issues found.
        
        Returns:
            List of configuration issues (empty if no issues)
        """
        issues = []
        
        # Check for reasonable minimum requirements
        for nutrient, min_val in self.min_req.items():
            if min_val <= 0:
                issues.append(f"Minimum requirement for {nutrient} is not positive: {min_val}")
            
            opt_val = self.optimal.get(nutrient, min_val)
            if opt_val <= 0:
                issues.append(f"Optimal target for {nutrient} is not positive: {opt_val}")
        
        # Check weights
        for nutrient, weight in self.weights.items():
            if weight <= 0:
                issues.append(f"Weight for {nutrient} is not positive: {weight}")
        
        # Check budget
        if self.cost_cap <= 0:
            issues.append(f"Cost cap is not positive: {self.cost_cap}")
        
        # Check food data
        if not self.foods:
            issues.append("No food items provided")
        else:
            for i, food in enumerate(self.foods):
                if food.price <= 0:
                    issues.append(f"Food {i} ({food.name}) has non-positive price: {food.price}")
                for nutrient, value in food.nutrients.items():
                    if value < 0:
                        issues.append(f"Food {i} ({food.name}) has negative {nutrient}: {value}")
        
        return issues

    def __str__(self) -> str:
        """String representation for debugging."""
        return f"FitnessEvaluator(foods={len(self.foods)}, budget={self.cost_cap:,.0f})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"FitnessEvaluator(foods={len(self.foods)}, "
                f"nutrients={list(self.min_req.keys())}, "
                f"budget={self.cost_cap:,.0f})")