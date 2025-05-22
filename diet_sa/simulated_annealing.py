from .data_loader import DataLoader
from .sa_models import Solution
from .fitness import FitnessEvaluator
from .plotting import Plotter
from typing import List, Dict, Tuple
import numpy as np
import math
import random
import os

class SimulatedAnnealing:
    """
    Simulated Annealing algorithm for monthly diet optimization.
    
    This class implements the SA algorithm to find an optimal food combination
    that meets nutritional requirements while staying within budget constraints.
    """
    
    def __init__(
        self,
        csv_path: str,
        max_iterations: int = 10000,
        initial_temp: float = 1000.0,
        final_temp: float = 0.1,
        cooling_rate: float = 0.95,
        cost_cap: float = 4_000_000.0,
        step_size: float = 2.0,
    ):
        """
        Initialize the Simulated Annealing algorithm
        
        Args:
            csv_path: Path to the CSV file with food data
            max_iterations: Maximum number of iterations to run
            initial_temp: Initial temperature for SA
            final_temp: Final (minimum) temperature
            cooling_rate: Rate at which temperature decreases (0 < cooling_rate < 1)
            cost_cap: Maximum cost constraint in Toman
            step_size: Maximum change in food quantity per perturbation (kg)
        """
        try:
            # Check if file exists, try alternative paths if needed
            if not os.path.exists(csv_path):
                print(f"Warning: File '{csv_path}' not found. Checking for alternative paths...")
                alt_paths = [
                    os.path.join(os.path.dirname(__file__), 'foods.csv'),
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'foods.csv')
                ]
                for path in alt_paths:
                    if os.path.exists(path):
                        print(f"Found alternative file at '{path}'")
                        csv_path = path
                        break
                else:
                    raise FileNotFoundError(f"Could not find 'foods.csv' in any of the expected locations.")
                
            # Load food data
            foods = DataLoader.load_foods(csv_path)
            gene_count = len(foods)
            
            if gene_count == 0:
                raise ValueError("No food items were loaded from the CSV file.")

            # Define daily nutritional requirements (same as GA)
            daily = {
                'calories': 2000, 'protein': 100, 'fat': 60,
                'carbs': 250,     'fiber': 25,  'calcium': 1000,
                'iron': 18,
            }
            
            # Convert to monthly requirements
            min_req = {nut: val * 30 for nut, val in daily.items()}
            
            # Define optimal values (per month) - same as GA
            optimal = {
                'calories': 1700*30, 'protein': 140*30, 'fat': 45*30,
                'carbs': 210*30,     'fiber': 35*30,  'calcium': 1200*30,
                'iron': 23*30,
            }
            
            # Define weights for nutrients in fitness function - same as GA
            weights = {
                'calories': 1.0, 'protein': 1.2, 'fat': 1.8,
                'carbs': 1.2,     'fiber': 1.2,  'calcium': 1.2,
                'iron': 3.0,
            }
            
            # Create fitness evaluator (reuse the same one from GA)
            self.evaluator = FitnessEvaluator(
                foods, min_req, optimal, weights, cost_cap,
            )
            
            # Store algorithm parameters
            self.foods = foods
            self.gene_count = gene_count
            self.max_iterations = max_iterations
            self.initial_temp = initial_temp
            self.final_temp = final_temp
            self.cooling_rate = cooling_rate
            self.cost_cap = cost_cap
            self.step_size = step_size
            
            # Store nutritional requirements for reporting
            self.min_daily = daily
            self.optimal_daily = {nut: ov/30 for nut, ov in optimal.items()}
            
            # Initialize tracking variables
            self.temp_history = []
            self.fitness_history = []
            self.cost_history = []
            self.acceptance_history = []
            self.nut_history = {nut: [] for nut in daily if nut != 'calories'}
            
            # Create plotter instance
            self.plotter = Plotter()
            
            # Print available food items
            self.print_foods_details()
            
        except Exception as e:
            print(f"Error initializing Simulated Annealing: {e}")
            raise
            
    def print_foods_details(self):
        """
        Print detailed information about all available food items.
        This helps users understand what foods are available for optimization.
        """
        print("\nAvailable Food Items:")
        print("-" * 110)
        print(f"{'Food Name':^25} | {'Calories':^8} | {'Protein':^8} | {'Fat':^6} | {'Carbs':^6} | {'Fiber':^6} | {'Ca':^5} | {'Fe':^5} | {'Price/kg':^12} | {'P/¥':^7}")
        print("-" * 110)
        
        # Sort foods by price per protein content (value metric)
        for food in self.foods:
            # Calculate protein per unit cost (value metric)
            protein_value = (food.nutrients['protein'] / food.price * 1000) if food.price > 0 else 0
            
            print(f"{food.name:25} | {food.nutrients['calories']:8.1f} | {food.nutrients['protein']:8.1f} | "
                  f"{food.nutrients['fat']:6.1f} | {food.nutrients['carbs']:6.1f} | {food.nutrients['fiber']:6.1f} | "
                  f"{food.nutrients['calcium']:5.1f} | {food.nutrients['iron']:5.1f} | {food.price:12,.0f} | {protein_value:7.2f}")
            
        print("-" * 110)
        print("P/¥: Protein value (g protein per 1000 Toman)")
        print()

    def generate_initial_solution(self) -> Solution:
        """
        Generate a random initial solution.
        
        Each food item gets a random quantity between 0 and 50 kg per month.
        This provides a good starting point for the SA algorithm.
        
        Returns:
            A randomly generated Solution object
        """
        # Generate random quantities for each food item (0 to 50 kg per month)
        quantities = [random.uniform(0, 50.0) for _ in range(self.gene_count)]
        return Solution(quantities, self.evaluator)

    def generate_neighbor(self, current_solution: Solution) -> Solution:
        """
        Generate a neighbor solution by perturbing the current solution.
        
        This function creates a new solution by randomly modifying some food quantities.
        The perturbation strategy:
        1. Select a random subset of food items to modify
        2. Apply random changes within the step_size parameter
        3. Ensure all quantities remain non-negative
        
        Args:
            current_solution: The current solution to perturb
            
        Returns:
            A new neighboring Solution object
        """
        # Copy current quantities
        new_quantities = current_solution.quantities.copy()
        
        # Decide how many items to perturb (1 to 5 items randomly)
        num_changes = random.randint(1, min(5, self.gene_count))
        
        # Select random indices to modify
        indices_to_change = random.sample(range(self.gene_count), num_changes)
        
        for idx in indices_to_change:
            # Generate a random change within step_size
            change = random.uniform(-self.step_size, self.step_size)
            new_quantities[idx] = max(0.0, new_quantities[idx] + change)
            
            # Occasionally set to zero (helps remove unnecessary foods)
            if random.random() < 0.1:
                new_quantities[idx] = 0.0
            
            # Occasionally set to a completely new random value (exploration)
            if random.random() < 0.05:
                new_quantities[idx] = random.uniform(0, 30.0)
        
        return Solution(new_quantities, self.evaluator)

    def acceptance_probability(self, current_fitness: float, new_fitness: float, temperature: float) -> float:
        """
        Calculate the probability of accepting a worse solution.
        
        This is the core of the Simulated Annealing algorithm. It allows the algorithm
        to accept worse solutions with a probability that decreases as temperature decreases.
        
        Formula: P(accept) = exp((new_fitness - current_fitness) / temperature)
        
        Args:
            current_fitness: Fitness of the current solution
            new_fitness: Fitness of the candidate new solution
            temperature: Current temperature of the system
            
        Returns:
            Probability of accepting the new solution (0 to 1)
        """
        if new_fitness > current_fitness:
            # Always accept better solutions
            return 1.0
        else:
            # Accept worse solutions with probability based on temperature
            # Higher temperature = higher probability of accepting worse solutions
            delta = new_fitness - current_fitness
            return math.exp(delta / temperature) if temperature > 0 else 0.0

    def update_temperature(self, iteration: int) -> float:
        """
        Update the temperature based on the current iteration.
        
        This function implements exponential cooling schedule:
        T(i) = T_initial * (cooling_rate ^ iteration)
        
        Alternative cooling schedules could be:
        - Linear: T(i) = T_initial - (T_initial - T_final) * i / max_iterations
        - Logarithmic: T(i) = T_initial / log(1 + i)
        
        Args:
            iteration: Current iteration number
            
        Returns:
            New temperature value
        """
        # Exponential cooling schedule
        temp = self.initial_temp * (self.cooling_rate ** iteration)
        
        # Ensure temperature doesn't go below final_temp
        return max(temp, self.final_temp)

    def run(self) -> Tuple[List[float], float, float, Dict[str, float]]:
        """
        Run the Simulated Annealing algorithm.
        
        The main SA loop:
        1. Start with a random solution
        2. For each iteration:
           a. Generate a neighbor solution
           b. Calculate fitness difference
           c. Accept/reject based on acceptance probability
           d. Update temperature
           e. Track best solution found so far
        3. Return the best solution found
        
        Returns:
            Tuple of (best_quantities, best_fitness, cost, nutrient_totals)
        """
        try:
            # Initialize with random solution
            current_solution = self.generate_initial_solution()
            best_solution = Solution(current_solution.quantities.copy(), self.evaluator)
            
            # Initialize temperature
            temperature = self.initial_temp
            
            # Tracking variables
            accepted_moves = 0
            total_moves = 0
            
            print("\nStarting Simulated Annealing Optimization:")
            print(f"{'-'*90}")
            print(f"{'Iter':^6}|{'Temp':^8}|{'Current':^12}|{'Best':^12}|{'Cost (T)':^15}|{'Acc%':^6}|{'Cal':^8}|{'Pro':^8}|{'Fat':^8}")
            print(f"{'-'*90}")
            
            # Main SA loop
            for iteration in range(self.max_iterations):
                # Generate neighbor solution
                neighbor = self.generate_neighbor(current_solution)
                
                # Calculate acceptance probability
                accept_prob = self.acceptance_probability(
                    current_solution.fitness, neighbor.fitness, temperature
                )
                
                total_moves += 1
                
                # Accept or reject the neighbor
                if random.random() < accept_prob:
                    current_solution = neighbor
                    accepted_moves += 1
                    
                    # Update best solution if needed
                    if current_solution.fitness > best_solution.fitness:
                        best_solution = Solution(current_solution.quantities.copy(), self.evaluator)
                
                # Update temperature
                temperature = self.update_temperature(iteration)
                
                # Record history for plotting
                self.temp_history.append(temperature)
                self.fitness_history.append(current_solution.fitness)
                
                # Calculate current cost and nutrition for display
                cost, nutrients = self._compute_cost_and_totals(current_solution)
                self.cost_history.append(cost)
                
                # Record nutrition history
                self._record_nutrition_history(nutrients)
                
                # Record acceptance rate
                acceptance_rate = (accepted_moves / total_moves) * 100 if total_moves > 0 else 0
                self.acceptance_history.append(acceptance_rate)
                
                # Print progress every 500 iterations or at key milestones
                if iteration % 500 == 0 or iteration == self.max_iterations - 1:
                    print(f"{iteration:6d}|{temperature:8.2f}|{current_solution.fitness:12.2f}|{best_solution.fitness:12.2f}|"
                          f"{cost:15,.0f}|{acceptance_rate:6.1f}|{nutrients['calories']:8.1f}|{nutrients['protein']:8.1f}|{nutrients['fat']:8.1f}")
                
                # Early stopping if temperature is very low and no improvement
                if temperature < self.final_temp and iteration > 1000:
                    if len(self.fitness_history) > 1000:
                        recent_improvement = max(self.fitness_history[-1000:]) - min(self.fitness_history[-1000:])
                        if recent_improvement < 0.1:  # Very small improvement in last 1000 iterations
                            print(f"\nEarly stopping at iteration {iteration} due to convergence.")
                            break
            
            print(f"{'-'*90}")
            print(f"Optimization completed. Total accepted moves: {accepted_moves}/{total_moves} ({acceptance_rate:.1f}%)")
            
            # Calculate final results
            final_cost, final_nutrients = self._compute_cost_and_totals(best_solution)
            
            # Generate plots
            self._generate_plots(final_nutrients)
            
            # Return results in the same format as GA
            return (
                best_solution.quantities,
                best_solution.fitness,
                final_cost,
                final_nutrients
            )
            
        except Exception as e:
            print(f"Error running Simulated Annealing: {e}")
            raise

    def _compute_cost_and_totals(self, solution: Solution) -> Tuple[float, Dict[str, float]]:
        """
        Compute the total cost and daily nutrient totals for a solution.
        
        This helper function calculates:
        1. Total monthly cost of the food basket
        2. Daily average nutrient intake for each nutrient
        
        Args:
            solution: The solution to analyze
            
        Returns:
            Tuple of (total_cost, daily_nutrient_totals)
        """
        totals = {nut: 0.0 for nut in self.evaluator.min_req}
        cost = 0.0
        
        # Calculate totals
        for qty, food in zip(solution.quantities, self.evaluator.foods):
            cost += qty * food.price
            for nut, per100g in food.nutrients.items():
                # Convert per 100g to per kg (multiply by 10), then to daily (divide by 30)
                totals[nut] += (per100g * 10 * qty) / 30
                
        return cost, {n: round(v, 1) for n, v in totals.items()}

    def _record_nutrition_history(self, nutrients: Dict[str, float]):
        """
        Record the nutrient-to-calorie ratios for tracking optimization progress.
        
        This helps visualize how the nutritional quality changes over iterations.
        
        Args:
            nutrients: Dictionary of daily nutrient values
        """
        calories = nutrients['calories']
        if calories > 0:
            for nut, val in nutrients.items():
                if nut != 'calories' and nut in self.nut_history:
                    self.nut_history[nut].append(val / calories)

    def _generate_plots(self, final_nutrients: Dict[str, float]):
        """
        Generate visualization plots for the SA optimization results.
        
        Creates several plots:
        1. Temperature and fitness progress over iterations
        2. Nutrition comparison (min vs optimal vs actual)
        3. Nutrient progress over iterations
        4. Acceptance rate over iterations
        
        Args:
            final_nutrients: Final nutrient values achieved
        """
        try:
            # Plot SA-specific metrics (temperature, acceptance rate)
            self._plot_sa_metrics()
            
            # Use existing plotter methods for common visualizations
            self.plotter.plot_nutrition_comparison(
                self.min_daily, self.optimal_daily, final_nutrients
            )
            
            if self.nut_history:
                self.plotter.plot_nutrition_progress(self.nut_history)
                
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")

    def _plot_sa_metrics(self):
        """
        Plot Simulated Annealing specific metrics.
        
        Creates a comprehensive plot showing:
        1. Temperature decay over iterations
        2. Fitness progress over iterations
        3. Acceptance rate over iterations
        4. Cost progress over iterations
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig)
        
        iterations = range(len(self.temp_history))
        
        # 1. Temperature decay
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(iterations, self.temp_history, 'r-', linewidth=2, label='Temperature')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Temperature')
        ax1.set_title('Temperature Decay (Cooling Schedule)')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale for better visualization
        
        # 2. Fitness progress
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(iterations, self.fitness_history, 'b-', linewidth=2, label='Current Fitness')
        
        # Also plot best fitness found so far
        best_so_far = []
        current_best = -float('inf')
        for fit in self.fitness_history:
            if fit > current_best:
                current_best = fit
            best_so_far.append(current_best)
        
        ax2.plot(iterations, best_so_far, 'g-', linewidth=2, label='Best So Far')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Fitness Score')
        ax2.set_title('Fitness Progress')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Acceptance rate
        ax3 = fig.add_subplot(gs[1, 0])
        # Smooth the acceptance rate with a moving average
        window_size = min(100, len(self.acceptance_history) // 10)
        if window_size > 1:
            smoothed_acc = []
            for i in range(len(self.acceptance_history)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(self.acceptance_history), i + window_size // 2)
                smoothed_acc.append(np.mean(self.acceptance_history[start_idx:end_idx]))
            ax3.plot(iterations, smoothed_acc, 'purple', linewidth=2, label=f'Acceptance Rate (MA-{window_size})')
        else:
            ax3.plot(iterations, self.acceptance_history, 'purple', linewidth=2, label='Acceptance Rate')
        
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Acceptance Rate (%)')
        ax3.set_title('Solution Acceptance Rate Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # 4. Cost progress
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(iterations, [c/1_000_000 for c in self.cost_history], 'orange', linewidth=2, label='Cost (Million Toman)')
        ax4.axhline(y=self.cost_cap/1_000_000, color='red', linestyle='--', alpha=0.7, label='Budget Cap')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Cost (Million Toman)')
        ax4.set_title('Cost Progress')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Simulated Annealing Optimization Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('sa_optimization_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()