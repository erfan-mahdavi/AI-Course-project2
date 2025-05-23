from .data_loader import DataLoader
from .models import Solution
from .fitness import FitnessEvaluator
from .plotting import Plotter
from typing import List, Dict, Tuple
import numpy as np
import math
import random

class SimulatedAnnealing:
    """
    Simulated Annealing algorithm for monthly diet optimization.
    
    This class implements the SA algorithm to find an optimal food combination
    that meets nutritional requirements while staying within budget constraints.
    """
    
    def __init__(
        self,
        csv_path: str,
        max_iterations: int = 15000,
        initial_temp: float = 2000.0,
        final_temp: float = 0.01,
        cooling_rate: float = 0.99,
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

        # Load food data
        foods = DataLoader.load_foods(csv_path)
        food_count = len(foods)

        daily = {
            'calories': 2000, 'protein': 100, 'fat': 60,
            'carbs': 250,     'fiber': 25,  'calcium': 1000,
            'iron': 18,
        }

        min_req = {
            'calories': 2000*30, 'protein': 100*30, 'fat': 60*30,
            'carbs': 250*30,     'fiber': 25*30,  'calcium': 1000*30,
            'iron': 18*30,
        }
        
        # Define optimal values (per month)
        optimal = {
            'calories': 1700*30, 'protein': 140*30, 'fat': 45*30,
            'carbs': 210*30,     'fiber': 35*30,  'calcium': 1200*30,
            'iron': 23*30,
        }
        
        weights = {
                'calories': 1.0, 'protein': 1.2, 'fat': 1.8,
                'carbs': 1.2,     'fiber': 1.2,  'calcium': 1.2,
                'iron': 3.0
            }
        
        # Create fitness evaluator
        self.evaluator = FitnessEvaluator(
            foods, min_req, optimal, weights, cost_cap,
        )
        
        # Store algorithm parameters
        self.foods = foods
        self.food_count = food_count
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.cost_cap = cost_cap
        self.step_size = step_size
        self.weights = weights
        self.min_daily = daily
        
        # Store nutritional requirements for reporting
        self.optimal_daily = {nut: ov/30 for nut, ov in min_req.items()}
        
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
        Generate a smarter initial solution using food priorities.
        
        Creates an initial solution that considers food cost-effectiveness
        and nutritional density to start with a better baseline.
        
        Returns:
            A strategically generated Solution object
        """

        return Solution.random_solution(self.food_count, self.evaluator)

    def generate_neighbor(self, current_solution: Solution) -> Solution:
        """
        Generate a neighbor solution using intelligent perturbation strategies.
        
        This enhanced neighbor generation uses multiple strategies:
        1. Random perturbation (exploration)
        2. Nutrient-focused adjustments (exploitation)
        3. Cost-aware modifications (constraint handling)
        
        Args:
            current_solution: The current solution to perturb
            
        Returns:
            A new neighboring Solution object
        """
        # Choose perturbation strategy based on current solution quality
        current_cost = current_solution.get_total_cost()
        budget_ratio = current_cost / self.cost_cap
        deficient_nutrients = current_solution.get_deficient_nutrients()
        
        # Strategy selection
        if budget_ratio > 0.95:  # Near budget limit
            return current_solution.perturb_cost_aware(budget_pressure=0.8, step_size=self.step_size)
        elif len(deficient_nutrients) > 0:  # Has nutrient deficiencies
            # Focus on the most critical deficient nutrient
            focus_nutrient = deficient_nutrients[0]
            return current_solution.perturb_focused(focus_nutrient, self.step_size)
        elif random.random() < 0.3:  # 30% chance of local optimization
            return current_solution.local_optimization_step(self.step_size * 0.5)
        else:  # Default random perturbation
            return current_solution.perturb_random(self.step_size)

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
        Update the temperature using adaptive cooling schedule.
        
        This function implements an adaptive exponential cooling schedule that
        adjusts based on the progress of the optimization.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            New temperature value
        """
        # Basic exponential cooling
        temp = self.initial_temp * (self.cooling_rate ** iteration)
        
        # Adaptive adjustment based on recent progress
        if hasattr(self, 'fitness_history') and len(self.fitness_history) > 100:
            recent_improvement = max(self.fitness_history[-50:]) - max(self.fitness_history[-100:-50])
            if recent_improvement > 0.1:  # Good improvement - cool slower
                temp *= 1.1
            elif recent_improvement < 0.01:  # Poor improvement - cool faster
                temp *= 0.9
        
        # Ensure temperature doesn't go below final_temp
        return max(temp, self.final_temp)

    def run(self) -> Tuple[List[float], float, float, Dict[str, float]]:
        """
        Run the Simulated Annealing algorithm with enhanced features.
        
        The main SA loop with improvements:
        1. Smart initial solution generation
        2. Adaptive neighbor generation
        3. Progress tracking and early stopping
        4. Detailed logging and analysis
        
        Returns:
            Tuple of (best_quantities, best_fitness, cost, nutrient_totals)
        """
        try:
            # Initialize with smart solution
            current_solution = self.generate_initial_solution()
            best_solution = current_solution.copy()
            
            # Initialize temperature
            temperature = self.initial_temp
            
            # Tracking variables
            accepted_moves = 0
            total_moves = 0
            stagnation_counter = 0
            last_improvement_iter = 0
            
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
                        best_solution = current_solution.copy()
                        last_improvement_iter = iteration
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1
                
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
                
                # Print progress every 200 iterations or at key milestones
                if iteration % 200 == 0 or iteration == self.max_iterations - 1:
                    print(f"{iteration:6d}|{temperature:8.2f}|{current_solution.fitness:12.2f}|{best_solution.fitness:12.2f}|"
                          f"{cost:15,.0f}|{acceptance_rate:6.1f}|{nutrients['calories']:8.1f}|{nutrients['protein']:8.1f}|{nutrients['fat']:8.1f}")
                
                # Enhanced early stopping conditions
                if self._should_terminate(iteration, temperature, stagnation_counter, last_improvement_iter):
                    print(f"\nEarly stopping at iteration {iteration}")
                    break
            
            print(f"{'-'*90}")
            print(f"Optimization completed. Total accepted moves: {accepted_moves}/{total_moves} ({acceptance_rate:.1f}%)")
            
            # Calculate final results
            final_cost, final_nutrients = self._compute_cost_and_totals(best_solution)
            
            # Generate plots (with error handling)
            try:
                self._generate_plots(final_nutrients)
            except Exception as e:
                print(f"Warning: Could not generate plots: {e}")
            
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

    def _should_terminate(self, iteration: int, temperature: float, stagnation_counter: int, last_improvement_iter: int) -> bool:
        """
        Determine if the algorithm should terminate early.
        
        Args:
            iteration: Current iteration
            temperature: Current temperature
            stagnation_counter: Iterations without improvement
            last_improvement_iter: Iteration of last improvement
            
        Returns:
            True if algorithm should terminate
        """
        # Temperature-based termination
        if temperature < self.final_temp and iteration > 1000:
            return True
        
        # Stagnation-based termination
        if stagnation_counter > 2000 and iteration > 5000:
            return True
        
        # Long-term stagnation
        if iteration - last_improvement_iter > 3000 and iteration > 5000:
            return True
        
        # Convergence detection
        if len(self.fitness_history) > 1000:
            recent_variance = np.var(self.fitness_history[-1000:])
            if recent_variance < 0.01 and iteration > 2000:
                return True
        
        return False

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
        try:
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
            
        except ImportError:
            print("Warning: matplotlib not available, skipping SA metrics plot")
        except Exception as e:
            print(f"Warning: Could not plot SA metrics: {e}")

    def get_optimization_summary(self) -> str:
        """
        Generate a comprehensive summary of the optimization run.
        
        Returns:
            Formatted string with optimization details
        """
        if not hasattr(self, 'fitness_history') or not self.fitness_history:
            return "No optimization data available"
        
        lines = []
        lines.append("=" * 80)
        lines.append("SIMULATED ANNEALING OPTIMIZATION SUMMARY")
        lines.append("=" * 80)
        
        # Basic statistics
        lines.append(f"Algorithm Parameters:")
        lines.append(f"  Max iterations: {self.max_iterations}")
        lines.append(f"  Initial temperature: {self.initial_temp}")
        lines.append(f"  Final temperature: {self.final_temp}")
        lines.append(f"  Cooling rate: {self.cooling_rate}")
        lines.append(f"  Step size: {self.step_size} kg")
        lines.append(f"  Budget cap: {self.cost_cap:,.0f} Toman")
        
        # Performance statistics
        iterations_run = len(self.fitness_history)
        final_temp = self.temp_history[-1] if self.temp_history else 0
        final_acceptance = self.acceptance_history[-1] if self.acceptance_history else 0
        
        lines.append(f"\nPerformance Statistics:")
        lines.append(f"  Iterations completed: {iterations_run:,}")
        lines.append(f"  Final temperature: {final_temp:.4f}")
        lines.append(f"  Final acceptance rate: {final_acceptance:.1f}%")
        
        # Fitness progression
        initial_fitness = self.fitness_history[0]
        final_fitness = max(self.fitness_history)
        improvement = final_fitness - initial_fitness
        
        lines.append(f"\nFitness Progression:")
        lines.append(f"  Initial fitness: {initial_fitness:.2f}")
        lines.append(f"  Final fitness: {final_fitness:.2f}")
        lines.append(f"  Total improvement: {improvement:.2f}")
        lines.append(f"  Improvement rate: {improvement/iterations_run*1000:.3f} per 1000 iterations")
        
        lines.append("=" * 80)
        return "\n".join(lines)