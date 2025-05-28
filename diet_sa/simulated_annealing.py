from .data_loader import DataLoader
from .models import Solution
from .fitness import FitnessEvaluator
from .plotting import Plotter
from typing import List, Dict, Tuple
import math
import random

random.seed(42)
import numpy as np

class SimulatedAnnealing:
    """
    Simulated Annealing algorithm for diet optimization.
    """
    
    def __init__(
        self,
        csv_path: str,
        max_iterations: int = 10000,
        initial_temp: float = 1000.0,  
        final_temp: float = 0.1,
        cooling_rate: float = 0.995,
        cost_cap: float = 4_000_000.0,
        step_size: float = 1.5,         
    ):
        # Load food data
        foods, columns = DataLoader.load_foods(csv_path)
        
        # Monthly nutritional requirements
        daily_requirements = {
            'calories': 2000,
            'protein': 100,
            'fat': 60,
            'carbs': 250,
            'fiber': 25,
            'calcium': 1000,
            'iron': 18,
        }
        min_req = {nut: val * 30 for nut, val in daily_requirements.items()}
        
        # Optimal targets - these are what Lili really wants for best health
        optimal_daily = {
            'calories': 1990,   # Less calories for weight management 
            'protein': 110,     # More protein for muscle strength
            'fat': 50,          # Less fat for healthy diet
            'carbs': 235,       # Less carbs to control sugar intake
            'fiber': 33,        # More fiber for digestion
            'calcium': 1025,    # More calcium for strong bones
            'iron': 25,         # More iron to prevent anemia
        }
        optimal = {nut: val * 30 for nut, val in optimal_daily.items()}
        
        # here
        # Nutrient importance weights
        # weights = {
        #     'calories': 1.55,
        #     'protein': 3.0,
        #     'fat': 1.5,
        #     'carbs': 1.55,
        #     'fiber': 3.5,
        #     'calcium': 3.0,
        #     'iron': 3.0,
        # }
        weights = {
                'calories': 0.95, 
                'protein': 1.23, #2-5
                'fat': 1.2,
                'carbs': 1.2,     
                'fiber': 1.25,  #5-7
                'calcium': 1.3,
                'iron': 1.2,
            }
        
        # Create fitness evaluator
        self.evaluator = FitnessEvaluator(foods, min_req, optimal, weights, cost_cap)
        
        # Store parameters
        self.foods = foods
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.cost_cap = cost_cap
        self.step_size = step_size
        self.columns = columns
        
        # Store requirements for reporting
        self.min_daily = daily_requirements
        self.optimal_daily = optimal_daily
        
        # Tracking variables
        self.temp_history = []
        self.fitness_history = []
        self.cost_history = []
        self.best_fitness_history = []
        
        self.plotter = Plotter()

    def generate_initial_solution(self) -> Solution:
        """Generate smart initial solution."""
        s_initial = Solution([0.0]*len(self.foods), self.evaluator)
        return s_initial.random_solution(len(self.foods), self.evaluator)

    def generate_neighbor(self, current_solution: Solution) -> Solution:
        """
        Generate neighbor solution using multiple strategies.
        """
        # current_cost = current_solution.get_total_cost()
        # deficient_nutrients = current_solution.get_deficient_nutrients()
        
        return current_solution.perturb_simple(self.step_size)
    
        # # Choose strategy based on current state
        # if current_cost >= self.cost_cap:  # Near budget limit
        #     return current_solution._reduce_cost_neighbor(self.foods.copy(), self.step_size)
        # elif len(deficient_nutrients) > 0:  # Has deficiencies
        #     return current_solution.perturb_focused(deficient_nutrients, self.step_size)
        # else:  # Normal case
        #     return current_solution.perturb_simple(self.step_size)

    def acceptance_probability(self, current_fitness: float, new_fitness: float, temperature: float) -> float:
        """Calculate acceptance probability."""
        if new_fitness > current_fitness:
            return 1.0  # Always accept better solutions
        
        if temperature <= self.final_temp:
            return 0.0
        
        delta = new_fitness - current_fitness
        return math.exp(delta / temperature)

    def _check_stopping_conditions(self, iteration: int, temperature: float, 
                                  stagnation_counter: int, last_improvement_iteration: int,
                                  consecutive_no_accept: int, acceptance_rate: float) -> Tuple[bool, str]:
        """
        Check multiple stopping conditions and return whether to stop and why.
        """
        # Don't stop too early - allow at least 350 iterations
        if iteration < 1000:
            return False, ""
        
        # 1. Temperature-based stopping
        if temperature < self.final_temp:
            return True, f"Temperature too low ({temperature:.6f})"
        
        # # 2. Very low acceptance rate for extended period
        # if iteration > 2000 and acceptance_rate < 1.0 and consecutive_no_accept > 500:
        #     return True, f"Very low acceptance rate ({acceptance_rate:.1f}%) for {consecutive_no_accept} iterations"
        
        # # 3. Long stagnation without improvement
        # iterations_since_improvement = iteration - last_improvement_iteration
        # if iterations_since_improvement > 3000 and iteration > 3000:
        #     return True, f"No improvement for {iterations_since_improvement} iterations"
        
        # # 4. Excessive stagnation counter
        # if stagnation_counter > 5000 and iteration > 5000:
        #     return True, f"Excessive stagnation ({stagnation_counter} iterations)"
        
        # 5. Convergence detection - check if best fitness is stable
        if len(self.best_fitness_history) > 2000 and iteration > 2000:
            recent_window = min(1000, len(self.best_fitness_history))
            recent_best = self.best_fitness_history[-recent_window:]
            fitness_variance = np.var(recent_best)
            
            if fitness_variance < 0.01:  # Very small variance in fitness
                return True, f"Converged - fitness variance: {fitness_variance:.6f}"
        
        # # 6. Check if we've reached a good enough solution
        # if self.best_fitness_history and iteration > 2000:
        #     current_best = max(self.best_fitness_history)
        #     # If fitness is positive and high, we might have found a good solution
        #     if current_best > 100 and iterations_since_improvement > 2000:
        #         return True, f"Good solution found (fitness: {current_best:.2f}) with no recent improvement"
        
        return False, ""

    def update_temperature(self, iteration: int) -> float:
        """Update temperature using adaptive geometric cooling."""
        # Basic geometric cooling
        temp = self.initial_temp * (self.cooling_rate ** iteration)
        
        # Adaptive cooling based on recent performance
        if len(self.fitness_history) > 200:
            # Check recent improvement
            recent_improvement = max(self.fitness_history[-50:]) - max(self.fitness_history[-200:-150])
            
            if recent_improvement > 1.0:  # Good improvement
                # Cool slower to allow more exploration
                temp *= 1.02
            elif recent_improvement < 0.1:  # Poor improvement
                # Cool faster to focus search
                temp *= 0.98
        
        # Ensure temperature doesn't go below minimum
        return max(temp, self.final_temp)

    def run(self) -> Tuple[List[float], float, float, Dict[str, float]]:
        """Run the Simulated Annealing algorithm."""
        # Initialize
        current_solution = self.generate_initial_solution()
        best_solution = current_solution.copy()
        temperature = self.initial_temp
        
        accepted_moves = 0
        total_moves = 0
        last_improvement_iteration = 0
        stagnation_counter = 0
        consecutive_no_accept = 0
        
        print("Starting Simulated Annealing Optimization:")
        print(f"{'Iteration':>8} | {'Temperature':>10} | {'Current Fitness':>14} | {'Best Fitness':>12} | {'Cost (M)':>9} | {'Accept%':>7}")
        print("-" * 85)
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Generate neighbor
            neighbor = self.generate_neighbor(current_solution)
            
            # Calculate acceptance probability
            accept_prob = self.acceptance_probability(
                current_solution.fitness, neighbor.fitness, temperature
            )
            
            total_moves += 1
            
            # Accept or reject
            if random.uniform(0,1) <= accept_prob:
                current_solution = neighbor
                accepted_moves += 1
                consecutive_no_accept = 0
                
                # Update best solution
                if current_solution.fitness > best_solution.fitness:
                    best_solution = current_solution.copy()
                    last_improvement_iteration = iteration
                    stagnation_counter = 0
                    print(f"    -> New best fitness: {best_solution.fitness:.2f} at iteration {iteration}")
                else:
                    stagnation_counter += 1
            else:
                consecutive_no_accept += 1
                stagnation_counter += 1
            
            # Update temperature
            temperature = self.update_temperature(iteration)
            
            # Record history
            self.temp_history.append(temperature)
            self.fitness_history.append(current_solution.fitness)
            self.best_fitness_history.append(best_solution.fitness)
            
            cost = current_solution.get_total_cost()
            self.cost_history.append(cost)
            
            # Calculate acceptance rate
            acceptance_rate = ((accepted_moves * 100) / total_moves)  if total_moves > 0 else 0
            
            # Print progress
            if iteration % 1000 == 0 or iteration == self.max_iterations - 1:
                print(f"{iteration:8d} | {temperature:10.3f} | {current_solution.fitness:14.2f} | {best_solution.fitness:12.2f} | {cost/1_000_000:9.2f} | {acceptance_rate:6.1f}%")
            
            # Improved early stopping conditions
            should_stop, stop_reason = self._check_stopping_conditions(
                iteration, temperature, stagnation_counter, 
                last_improvement_iteration, consecutive_no_accept, acceptance_rate
            )
            
            if should_stop:
                print(f"\nStopping early at iteration {iteration}: {stop_reason}")
                break
        
        print("-" * 85)
        print(f"Optimization completed after {iteration + 1} iterations")
        print(f"Accepted moves: {accepted_moves}/{total_moves} ({acceptance_rate:.1f}%)")
        print(f"Best fitness: {best_solution.fitness:.2f}")
        print(f"Last improvement at iteration: {last_improvement_iteration}")
        
        # Calculate final results
        final_cost = best_solution.get_total_cost()
        final_nutrients = best_solution.get_daily_nutrient_totals()
        
        # Generate plots
        try:
            self._plot_convergence()
            self.plotter.plot_nutrition_comparison(
                self.min_daily, self.optimal_daily, final_nutrients
            )
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
        
        return best_solution.quantities, best_solution.fitness, final_cost, final_nutrients

    def _plot_convergence(self):
        """Plot SA convergence metrics."""
        try:
            import matplotlib.pyplot as plt
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            iterations = range(len(self.temp_history))
            
            # Temperature decay
            ax1.plot(iterations, self.temp_history, 'r-', linewidth=2)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Temperature')
            ax1.set_title('Temperature Decay')
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
            
            # Fitness progress
            ax2.plot(iterations, self.fitness_history, 'b-', alpha=0.6, label='Current')
            ax2.plot(iterations, self.best_fitness_history, 'g-', linewidth=2, label='Best')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Fitness Score')
            ax2.set_title('Fitness Progress')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Cost progress
            ax3.plot(iterations, [c/1_000_000 for c in self.cost_history], 'orange', linewidth=2)
            ax3.axhline(y=self.cost_cap/1_000_000, color='red', linestyle='--', label='Budget Limit')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Cost (Million Toman)')
            ax3.set_title('Cost Evolution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Fitness improvement trend
            if len(self.best_fitness_history) > 100:
                window = 100
                smoothed = []
                for i in range(window, len(self.best_fitness_history)):
                    smoothed.append(max(self.best_fitness_history[i-window:i]))
                ax4.plot(range(window, len(self.best_fitness_history)), smoothed, 'purple', linewidth=2)
                ax4.set_xlabel('Iteration')
                ax4.set_ylabel('Best Fitness (Smoothed)')
                ax4.set_title('Fitness Improvement Trend')
                ax4.grid(True, alpha=0.3)
            
            plt.suptitle('Simulated Annealing Convergence Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('sa_convergence.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except ImportError:
            print("matplotlib not available for plotting")
        except Exception as e:
            print(f"Error plotting convergence: {e}")