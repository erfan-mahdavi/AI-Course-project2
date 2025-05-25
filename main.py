import argparse
import os
import sys
import time
from typing import Dict, List, Tuple, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from diet_ga.genetic_algorithm import GeneticAlgorithm
    GA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Genetic Algorithm not available: {e}")
    GA_AVAILABLE = False

try:
    from diet_sa.simulated_annealing import SimulatedAnnealing
    SA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Simulated Annealing not available: {e}")
    SA_AVAILABLE = False

def get_algorithm_choice():
    """Ask user to choose between GA and SA algorithms."""
    print("\n" + "="*60)
    print("LILI'S MONTHLY DIET OPTIMIZATION")
    print("="*60)
    print("Available optimization algorithms:")
    
    available_algorithms = []
    if GA_AVAILABLE:
        print("  1. GA  - Genetic Algorithm (Population-based)")
        available_algorithms.append(('1', 'ga'))
    if SA_AVAILABLE:
        print("  2. SA  - Simulated Annealing (Single-solution)")
        available_algorithms.append(('2', 'sa'))
    
    if not available_algorithms:
        print("Error: No optimization algorithms are available!")
        sys.exit(1)
    
    print("-"*60)
    
    while True:
        choice = input("Choose algorithm (1 for GA, 2 for SA, or 'ga'/'sa'): ").strip().lower()
        
        if GA_AVAILABLE and choice in ['1', 'ga']:
            print(f"Selected: Genetic Algorithm")
            return 'ga'
        elif SA_AVAILABLE and choice in ['2', 'sa']:
            print(f"Selected: Simulated Annealing")
            return 'sa'
        else:
            available_options = []
            if GA_AVAILABLE:
                available_options.extend(['1', 'ga'])
            if SA_AVAILABLE:
                available_options.extend(['2', 'sa'])
            print(f"Invalid choice. Please enter one of: {', '.join(available_options)}")

def find_foods_file(provided_path: str) -> str:
    """Find the foods.csv file in various locations."""
    if os.path.exists(provided_path):
        return provided_path

def run_genetic_algorithm(args) -> Tuple[List[float], float, float, Dict[str, float], float, Any]:
    """Run the Genetic Algorithm with given parameters."""
    if not GA_AVAILABLE:
        raise ImportError("Genetic Algorithm is not available")
    
    print("\n" + "="*80)
    print(f"{'MONTHLY DIET OPTIMIZATION VIA GENETIC ALGORITHM':^80}")
    print("="*80)
    print(f"GA Parameters:")
    print(f"  - Population size: {args.pop}")
    print(f"  - Generations: {args.gens}")
    print(f"  - Mutation rate: {args.mut} → {args.final_mut}")
    print(f"  - Elite size: {args.retain}")
    print(f"  - Budget cap: {args.cost_cap:,.0f} Toman")
    print("-"*80)
    
    start_time = time.time()
    
    try:
        ga = GeneticAlgorithm(
            csv_path=args.foods,
            pop_size=args.pop,
            generations=args.gens,
            init_mut_rate=args.mut,
            final_mut_rate=args.final_mut,
            retain=args.retain,
            cost_cap=args.cost_cap
        )
        
        best_chromosome, best_fitness, actual_cost, nutrient_totals = ga.run()
        elapsed_time = time.time() - start_time
        
        return best_chromosome, best_fitness, actual_cost, nutrient_totals, elapsed_time, ga
        
    except Exception as e:
        print(f"Error running Genetic Algorithm: {e}")
        raise

def run_simulated_annealing(args) -> Tuple[List[float], float, float, Dict[str, float], float, Any]:
    """Run the Simulated Annealing algorithm with given parameters."""
    if not SA_AVAILABLE:
        raise ImportError("Simulated Annealing is not available")
    
    print("\n" + "="*80)
    print(f"{'MONTHLY DIET OPTIMIZATION VIA SIMULATED ANNEALING':^80}")
    print("="*80)
    print(f"SA Parameters:")
    print(f"  - Max iterations: {args.iterations}")
    print(f"  - Temperature: {args.temp} → {args.final_temp}")
    print(f"  - Cooling rate: {args.cooling}")
    print(f"  - Step size: {args.step_size} kg")
    print(f"  - Budget cap: {args.cost_cap:,.0f} Toman")
    print("-"*80)
    
    start_time = time.time()
    
    try:
        sa = SimulatedAnnealing(
            csv_path=args.foods,
            max_iterations=args.iterations,
            initial_temp=args.temp,
            final_temp=args.final_temp,
            cooling_rate=args.cooling,
            cost_cap=args.cost_cap,
            step_size=args.step_size
        )
        
        best_solution, best_fitness, actual_cost, nutrient_totals = sa.run()
        elapsed_time = time.time() - start_time
        
        return best_solution, best_fitness, actual_cost, nutrient_totals, elapsed_time, sa
        
    except Exception as e:
        print(f"Error running Simulated Annealing: {e}")
        raise

def get_daily_requirements(optimizer):
    """Get daily nutritional requirements from optimizer."""
    if hasattr(optimizer, 'min_daily'):
        return optimizer.min_daily, optimizer.optimal_daily
    elif hasattr(optimizer, 'evaluator'):
        evaluator = optimizer.evaluator
        min_daily = {nut: req/30 for nut, req in evaluator.min_req.items()}
        optimal_daily = {nut: opt/30 for nut, opt in evaluator.optimal.items()}
        return min_daily, optimal_daily
    else:
        # Default values
        return {
            'calories': 2000, 'protein': 100, 'fat': 60, 'carbs': 250,
            'fiber': 25, 'calcium': 1000, 'iron': 18
        }, {
            'calories': 1800, 'protein': 110, 'fat': 55, 'carbs': 225,
            'fiber': 30, 'calcium': 1100, 'iron': 20
        }

def get_foods_list(optimizer):
    """Get foods list from optimizer."""
    if hasattr(optimizer, 'foods'):
        return optimizer.foods
    elif hasattr(optimizer, 'evaluator') and hasattr(optimizer.evaluator, 'foods'):
        return optimizer.evaluator.foods
    else:
        return []

def get_budget_cap(optimizer):
    """Get budget cap from optimizer."""
    if hasattr(optimizer, 'cost_cap'):
        return optimizer.cost_cap
    elif hasattr(optimizer, 'evaluator') and hasattr(optimizer.evaluator, 'cost_cap'):
        return optimizer.evaluator.cost_cap
    else:
        return 4_000_000.0

def print_results(algorithm: str, best_solution: List[float], best_fitness: float, 
                 actual_cost: float, nutrient_totals: Dict[str, float], 
                 elapsed_time: float, optimizer: Any) -> None:
    """Print optimization results."""
    
    print("\n" + "="*80)
    print(f"{'OPTIMIZATION RESULTS':^80}")
    print("="*80)
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Execution time: {elapsed_time:.2f} seconds")
    print(f"Best fitness score: {best_fitness:.2f}")
    print(f"Total monthly cost: {actual_cost:,.0f} Toman")
    
    budget_cap = get_budget_cap(optimizer)
    print(f"Budget cap: {budget_cap:,.0f} Toman")
    
    budget_utilization = (actual_cost / budget_cap) * 100
    print(f"Budget utilization: {budget_utilization:.1f}%")
    
    # Get nutritional requirements
    min_daily, optimal_daily = get_daily_requirements(optimizer)
    
    # Nutrient summary
    print("\nDaily Nutrient Summary:")
    print("-"*80)
    print(f"{'Nutrient':^12} | {'Required':^10} | {'Actual':^10} | {'Optimal':^10} | {'% of Req':^10} | {'Status':^10}")
    print("-"*80)
    
    for nut, actual in nutrient_totals.items():
        required = min_daily.get(nut, 0)
        optimal = optimal_daily.get(nut, required)
        pct_req = (actual / required) * 100 if required > 0 else 0
        
        if actual >= required:
            status = "✓ GOOD"
        else:
            status = "✗ LOW"
        
        print(f"{nut.capitalize():12} | {required:10.1f} | {actual:10.1f} | {optimal:10.1f} | {pct_req:9.1f}% | {status:10}")
    
    # Food basket summary
    print("\nOptimal Food Basket (Monthly):")
    print("-"*80)
    print(f"{'Food Item':^25} | {'Quantity (kg)':^12} | {'Daily (g)':^10} | {'Cost (T)':^12} | {'% Budget':^10}")
    print("-"*80)
    
    foods = get_foods_list(optimizer)
    food_data = []
    
    for i, qty in enumerate(best_solution):
        if i < len(foods) and qty > 0.01:  # Only significant quantities
            food = foods[i]
            daily_grams = (qty * 1000) / 30
            item_cost = qty * food.price
            budget_pct = (item_cost / actual_cost) * 100 if actual_cost > 0 else 0
            food_data.append((food.name, qty, daily_grams, item_cost, budget_pct))
    
    # Sort by quantity descending
    food_data.sort(key=lambda x: x[1], reverse=True)
    
    for name, qty, daily_g, cost, budget_pct in food_data:
        print(f"{name:25} | {qty:12.2f} | {daily_g:10.1f} | {cost:12,.0f} | {budget_pct:9.1f}%")
    
    print("-"*80)
    print(f"Total food weight: {sum(best_solution):.1f} kg/month")
    
    # Algorithm-specific info
    if algorithm == 'ga':
        print_ga_info(optimizer, elapsed_time)
    else:
        print_sa_info(optimizer, elapsed_time)
    
    print("="*80)

def print_ga_info(ga, elapsed_time: float):
    """Print GA-specific information."""
    try:
        print(f"\nGenetic Algorithm Performance:")
        if hasattr(ga, 'generations'):
            print(f"- Generations completed: {ga.generations}")
            if ga.generations > 0:
                print(f"- Time per generation: {elapsed_time/ga.generations:.3f} seconds")
        if hasattr(ga, 'pop_size'):
            print(f"- Population size: {ga.pop_size}")
    except:
        pass

def print_sa_info(sa, elapsed_time: float):
    """Print SA-specific information."""
    try:
        print(f"\nSimulated Annealing Performance:")
        iterations = len(getattr(sa, 'temp_history', [])) or getattr(sa, 'max_iterations', 0)
        print(f"- Iterations completed: {iterations}")
        if iterations > 0:
            print(f"- Time per iteration: {elapsed_time/iterations:.4f} seconds")
        
        if hasattr(sa, 'temp_history') and sa.temp_history:
            print(f"- Temperature: {sa.temp_history[0]:.1f} → {sa.temp_history[-1]:.4f}")
        
        if hasattr(sa, 'acceptance_history') and sa.acceptance_history:
            print(f"- Final acceptance rate: {sa.acceptance_history[-1]:.1f}%")
    except:
        pass

def main():
    """Main function for running diet optimization."""
    
    parser = argparse.ArgumentParser(
        description='Monthly diet optimization via GA or SA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Algorithm Information:
  GA (Genetic Algorithm): Population-based evolutionary optimization
  SA (Simulated Annealing): Single-solution temperature-based optimization
        """)
    
    # Common arguments
    parser.add_argument('--foods', default='foods.csv',
                        help='Path to CSV file containing food data')
    parser.add_argument('--cost-cap', type=float, default=4_000_000.0,
                        help='Monthly budget cap in Toman')
    
    # GA arguments
    ga_group = parser.add_argument_group('Genetic Algorithm Options')
    ga_group.add_argument('--pop', type=int, default=1600,
                         help='Population size for GA')
    ga_group.add_argument('--gens', type=int, default=99,
                         help='Number of generations for GA')
    ga_group.add_argument('--mut', type=float, default=0.25,
                         help='Initial mutation rate for GA')
    ga_group.add_argument('--final-mut', type=float, default=0.01,
                         help='Final mutation rate for GA')
    ga_group.add_argument('--retain', type=int, default=80,
                         help='Number of elite individuals to retain')
    
    # SA arguments
    sa_group = parser.add_argument_group('Simulated Annealing Options')
    sa_group.add_argument('--iterations', type=int, default=900,
                         help='Maximum iterations for SA')
    sa_group.add_argument('--temp', type=float, default=9000000.0,
                         help='Initial temperature for SA')
    sa_group.add_argument('--final-temp', type=float, default=0.001,
                         help='Final temperature for SA')
    sa_group.add_argument('--cooling', type=float, default=0.9,
                         help='Cooling rate for SA')
    sa_group.add_argument('--step-size', type=float, default=1.5,
                         help='Step size for SA (kg)')
    
    args = parser.parse_args()

    # Check if at least one algorithm is available
    if not GA_AVAILABLE and not SA_AVAILABLE:
        print("Error: Neither GA nor SA is available!")
        return 1

    try:
        # Find foods file
        args.foods = find_foods_file(args.foods)
        
        # Choose algorithm
        algorithm = get_algorithm_choice()
        
        # Run selected algorithm
        if algorithm == 'ga':
            if not GA_AVAILABLE:
                print("Error: Genetic Algorithm is not available!")
                return 1
            best_solution, best_fitness, actual_cost, nutrient_totals, elapsed_time, optimizer = run_genetic_algorithm(args)
        else:  # SA
            if not SA_AVAILABLE:
                print("Error: Simulated Annealing is not available!")
                return 1
            best_solution, best_fitness, actual_cost, nutrient_totals, elapsed_time, optimizer = run_simulated_annealing(args)
        
        # Print results
        print_results(algorithm, best_solution, best_fitness, actual_cost, 
                     nutrient_totals, elapsed_time, optimizer)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        return 1
    except FileNotFoundError as e:
        print(f"File error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)