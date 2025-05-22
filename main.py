import argparse
import os
import time
from diet_ga.genetic_algorithm import GeneticAlgorithm
from diet_sa.simulated_annealing import SimulatedAnnealing

def get_algorithm_choice():
    """
    Interactively ask user to choose between GA and SA algorithms.
    
    Returns:
        str: Either 'ga' for Genetic Algorithm or 'sa' for Simulated Annealing
    """
    print("\n" + "="*60)
    print("MONTHLY DIET OPTIMIZATION")
    print("="*60)
    print("Available optimization algorithms:")
    print("  1. GA  - Genetic Algorithm (Population-based evolutionary approach)")
    print("  2. SA  - Simulated Annealing (Single-solution temperature-based approach)")
    print("-"*60)
    
    while True:
        choice = input("Choose algorithm (1 for GA, 2 for SA, or 'ga'/'sa'): ").strip().lower()
        
        if choice in ['1', 'ga']:
            print(f"Selected: Genetic Algorithm")
            return 'ga'
        elif choice in ['2', 'sa']:
            print(f"Selected: Simulated Annealing")
            return 'sa'
        else:
            print("Invalid choice. Please enter '1', '2', 'ga', or 'sa'.")

def run_genetic_algorithm(args):
    """
    Run the Genetic Algorithm with given parameters.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        tuple: (best_solution, best_fitness, actual_cost, nutrient_totals, elapsed_time)
    """
    print("\n" + "="*80)
    print(f"{'MONTHLY DIET OPTIMIZATION VIA GENETIC ALGORITHM':^80}")
    print("="*80)
    print(f"GA Parameters:")
    print(f"  - Population size: {args.pop}")
    print(f"  - Generations: {args.gens}")
    print(f"  - Initial mutation rate: {args.mut}")
    print(f"  - Final mutation rate: {args.final_mut}")
    print(f"  - Elite size (retained): {args.retain}")
    print(f"  - Budget cap: {args.cost_cap:,.0f} Toman")
    print(f"  - Foods data: {args.foods}")
    print("-"*80)
    
    # Start time measurement
    start_time = time.time()
    
    # Initialize and run GA
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
    
    # End time measurement
    elapsed_time = time.time() - start_time
    
    return best_chromosome, best_fitness, actual_cost, nutrient_totals, elapsed_time, ga

def run_simulated_annealing(args):
    """
    Run the Simulated Annealing algorithm with given parameters.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        tuple: (best_solution, best_fitness, actual_cost, nutrient_totals, elapsed_time)
    """
    print("\n" + "="*80)
    print(f"{'MONTHLY DIET OPTIMIZATION VIA SIMULATED ANNEALING':^80}")
    print("="*80)
    print(f"SA Parameters:")
    print(f"  - Max iterations: {args.iterations}")
    print(f"  - Initial temperature: {args.temp}")
    print(f"  - Final temperature: {args.final_temp}")
    print(f"  - Cooling rate: {args.cooling}")
    print(f"  - Step size: {args.step_size} kg")
    print(f"  - Budget cap: {args.cost_cap:,.0f} Toman")
    print(f"  - Foods data: {args.foods}")
    print("-"*80)
    
    # Start time measurement
    start_time = time.time()
    
    # Initialize and run SA
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
    
    # End time measurement
    elapsed_time = time.time() - start_time
    
    return best_solution, best_fitness, actual_cost, nutrient_totals, elapsed_time, sa

def print_results(algorithm, best_solution, best_fitness, actual_cost, nutrient_totals, elapsed_time, optimizer):
    """
    Print the optimization results in a formatted way.
    
    Args:
        algorithm: Algorithm used ('ga' or 'sa')
        best_solution: Best solution found
        best_fitness: Best fitness score
        actual_cost: Total cost of the solution
        nutrient_totals: Dictionary of nutrient totals
        elapsed_time: Time taken for optimization
        optimizer: The optimizer instance (GA or SA)
    """
    print("\n" + "="*80)
    print(f"{'OPTIMIZATION RESULTS':^80}")
    print("="*80)
    print(f"Algorithm used: {algorithm.upper()}")
    print(f"Execution time: {elapsed_time:.2f} seconds")
    print(f"Best fitness score: {best_fitness:.2f}")
    print(f"Total monthly cost: {actual_cost:,.0f} Toman (Budget: {optimizer.evaluator.cost_cap if hasattr(optimizer, 'evaluator') else optimizer.population.evaluator.cost_cap:,.0f} Toman)")
    
    # Calculate budget utilization
    budget_cap = optimizer.evaluator.cost_cap if hasattr(optimizer, 'evaluator') else optimizer.population.evaluator.cost_cap
    budget_utilization = (actual_cost / budget_cap) * 100
    print(f"Budget utilization: {budget_utilization:.1f}%")
    
    # Print nutrient summary
    print_nutrient_summary(nutrient_totals, optimizer)
    
    # Print food basket
    print_food_basket(best_solution, actual_cost, optimizer)
    
    # Print algorithm-specific information
    if algorithm == 'ga':
        print_ga_specific_info(optimizer, elapsed_time)
    else:
        print_sa_specific_info(optimizer, elapsed_time)
    
    print("="*80)

def print_nutrient_summary(nutrient_totals, optimizer):
    """
    Print detailed nutrient analysis.
    
    Args:
        nutrient_totals: Dictionary of daily nutrient totals
        optimizer: The optimizer instance
    """
    print("\nNutrient Summary (Daily Values):")
    print("-"*80)
    print(f"{'Nutrient':^15} | {'Minimum':^12} | {'Actual':^12} | {'Optimal':^12} | {'% of Min':^12} | {'% of Opt':^12} | {'Status':^10}")
    print("-"*80)
    
    # Get daily requirements from optimizer
    if hasattr(optimizer, 'min_daily'):
        min_daily = optimizer.min_daily
        optimal_daily = optimizer.optimal_daily
    else:
        # For SA, calculate from evaluator
        evaluator = optimizer.evaluator if hasattr(optimizer, 'evaluator') else optimizer.population.evaluator
        min_daily = {nut: req/30 for nut, req in evaluator.min_req.items()}
        optimal_daily = {nut: opt/30 for nut, opt in evaluator.optimal.items()}
    
    # Get directions
    evaluator = optimizer.evaluator if hasattr(optimizer, 'evaluator') else optimizer.population.evaluator
    directions = evaluator.directions
    
    for nut, val in nutrient_totals.items():
        min_val = min_daily[nut]
        opt_val = optimal_daily.get(nut, min_val)
        pct_min = (val / min_val) * 100 if min_val > 0 else 0
        pct_opt = (val / opt_val) * 100 if opt_val > 0 else 0
        
        # Determine status based on nutrient direction
        direction = directions.get(nut, 'max')
        if direction == 'max':  # Higher is better
            if val < min_val:
                status = "DEFICIT"
            elif val <= opt_val:
                status = "GOOD"
            else:
                status = "EXCESS"
        else:  # 'min' direction - lower is better
            if val > min_val:
                status = "HIGH"
            elif val >= opt_val:
                status = "GOOD"
            else:
                status = "LOW"
        
        print(f"{nut.capitalize():15} | {min_val:12.1f} | {val:12.1f} | {opt_val:12.1f} | {pct_min:11.1f}% | {pct_opt:11.1f}% | {status:10}")

def print_food_basket(best_solution, actual_cost, optimizer):
    """
    Print the optimal food basket.
    
    Args:
        best_solution: List of food quantities
        actual_cost: Total cost
        optimizer: The optimizer instance
    """
    print("\nOptimal Food Basket (Monthly Quantities):")
    print("-"*80)
    print(f"{'Food Item':^30} | {'Quantity (kg)':^15} | {'Daily (g)':^12} | {'Total Cost (T)':^18} | {'% of Budget':^15}")
    print("-"*80)
    
    # Get foods from optimizer
    foods = optimizer.foods if hasattr(optimizer, 'foods') else optimizer.evaluator.foods
    
    # Sort foods by quantity in descending order
    food_quantities = [(food.name, qty, food.price * qty) 
                     for food, qty in zip(foods, best_solution)]
    food_quantities.sort(key=lambda x: x[1], reverse=True)
    
    for name, qty, cost in food_quantities:
        if qty > 0.01:  # Only show foods with significant quantities
            daily_grams = (qty * 1000) / 30  # Convert to grams per day
            budget_pct = (cost / actual_cost) * 100 if actual_cost > 0 else 0
            print(f"{name:30} | {qty:15.2f} | {daily_grams:12.1f} | {cost:18,.0f} | {budget_pct:14.1f}%")
    
    print("-"*80)
    print(f"Total monthly cost: {actual_cost:,.0f} Toman")

def print_ga_specific_info(ga, elapsed_time):
    """
    Print Genetic Algorithm specific performance information.
    
    Args:
        ga: GeneticAlgorithm instance
        elapsed_time: Total execution time
    """
    print(f"\nGenetic Algorithm Performance:")
    print(f"- Generations completed: {ga.generations}")
    print(f"- Population size: {ga.pop_size}")
    print(f"- Time per generation: {elapsed_time/ga.generations:.3f} seconds")
    
    if hasattr(ga, 'diversity_hist') and ga.diversity_hist:
        print(f"- Final population diversity: {ga.diversity_hist[-1]:.3f}")
        print(f"- Initial population diversity: {ga.diversity_hist[0]:.3f}")
    
    # Population fitness statistics
    if hasattr(ga, 'population') and ga.population.individuals:
        fitness_values = [ind.fitness for ind in ga.population.individuals 
                         if ind.fitness > -float('inf')]
        if len(fitness_values) > 1:
            fitness_range = max(fitness_values) - min(fitness_values)
            print(f"- Final population fitness range: {fitness_range:.2f}")
            if fitness_range < 1.0:
                print("- Status: Population has converged")
            else:
                print("- Status: Population maintains diversity")

def print_sa_specific_info(sa, elapsed_time):
    """
    Print Simulated Annealing specific performance information.
    
    Args:
        sa: SimulatedAnnealing instance
        elapsed_time: Total execution time
    """
    iterations_completed = len(sa.temp_history) if hasattr(sa, 'temp_history') else sa.max_iterations
    print(f"\nSimulated Annealing Performance:")
    print(f"- Iterations completed: {iterations_completed}")
    print(f"- Time per iteration: {elapsed_time/iterations_completed:.4f} seconds")
    
    if hasattr(sa, 'temp_history') and sa.temp_history:
        print(f"- Initial temperature: {sa.temp_history[0]:.2f}")
        print(f"- Final temperature: {sa.temp_history[-1]:.4f}")
        temp_reduction = sa.temp_history[-1] / sa.temp_history[0]
        print(f"- Temperature reduction ratio: {temp_reduction:.6f}")
    
    if hasattr(sa, 'acceptance_history') and sa.acceptance_history:
        final_acceptance = sa.acceptance_history[-1]
        print(f"- Final acceptance rate: {final_acceptance:.1f}%")
    
    # Convergence analysis
    if hasattr(sa, 'fitness_history') and sa.fitness_history:
        recent_window = min(1000, len(sa.fitness_history))
        if recent_window > 1:
            recent_fitness = sa.fitness_history[-recent_window:]
            fitness_improvement = max(recent_fitness) - min(recent_fitness)
            print(f"- Recent fitness improvement (last {recent_window} iter): {fitness_improvement:.2f}")
            
            if fitness_improvement < 0.1:
                print("- Status: Algorithm has converged")
            else:
                print("- Status: Still finding improvements")

def main():
    """
    Main function that interactively selects and runs the chosen optimization algorithm.
    """
    parser = argparse.ArgumentParser(
        description='Monthly diet optimization via GA or SA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Algorithm Information:
  GA (Genetic Algorithm): Population-based evolutionary optimization
    - Uses multiple solutions that evolve over generations
    - Good for global optimization and exploration
    - Slower but more thorough search
    
  SA (Simulated Annealing): Single-solution temperature-based optimization
    - Uses one solution that is iteratively improved
    - Good balance of exploration and exploitation
    - Faster iterations, good for local optimization with escape mechanism

The program will ask you to choose which algorithm to use interactively.
        """)
    
    # Common arguments
    default_foods_path = os.path.join(os.path.dirname(__file__), 'foods.csv')
    parser.add_argument('--foods', default=default_foods_path,
                        help='Path to CSV file containing food data')
    parser.add_argument('--cost-cap', type=float, default=4_000_000.0,
                        help='Monthly budget cap in Toman')
    
    # Genetic Algorithm arguments
    ga_group = parser.add_argument_group('Genetic Algorithm Options')
    ga_group.add_argument('--pop', type=int, default=1500,
                         help='Population size for GA')
    ga_group.add_argument('--gens', type=int, default=90,
                         help='Number of generations for GA')
    ga_group.add_argument('--mut', type=float, default=0.25,
                         help='Initial mutation rate for GA')
    ga_group.add_argument('--final-mut', type=float, default=0.01,
                         help='Final (minimum) mutation rate for GA')
    ga_group.add_argument('--retain', type=int, default=70,
                         help='Number of best individuals to retain each generation')
    
    # Simulated Annealing arguments
    sa_group = parser.add_argument_group('Simulated Annealing Options')
    sa_group.add_argument('--iterations', type=int, default=10000,
                         help='Maximum number of iterations for SA')
    sa_group.add_argument('--temp', type=float, default=1000.0,
                         help='Initial temperature for SA')
    sa_group.add_argument('--final-temp', type=float, default=0.1,
                         help='Final (minimum) temperature for SA')
    sa_group.add_argument('--cooling', type=float, default=0.95,
                         help='Cooling rate for SA (0 < rate < 1)')
    sa_group.add_argument('--step-size', type=float, default=2.0,
                         help='Maximum change in food quantity per perturbation (kg)')
    
    args = parser.parse_args()

    # Check if food data file exists
    if not os.path.exists(args.foods):
        print(f"Error: File '{args.foods}' not found!")
        print(f"Current directory: {os.getcwd()}")
        print(f"Available files: {os.listdir()}")
        return

    try:
        # Interactive algorithm selection
        algorithm = get_algorithm_choice()
        
        # Run the selected algorithm
        if algorithm == 'ga':
            best_solution, best_fitness, actual_cost, nutrient_totals, elapsed_time, optimizer = run_genetic_algorithm(args)
        else:  # algorithm == 'sa'
            best_solution, best_fitness, actual_cost, nutrient_totals, elapsed_time, optimizer = run_simulated_annealing(args)
        
        # Print results
        print_results(algorithm, best_solution, best_fitness, actual_cost, nutrient_totals, elapsed_time, optimizer)
        
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
    except Exception as e:
        print(f"Runtime error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()