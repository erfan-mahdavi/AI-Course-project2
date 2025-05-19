import argparse
import os
import time
from diet_ga.genetic_algorithm import GeneticAlgorithm

def main():
    parser = argparse.ArgumentParser(
        description='Monthly diet optimization via GA'
    )
    # *** Use relative path from the script location
    default_foods_path = os.path.join(os.path.dirname(__file__), 'foods.csv')
    parser.add_argument('--foods', default=default_foods_path)
    parser.add_argument('--pop', type=int, default=10)
    parser.add_argument('--gens', type=int, default=10)
    parser.add_argument('--mut', type=float, default=0.15)
    parser.add_argument('--final-mut', type=float, default=0.005,
                        help='Final (minimum) mutation rate')
    parser.add_argument('--retain', type=int, default=800)
    parser.add_argument('--cost-cap', type=float, default=4_000_000.0,
                        help='Monthly budget cap in Toman')
    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.foods):
        print(f"Error: File '{args.foods}' not found!")
        print(f"Current directory: {os.getcwd()}")
        print(f"Available files: {os.listdir()}")
        return

    try:
        print("\n" + "="*80)
        print(f"{'MONTHLY DIET OPTIMIZATION VIA GENETIC ALGORITHM':^80}")
        print("="*80)
        print(f"Parameters:")
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
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Print summary of results
        print("\n" + "="*80)
        print(f"{'OPTIMIZATION RESULTS':^80}")
        print("="*80)
        print(f"Execution time: {elapsed_time:.2f} seconds")
        print(f"Best fitness score: {best_fitness:.2f}")
        print(f"Total monthly cost: {actual_cost:,.0f} Toman (Budget: {args.cost_cap:,.0f} Toman)")
        print(f"Budget utilization: {(actual_cost/args.cost_cap)*100:.1f}%")
        
        # Print nutrient summary
        print("\nNutrient Summary (Daily Values):")
        print("-"*80)
        print(f"{'Nutrient':^15} | {'Minimum':^12} | {'Actual':^12} | {'Optimal':^12} | {'% of Min':^12} | {'% of Opt':^12}")
        print("-"*80)
        
        for nut, val in nutrient_totals.items():
            min_val = ga.min_daily[nut]
            opt_val = ga.optimal_daily[nut]
            pct_min = (val / min_val) * 100
            pct_opt = (val / opt_val) * 100
            
            # Color indicators for console (if supported)
            if pct_min < 100:
                min_indicator = "DEFICIT"
            else:
                min_indicator = "OK"
                
            if nut in ['calories', 'fat', 'carbs']:  # 'min' direction nutrients
                if val <= opt_val:
                    opt_indicator = "OPTIMAL"
                elif val <= min_val:
                    opt_indicator = "GOOD"
                else:
                    opt_indicator = "HIGH"
            else:  # 'max' direction nutrients
                if val >= opt_val:
                    opt_indicator = "OPTIMAL"
                elif val >= min_val:
                    opt_indicator = "GOOD"
                else:
                    opt_indicator = "LOW"
            
            print(f"{nut.capitalize():15} | {min_val:12.1f} | {val:12.1f} | {opt_val:12.1f} | {pct_min:11.1f}% | {pct_opt:11.1f}% | {opt_indicator}")
        
        print("-"*80)
        
        # Print food basket
        print("\nOptimal Food Basket (Monthly Quantities):")
        print("-"*80)
        print(f"{'Food Item':^30} | {'Quantity (kg)':^15} | {'Total Cost (Toman)':^20} | {'% of Budget':^15}")
        print("-"*80)
        
        # Sort foods by quantity in descending order
        food_quantities = [(food.name, qty, food.price * qty) 
                         for food, qty in zip(ga.foods, best_chromosome)]
        food_quantities.sort(key=lambda x: x[1], reverse=True)
        
        for name, qty, cost in food_quantities:
            if qty > 0.01:  # Only show foods with significant quantities
                print(f"{name:30} | {qty:15.2f} | {cost:20,.0f} | {(cost/actual_cost)*100:14.1f}%")
        
        print("-"*80)
        print(f"Total monthly cost: {actual_cost:,.0f} Toman")
        print("="*80)
        
    except Exception as e:
        print(f"Runtime error: {e}")

if __name__ == '__main__':
    main()