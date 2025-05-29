# File: genetic_algorithm.py
from .data_loader import DataLoader
from .models import Population
from .fitness2 import FitnessEvaluator
from .plotting import Plotter
from typing import List, Dict
import numpy as np
import math
import os

class GeneticAlgorithm:
    def __init__(
        self,
        csv_path: str,
        pop_size: int = 10,
        generations: int = 10,
        init_mut_rate: float = 0.15,
        final_mut_rate: float = 0.01,
        retain: int = 800,
        cost_cap: float = 4_000_000.0
    ):
        """
        Initialize the Genetic Algorithm
        
        Args:
            csv_path: Path to the CSV file with food data
            pop_size: Population size
            generations: Number of generations to run
            init_mut_rate: Initial mutation rate
            final_mut_rate: Final mutation rate
            retain: Number of best individuals to retain in each generation
            cost_cap: Maximum cost constraint
        """
        try:
            # Make sure the file exists
            if not os.path.exists(csv_path):
                print(f"Warning: File '{csv_path}' not found. Checking for alternative paths...")
                # Try alternative paths
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
                
            foods = DataLoader.load_foods(csv_path)
            gene_count = len(foods)
            
            if gene_count == 0:
                raise ValueError("No food items were loaded from the CSV file.")

            # Define daily requirements
            daily = {
                'calories': 2000, 'protein': 100, 'fat': 60,
                'carbs': 250,     'fiber': 25,  'calcium': 1000,
                'iron': 18
            }
            
            # Convert to monthly requirements
            min_req = {nut: val * 30 for nut, val in daily.items()}
            
            # Define optimal values (per month)
            optimal = {
                'calories': 1700*30, 'protein': 140*30, 'fat': 45*30,
                'carbs': 210*30,     'fiber': 35*30,  'calcium': 1200*30,
                'iron': 23*30
            }
            
            # Define weights for nutrients in fitness function
            weights = {
                'calories': 0.8, 'protein': 1.2, 'fat': 1.2,
                'carbs': 1.2,     'fiber': 1.22,  'calcium': 1.3,
                'iron': 1.2
            }
            
            # Create fitness evaluator
            evaluator = FitnessEvaluator(
                foods, min_req, optimal, weights, cost_cap
            )
            
            # Initialize population
            self.population = Population(pop_size, gene_count, evaluator)
            
            # Store parameters for later use
            self.foods = foods
            self.min_daily = daily
            self.optimal_daily = {nut: ov/30 for nut, ov in optimal.items()}
            self.nut_history = {nut: [] for nut in daily if nut != 'calories'}
            self.diversity_hist = []
            self.plotter = Plotter()
            self.generations = generations
            self.init_mut_rate = init_mut_rate
            self.final_mut_rate = final_mut_rate
            self.retain = retain
            self.pop_size = pop_size
            
            # Print available food items
            self.print_foods_details()
            
        except Exception as e:
            print(f"Error initializing Genetic Algorithm: {e}")
            raise
            
    def print_foods_details(self):
        """
        Print detailed information about all available food items
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

    def run(self) -> tuple:
        """
        Run the genetic algorithm
        
        Returns:
            Tuple of (best_chromosome, best_fitness, cost, nutrient_totals)
        """
        best_history, avg_history = [], []
        
        try:
            # Print header for generation progress
            print("\nOptimization Progress:")
            print(f"{'-'*80}")
            print(f"{'Gen':^5}|{'Fitness':^12}|{'Cost (Toman)':^15}|{'Calories':^10}|{'Protein':^9}|{'Fat':^8}|{'Carbs':^8}|{'Fiber':^8}|{'Avg Fit':^10}")
            print(f"{'-'*80}")
            
            # Initial evaluation
            self.population.evaluate()
            
            # Main loop
            for gen in range(self.generations + 1):
                if gen > 0:
                    # Calculate mutation rate (linear decrease)
                    frac = gen/self.generations
                    mr = (1-frac)*self.init_mut_rate + frac*self.final_mut_rate
                    
                    # Evolve population
                    self.population.evolve(self.retain, self.pop_size, mr)
                    self.population.evaluate()
                
                # Get best individual
                best = self.population.individuals[0]
                
                # Calculate cost and nutrient totals
                cost, totals = self._compute_cost_and_totals(best)
                
                # Calculate average fitness (excluding -inf values)
                vals = [i.fitness for i in self.population.individuals if i.fitness > -np.inf]
                avg = np.mean(vals) if vals else -np.inf
                
                # Record history
                best_history.append(best.fitness)
                avg_history.append(avg)
                
                # Record nutrition information for this generation
                self._record_generation(totals, best.chromosome)
                
                # Print progress (more readable format)
                print(f"{gen:5d}|{best.fitness:12.2f}|{cost:15,.0f}|{totals['calories']:10.1f}|{totals['protein']:9.1f}|{totals['fat']:8.1f}|{totals['carbs']:8.1f}|{totals['fiber']:8.1f}|{avg:10.2f}")
                
                # Print a separator line every 10 generations
                if gen % 10 == 9:
                    print(f"{'-'*80}")
            
            
            def within_constraints(ind):
                totals_month = {nut:0.0 for nut in self.population.evaluator.min_req}
                for qty, food in zip(ind.chromosome, self.population.evaluator.foods):
                    for nut, per100 in food.nutrients.items():
                        totals_month[nut] += per100*10*qty
                for nut, min_val in self.population.evaluator.min_req.items():
                    dirn = self.population.evaluator.directions[nut]
                    val = totals_month[nut]
                    # remove if above min_req for 'min' nutrients
                    if dirn=='min' and val>min_val:
                        return False
                    # remove if below min_req for 'max' nutrients
                    if dirn=='max' and val<min_val:
                        return False
                return True

            filtered = list(filter(within_constraints, self.population.individuals))
            best_final = filtered[0] if filtered else self.population.individuals[0]
            cost_last, totals_last = self._compute_cost_and_totals(best_final)

            
            # Generate plots
            self.plotter.plot(best_history, avg_history)
            self.plotter.plot_nutrition_comparison(self.min_daily, self.optimal_daily, totals_last)
            self.plotter.plot_nutrition_progress(self.nut_history)
            self.plotter.plot_diversity(self.diversity_hist)
            
            # Return best solution along with additional information
            return (
                  best_final.chromosome,
                  best_final.fitness,
                  cost_last,
                  totals_last
                    )

            
        except Exception as e:
            print(f"Error running Genetic Algorithm: {e}")
            raise

    def _compute_cost_and_totals(self, ind):
        """
        Compute the cost and nutrient totals for an individual
        
        Args:
            ind: The individual
            
        Returns:
            Tuple of (cost, nutrient_totals)
        """
        ev = self.population.evaluator
        totals, cost = {nut: 0.0 for nut in ev.min_req}, 0.0
        
        for q, f in zip(ind.chromosome, ev.foods):
            cost += q * f.price
            for nut, per100 in f.nutrients.items():
                totals[nut] += (per100 * 10 * q) / 30  # Convert to daily values
                
        return cost, {n: round(v, 1) for n, v in totals.items()}

    def _record_generation(self, totals_daily, chromo):
        """
        Record nutrition and diversity data for this generation
        
        Args:
            totals_daily: Daily nutrient totals
            chromo: Chromosome of the best individual
        """
        # Record nutrient-to-calorie ratios
        cal = totals_daily['calories']
        for nut, val in totals_daily.items():
            if nut == 'calories': 
                continue
            self.nut_history[nut].append(val / cal if cal > 0 else 0)
        
        # Calculate diversity using entropy
        tot = sum(chromo)
        if tot > 0:
            # Calculate Shannon entropy
            ps = [q/tot for q in chromo if q > 0]
            H = -sum(p * math.log(p) for p in ps)
            Hm = math.log(len(chromo))
            div = H / Hm if Hm > 0 else 0
        else: 
            div = 0
            
        self.diversity_hist.append(div)