import random
from typing import List, Callable
import tabulate
import numpy as np

class FoodItem:
    def __init__(self, name: str, nutrients: dict, price_per_kg: float):
        self.name = name
        self.nutrients = nutrients
        self.price = price_per_kg

class Individual:
    def __init__(self, chromosome: List[float]):
        self.chromosome = chromosome  # kg of each food item monthly
        self.fitness: float = -float('inf')

    @classmethod
    def random(cls, gene_count: int, max_qty: float = 50.0) -> 'Individual': 
        """
        Create a random individual with random quantities for each food item
        
        Args:
            gene_count: Number of genes (food items)
            max_qty: Maximum quantity for each food item in kg
            
        Returns:
            A new Individual with random chromosome
        """
        return cls([random.uniform(0, max_qty) for _ in range(gene_count)])

    def crossover(self, other: 'Individual') -> 'Individual':
        """
        Perform crossover with another individual
        
        Args:
            other: The other parent individual
            
        Returns:
            A new child Individual
        """
        pivot = random.randrange(1, len(self.chromosome))
        child_genes = self.chromosome[:pivot] + other.chromosome[pivot:]
        return Individual(child_genes)

    def mutate(self, rate: float, max_qty: float = 50.0) -> None:
        """
        Mutate the chromosome with the given rate
        
        Args:
            rate: Mutation rate (probability of mutating each gene)
            max_qty: Maximum quantity for each food item in kg
        """
        for i in range(len(self.chromosome)):
            if random.random() < rate:
                self.chromosome[i] = random.uniform(0, max_qty)
    
    def format_basket(self, food_names) -> str:
        """
        Format the food basket in a readable table
        
        Args:
            food_names: List of food names
            
        Returns:
            Formatted table string
        """
        # Filter out items with zero or very small quantities
        non_zero_items = [(name, qty) for name, qty in zip(food_names, self.chromosome) if qty > 0.01]
        
        # Sort by quantity (descending)
        sorted_items = sorted(non_zero_items, key=lambda x: x[1], reverse=True)
        
        # Create table data
        table_data = []
        for name, qty in sorted_items:
            table_data.append([name, f"{qty:.2f} kg", f"{qty*1000/30:.1f} g/day"])
        
        # Generate table
        headers = ["Food Item", "Monthly Qty", "Daily Avg"]
        return tabulate.tabulate(table_data, headers=headers, tablefmt="grid")

class Population:
    def __init__(self, size: int, gene_count: int, evaluator: Callable[['Individual'], float]):
        self.individuals: List[Individual] = [Individual.random(gene_count) for _ in range(size)]
        self.evaluator = evaluator

    def evaluate(self) -> None:
        """
        Evaluate fitness of all individuals in the population
        """
        for ind in self.individuals:
            ind.fitness = self.evaluator(ind)
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)

    def select(self, tournament_size: int = None) -> Individual:
        """
        Select an individual using tournament selection
        
        Args:
            tournament_size: Size of the tournament (default is population size // 20)
            
        Returns:
            The selected Individual
        """
        # If tournament_size is not provided, use 5% of population size (min 2)
        if tournament_size is None:
            tournament_size = max(2, len(self.individuals) // 20)
            
        aspirants = random.sample(self.individuals, tournament_size)
        return max(aspirants, key=lambda x: x.fitness)

    def evolve(self, retain: int, pop_size: int, mutate_rate: float) -> None:
        """
        Evolve the population to the next generation
        
        Args:
            retain: Number of best individuals to retain for the next generation
            pop_size: Target population size
            mutate_rate: Mutation rate
        """
        # Keep the best individuals
        next_gen = self.individuals[:retain]
        
        # Create new individuals to fill the population
        while len(next_gen) < pop_size:
            # Select parents using tournament selection
            p1 = self.select()
            p2 = self.select()
            
            # Create child and mutate
            child = p1.crossover(p2)
            child.mutate(mutate_rate)
            next_gen.append(child)
            
        self.individuals = next_gen
        
    def get_diversity_metrics(self) -> dict:
        """
        Calculate diversity metrics for the population
        
        Returns:
            Dictionary of diversity metrics
        """
        # Calculate standard deviation for each gene position
        gene_count = len(self.individuals[0].chromosome)
        gene_stds = []
        
        for i in range(gene_count):
            values = [ind.chromosome[i] for ind in self.individuals]
            gene_stds.append(np.std(values))
        
        # Calculate fitness diversity
        fitness_values = [ind.fitness for ind in self.individuals 
                         if ind.fitness > -float('inf')]
        
        if not fitness_values:
            fitness_std = 0
            fitness_range = 0
        else:
            fitness_std = np.std(fitness_values)
            fitness_range = max(fitness_values) - min(fitness_values)
        
        return {
            'gene_std_avg': np.mean(gene_stds),
            'gene_std_max': np.max(gene_stds),
            'gene_std_min': np.min(gene_stds),
            'fitness_std': fitness_std,
            'fitness_range': fitness_range
        }