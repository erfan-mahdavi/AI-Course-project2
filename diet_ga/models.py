import random
from typing import List, Callable

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
    def random(cls, gene_count: int, max_qty: float = 50.0) -> 'Individual': # 200 came from 4 milion/ 20k milk price lowered to 50
        return cls([random.uniform(0, max_qty) for _ in range(gene_count)])

    def crossover(self, other: 'Individual') -> 'Individual':
        pivot = random.randrange(1, len(self.chromosome))
        child_genes = self.chromosome[:pivot] + other.chromosome[pivot:]
        return Individual(child_genes)

    def mutate(self, rate: float, max_qty: float = 200.0) -> None:
        for i in range(len(self.chromosome)):
            if random.random() < rate:
                self.chromosome[i] = random.uniform(0, max_qty)

class Population:
    def __init__(self, size: int, gene_count: int, evaluator: Callable[['Individual'], float]):
        self.individuals: List[Individual] = [Individual.random(gene_count) for _ in range(size)]
        self.evaluator = evaluator

    def evaluate(self) -> None:
        for ind in self.individuals:
            ind.fitness = self.evaluator(ind)
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)

    def select(self, tournament_size: int) -> Individual:
        aspirants = random.sample(self.individuals, tournament_size)
        return max(aspirants, key=lambda x: x.fitness)

    def evolve(self, retain: int, pop_size: int, mutate_rate: float) -> None:
        next_gen = self.individuals[:retain]
        while len(next_gen) < pop_size:
            p1 = self.select(50)
            p2 = self.select(50)
            child = p1.crossover(p2)
            child.mutate(mutate_rate)
            next_gen.append(child)
        self.individuals = next_gen