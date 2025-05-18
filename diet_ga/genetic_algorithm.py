# File: genetic_algorithm.py
from .data_loader import DataLoader
from .models import Population
from .fitness import FitnessEvaluator
from .plotting import Plotter
from typing import List
import numpy as np
import math

class GeneticAlgorithm:
    def __init__(
        self,
        csv_path: str,
        pop_size: int = 10000,
        generations: int = 60,
        init_mut_rate: float = 0.15,
        final_mut_rate: float = 0.01,
        retain: int = 800,
        cost_cap: float = 4_000_000.0
    ):
        foods = DataLoader.load_foods(csv_path)
        gene_count = len(foods)

        daily = {
            'calories':2000, 'protein':100, 'fat':60,
            'carbs':250,     'fiber':25,  'calcium':1000,
            'iron':18
        }
        min_req = {nut: val * 30 for nut, val in daily.items()}
        optimal = {
            'calories':1700*30, 'protein':140*30, 'fat':45*30,
            'carbs':210*30,     'fiber':35*30,  'calcium':1200*30,
            'iron':23*30
        }
        weights = {
            'calories':1.0, 'protein':1.2, 'fat':1.3,
            'carbs':1.2,     'fiber':1.2,  'calcium':1.2,
            'iron':1.5
        }
        evaluator = FitnessEvaluator(
            foods, min_req, optimal, weights, cost_cap
        )
        self.population = Population(pop_size, gene_count, evaluator)
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

    def run(self) -> List[float]:
        best_history, avg_history = [], []
        self.population.evaluate()
        for gen in range(self.generations + 1):
            if gen > 0:
                frac = gen/self.generations
                mr = (1-frac)*self.init_mut_rate + frac*self.final_mut_rate
                self.population.evolve(self.retain, self.pop_size, mr)
                self.population.evaluate()
            best = self.population.individuals[0]
            _, totals = self._compute_cost_and_totals(best)
            vals = [i.fitness for i in self.population.individuals if i.fitness>-np.inf]
            avg = np.mean(vals) if vals else -np.inf
            best_history.append(best.fitness)
            avg_history.append(avg)
            self._record_generation(totals, best.chromosome)
            print(f"Gen {gen:3d}: fitness={best.fitness:.2f}, cost={_:.0f}, nutrients={totals}, avg={avg:.2f}")
        cost_last, totals_last = self._compute_cost_and_totals(self.population.individuals[0])
        self.plotter.plot(best_history, avg_history)
        self.plotter.plot_nutrition_comparison(self.min_daily, self.optimal_daily, totals_last)
        self.plotter.plot_nutrition_progress(self.nut_history)
        self.plotter.plot_diversity(self.diversity_hist)
        return self.population.individuals[0].chromosome

    def _compute_cost_and_totals(self, ind):
        ev = self.population.evaluator
        totals, cost = {nut:0.0 for nut in ev.min_req}, 0.0
        for q, f in zip(ind.chromosome, ev.foods):
            cost += q*f.price
            for nut, per100 in f.nutrients.items():
                totals[nut] += (per100*10*q)/30
        return cost, {n:round(v,1) for n,v in totals.items()}

    def _record_generation(self, totals_daily, chromo):
        cal = totals_daily['calories']
        for nut,val in totals_daily.items():
            if nut=='calories': continue
            self.nut_history[nut].append(val/cal if cal>0 else 0)
        tot=sum(chromo)
        if tot>0:
            ps=[q/tot for q in chromo if q>0]
            H=-sum(p*math.log(p) for p in ps)
            Hm=math.log(len(chromo))
            div=H/Hm if Hm>0 else 0
        else: div=0
        self.diversity_hist.append(div)
