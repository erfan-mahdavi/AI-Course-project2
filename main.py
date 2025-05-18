import argparse
from diet_ga.genetic_algorithm import GeneticAlgorithm

def main():
    parser = argparse.ArgumentParser(
        description='Monthly diet optimization via GA'
    )
    parser.add_argument('--foods', default='foods.csv')
    parser.add_argument('--pop', type=int, default=10000)
    parser.add_argument('--gens', type=int, default=60)
    parser.add_argument('--mut', type=float, default=0.15)
    parser.add_argument('--final-mut', type=float, default=0.005,
                        help='Final (minimum) mutation rate')
    parser.add_argument('--retain', type=int, default=800)
    parser.add_argument('--cost-cap', type=float, default=4_000_000.0,
                        help='Monthly budget cap in Toman')
    args = parser.parse_args()

    ga = GeneticAlgorithm(
        csv_path=args.foods,
        pop_size=args.pop,
        generations=args.gens,
        init_mut_rate=args.mut,
        final_mut_rate=args.final_mut,
        retain=args.retain,
        cost_cap=args.cost_cap
    )
    best = ga.run()
    print('Best basket (kg per item):', best)

if __name__ == '__main__':
    main()
