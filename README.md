# Diet Optimization System Documentation

## Overview

This project implements an intelligent diet optimization system designed to create optimal monthly food plans that meet nutritional requirements while staying within budget constraints. The system offers two optimization algorithms: Genetic Algorithm (GA) and Simulated Annealing (SA), allowing users to choose the approach that best fits their needs.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Algorithms](#algorithms)
3. [Core Components](#core-components)
4. [Installation & Setup](#installation--setup)
5. [Usage Guide](#usage-guide)
6. [Configuration](#configuration)
7. [Data Format](#data-format)
8. [Optimization Details](#optimization-details)
9. [Output & Visualization](#output--visualization)
10. [Performance Considerations](#performance-considerations)
11. [Troubleshooting](#troubleshooting)

## System Architecture

The system is organized into two main modules:

### Diet GA Module
- **Genetic Algorithm Implementation**: Population-based evolutionary optimization
- **Multi-generational Evolution**: Iterative improvement through selection, crossover, and mutation
- **Parallel Solution Evaluation**: Evaluates multiple diet combinations simultaneously

### Diet SA Module  
- **Simulated Annealing Implementation**: Single-solution temperature-based optimization
- **Adaptive Cooling**: Dynamic temperature adjustment based on performance
- **Neighborhood Exploration**: Intelligent perturbation strategies

### Shared Components
- **Data Loading**: CSV-based food database management
- **Fitness Evaluation**: Multi-objective constraint handling
- **Visualization**: Comprehensive plotting and analysis tools

## Algorithms

### Genetic Algorithm (GA)
**Best for**: Finding globally optimal solutions when computational time is available

**Characteristics**:
- Population size: 10-1600 individuals (configurable)
- Generations: 10-100+ iterations
- Selection: Tournament-based selection
- Crossover: Single-point chromosome crossover
- Mutation: Adaptive rate (0.25 → 0.01)
- Elite retention: Preserves best solutions

**Advantages**:
- Excellent global search capability
- Robust against local optima
- Parallel evaluation of multiple solutions
- Good for complex constraint satisfaction

### Simulated Annealing (SA)
**Best for**: Fast convergence and real-time optimization

**Characteristics**:
- Single solution evolution
- Temperature-based acceptance (1000.0 → 0.1)
- Adaptive cooling rate (0.995)
- Multiple perturbation strategies
- Early stopping conditions

**Advantages**:
- Faster execution per iteration
- Lower memory requirements
- Good local optimization
- Suitable for real-time applications

## Core Components

### 1. Data Loader (`data_loader.py`)

Handles food database loading and validation:

```python
# Expected CSV format
foods = DataLoader.load_foods('foods.csv')
```

**Features**:
- Automatic data validation
- Missing value handling
- Nutritional summary generation
- Price range analysis

### 2. Food Item Model (`models.py`)

Core data structures:

```python
class FoodItem:
    def __init__(self, name: str, nutrients: Dict[str, float], price_per_kg: float)
    def get_nutrient_density(self, nutrient: str) -> float  # nutrients per cost unit
```

**Nutrient Categories**:
- **Minimize**: calories, fat, carbs
- **Maximize**: protein, fiber, calcium, iron

### 3. Fitness Evaluation (`fitness.py` / `fitness2.py`)

Multi-objective optimization with constraint handling:

**Objective Components**:
1. **Budget Constraint**: Hard limit at 4M Toman/month
2. **Nutritional Requirements**: Daily minimums × 30 days
3. **Directional Preferences**: Min/max nutrient targets
4. **Cost Efficiency**: Bonus for staying under budget

**Scoring System**:
- Heavy penalties for constraint violations (-5000 to -10000)
- Moderate penalties for suboptimal values (-500 to -1000)
- Efficiency bonuses for meeting targets (+100 to +500)

### 4. Solution Representation

**GA Individual**:
```python
class Individual:
    chromosome: List[float]  # kg quantities for each food
    fitness: float          # evaluated score
```

**SA Solution**:
```python
class Solution:
    quantities: List[float]  # kg quantities for each food
    fitness: float          # evaluated score
```

## Installation & Setup

### Prerequisites
```bash
pip install pandas numpy matplotlib colorama tabulate
```

### File Structure
```
project/
├── diet_ga/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── models.py
│   ├── fitness2.py
│   ├── genetic_algorithm.py
│   └── plotting.py
├── diet_sa/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── models.py
│   ├── fitness.py
│   ├── simulated_annealing.py
│   └── plotting.py
├── main.py
└── foods.csv
```
┌─────────────────────────────────────────────────────────────┐
│                    Main Controller                          │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │  Algorithm      │    │    User Interface &             │ │
│  │  Selection      │    │    Parameter Management         │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                  Algorithm Modules                          │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │   GA Module     │    │       SA Module                 │ │
│  │                 │    │                                 │ │
│  │ • Population    │    │ • Solution                      │ │
│  │ • Individual    │    │ • Neighbor Generation           │ │
│  │ • Evolution     │    │ • Temperature Control           │ │
│  │ • Selection     │    │ • Acceptance Criteria           │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                  Shared Infrastructure                      │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────────┐   │
│  │ Data Loader  │ │ Fitness      │ │ Visualization &    │   │
│  │              │ │ Evaluator    │ │ Reporting          │   │
│  │ • CSV Parse  │ │              │ │                    │   │
│  │ • Validation │ │ • Multi-obj  │ │ • Matplotlib       │   │
│  │ • Food Model │ │ • Constraints│ │ • Progress Track   │   │
│  └──────────────┘ └──────────────┘ └────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

## Usage Guide

### Command Line Interface

```bash
# Basic usage - interactive algorithm selection
python main.py

# GA with custom parameters
python main.py --pop 800 --gens 50 --mut 0.2 --cost-cap 3500000

# SA with custom parameters  
python main.py --iterations 50000 --temp 500 --cooling 0.99 --step-size 1.5

# Custom food database
python main.py --foods my_foods.csv
```

### Parameters

**Genetic Algorithm**:
- `--pop`: Population size (default: 1600)
- `--gens`: Number of generations (default: 99)
- `--mut`: Initial mutation rate (default: 0.25)
- `--final-mut`: Final mutation rate (default: 0.01)
- `--retain`: Elite individuals to retain (default: 80)

**Simulated Annealing**:
- `--iterations`: Maximum iterations (default: 1,000,000)
- `--temp`: Initial temperature (default: 1.0)
- `--final-temp`: Final temperature (default: 1e-12)
- `--cooling`: Cooling rate (default: 0.99999)
- `--step-size`: Perturbation step size (default: 1.0 kg)

**Common**:
- `--cost-cap`: Budget limit in Toman (default: 4,000,000)
- `--foods`: Path to CSV food database (default: foods.csv)

## Configuration

### Nutritional Requirements

**Daily Targets**:
```python
daily_requirements = {
    'calories': 2000,   # kcal
    'protein': 100,     # g  
    'fat': 60,          # g
    'carbs': 250,       # g
    'fiber': 25,        # g
    'calcium': 1000,    # mg
    'iron': 18,         # mg
}
```

**Optimal Targets** (GA):
```python
optimal_daily = {
    'calories': 1700,   # Lower for weight management
    'protein': 140,     # Higher for muscle health
    'fat': 45,          # Moderate healthy fats
    'carbs': 210,       # Controlled carb intake
    'fiber': 35,        # Higher for digestion
    'calcium': 1200,    # Higher for bone health
    'iron': 23,         # Higher to prevent anemia
}
```

### Fitness Weights

```python
weights = {
    'calories': 0.8,    # Lower priority
    'protein': 1.2,     # High priority
    'fat': 1.2,         # Moderate priority
    'carbs': 1.2,       # Moderate priority
    'fiber': 1.25,      # High priority
    'calcium': 1.3,     # Highest priority
    'iron': 1.2,        # High priority
}
```

## Data Format

### Food Database CSV Structure

```csv
Food,Calories_kcal,Protein_g,Fat_g,Carbohydrates_g,Fiber_g,Calcium_mg,Iron_mg,Price_Toman_per_kg
Rice,130,2.7,0.3,28,0.4,10,0.8,45000
Chicken Breast,165,31,3.6,0,0,14,0.9,180000
Lentils,116,9,0.4,20,8,19,3.3,65000
Spinach,23,2.9,0.4,3.6,2.2,99,2.7,35000
```

**Required Columns**:
- `Food`: Food item name
- `Calories_kcal`: Energy per 100g
- `Protein_g`: Protein content per 100g
- `Fat_g`: Fat content per 100g  
- `Carbohydrates_g`: Carb content per 100g
- `Fiber_g`: Fiber content per 100g
- `Calcium_mg`: Calcium content per 100g
- `Iron_mg`: Iron content per 100g
- `Price_Toman_per_kg`: Price per kilogram in Toman

## Optimization Details

### Problem Formulation

**Objective**: Minimize cost while satisfying nutritional constraints

**Decision Variables**: x_i = kg of food i per month (i = 1, 2, ..., n)

**Constraints**:
1. Budget: Σ(x_i × price_i) ≤ budget_cap
2. Nutrition: nutrient_j ≥ requirement_j (for max nutrients)
3. Nutrition: nutrient_j ≤ requirement_j (for min nutrients)
4. Non-negativity: x_i ≥ 0

### Algorithm-Specific Details

**GA Evolution Process**:
1. Initialize random population
2. Evaluate fitness for all individuals
3. Select parents via tournament selection
4. Create offspring through crossover
5. Apply mutation with adaptive rate
6. Retain elite individuals
7. Repeat until convergence

**SA Optimization Process**:
1. Generate intelligent initial solution
2. Create neighbor through perturbation
3. Calculate acceptance probability
4. Accept/reject based on temperature
5. Update temperature (cooling)
6. Apply early stopping conditions
7. Return best solution found

### Perturbation Strategies (SA)

1. **Simple Perturbation**: Random changes to 1-3 food quantities
2. **Focused Perturbation**: Target deficient nutrients specifically
3. **Cost Reduction**: Reduce expensive food quantities when over budget

## Output & Visualization

### Console Output

```
Generation Progress:
Gen  | Fitness    | Cost (Toman)  | Calories | Protein | Fat   | Carbs | Fiber | Avg Fit
-----|------------|---------------|----------|---------|-------|-------|-------|--------
   0 |    -45.67  |   3,874,520   |  1847.3  |  98.7   | 52.1  | 234.8 | 31.2  |  -156.8
  10 |     89.23  |   3,654,210   |  1923.4  |  112.5  | 48.9  | 221.7 | 34.6  |   34.7
  20 |    156.78  |   3,543,890   |  1889.6  |  118.9  | 45.3  | 208.4 | 38.1  |   87.2
```

### Generated Visualizations

1. **Fitness Progress**: Convergence tracking over generations/iterations
2. **Nutrient Comparison**: Actual vs. required vs. optimal values
3. **Nutrition Progress**: Nutrient-to-calorie ratios over time
4. **Population Diversity**: Genetic diversity maintenance (GA only)
5. **Convergence Analysis**: Temperature decay and acceptance rates (SA only)

### Results Summary

```
OPTIMIZATION RESULTS
Algorithm: GA
Execution time: 45.23 seconds
Best fitness score: 234.56
Total monthly cost: 3,654,210 Toman
Budget utilization: 91.4%

Daily Nutrient Summary:
Nutrient     | Required   | Actual     | Optimal    | % of Req
-------------|------------|------------|------------|----------
Calories     |    2000.0  |    1889.6  |    1700.0  |     94.5%
Protein      |     100.0  |     118.9  |     140.0  |    118.9%
Fat          |      60.0  |      45.3  |      45.0  |     75.5%
```

## Performance Considerations

### Computational Complexity

**Genetic Algorithm**:
- Time: O(G × P × N × F) where G=generations, P=population, N=foods, F=fitness
- Space: O(P × N) for population storage
- Typical runtime: 30-120 seconds

**Simulated Annealing**:
- Time: O(I × N × F) where I=iterations, N=foods, F=fitness  
- Space: O(N) for single solution
- Typical runtime: 15-60 seconds

### Scalability Guidelines

**Food Database Size**:
- Small (< 50 foods): Both algorithms perform well
- Medium (50-200 foods): SA may be faster
- Large (> 200 foods): Consider parameter tuning

**Optimization Recommendations**:
- For quick results: Use SA with moderate iterations (10,000-50,000)
- For best quality: Use GA with large population (800-1600)
- For balanced approach: Use SA with high iterations (100,000+)

## Troubleshooting

### Common Issues

**FileNotFoundError: foods.csv not found**
```bash
# Solution: Specify correct path
python main.py --foods /path/to/foods.csv
```

**Poor convergence / Low fitness scores**
```bash
# For GA: Increase population or generations
python main.py --pop 1600 --gens 100

# For SA: Adjust temperature or cooling rate  
python main.py --temp 1000 --cooling 0.999
```

**Budget constraint violations**
- Check food prices in CSV
- Increase budget cap: `--cost-cap 5000000`
- Verify currency units (Toman)

**Nutritional requirement not met**
- Review daily requirements in code
- Check food nutritional data accuracy
- Consider relaxing some constraints

### Debugging Tips

1. **Enable detailed logging**: Modify print statements in algorithms
2. **Check intermediate results**: Monitor fitness components
3. **Validate input data**: Ensure CSV format correctness
4. **Parameter sensitivity**: Test different parameter combinations
5. **Memory issues**: Reduce population size for large datasets

### Performance Optimization

**For faster execution**:
- Reduce population size (GA) or iterations (SA)
- Implement early stopping conditions
- Use compiled libraries (NumPy operations)

**For better results**:
- Increase computational budget
- Fine-tune algorithm parameters
- Improve initial solution generation
- Add problem-specific heuristics

## Advanced Usage

### Custom Fitness Functions

Modify `fitness.py` or `fitness2.py` to implement custom scoring:

```python
def custom_nutrient_score(self, actual, minimum, optimal, direction, weight):
    # Implement custom scoring logic
    return score
```

### Algorithm Hybridization

Combine GA and SA for multi-stage optimization:

1. Use GA for global exploration (50 generations)
2. Use SA for local refinement (10,000 iterations)
3. Take best solution from either algorithm

### Integration with External Systems

The system can be integrated with:
- Meal planning applications
- Nutritional databases (USDA, local databases)
- Cost tracking systems
- Health monitoring platforms

---

## Conclusion

This diet optimization system provides a robust, flexible platform for automated meal planning that balances nutritional requirements with budget constraints. The dual-algorithm approach allows users to choose between comprehensive global search (GA) and efficient local optimization (SA) based on their specific needs and computational resources.

The system's modular design facilitates easy customization and extension, making it suitable for both research applications and practical meal planning scenarios.