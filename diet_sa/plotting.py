import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

class Plotter:
    """
    Plotting utility class specifically designed for Simulated Annealing diet optimization.
    
    This class provides visualization methods tailored to SA's single-solution approach,
    including temperature tracking, acceptance rates, and convergence analysis.
    """
    
    @staticmethod
    def plot_sa_convergence(fitness_history: List[float], temp_history: List[float], 
                           acceptance_history: List[float]) -> None:
        """
        Plot Simulated Annealing convergence metrics in a comprehensive dashboard.
        
        This method creates a multi-panel plot showing:
        1. Fitness progress over iterations
        2. Temperature decay schedule
        3. Acceptance rate evolution
        4. Best fitness found so far
        
        Args:
            fitness_history: List of fitness values for each iteration
            temp_history: List of temperature values for each iteration
            acceptance_history: List of acceptance rates for each iteration
        """
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        iterations = range(len(fitness_history))
        
        # 1. Fitness Progress
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Plot current fitness
        ax1.plot(iterations, fitness_history, 'b-', alpha=0.7, linewidth=1.5, 
                label='Current Fitness')
        
        # Plot best fitness found so far
        best_so_far = []
        current_best = -float('inf')
        for fit in fitness_history:
            if fit > current_best:
                current_best = fit
            best_so_far.append(current_best)
        
        ax1.plot(iterations, best_so_far, 'g-', linewidth=2.5, 
                label='Best So Far', marker='o', markevery=max(1, len(iterations)//20))
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Fitness Score')
        ax1.set_title('Fitness Evolution During SA Optimization')
        ax1.legend()
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Annotate final best fitness
        if best_so_far:
            final_best = best_so_far[-1]
            ax1.annotate(f'Final Best: {final_best:.2f}', 
                        xy=(len(iterations)-1, final_best),
                        xytext=(len(iterations)*0.7, final_best*1.05),
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='green'))
        
        # 2. Temperature Decay
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(iterations, temp_history, 'r-', linewidth=2.5, label='Temperature')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Temperature')
        ax2.set_title('Temperature Cooling Schedule')
        ax2.set_yscale('log')  # Log scale for better visualization
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Annotate temperature phases
        if len(temp_history) > 1:
            initial_temp = temp_history[0]
            final_temp = temp_history[-1]
            ax2.annotate(f'Initial: {initial_temp:.1f}', 
                        xy=(0, initial_temp),
                        xytext=(len(iterations)*0.1, initial_temp*2),
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2", fc="orange", alpha=0.8))
            ax2.annotate(f'Final: {final_temp:.3f}', 
                        xy=(len(iterations)-1, final_temp),
                        xytext=(len(iterations)*0.8, final_temp*10),
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2", fc="red", alpha=0.8))
        
        # 3. Acceptance Rate
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Smooth acceptance rate for better visualization
        window_size = min(100, len(acceptance_history) // 10)
        if window_size > 1:
            smoothed_acceptance = []
            for i in range(len(acceptance_history)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(acceptance_history), i + window_size // 2)
                smoothed_acceptance.append(np.mean(acceptance_history[start_idx:end_idx]))
            
            ax3.plot(iterations, smoothed_acceptance, 'purple', linewidth=2.5, 
                    label=f'Acceptance Rate (MA-{window_size})')
        else:
            ax3.plot(iterations, acceptance_history, 'purple', linewidth=2.5, 
                    label='Acceptance Rate')
        
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Acceptance Rate (%)')
        ax3.set_title('Solution Acceptance Rate Over Time')
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # Add reference lines for acceptance rate phases
        ax3.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% Acceptance')
        ax3.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% Acceptance')
        
        # 4. Convergence Analysis
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Calculate improvement rate (fitness improvement over rolling window)
        window = min(500, len(fitness_history) // 5)
        improvement_rate = []
        
        for i in range(window, len(fitness_history)):
            recent_max = max(fitness_history[i-window:i])
            older_max = max(fitness_history[max(0, i-2*window):i-window]) if i >= 2*window else recent_max
            improvement = recent_max - older_max
            improvement_rate.append(improvement)
        
        if improvement_rate:
            improvement_iterations = range(window, len(fitness_history))
            ax4.plot(improvement_iterations, improvement_rate, 'green', linewidth=2, 
                    label=f'Improvement Rate (window={window})')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Fitness Improvement')
            ax4.set_title('Convergence Analysis')
            ax4.legend()
            ax4.grid(True, alpha=0.3, linestyle='--')
            
            # Identify convergence point
            convergence_threshold = 0.01
            convergence_iter = None
            for i, improvement in enumerate(improvement_rate):
                if abs(improvement) < convergence_threshold:
                    convergence_iter = improvement_iterations[i]
                    break
            
            if convergence_iter:
                ax4.axvline(x=convergence_iter, color='red', linestyle='--', alpha=0.7)
                ax4.annotate(f'Convergence: {convergence_iter}', 
                            xy=(convergence_iter, 0),
                            xytext=(convergence_iter, max(improvement_rate)*0.5),
                            fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.8),
                            arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.suptitle('Simulated Annealing Optimization Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('sa_convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_nutrition_comparison(min_daily: Dict[str, float], opt_daily: Dict[str, float], 
                                actual: Dict[str, float]) -> None:
        """
        Plot comparison of minimum, optimal, and actual nutrient values.
        
        This method is identical to the GA version but works with SA solutions.
        
        Args:
            min_daily: Minimum required daily nutrient values
            opt_daily: Optimal daily nutrient values
            actual: Actual daily nutrient values from SA solution
        """
        nutrients = list(actual.keys())
        
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 4)
        
        colors = ['#3498db', '#2ecc71', '#f1c40f']  # Blue, Green, Yellow
        bar_labels = ['Minimum', 'Actual', 'Optimal']
        
        for i, nut in enumerate(nutrients):
            row, col = i // 4, i % 4
            ax = fig.add_subplot(gs[row, col])
            
            values = [min_daily[nut], actual[nut], opt_daily[nut]]
            bars = ax.bar(bar_labels, values, color=colors, width=0.6, 
                    edgecolor='gray', linewidth=0.5)
            
            # Color code the actual value based on adequacy
            if actual[nut] >= min_daily[nut]:
                bars[1].set_color('#2ecc71')  # Green if meeting minimum
            else:
                bars[1].set_color('#e74c3c')  # Red if below minimum
            
            # Add value labels on bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                        f'{values[j]:.1f}', ha='center', va='bottom', 
                        fontsize=9, fontweight='bold')
            
            # Highlight optimal range
            if nut in ['protein', 'fiber', 'calcium', 'iron']:
                # Max nutrients: highlight between minimum and optimal
                ax.axhspan(min_daily[nut], opt_daily[nut], alpha=0.1, color='green')
            else:
                # Min nutrients: highlight between optimal and minimum
                ax.axhspan(opt_daily[nut], min_daily[nut], alpha=0.1, color='green')
            
            # Show percentage of optimal
            if opt_daily[nut] > 0:
                pct = (actual[nut] / opt_daily[nut]) * 100
                ax.text(1, values[1] * 0.5, f"{pct:.1f}%", ha='center',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
                        fontsize=9)
            
            ax.set_title(nut.capitalize(), fontsize=12, fontweight='bold')
            ax.set_ylabel('Daily Value', fontsize=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.suptitle('SA Solution: Nutrient Comparison (Minimum vs. Actual vs. Optimal)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('sa_nutrient_comparison.png', dpi=300)
        plt.show()

    @staticmethod
    def plot_food_distribution(solution, food_names: List[str]) -> None:
        """
        Plot the distribution of food items in the SA solution.
        
        Args:
            solution: SA Solution object
            food_names: List of food item names
        """
        foods = solution.evaluator.foods
        quantities = solution.quantities
        
        # Filter foods with significant quantities
        threshold = 0.1  # kg
        significant_foods = []
        for i, qty in enumerate(quantities):
            if qty > threshold:
                cost = qty * foods[i].price
                significant_foods.append((food_names[i], qty, cost))
        
        if not significant_foods:
            print("No significant foods to plot.")
            return
        
        # Sort by quantity (descending)
        significant_foods.sort(key=lambda x: x[1], reverse=True)
        
        food_names_filtered = [item[0] for item in significant_foods]
        quantities_filtered = [item[1] for item in significant_foods]
        costs_filtered = [item[2] for item in significant_foods]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        
        colors = cm.viridis(np.linspace(0, 0.8, len(food_names_filtered)))
        
        # Quantity distribution
        bars1 = ax1.barh(food_names_filtered, quantities_filtered, color=colors)
        ax1.set_xlabel('Quantity (kg/month)', fontsize=12, fontweight='bold')
        ax1.set_title('Food Quantities in SA Solution', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            daily_g = (width * 1000) / 30  # Convert to grams per day
            ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}kg ({daily_g:.0f}g/day)', ha='left', va='center', fontsize=9)
        
        # Cost distribution
        bars2 = ax2.barh(food_names_filtered, costs_filtered, color=colors)
        ax2.set_xlabel('Cost (Toman/month)', fontsize=12, fontweight='bold')
        ax2.set_title('Cost Distribution in SA Solution', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        total_cost = sum(costs_filtered)
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            percentage = (width / total_cost) * 100
            ax2.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:,.0f}T ({percentage:.1f}%)', ha='left', va='center', fontsize=9)
        
        plt.suptitle('SA Solution: Food Distribution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('sa_food_distribution.png', dpi=300)
        plt.show()

    @staticmethod
    def plot_solution_dashboard(solution, temp_history: List[float], 
                              fitness_history: List[float]) -> None:
        """
        Create a comprehensive dashboard for SA solution analysis.
        
        Args:
            solution: SA Solution object
            temp_history: Temperature history from SA run
            fitness_history: Fitness history from SA run
        """
        analysis = solution.get_detailed_nutritional_analysis()
        
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        # 1. Overall Metrics
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['Fitness', 'Cost (M Toman)', 'Weight (kg)', 'Diversity']
        values = [
            solution.fitness,
            solution.get_total_cost() / 1_000_000,
            solution.get_total_weight(),
            solution.get_food_diversity_score()
        ]
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
        bars = ax1.bar(metrics, values, color=colors)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_title('Solution Overview', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 2. Budget Analysis
        ax2 = fig.add_subplot(gs[0, 1])
        budget_data = [
            analysis['total_monthly_cost'],
            analysis['budget_cap'] - analysis['total_monthly_cost']
        ]
        budget_labels = ['Used', 'Remaining']
        budget_colors = ['#e74c3c' if not analysis['within_budget'] else '#2ecc71', '#ecf0f1']
        
        wedges, texts, autotexts = ax2.pie(budget_data, labels=budget_labels, colors=budget_colors,
                                          autopct='%1.1f%%', startangle=90)
        ax2.set_title(f"Budget Utilization\n({analysis['budget_utilization_percent']:.1f}% of {analysis['budget_cap']:,.0f}T)")
        
        # 3. Nutritional Status
        ax3 = fig.add_subplot(gs[0, 2])
        status_counts = {}
        for nut_info in analysis['nutritional_status'].values():
            status = nut_info['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        status_labels = list(status_counts.keys())
        status_values = list(status_counts.values())
        status_colors = {'OPTIMAL': '#2ecc71', 'DEFICIENT': '#e74c3c', 'EXCESSIVE': '#f39c12'}
        colors = [status_colors.get(status, '#95a5a6') for status in status_labels]
        
        ax3.pie(status_values, labels=status_labels, colors=colors, autopct='%1.0f')
        ax3.set_title('Nutritional Status Distribution')
        
        # 4. Fitness Progress
        ax4 = fig.add_subplot(gs[1, :])
        iterations = range(len(fitness_history))
        ax4.plot(iterations, fitness_history, 'b-', alpha=0.6, linewidth=1, label='Current')
        
        # Best so far
        best_so_far = []
        current_best = -float('inf')
        for fit in fitness_history:
            current_best = max(current_best, fit)
            best_so_far.append(current_best)
        
        ax4.plot(iterations, best_so_far, 'g-', linewidth=2, label='Best So Far')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Fitness')
        ax4.set_title('SA Fitness Evolution')
        ax4.legend()
        ax4.grid(True, alpha=0.3, linestyle='--')
        
        # 5. Temperature Decay
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(range(len(temp_history)), temp_history, 'r-', linewidth=2)
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Temperature')
        ax5.set_title('Temperature Cooling')
        ax5.set_yscale('log')
        ax5.grid(True, alpha=0.3, linestyle='--')
        
        # 6. Top Foods
        ax6 = fig.add_subplot(gs[2, 1])
        food_names = [food.name for food in solution.evaluator.foods]
        top_foods = []
        for i, qty in enumerate(solution.quantities):
            if qty > 0.1:
                top_foods.append((food_names[i], qty))
        
        top_foods.sort(key=lambda x: x[1], reverse=True)
        top_foods = top_foods[:8]  # Top 8 foods
        
        if top_foods:
            names = [item[0] for item in top_foods]
            quantities = [item[1] for item in top_foods]
            
            bars = ax6.barh(names, quantities, color='skyblue')
            ax6.set_xlabel('Quantity (kg)')
            ax6.set_title('Top Food Items')
            ax6.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 7. Nutrient Adequacy
        ax7 = fig.add_subplot(gs[2, 2])
        nutrients = list(analysis['nutritional_status'].keys())
        adequacy_scores = []
        
        for nut in nutrients:
            status_info = analysis['nutritional_status'][nut]
            if status_info['status'] == 'OPTIMAL':
                score = 100
            elif status_info['status'] == 'DEFICIENT':
                score = status_info['percent_of_minimum']
            else:  # EXCESSIVE
                score = max(0, 100 - (status_info['percent_of_optimal'] - 100))
            adequacy_scores.append(score)
        
        colors = ['#2ecc71' if score >= 90 else '#f1c40f' if score >= 70 else '#e74c3c' 
                 for score in adequacy_scores]
        
        bars = ax7.barh([n.capitalize() for n in nutrients], adequacy_scores, color=colors)
        ax7.set_xlabel('Adequacy Score (%)')
        ax7.set_title('Nutrient Adequacy')
        ax7.set_xlim(0, 110)
        ax7.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.suptitle('Simulated Annealing Solution Dashboard', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig('sa_solution_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_comparison_with_random(solution, num_random_solutions: int = 100) -> None:
        """
        Compare the SA solution with random solutions to show optimization effectiveness.
        
        Args:
            solution: Optimized SA solution
            num_random_solutions: Number of random solutions to generate for comparison
        """
        # Generate random solutions for comparison
        random_fitness_scores = []
        random_costs = []
        
        for _ in range(num_random_solutions):
            random_sol = solution.__class__.random_solution(
                len(solution.quantities), 
                solution.evaluator
            )
            random_fitness_scores.append(random_sol.fitness)
            random_costs.append(random_sol.get_total_cost())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Fitness comparison
        ax1.hist(random_fitness_scores, bins=20, alpha=0.7, color='lightcoral', 
                label='Random Solutions', density=True)
        ax1.axvline(solution.fitness, color='green', linewidth=3, 
                   label=f'SA Solution ({solution.fitness:.2f})')
        ax1.set_xlabel('Fitness Score')
        ax1.set_ylabel('Density')
        ax1.set_title('Fitness: SA vs Random Solutions')
        ax1.legend()
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Cost comparison
        ax2.hist([cost/1_000_000 for cost in random_costs], bins=20, alpha=0.7, 
                color='lightblue', label='Random Solutions', density=True)
        ax2.axvline(solution.get_total_cost()/1_000_000, color='red', linewidth=3,
                   label=f'SA Solution ({solution.get_total_cost()/1_000_000:.2f}M)')
        ax2.axvline(solution.evaluator.cost_cap/1_000_000, color='orange', 
                   linestyle='--', linewidth=2, label='Budget Cap')
        ax2.set_xlabel('Cost (Million Toman)')
        ax2.set_ylabel('Density')
        ax2.set_title('Cost: SA vs Random Solutions')
        ax2.legend()
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        plt.suptitle('SA Optimization Effectiveness', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('sa_vs_random_comparison.png', dpi=300)
        plt.show()
        
        # Print statistics
        print(f"\nOptimization Effectiveness Analysis:")
        print(f"SA Solution Fitness: {solution.fitness:.2f}")
        print(f"Random Solutions - Mean: {np.mean(random_fitness_scores):.2f}, "
              f"Std: {np.std(random_fitness_scores):.2f}")
        print(f"SA solution is {(solution.fitness - np.mean(random_fitness_scores)):.2f} "
              f"points better than average random solution")
        
        percentile = (sum(1 for score in random_fitness_scores if score < solution.fitness) 
                     / len(random_fitness_scores)) * 100
        print(f"SA solution is better than {percentile:.1f}% of random solutions")