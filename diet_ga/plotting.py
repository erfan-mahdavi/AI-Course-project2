import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm

class Plotter:
    
    @staticmethod
    def plot(best: List[float], avg: List[float]) -> None:
        """
        Plot the fitness history with enhanced readability
        
        Args:
            best: List of best fitness values per generation
            avg: List of average fitness values per generation
        """
        plt.figure(figsize=(12, 7))
        
        best_color = '#1f77b4'  # Blue
        avg_color = '#ff7f0e'   # Orange
        
        plt.plot(best, label='Best Fitness', color=best_color, linewidth=2.5, marker='o', 
                 markevery=max(1, len(best)//10), markersize=6)
        plt.plot(avg, label='Average Fitness', color=avg_color, linewidth=2, marker='s', 
                 markevery=max(1, len(avg)//10), markersize=5, alpha=0.8)
        
        plt.fill_between(range(len(best)), best, avg, alpha=0.1, color=best_color)
        
        if len(best) > 10:
            start_idx = int(len(best) * 0.7)
            x = np.array(range(start_idx, len(best)))
            y = np.array(best[start_idx:])
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), "r--", alpha=0.7, linewidth=1.5, label="Trend")
        
        plt.xlabel('Generation', fontsize=12, fontweight='bold')
        plt.ylabel('Fitness Score', fontsize=12, fontweight='bold')
        plt.title('Fitness Progress During Optimization', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best', framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        max_best_idx = np.argmax(best)
        max_best = best[max_best_idx]
        plt.annotate(f'Max: {max_best:.2f}', 
                     xy=(max_best_idx, max_best),
                     xytext=(max_best_idx, max_best*1.03),
                     fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                     ha='center')
        
        plt.tight_layout()
        plt.savefig('fitness_progress.png', dpi=300)
        plt.show()

    @staticmethod
    def plot_nutrition_comparison(min_daily: Dict[str, float], opt_daily: Dict[str, float], 
                                actual: Dict[str, float]) -> None:
        """
        Plot comparison of minimum, optimal, and actual nutrient values with enhanced readability
        
        Args:
            min_daily: Minimum required nutrient values
            opt_daily: Optimal nutrient values
            actual: Actual nutrient values
        """
        nuts = list(actual.keys())
        
        fig = plt.figure(figsize=(16, 10))
        
        gs = gridspec.GridSpec(2, 4)
        
        colors = ['#3498db', '#2ecc71', '#f1c40f']  # Blue, Green, Yellow
        bar_labels = ['Minimum', 'Actual', 'Optimal']
        
        for i, nut in enumerate(nuts):
            row, col = i // 4, i % 4
            ax = fig.add_subplot(gs[row, col])
            
            values = [min_daily[nut], actual[nut], opt_daily[nut]]
            bars = ax.bar(bar_labels, values, color=colors, width=0.6, 
                    edgecolor='gray', linewidth=0.5)
            
            # if actual[nut] >= min_daily[nut]:
            #     bars[1].set_color('#2ecc71')  # Green if satisfying constraint
            # else:
            #     bars[1].set_color('#e74c3c')  # Red if not satisfying constraint
            
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{values[j]:.1f}', ha='center', va='bottom', 
                        fontsize=9, fontweight='bold')
            
            if nut in ['protein', 'fiber', 'calcium', 'iron']:
                # Max nutrients: highlight the band between minimum and optimal
                ax.axhspan(min_daily[nut], opt_daily[nut], alpha=0.1, color='green')
            else:
                # Min nutrients: highlight the band between optimal and minimum
                ax.axhspan(opt_daily[nut], min_daily[nut], alpha=0.1, color='green')
            
            if nut not in ['calories']:
                pct = (actual[nut] / opt_daily[nut]) * 100
                ax.text(1, values[1] * 0.5, f"{pct:.1f}%", ha='center',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
                        fontsize=9)
            
            ax.set_title(nut.capitalize(), fontsize=12, fontweight='bold')
            ax.set_ylabel('Daily Value', fontsize=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.suptitle('Nutrient Comparison: Minimum vs. Actual vs. Optimal', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('nutrient_comparison.png', dpi=300)
        plt.show()

    @staticmethod
    def plot_nutrition_progress(hist: Dict[str, List[float]]) -> None:
        """
        Plot the progress of nutrient-to-calorie ratios over generations with enhanced readability
        
        Args:
            hist: Dictionary of nutrient histories
        """
        plt.figure(figsize=(14, 8))
        
        colors = cm.tab10(np.linspace(0, 1, len(hist)))
        
        for i, (nut, vals) in enumerate(hist.items()):
            marker_interval = max(1, len(vals) // 15)
            plt.plot(vals, label=nut.capitalize(), color=colors[i], linewidth=2.5, 
                    marker='o', markevery=marker_interval, markersize=5)
        
        plt.xlabel('Generation', fontsize=12, fontweight='bold')
        plt.ylabel('Nutrient/Calorie Ratio', fontsize=12, fontweight='bold')
        plt.title('Nutrient to Calorie Ratio Optimization Progress', 
                 fontsize=14, fontweight='bold')
        
        plt.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  ncol=min(4, len(hist)), framealpha=0.9)
        
        plt.grid(True, alpha=0.3, linestyle='--')
        
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        for i, (nut, vals) in enumerate(hist.items()):
            if len(vals) > 0:
                plt.annotate(f'{vals[-1]:.4f}', 
                            xy=(len(vals)-1, vals[-1]),
                            xytext=(len(vals)-1, vals[-1]*1.05),
                            fontsize=9,
                            color=colors[i],
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7),
                            ha='center')
        
        plt.tight_layout()
        plt.savefig('nutrient_progress.png', dpi=300)
        plt.show()

    @staticmethod
    def plot_diversity(div: List[float]) -> None:
        """
        Plot the diversity of the population over generations with enhanced readability
        
        Args:
            div: List of diversity values
        """
        plt.figure(figsize=(12, 7))
        
        plt.plot(div, color='purple', linewidth=2.5, marker='o', 
                markevery=max(1, len(div)//10), markersize=6)
        
        plt.fill_between(range(len(div)), div, alpha=0.2, color='purple')
        
        plt.xlabel('Generation', fontsize=12, fontweight='bold')
        plt.ylabel('Diversity (Normalized Entropy)', fontsize=12, fontweight='bold')
        plt.title('Population Diversity Over Optimization Process', 
                 fontsize=14, fontweight='bold')
        
        plt.ylim(0, max(1.0, max(div) * 1.1))
        
        plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='High Diversity')
        plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium Diversity')
        plt.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Low Diversity')
        
        plt.legend(fontsize=10, loc='best')
        
        plt.grid(True, alpha=0.3, linestyle='--')
        
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        if len(div) > 1:
            plt.annotate(f'Initial: {div[0]:.3f}', 
                        xy=(0, div[0]),
                        xytext=(len(div)*0.05, div[0]),
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                        ha='left')
            
            plt.annotate(f'Final: {div[-1]:.3f}', 
                        xy=(len(div)-1, div[-1]),
                        xytext=(len(div)*0.95, div[-1]),
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                        ha='right')
        
        plt.tight_layout()
        plt.savefig('diversity_progress.png', dpi=300)
        plt.show()
    
    @staticmethod
    def plot_food_distribution(foods, quantities, cost_per_item):
        """
        Plot the distribution of food items in the final solution
        
        Args:
            foods: List of food items
            quantities: List of quantities (kg)
            cost_per_item: List of costs for each food item
        """
        threshold = 0.1  # kg
        filtered_indices = [i for i, q in enumerate(quantities) if q > threshold]
        filtered_foods = [foods[i].name for i in filtered_indices]
        filtered_quantities = [quantities[i] for i in filtered_indices]
        filtered_costs = [cost_per_item[i] for i in filtered_indices]
        
        sorted_indices = np.argsort(filtered_quantities)[::-1]
        sorted_foods = [filtered_foods[i] for i in sorted_indices]
        sorted_quantities = [filtered_quantities[i] for i in sorted_indices]
        sorted_costs = [filtered_costs[i] for i in sorted_indices]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        
        colors = cm.viridis(np.linspace(0, 0.8, len(sorted_foods)))
        
        bars1 = ax1.barh(sorted_foods, sorted_quantities, color=colors)
        ax1.set_xlabel('Quantity (kg)', fontsize=12, fontweight='bold')
        ax1.set_title('Food Quantities in Optimal Diet', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f} kg', ha='left', va='center', fontsize=9)
        
        bars2 = ax2.barh(sorted_foods, sorted_costs, color=colors)
        ax2.set_xlabel('Cost (Toman)', fontsize=12, fontweight='bold')
        ax2.set_title('Cost Distribution in Optimal Diet', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        total_cost = sum(sorted_costs)
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            percentage = (width / total_cost) * 100
            ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{width:.0f} ({percentage:.1f}%)', ha='left', va='center', fontsize=9)
        
        plt.suptitle('Optimal Food Distribution', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('food_distribution.png', dpi=300)
        plt.show()
        
    @staticmethod
    def plot_summary(best_fitness, cost, min_daily, optimal_daily, actual_daily, total_food_qty):
        """
        Create a summary dashboard with key metrics
        
        Args:
            best_fitness: Best fitness value achieved
            cost: Total cost of the solution
            min_daily, optimal_daily, actual_daily: Nutrient values
            total_food_qty: Total quantity of food in kg
        """
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 2)
        
        # 1. Overall metrics
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['Fitness', 'Cost (Million T)', 'Total Food (kg)']
        values = [best_fitness, cost/1_000_000, total_food_qty]
        
        bars = ax1.barh(metrics, values, color=['#3498db', '#e74c3c', '#2ecc71'])
        
        # Add labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width * 1.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.2f}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        ax1.set_title('Overall Metrics', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 2. Nutrient fulfillment percentage
        ax2 = fig.add_subplot(gs[0, 1])
        nutrients = list(min_daily.keys())
        percentages = []
        
        for nut in nutrients:
            if nut in ['protein', 'fiber', 'calcium', 'iron']:
                # For 'max' nutrients, we want to exceed minimum
                pct = (actual_daily[nut] / optimal_daily[nut]) * 100
            else:
                # For 'min' nutrients, we want to be below minimum
                target = min(min_daily[nut], optimal_daily[nut])
                pct = min(100, (target / actual_daily[nut]) * 100 if actual_daily[nut] > 0 else 0)
            percentages.append(pct)
        
        bar_colors = ['#2ecc71' if p >= 90 else '#f1c40f' if p >= 70 else '#e74c3c' for p in percentages]
        
        bars = ax2.barh([n.capitalize() for n in nutrients], percentages, color=bar_colors)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%', ha='left', va='center', fontsize=9, fontweight='bold')
        
        ax2.set_title('Nutrient Optimization Score', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 110)
        ax2.axvline(x=100, color='green', linestyle='--', alpha=0.7)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 3. Nutrient actual vs. target
        ax3 = fig.add_subplot(gs[1, :])
        
        x = np.arange(len(nutrients))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, [actual_daily[nut] for nut in nutrients], width, 
                      label='Actual', color='#3498db')
        bars2 = ax3.bar(x + width/2, [optimal_daily[nut] for nut in nutrients], width, 
                      label='Optimal', color='#f1c40f')
        
        ax3.set_title('Nutrients: Actual vs. Optimal Values', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([n.capitalize() for n in nutrients])
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        
        for i, (bars, values) in enumerate(zip([bars1, bars2], 
                                           [[actual_daily[nut] for nut in nutrients], 
                                           [optimal_daily[nut] for nut in nutrients]])):
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{values[j]:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('Optimization Results Summary', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('optimization_summary.png', dpi=300)
        plt.show()