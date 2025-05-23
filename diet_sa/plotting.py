import matplotlib.pyplot as plt
from typing import Dict
import matplotlib.gridspec as gridspec

class Plotter:
    """
    Plotting utility class specifically designed for Simulated Annealing diet optimization.
    
    This class provides visualization methods tailored to SA's single-solution approach,
    including temperature tracking, acceptance rates, and convergence analysis.
    """
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