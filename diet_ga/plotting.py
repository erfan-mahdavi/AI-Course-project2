import matplotlib.pyplot as plt
from typing import Dict, List

class Plotter:
    @staticmethod
    def plot(best: List[float], avg: List[float]) -> None:
        plt.figure();plt.plot(best,label='Best');plt.plot(avg,label='Avg')
        plt.xlabel('Gen');plt.ylabel('Fitness');plt.legend();plt.tight_layout();plt.show()

    @staticmethod
    def plot_nutrition_comparison(min_daily: Dict[str,float], opt_daily: Dict[str,float], actual: Dict[str,float]) -> None:
        nuts=list(actual.keys())
        fig, axes = plt.subplots(1,len(nuts),figsize=(4*len(nuts),4))
        for ax,nut in zip(axes,nuts):
            ax.bar(['min','actual','opt'],[min_daily[nut],actual[nut],opt_daily[nut]])
            ax.set_title(nut);ax.set_ylabel('per-day')
        plt.tight_layout();plt.show()

    @staticmethod
    def plot_nutrition_progress(hist: Dict[str,List[float]]) -> None:
        plt.figure()
        for nut,vals in hist.items(): plt.plot(vals,label=nut)
        plt.xlabel('Gen');plt.ylabel('Nut/Cal');plt.legend();plt.tight_layout();plt.show()

    @staticmethod
    def plot_diversity(div: List[float]) -> None:
        plt.figure();plt.plot(div);plt.xlabel('Gen');plt.ylabel('Diversity');plt.tight_layout();plt.show()
