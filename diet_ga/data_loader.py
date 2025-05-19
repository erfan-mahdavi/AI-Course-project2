import pandas as pd
import os
from typing import List, Dict
from .models import FoodItem
from colorama import Fore, Style, init
import tabulate

# Initialize colorama
init(autoreset=True)

class DataLoader:
    @staticmethod
    def load_foods(csv_path: str) -> List[FoodItem]:
        """
        Load food items from a CSV file
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            List of FoodItem objects
            
        Raises:
            FileNotFoundError: If the CSV file does not exist
            ValueError: If required columns are missing from the CSV
        """
        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"File '{csv_path}' not found.")
                
            df = pd.read_csv(csv_path)
            
            # Verify required columns
            required_columns = ['Food', 'Calories_kcal', 'Protein_g', 'Fat_g', 
                               'Carbohydrates_g', 'Fiber_g', 'Calcium_mg', 
                               'Iron_mg', 'Price_Toman_per_kg']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in CSV: {', '.join(missing_cols)}")
            
            items: List[FoodItem] = []
            for _, row in df.iterrows():
                nutrients = {
                    'calories':    row['Calories_kcal'],
                    'protein':     row['Protein_g'],
                    'fat':         row['Fat_g'],
                    'carbs':       row['Carbohydrates_g'],
                    'fiber':       row['Fiber_g'],
                    'calcium':     row['Calcium_mg'],
                    'iron':        row['Iron_mg'],
                }
                items.append(
                    FoodItem(
                        name=row['Food'],
                        nutrients=nutrients,
                        price_per_kg=row['Price_Toman_per_kg']
                    )
                )
            
            if not items:
                print(f"{Fore.YELLOW}Warning: No food items were loaded from the CSV file.")
                
            print(f"{Fore.GREEN}Successfully loaded {len(items)} food items from '{csv_path}'")
            DataLoader.print_food_summary(items)
            
            return items
                
        except Exception as e:
            print(f"{Fore.RED}Error loading food data: {e}")
            raise
    
    @staticmethod
    def print_food_summary(items: List[FoodItem]) -> None:
        """
        Print a summary of loaded food items
        
        Args:
            items: List of food items
        """
        # Prepare table data
        table_data = []
        for food in items:
            table_data.append([
                food.name,
                food.nutrients['calories'],
                food.nutrients['protein'],
                food.nutrients['fat'],
                food.nutrients['carbs'],
                food.nutrients['fiber'],
                food.nutrients['calcium'],
                food.nutrients['iron'],
                f"{food.price:,}"
            ])
        
        # Sort by food name
        table_data.sort(key=lambda x: x[0])
        
        # Show only first 10 items if there are more
        if len(table_data) > 10:
            display_data = table_data[:10]
            print(f"\n{Fore.CYAN}Food Item Summary (Showing first 10 of {len(table_data)} items):")
        else:
            display_data = table_data
            print(f"\n{Fore.CYAN}Food Item Summary (All {len(table_data)} items):")
        
        # Generate table
        headers = ["Food", "Cal", "Protein", "Fat", "Carbs", "Fiber", "Calcium", "Iron", "Price"]
        print(tabulate.tabulate(display_data, headers=headers, tablefmt="grid"))
        
        # Print price range
        prices = [food.price for food in items]
        print(f"{Fore.CYAN}Price range: {min(prices):,} to {max(prices):,} Toman per kg\n")