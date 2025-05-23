import pandas as pd
import os
from typing import List
from .models import FoodItem
from colorama import Fore, init
import tabulate

# Initialize colorama for colored terminal output
init(autoreset=True)

class DataLoader:
    """
    Data loading utility class for diet optimization.
    
    This class handles loading and validating food data from CSV files.
    It ensures data integrity and provides useful feedback about the loaded data.
    """
    
    @staticmethod
    def load_foods(csv_path: str) -> List[FoodItem]:
        """
        Load food items from a CSV file.
        
        This method reads a CSV file containing food data and creates FoodItem objects
        for each food. It validates the data structure and handles errors gracefully.
        
        Expected CSV format:
        - Food: Name of the food item
        - Calories_kcal: Calories per 100g
        - Protein_g: Protein content per 100g
        - Fat_g: Fat content per 100g  
        - Carbohydrates_g: Carbohydrate content per 100g
        - Fiber_g: Fiber content per 100g
        - Calcium_mg: Calcium content per 100g
        - Iron_mg: Iron content per 100g
        - Price_Toman_per_kg: Price in Toman per kilogram
        
        Args:
            csv_path: Path to the CSV file containing food data
            
        Returns:
            List of FoodItem objects loaded from the CSV file
            
        Raises:
            FileNotFoundError: If the CSV file does not exist
            Exception: For other data loading errors
        """
        try:
            # Check if file exists
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"File '{csv_path}' not found.")
                
            # Load CSV file
            df = pd.read_csv(csv_path)
            
            # Check for empty dataframe
            if df.empty:
                raise ValueError("CSV file is empty or contains no valid data.")
            
            # Create FoodItem objects
            items: List[FoodItem] = []
            
            for index, row in df.iterrows():
                try:
                    # Extract nutrient information per 100g
                    nutrients = {
                        'calories':    float(row['Calories_kcal']),
                        'protein':     float(row['Protein_g']),
                        'fat':         float(row['Fat_g']),
                        'carbs':       float(row['Carbohydrates_g']),
                        'fiber':       float(row['Fiber_g']),
                        'calcium':     float(row['Calcium_mg']),
                        'iron':        float(row['Iron_mg']),
                    }
                    
                    price = float(row['Price_Toman_per_kg'])
                    
                    # Create FoodItem object
                    food_item = FoodItem(
                        name=str(row['Food']).strip(),
                        nutrients=nutrients,
                        price_per_kg=price
                    )
                    
                    items.append(food_item)
                    
                except (ValueError, KeyError) as e:
                    print(f"{Fore.YELLOW}Warning: Error processing row {index+1} ({row.get('Food', 'Unknown')}): {e}")
                    continue
            
            # Print summary of loaded data
            DataLoader.print_food_summary(items)
            
            return items
                
        except FileNotFoundError as e:
            print(f"{Fore.RED}Error: {e}")
            raise
    
    @staticmethod
    def print_food_summary(items: List[FoodItem]) -> None:
        """
        Print a summary of loaded food items in a formatted table.
        
        This method displays the first 10 food items (or all if fewer than 10)
        in a nicely formatted table showing nutritional content and prices.
        
        Args:
            items: List of FoodItem objects to summarize
        """
        if not items:
            print(f"{Fore.YELLOW}No food items to display.")
            return
        
        # Prepare table data
        table_data = []
        for food in items:
            table_data.append([
                food.name,
                f"{food.nutrients['calories']:.1f}",
                f"{food.nutrients['protein']:.1f}",
                f"{food.nutrients['fat']:.1f}",
                f"{food.nutrients['carbs']:.1f}",
                f"{food.nutrients['fiber']:.1f}",
                f"{food.nutrients['calcium']:.1f}",
                f"{food.nutrients['iron']:.1f}",
                f"{food.price:,}"
            ])
        
        # Sort by food name for consistent display
        table_data.sort(key=lambda x: x[0])
        
        # Show only first 10 items if there are more than 10
        if len(table_data) > 10:
            display_data = table_data[:10]
            print(f"\n{Fore.CYAN}Food Item Summary (Showing first 10 of {len(table_data)} items):")
        else:
            display_data = table_data
            print(f"\n{Fore.CYAN}Food Item Summary (All {len(table_data)} items):")
        
        # Generate and display table
        headers = ["Food", "Cal", "Protein", "Fat", "Carbs", "Fiber", "Calcium", "Iron", "Price (T/kg)"]
        print(tabulate.tabulate(display_data, headers=headers, tablefmt="grid"))
        
        # Print additional statistics
        prices = [food.price for food in items]
        calories = [food.nutrients['calories'] for food in items]
        proteins = [food.nutrients['protein'] for food in items]
        
        print(f"\n{Fore.CYAN}Data Statistics:")
        print(f"Price range: {min(prices):,} to {max(prices):,} Toman per kg")
        print(f"Average price: {sum(prices)/len(prices):,.0f} Toman per kg")
        print(f"Calories range: {min(calories):.1f} to {max(calories):.1f} kcal per 100g")
        print(f"Protein range: {min(proteins):.1f} to {max(proteins):.1f} g per 100g")
        
        print()

