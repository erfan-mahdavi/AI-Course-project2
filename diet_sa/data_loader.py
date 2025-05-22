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
            ValueError: If required columns are missing from the CSV
            Exception: For other data loading errors
        """
        try:
            # Check if file exists
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"File '{csv_path}' not found.")
                
            # Load CSV file
            df = pd.read_csv(csv_path)
            
            # Verify required columns exist
            required_columns = ['Food', 'Calories_kcal', 'Protein_g', 'Fat_g', 
                               'Carbohydrates_g', 'Fiber_g', 'Calcium_mg', 
                               'Iron_mg', 'Price_Toman_per_kg']
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in CSV: {', '.join(missing_cols)}")
            
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
                    
                    # Validate nutrient values (should be non-negative)
                    for nutrient, value in nutrients.items():
                        if value < 0:
                            print(f"{Fore.YELLOW}Warning: Negative {nutrient} value for {row['Food']}: {value}")
                            nutrients[nutrient] = 0  # Set to zero if negative
                    
                    # Validate price
                    price = float(row['Price_Toman_per_kg'])
                    if price <= 0:
                        print(f"{Fore.YELLOW}Warning: Invalid price for {row['Food']}: {price}. Skipping item.")
                        continue
                    
                    # Create FoodItem object
                    food_item = FoodItem(
                        name=str(row['Food']).strip(),  # Remove whitespace
                        nutrients=nutrients,
                        price_per_kg=price
                    )
                    
                    items.append(food_item)
                    
                except (ValueError, KeyError) as e:
                    print(f"{Fore.YELLOW}Warning: Error processing row {index+1} ({row.get('Food', 'Unknown')}): {e}")
                    continue
            
            # Check if any items were successfully loaded
            if not items:
                print(f"{Fore.YELLOW}Warning: No food items were loaded from the CSV file.")
                return []
                
            print(f"{Fore.GREEN}Successfully loaded {len(items)} food items from '{csv_path}'")
            
            # Print summary of loaded data
            DataLoader.print_food_summary(items)
            
            return items
                
        except FileNotFoundError as e:
            print(f"{Fore.RED}Error: {e}")
            raise
        except ValueError as e:
            print(f"{Fore.RED}Data validation error: {e}")
            raise
        except Exception as e:
            print(f"{Fore.RED}Error loading food data: {e}")
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
        
        # Find most and least expensive items
        most_expensive = max(items, key=lambda x: x.price)
        least_expensive = min(items, key=lambda x: x.price)
        
        print(f"\nMost expensive: {most_expensive.name} ({most_expensive.price:,} T/kg)")
        print(f"Least expensive: {least_expensive.name} ({least_expensive.price:,} T/kg)")
        
        # Find highest protein foods
        highest_protein = max(items, key=lambda x: x.nutrients['protein'])
        print(f"Highest protein: {highest_protein.name} ({highest_protein.nutrients['protein']:.1f} g/100g)")
        
        print()  # Empty line for better formatting
    
    @staticmethod
    def validate_food_data(items: List[FoodItem]) -> bool:
        """
        Validate the loaded food data for common issues.
        
        This method performs additional validation checks on the loaded food items
        to ensure data quality and consistency.
        
        Args:
            items: List of FoodItem objects to validate
            
        Returns:
            True if all data passes validation, False if issues are found
        """
        if not items:
            print(f"{Fore.RED}Validation failed: No food items to validate.")
            return False
        
        issues_found = 0
        
        print(f"{Fore.CYAN}Validating food data...")
        
        for i, food in enumerate(items):
            # Check for extremely high or low values that might be data entry errors
            
            # Calories should be reasonable (50-900 kcal per 100g for most foods)
            if food.nutrients['calories'] > 900:
                print(f"{Fore.YELLOW}Warning: Very high calories for {food.name}: {food.nutrients['calories']} kcal/100g")
                issues_found += 1
            elif food.nutrients['calories'] < 5:
                print(f"{Fore.YELLOW}Warning: Very low calories for {food.name}: {food.nutrients['calories']} kcal/100g")
                issues_found += 1
            
            # Protein should be reasonable (0-50g per 100g for most foods)
            if food.nutrients['protein'] > 50:
                print(f"{Fore.YELLOW}Warning: Very high protein for {food.name}: {food.nutrients['protein']} g/100g")
                issues_found += 1
            
            # Fat should be reasonable (0-100g per 100g)
            if food.nutrients['fat'] > 100:
                print(f"{Fore.YELLOW}Warning: Fat content exceeds 100g per 100g for {food.name}: {food.nutrients['fat']} g/100g")
                issues_found += 1
            
            # Check for unrealistic prices (too high or too low)
            if food.price > 1_000_000:  # More than 1 million Toman per kg
                print(f"{Fore.YELLOW}Warning: Very expensive food {food.name}: {food.price:,} T/kg")
                issues_found += 1
            elif food.price < 1000:  # Less than 1000 Toman per kg
                print(f"{Fore.YELLOW}Warning: Very cheap food {food.name}: {food.price:,} T/kg")
                issues_found += 1
            
            # Check for duplicate food names
            for j, other_food in enumerate(items[i+1:], i+1):
                if food.name.lower() == other_food.name.lower():
                    print(f"{Fore.YELLOW}Warning: Duplicate food name found: {food.name} (rows {i+1} and {j+1})")
                    issues_found += 1
        
        if issues_found == 0:
            print(f"{Fore.GREEN}Data validation passed: No issues found.")
            return True
        else:
            print(f"{Fore.YELLOW}Data validation completed: {issues_found} potential issues found.")
            return False
    
    @staticmethod
    def save_food_summary(items: List[FoodItem], output_path: str) -> None:
        """
        Save a summary of food items to a text file.
        
        Args:
            items: List of FoodItem objects to summarize
            output_path: Path where to save the summary file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("FOOD DATA SUMMARY\n")
                f.write("="*50 + "\n\n")
                f.write(f"Total food items: {len(items)}\n")
                f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Write detailed food list
                for food in sorted(items, key=lambda x: x.name):
                    f.write(f"Food: {food.name}\n")
                    f.write(f"  Calories: {food.nutrients['calories']:.1f} kcal/100g\n")
                    f.write(f"  Protein: {food.nutrients['protein']:.1f} g/100g\n")
                    f.write(f"  Fat: {food.nutrients['fat']:.1f} g/100g\n")
                    f.write(f"  Carbs: {food.nutrients['carbs']:.1f} g/100g\n")
                    f.write(f"  Fiber: {food.nutrients['fiber']:.1f} g/100g\n")
                    f.write(f"  Calcium: {food.nutrients['calcium']:.1f} mg/100g\n")
                    f.write(f"  Iron: {food.nutrients['iron']:.1f} mg/100g\n")
                    f.write(f"  Price: {food.price:,} Toman/kg\n\n")
            
            print(f"{Fore.GREEN}Food summary saved to: {output_path}")
            
        except Exception as e:
            print(f"{Fore.RED}Error saving food summary: {e}")