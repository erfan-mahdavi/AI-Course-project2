import pandas as pd
from typing import List
from .models import FoodItem

class DataLoader:
    @staticmethod
    def load_foods(csv_path: str) -> List[FoodItem]:
        df = pd.read_csv(csv_path)
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
        return items