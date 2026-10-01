import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_data(num_days=90):
    np.random.seed(42)
    start_date = datetime.now() - timedelta(days=num_days)
    
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    meal_types = ['Lunch', 'Dinner']
    menu_items = ['Rice', 'Dal', 'Roti', 'Vegetable Curry', 'Chicken Curry']
    
    data = []
    
    for d in dates:
        # Determine day of week and basic demand
        is_weekend = d.weekday() >= 5
        base_customers = 200 if not is_weekend else 50
        
        # Optional variables
        is_holiday = np.random.choice([0, 1], p=[0.95, 0.05]) if not is_weekend else 0
        has_event = np.random.choice([0, 1], p=[0.9, 0.1])
        temperature = np.random.normal(25, 5) # average 25C
        
        # Adjust customers based on factors
        expected_customers = base_customers
        if is_holiday:
            expected_customers *= 0.3
        if has_event:
            expected_customers *= 1.5
            
        expected_customers = int(np.random.normal(expected_customers, expected_customers * 0.1))
        expected_customers = max(0, expected_customers)
        
        for meal in meal_types:
            for item in menu_items:
                # Popularity of item
                popularity = {
                    'Rice': 0.9,
                    'Dal': 0.8,
                    'Roti': 0.7,
                    'Vegetable Curry': 0.6,
                    'Chicken Curry': 0.5
                }[item]
                
                # Meals served
                meals_served = int(expected_customers * popularity * np.random.normal(1, 0.1))
                meals_served = max(0, meals_served)
                
                # Meals prepared (kitchen overprepares based on habit)
                buffer = np.random.uniform(1.05, 1.20)
                meals_prepared = int(meals_served * buffer)
                
                # Waste calculation (kg)
                # Assume 0.2kg per unserved meal
                waste_kg = (meals_prepared - meals_served) * 0.2
                
                # Add some random noise to waste
                waste_kg = max(0, waste_kg + np.random.normal(0, 0.5))
                waste_kg = round(waste_kg, 2)
                
                data.append({
                    'date': d.strftime('%Y-%m-%d'),
                    'meal_type': meal,
                    'menu_item': item,
                    'meals_prepared': meals_prepared,
                    'meals_served': meals_served,
                    'food_waste_kg': waste_kg,
                    'temperature': round(temperature, 1),
                    'holiday': is_holiday,
                    'event': has_event,
                    'expected_customers': expected_customers
                })
                
    df = pd.DataFrame(data)
    
    # Create data directory if not exists
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/synthetic_historical_data.csv', index=False)
    print(f"Generated {len(df)} rows of synthetic data in data/synthetic_historical_data.csv")

if __name__ == '__main__':
    generate_synthetic_data()
