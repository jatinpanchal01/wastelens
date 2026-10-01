import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random

def generate_base_data(num_days=90):
    np.random.seed(42)
    start_date = datetime.now() - timedelta(days=num_days)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    meal_types = ['Lunch', 'Dinner']
    menu_items = ['Rice', 'Dal', 'Roti', 'Vegetable Curry', 'Chicken Curry']
    
    data = []
    for d in dates:
        is_weekend = d.weekday() >= 5
        base_customers = 200 if not is_weekend else 50
        is_holiday = np.random.choice([0, 1], p=[0.95, 0.05]) if not is_weekend else 0
        has_event = np.random.choice([0, 1], p=[0.9, 0.1])
        temperature = np.random.normal(25, 5)
        
        expected_customers = base_customers
        if is_holiday: expected_customers *= 0.3
        if has_event: expected_customers *= 1.5
        expected_customers = max(0, int(np.random.normal(expected_customers, expected_customers * 0.1)))
        
        for meal in meal_types:
            for item in menu_items:
                popularity = {'Rice': 0.9, 'Dal': 0.8, 'Roti': 0.7, 'Vegetable Curry': 0.6, 'Chicken Curry': 0.5}[item]
                meals_served = max(0, int(expected_customers * popularity * np.random.normal(1, 0.1)))
                buffer = np.random.uniform(1.05, 1.20)
                meals_prepared = int(meals_served * buffer)
                waste_kg = round(max(0, (meals_prepared - meals_served) * 0.2 + np.random.normal(0, 0.5)), 2)
                
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
    return pd.DataFrame(data)

def generate_test_datasets():
    os.makedirs('data/test_datasets', exist_ok=True)
    
    # 1. Golden Valid (30 days)
    df_golden = generate_base_data(30)
    df_golden.to_csv('data/test_datasets/test_golden_valid.csv', index=False)
    print("Created: test_golden_valid.csv (Perfect, clean dataset)")
    
    # 2. Sparse 14-day history (Barely meets threshold)
    df_sparse = generate_base_data(14)
    df_sparse.to_csv('data/test_datasets/test_sparse_14_days.csv', index=False)
    print("Created: test_sparse_14_days.csv (Minimum required history)")
    
    # 3. Invalid Rows (20% missing/corrupt values)
    df_invalid = generate_base_data(30)
    # Corrupt 20% of rows
    num_corrupt = int(len(df_invalid) * 0.2)
    corrupt_indices = random.sample(range(len(df_invalid)), num_corrupt)
    
    for idx in corrupt_indices:
        corruption_type = random.choice([1, 2, 3])
        if corruption_type == 1:
            df_invalid.loc[idx, 'meals_served'] = np.nan # Missing required column
        elif corruption_type == 2:
            df_invalid.loc[idx, 'date'] = 'invalid-date-format' # Bad date
        elif corruption_type == 3:
            df_invalid.loc[idx, 'food_waste_kg'] = 'string_instead_of_float' # Bad type
            
    df_invalid.to_csv('data/test_datasets/test_invalid_rows.csv', index=False)
    print("Created: test_invalid_rows.csv (20% corrupted rows to trigger validation warnings)")
    
    # 4. Adversarial values
    df_adv = generate_base_data(30)
    adv_indices = random.sample(range(len(df_adv)), 10)
    for idx in adv_indices:
        df_adv.loc[idx, 'food_waste_kg'] = -50.5  # Impossible negative waste
        df_adv.loc[idx, 'meals_prepared'] = -100 # Impossible negative prep
        df_adv.loc[idx, 'meals_served'] = 999999 # Impossible high spike
        
    df_adv.to_csv('data/test_datasets/test_adversarial.csv', index=False)
    print("Created: test_adversarial.csv (Contains extreme outliers and impossible negative values)")

if __name__ == '__main__':
    generate_test_datasets()
