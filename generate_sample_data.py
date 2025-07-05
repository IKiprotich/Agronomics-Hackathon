#!/usr/bin/env python3
"""
Sample Data Generator for Agricultural Dashboard

This script generates sample CSV files that can be used to test the agricultural dashboard.
Run this script to create sample_env_data.csv, sample_crop_data.csv, and sample_livestock_data.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_environmental_data(n_samples=1000):
    """Generate sample environmental data"""
    print("Generating environmental data...")
    
    np.random.seed(42)
    
    # Generate realistic environmental data
    data = []
    
    for i in range(n_samples):
        # Simulate seasonal temperature variations
        month = np.random.randint(1, 13)
        season_factor = np.sin(2 * np.pi * month / 12)
        
        temperature = 22 + 5 * season_factor + np.random.normal(0, 3)
        humidity = 65 + 10 * season_factor + np.random.normal(0, 8)
        
        # Pressure varies less
        pressure = 840 + np.random.normal(0, 5)
        
        # Time features
        hour = np.random.randint(0, 24)
        day_of_year = np.random.randint(1, 366)
        day_of_week = np.random.randint(0, 7)
        
        data.append({
            'temperature': round(temperature, 2),
            'humidity': round(max(0, min(100, humidity)), 2),
            'pressure': round(pressure, 2),
            'hour': hour,
            'month': month,
            'day_of_year': day_of_year,
            'day_of_week': day_of_week
        })
    
    return pd.DataFrame(data)

def generate_crop_data():
    """Generate sample crop data"""
    print("Generating crop data...")
    
    np.random.seed(42)
    
    # Kenyan counties
    counties = [
        'Nairobi', 'Kiambu', 'Nakuru', 'Meru', 'Kisumu', 'Eldoret', 'Mombasa', 
        'Nyeri', 'Kakamega', 'Kisii', 'Machakos', 'Kitui', 'Embu', 'Murang\'a',
        'Kirinyaga', 'Nyandarua', 'Laikipia', 'Kericho', 'Bomet', 'Narok'
    ]
    
    # Common crops in Kenya
    crops = [
        'Maize', 'Beans', 'Potatoes', 'Rice', 'Wheat', 'Sorghum', 
        'Millet', 'Cassava', 'Sweet Potatoes', 'Bananas', 'Coffee', 'Tea'
    ]
    
    data = []
    
    for county in counties:
        for crop in crops:
            # Different crops have different suitability in different counties
            county_factor = np.random.uniform(0.5, 1.5)
            crop_factor = np.random.uniform(0.7, 1.3)
            
            production = int(np.random.lognormal(6, 1.5) * county_factor * crop_factor)
            area_planted = int(production * np.random.uniform(0.1, 0.3))
            yield_per_hectare = round(production / max(area_planted, 1), 2)
            
            data.append({
                'county': county,
                'crop': crop,
                'production': production,
                'area_planted': area_planted,
                'yield_per_hectare': yield_per_hectare
            })
    
    return pd.DataFrame(data)

def generate_livestock_data():
    """Generate sample livestock data"""
    print("Generating livestock data...")
    
    np.random.seed(42)
    
    # Kenyan counties
    counties = [
        'Nairobi', 'Kiambu', 'Nakuru', 'Meru', 'Kisumu', 'Eldoret', 'Mombasa', 
        'Nyeri', 'Kakamega', 'Kisii', 'Machakos', 'Kitui', 'Embu', 'Murang\'a',
        'Kirinyaga', 'Nyandarua', 'Laikipia', 'Kericho', 'Bomet', 'Narok'
    ]
    
    # Common livestock in Kenya
    livestock_types = [
        'Indigenous cattle', 'Exotic cattle', 'Goats', 'Sheep', 
        'Indigenous Chicken', 'Exotic Chicken', 'Pigs', 'Donkeys', 
        'Camels', 'Rabbits', 'Ducks', 'Turkeys'
    ]
    
    data = []
    
    for county in counties:
        for livestock in livestock_types:
            # Different livestock have different suitability in different counties
            county_factor = np.random.uniform(0.3, 2.0)
            livestock_factor = np.random.uniform(0.5, 1.5)
            
            # Some livestock are more common than others
            if livestock in ['Indigenous cattle', 'Goats', 'Indigenous Chicken']:
                base_population = np.random.lognormal(8, 1.2)
            elif livestock in ['Exotic cattle', 'Sheep', 'Exotic Chicken']:
                base_population = np.random.lognormal(7, 1.0)
            else:
                base_population = np.random.lognormal(6, 1.5)
            
            population = int(base_population * county_factor * livestock_factor)
            productivity = round(np.random.uniform(0.6, 1.4), 2)
            
            data.append({
                'county': county,
                'livestock': livestock,
                'population': population,
                'productivity': productivity
            })
    
    return pd.DataFrame(data)

def main():
    """Generate and save sample data files"""
    print("🌾 Agricultural Dashboard - Sample Data Generator")
    print("="*50)
    
    # Create output directory if it doesn't exist
    output_dir = "sample_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Generate data
    env_data = generate_environmental_data(1000)
    crop_data = generate_crop_data()
    livestock_data = generate_livestock_data()
    
    # Save to CSV files
    env_file = os.path.join(output_dir, "sample_env_data.csv")
    crop_file = os.path.join(output_dir, "sample_crop_data.csv")
    livestock_file = os.path.join(output_dir, "sample_livestock_data.csv")
    
    env_data.to_csv(env_file, index=False)
    crop_data.to_csv(crop_file, index=False)
    livestock_data.to_csv(livestock_file, index=False)
    
    print(f"\n✅ Sample data generated successfully!")
    print(f"Files created:")
    print(f"  📊 Environmental data: {env_file} ({len(env_data)} rows)")
    print(f"  🌾 Crop data: {crop_file} ({len(crop_data)} rows)")
    print(f"  🐄 Livestock data: {livestock_file} ({len(livestock_data)} rows)")
    
    # Display preview
    print(f"\n📋 Data Preview:")
    print(f"\nEnvironmental Data Preview:")
    print(env_data.head())
    print(f"\nCrop Data Preview:")
    print(crop_data.head())
    print(f"\nLivestock Data Preview:")
    print(livestock_data.head())
    
    print(f"\n🚀 You can now use these files to test the agricultural dashboard!")
    print(f"Run: streamlit run agricultural_dashboard.py")

if __name__ == "__main__":
    main() 