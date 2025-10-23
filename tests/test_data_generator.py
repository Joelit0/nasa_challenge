import pandas as pd
import numpy as np

def generate_sample_exoplanet_data(n_samples=100, seed=42):
    """
    Generate sample exoplanet data for testing the app
    """
    np.random.seed(seed)
    
    # Generate realistic ranges based on actual exoplanet data
    data = {
        'planet_radius': np.random.lognormal(1.5, 0.8, n_samples),  # Earth radii
        'transit_depth': np.random.lognormal(7, 1.5, n_samples),    # ppm
        'transit_duration': np.random.uniform(1, 8, n_samples),      # hours
        'orbital_period': np.random.lognormal(1.5, 1.2, n_samples), # days
        'eq_temperature': np.random.normal(1200, 600, n_samples),    # Kelvin
        'stellar_temp': np.random.normal(5800, 1000, n_samples),     # Kelvin
        'stellar_radius': np.random.lognormal(0, 0.3, n_samples),    # Solar radii
        'logg': np.random.normal(4.4, 0.3, n_samples)                # log(cm/s²)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure positive values where needed
    df['planet_radius'] = df['planet_radius'].clip(lower=0.1)
    df['transit_depth'] = df['transit_depth'].clip(lower=10)
    df['orbital_period'] = df['orbital_period'].clip(lower=0.5)
    df['eq_temperature'] = df['eq_temperature'].clip(lower=100)
    df['stellar_temp'] = df['stellar_temp'].clip(lower=2800, upper=50000)
    df['stellar_radius'] = df['stellar_radius'].clip(lower=0.1)
    
    # Add some object IDs for reference
    df.insert(0, 'object_id', [f'TEST-{i:04d}' for i in range(n_samples)])
    
    return df

# Generate and save sample data
if __name__ == "__main__":
    # Generate test data
    test_data = generate_sample_exoplanet_data(n_samples=100)
    
    # Save to CSV
    test_data.to_csv('test_exoplanet_data.csv', index=False)
    print("✅ Test data generated: test_exoplanet_data.csv")
    print(f"\nShape: {test_data.shape}")
    print(f"\nFirst few rows:")
    print(test_data.head())
    print(f"\nStatistics:")
    print(test_data.describe())
