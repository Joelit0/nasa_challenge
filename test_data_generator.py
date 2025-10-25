import pandas as pd
import numpy as np


def generate_sample_exoplanet_data(n_samples=100, seed=42):
    """
    Generate realistic synthetic exoplanet data for testing the Streamlit app.
    Columns correspond to the model's feature set:
    ['planet_radius', 'transit_depth', 'transit_duration', 'orbital_period',
     'stellar_temp', 'stellar_radius', 'stellar_logg', 'snr', 'total_fp_flags']
    """
    np.random.seed(seed)

    data = {
        # Planetary features
        "planet_radius": np.random.lognormal(
            mean=1.2, sigma=0.6, size=n_samples
        ),  # ~3–5 Earth radii typical median
        "transit_depth": np.random.lognormal(
            mean=7.0, sigma=1.2, size=n_samples
        ),  # ~ppm scale
        "transit_duration": np.random.uniform(0.5, 12.0, size=n_samples),  # hours
        "orbital_period": np.random.lognormal(
            mean=1.5, sigma=1.0, size=n_samples
        ),  # days
        # Stellar features
        "stellar_temp": np.random.normal(
            loc=5700, scale=800, size=n_samples
        ),  # K (Sun-like)
        "stellar_radius": np.random.lognormal(
            mean=0.0, sigma=0.3, size=n_samples
        ),  # solar radii
        "stellar_logg": np.random.normal(loc=4.4, scale=0.2, size=n_samples),  # cgs
        # Observation-related features
        "snr": np.random.uniform(5, 100, size=n_samples),  # signal-to-noise ratio
        "total_fp_flags": np.random.randint(
            0, 4, size=n_samples
        ),  # sum of false-positive flags
    }

    df = pd.DataFrame(data)

    # Clip to realistic physical limits
    df["planet_radius"] = df["planet_radius"].clip(lower=0.5, upper=20)
    df["transit_depth"] = df["transit_depth"].clip(lower=50, upper=50000)
    df["orbital_period"] = df["orbital_period"].clip(lower=0.5, upper=500)
    df["stellar_temp"] = df["stellar_temp"].clip(lower=3000, upper=10000)
    df["stellar_radius"] = df["stellar_radius"].clip(lower=0.1, upper=10)
    df["stellar_logg"] = df["stellar_logg"].clip(lower=3.5, upper=5.0)
    df["snr"] = df["snr"].clip(lower=1, upper=200)

    # Add synthetic IDs
    df.insert(0, "object_id", [f"TEST-{i:04d}" for i in range(n_samples)])

    return df


if __name__ == "__main__":
    test_data = generate_sample_exoplanet_data(n_samples=100)
    test_data.to_csv("test_exoplanet_data.csv", index=False)

    print("✅ Test data generated: test_exoplanet_data.csv")
    print(f"\nShape: {test_data.shape}")
    print("\nFirst few rows:")
    print(test_data.head())
    print("\nStatistics:")
    print(test_data.describe())
