import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class PreviousValueRegressor:
    """
    Baseline regressor that always predicts the previous observed value.
    Simple incremental learning model that remembers only the last seen value.
    """
    def __init__(self):
        self.last_value = 0.0  # Default value when no data has been seen

    def learn_one(self, x, y):
        """River compatibility - learn from one sample"""
        self.last_value = y
        return self

    def predict_one(self, x):
        """River compatibility - predict for one sample"""
        return self.last_value
    

class SeasonalMeanRegressor(BaseEstimator, RegressorMixin):
    """
    sklearn-style regressor that predicts the mean value based on seasonality.
    """
    def __init__(self, freq_mins=60):
        # freq_mins: number of minutes between observations (e.g. 60 for hourly, 15 for quarter-hour)
        self.freq_mins = freq_mins
        self.periods_per_day = int(24*60 / self.freq_mins)
        self.periods_per_week = self.periods_per_day * 7
        self.season_stats = {}   # period_id -> [sum, count]
        self.sum_total = 0.0
        self.count_total = 0
        # to be set by fit()
        self.period_means = None
        self.default_mean = None

    def _get_period_id(self, dt):
        """Compute a unique period ID from a datetime or numpy datetime64."""
        dt = pd.to_datetime(dt)
        weekday = dt.weekday()      # 0 = Monday
        mins = dt.hour*60 + dt.minute
        slot = mins // self.freq_mins
        return weekday * self.periods_per_day + slot

    def fit(self, X, y):
        """Vectorized fit using pandas groupby."""
        df = pd.DataFrame({'timestamp': X[:, 0], 'y': y})  # Assuming X[:, 0] contains timestamps
        df['dt'] = pd.to_datetime(df['timestamp'])
        df['period_id'] = (
            df['dt'].dt.weekday * self.periods_per_day
            + ((df['dt'].dt.hour * 60 + df['dt'].dt.minute) // self.freq_mins)
        )
        self.period_means = df.groupby('period_id')['y'].mean()
        self.default_mean = df['y'].mean()
        return self

    def predict(self, X):
        """Vectorized predict using stored seasonal means."""
        df = pd.DataFrame({'timestamp': X[:, 0]})
        df['dt'] = pd.to_datetime(df['timestamp'])
        df['period_id'] = (
            df['dt'].dt.weekday * self.periods_per_day
            + ((df['dt'].dt.hour * 60 + df['dt'].dt.minute) // self.freq_mins)
        )
        return self.period_means.reindex(df['period_id']).fillna(self.default_mean).values

if __name__ == "__main__":
    print("Testing SeasonalMeanRegressor...")
    
    # Generate synthetic data with weekly patterns
    start_date = pd.Timestamp('2023-01-01')
    days = 21  # 3 weeks of data
    hours_per_day = 24
    total_hours = days * hours_per_day
    
    # Create timestamps for hourly data
    timestamps = [start_date + pd.Timedelta(hours=i) for i in range(total_hours)]
    timestamps_array = np.array([[ts] for ts in timestamps])
    
    # Generate synthetic values with daily and weekly patterns
    np.random.seed(42)
    values = []
    for ts in timestamps:
        # Base value
        value = 10
        
        # Daily pattern: higher during day, lower at night
        hour = ts.hour
        if 8 <= hour <= 20:  # Daytime hours
            value += 5
        
        # Weekly pattern: higher on weekends
        if ts.dayofweek >= 5:  # Saturday and Sunday
            value += 8
        
        # Add some noise
        value += np.random.normal(0, 1)
        values.append(value)
    
    values = np.array(values)
    
    # Split into train and test (first 2 weeks for training, last week for testing)
    train_size = 14 * hours_per_day  # 2 weeks
    
    X_train = timestamps_array[:train_size]
    y_train = values[:train_size]
    
    X_test = timestamps_array[train_size:]
    y_test = values[train_size:]
    
    # Fit the model
    model = SeasonalMeanRegressor(freq_mins=60)  # hourly data
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    
    # Plot results (if matplotlib is available)
    try:
        import matplotlib.pyplot as plt
        
        # Convert to datetime for better plotting
        test_dates = [pd.to_datetime(ts[0]) for ts in X_test]
        
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, y_test, label='Actual Values', marker='o', alpha=0.6)
        plt.plot(test_dates, y_pred, label='Predictions', marker='x', linestyle='--')
        plt.title('SeasonalMeanRegressor: Actual vs Predicted Values')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add day separators
        for i in range(7):
            day_start = train_size + i * hours_per_day
            if day_start < len(timestamps):
                plt.axvline(x=timestamps[day_start], color='gray', alpha=0.5, linestyle=':')
        
        plt.tight_layout()
        plt.show()
        
        # Plot period means to see the learned seasonal pattern
        periods = sorted(model.period_means.index)
        means = [model.period_means[p] for p in periods]
        
        plt.figure(figsize=(12, 6))
        plt.bar(periods, means)
        plt.title('Learned Seasonal Patterns')
        plt.xlabel('Period ID (hours within week)')
        plt.ylabel('Mean Value')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")



