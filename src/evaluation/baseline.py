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
    sklearn-style regressor that predicts values based on time-of-week patterns.
    
    This regressor calculates the mean value for each hour of each day of the week, 
    creating a 168-point (24 hours x 7 days) seasonality table. Any timestamp will 
    be mapped to its corresponding hour and day of week for prediction.
    """
    def __init__(self):
        """Initialize the seasonal mean regressor with a weekly seasonality table."""
        self.weekly_means = None    # (day_of_week, hour) -> mean value
        self.hour_means = None      # Hour of day (0-23) -> mean value (fallback)
        self.global_mean = None     # Fallback value for any missing slots
        
    def fit(self, X, y):
        """
        Fit the model by calculating mean values for each time slot.
        
        Creates a 168-point seasonality table (24 hours x 7 days).
        """
        df = pd.DataFrame({'hour': X[:, 0],  # X[:, 0] is the hour
                           'day_of_week': X[:, 1],  # X[:, 1] is the day of week
                           'y': y})

        # Calculate means for different seasonality patterns
        self.hour_means = df.groupby('hour')['y'].mean()
        self.weekly_means = df.groupby(['day_of_week', 'hour'])['y'].mean()
        self.global_mean = df['y'].mean()
        
        return self

    def predict(self, X):
        """
        Predict based on weekly seasonality patterns.
        
        Returns the mean value corresponding to the specific hour of the 
        specific day of week for each timestamp.
        """
        df = pd.DataFrame({'hour': X[:, 0],  # X[:, 0] is the hour
                           'day_of_week': X[:, 1]})  # X[:, 1] is the day of week

        # Use full weekly patterns (168 values)
        results = []
        for _, row in df.iterrows():
            day = row['day_of_week']
            hour = row['hour']
            try:
                # Try to get the specific day+hour mean
                val = self.weekly_means.loc[(day, hour)]
            except KeyError:
                print(f"Warning: No data for day {day}, hour {hour}. Using fallback...")
                # If not found, fall back to hour-only mean
                try:
                    val = self.hour_means.loc[hour]
                except KeyError:
                    # If still not found, use global mean
                    val = self.global_mean
            results.append(val)
        return np.array(results)


class ExponentialMovingAverageRegressor:
    """
    Incremental learning regressor that uses an exponential moving average to make predictions.
    Uses River-style interface with learn_one and predict_one methods.
    """
    def __init__(self, alpha=0.3):
        """
        Initialize the exponential moving average regressor.
        
        Parameters:
        -----------
        alpha : float, default=0.3
            Smoothing factor between 0 and 1.
            - Closer to 1: more weight on recent observations (adapts quickly)
            - Closer to 0: more weight on historical observations (more stable)
        """
        self.alpha = alpha
        self.current_average = None
    
    def learn_one(self, x, y):
        """River compatibility - learn from one sample"""
        if self.current_average is None:
            self.current_average = y
        else:
            self.current_average = self.alpha * y + (1 - self.alpha) * self.current_average
        return self
    
    def predict_one(self, x):
        """River compatibility - predict for one sample"""
        if self.current_average is None:
            return 0.0
        return self.current_average


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



