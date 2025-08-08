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
        self.weekly_means = None  # (day_of_week, hour) -> mean value
        self.hour_means = None  # Hour of day (0-23) -> mean value (fallback)
        self.global_mean = None  # Fallback value for any missing slots

    def fit(self, X, y):
        """
        Fit the model by calculating mean values for each time slot.

        Creates a 168-point seasonality table (24 hours x 7 days).
        """
        df = pd.DataFrame(
            {
                "hour": X[:, 0],  # X[:, 0] is the hour
                "day_of_week": X[:, 1],  # X[:, 1] is the day of week
                "y": y,
            }
        )

        # Calculate means for different seasonality patterns
        self.hour_means = df.groupby("hour")["y"].mean()
        self.weekly_means = df.groupby(["day_of_week", "hour"])["y"].mean()
        self.global_mean = df["y"].mean()

        return self

    def predict(self, X):
        """
        Predict based on weekly seasonality patterns.

        Returns the mean value corresponding to the specific hour of the
        specific day of week for each timestamp.
        """
        df = pd.DataFrame(
            {"hour": X[:, 0], "day_of_week": X[:, 1]}  # X[:, 0] is the hour
        )  # X[:, 1] is the day of week

        # Use full weekly patterns (168 values)
        results = []
        for _, row in df.iterrows():
            day = row["day_of_week"]
            hour = row["hour"]
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
            self.current_average = (
                self.alpha * y + (1 - self.alpha) * self.current_average
            )
        return self

    def predict_one(self, x):
        """River compatibility - predict for one sample"""
        if self.current_average is None:
            return 0.0
        return self.current_average


if __name__ == "__main__":
    print("Testing baseline regressors...")

    # Generate synthetic data with weekly patterns
    start_date = pd.Timestamp("2023-01-01")
    days = 21  # 3 weeks of data

    # Test with different granularities
    for freq_mins in [60, 15]:  # Test hourly and 15-minute data
        print(f"\n=== Testing with {freq_mins}-minute frequency data ===")

        # Calculate total periods
        periods_per_day = int(24 * 60 / freq_mins)
        total_periods = days * periods_per_day

        # Create timestamps for the given frequency
        timestamps = [
            start_date + pd.Timedelta(minutes=i * freq_mins)
            for i in range(total_periods)
        ]
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
        train_size = 14 * periods_per_day  # 2 weeks

        X_train = timestamps_array[:train_size]
        y_train = values[:train_size]

        X_test = timestamps_array[train_size:]
        y_test = values[train_size:]

        # Test SeasonalMeanRegressor
        seasonal_model = SeasonalMeanRegressor()
        seasonal_model.fit(X_train, y_train)
        seasonal_pred = seasonal_model.predict(X_test)

        seasonal_mae = np.mean(np.abs(y_test - seasonal_pred))
        seasonal_rmse = np.sqrt(np.mean((y_test - seasonal_pred) ** 2))

        print(f"\nSeasonalMeanRegressor Results:")
        print(f"Mean Absolute Error: {seasonal_mae:.4f}")
        print(f"Root Mean Squared Error: {seasonal_rmse:.4f}")

        # Test ExponentialMovingAverageRegressor with different alpha values (incremental learning)
        alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        ema_results = []

        print("\nExponentialMovingAverageRegressor Results (incremental learning):")
        for alpha in alpha_values:
            # Initialize model and predictions storage
            ema_model = ExponentialMovingAverageRegressor(alpha=alpha)
            ema_pred = []

            # Train incrementally on training data
            for i in range(len(X_train)):
                # First predict, then learn
                ema_model.learn_one(X_train[i], y_train[i])

            # Test incrementally
            for i in range(len(X_test)):
                # Predict and learn
                pred = ema_model.predict_one(X_test[i])
                ema_pred.append(pred)
                ema_model.learn_one(X_test[i], y_test[i])

            ema_pred = np.array(ema_pred)

            # Calculate metrics
            ema_mae = np.mean(np.abs(y_test - ema_pred))
            ema_rmse = np.sqrt(np.mean((y_test - ema_pred) ** 2))

            print(f"Alpha={alpha}: MAE={ema_mae:.4f}, RMSE={ema_rmse:.4f}")
            ema_results.append((alpha, ema_model, ema_pred, ema_mae, ema_rmse))

        # Find best alpha
        best_alpha_result = min(ema_results, key=lambda x: x[3])  # Sort by MAE
        print(
            f"\nBest alpha: {best_alpha_result[0]} with MAE: {best_alpha_result[3]:.4f}"
        )

        # Plot results (if matplotlib is available)
        try:
            import matplotlib.pyplot as plt

            # Convert to datetime for better plotting
            test_dates = [pd.to_datetime(ts[0]) for ts in X_test]

            # Plot model comparison
            plt.figure(figsize=(14, 7))
            plt.plot(test_dates, y_test, label="Actual Values", marker="o", alpha=0.6)
            plt.plot(
                test_dates,
                seasonal_pred,
                label=f"Weekly Pattern (MAE={seasonal_mae:.2f})",
                marker="d",
                linestyle=":",
            )

            # Plot best EMA predictions
            best_alpha, _, best_ema_pred, best_mae, _ = best_alpha_result
            plt.plot(
                test_dates,
                best_ema_pred,
                label=f"EMA α={best_alpha} (MAE={best_mae:.2f})",
                marker="+",
                linestyle="-.",
            )

            plt.title(
                f"Model Comparison ({freq_mins}-minute data): Actual vs Predicted Values"
            )
            plt.xlabel("Date")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Add day separators
            for i in range(7):
                day_start = train_size + i * periods_per_day
                if day_start < len(timestamps):
                    plt.axvline(
                        x=timestamps[day_start], color="gray", alpha=0.5, linestyle=":"
                    )

            plt.tight_layout()
            plt.show()

            # Also create a heatmap visualization of the weekly seasonality pattern
            if not freq_mins > 60:  # Only show for hourly or finer granularity
                try:
                    # Create a DataFrame from the weekly seasonality table
                    weekly_pattern = pd.DataFrame(
                        index=range(7), columns=range(24)  # days  # hours
                    )

                    # Fill with values
                    for (day, hour), value in seasonal_model.weekly_means.items():
                        weekly_pattern.loc[day, hour] = value

                    # Replace day numbers with names
                    day_names = [
                        "Monday",
                        "Tuesday",
                        "Wednesday",
                        "Thursday",
                        "Friday",
                        "Saturday",
                        "Sunday",
                    ]
                    weekly_pattern.index = day_names

                    plt.figure(figsize=(15, 8))
                    plt.title("Weekly Seasonality Pattern Heatmap")

                    # Use seaborn if available for better heatmaps
                    try:
                        import seaborn as sns

                        ax = sns.heatmap(
                            weekly_pattern,
                            cmap="YlGnBu",
                            annot=True,
                            fmt=".1f",
                            linewidths=0.5,
                        )
                        ax.set_xlabel("Hour of Day")
                        ax.set_ylabel("Day of Week")
                    except ImportError:
                        plt.imshow(weekly_pattern, cmap="viridis", aspect="auto")
                        plt.colorbar(label="Mean Value")
                        plt.xlabel("Hour of Day")
                        plt.ylabel("Day of Week")
                        plt.xticks(range(24))
                        plt.yticks(range(7), day_names)

                    plt.tight_layout()
                    plt.show()

                except Exception as e:
                    print(f"Could not create heatmap: {e}")

        except ImportError:
            print("Matplotlib not available for visualization")
            day_names = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            weekly_pattern.index = day_names

            plt.figure(figsize=(15, 8))
            plt.title("Weekly Seasonality Pattern Heatmap")

            # Use seaborn if available for better heatmaps
            try:
                import seaborn as sns

                ax = sns.heatmap(
                    weekly_pattern, cmap="YlGnBu", annot=True, fmt=".1f", linewidths=0.5
                )
                ax.set_xlabel("Hour of Day")
                ax.set_ylabel("Day of Week")
            except ImportError:
                plt.imshow(weekly_pattern, cmap="viridis", aspect="auto")
                plt.colorbar(label="Mean Value")
                plt.xlabel("Hour of Day")
                plt.ylabel("Day of Week")
                plt.xticks(range(24))
                plt.yticks(range(7), day_names)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Could not create heatmap: {e}")

        except ImportError:
            print("Matplotlib not available for visualization")
