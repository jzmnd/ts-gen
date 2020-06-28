import numpy as np
import pandas as pd
from scipy.stats import norm


class RandomDailyTSGenerator():

    """A random time series generator. This class creates daily time series data with the
    possibility to add: a linear or exponential trend, weekly seasonality, yearly seasonality,
    Gaussian noise, and yearly peak events. The seasonal components are sine functions. Uses
    include testing time series models where we wish to know the underlying data generation process.
    """

    days_in_week = 7
    days_in_year = 365
    date_col = 'date'
    y_col = 'y'

    def __init__(
        self,
        start,
        end,
        peak_event_length=3,
        noise_scaling=1.0,
        trend_scaling=(1.0, 2.0),
        peak_scaling=(2.0, 5.0),
        peak_list=None,
        weekly_seasonal_scaling=(1.0, 2.0),
        yearly_seasonal_scaling=(0.2, 2.0),
        exp_trend=True
    ):
        """Initialize class with simulation parameters.

        Args:
            start (str): Start date of data
            end (str): End date of data
            peak_event_length (int, optional): Length of peak events in days
            noise_scaling (float, optional): Noise scaling factors
            trend_scaling (tuple, optional): Trend scaling factors
            peak_scaling (tuple, optional): Peak event scaling factors
            peak_list (list[str], optional): List of peak event days
            weekly_seasonal_scaling (tuple, optional): Weekly seasonality scaling factor
            yearly_seasonal_scaling (tuple, optional): Yearly seasonality scaling factor
            exp_trend (bool, optional): Whether to use an exponential trend
        """
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)

        self.peak_event_length = peak_event_length
        self.noise_scaling = noise_scaling
        self.trend_scaling = trend_scaling
        self.peak_scaling = peak_scaling
        self.peak_list = peak_list
        self.weekly_seasonal_scaling = weekly_seasonal_scaling
        self.yearly_seasonal_scaling = yearly_seasonal_scaling
        self.exp_trend = exp_trend

        self.x_ts = pd.date_range(start, end, freq='D')
        self.x = np.arange(len(self.x_ts))
        self.x_size = len(self.x)

    @staticmethod
    def scaler(fixed_scaling, rand_scaling):
        """A combined scaler that includes a fixed and random scaling component.

        Args:
            fixed_scaling (float): Fixed scaling factor
            rand_scaling (float): Random scaling factor

        Returns:
            float
        """
        return fixed_scaling * np.random.uniform(low=1, high=rand_scaling)

    def linear_trend(self):
        """Simple linear trend scaled from 0 to 1.

        Returns:
            np.array
        """
        return self.x / max(self.x)

    def gaussian_noise(self, scale):
        """Generated Gaussian noise.

        Args:
            scale (float): Standard deviation of noise

        Returns:
            np.array
        """
        return np.random.normal(loc=0, scale=scale, size=self.x_size)

    def seasonal(self, period):
        """Seasonal component generator using a sine function.

        Args:
            period (int): Seasonal period in days

        Returns:
            np.array
        """
        return 0.5 * (1 + np.sin(2 * np.pi * self.x / period))

    @staticmethod
    def gaussian(x, loc, scale):
        """Gaussian peak function scaled from 0 to 1.

        Args:
            x (np.array): x array
            loc (int): Center of Gaussian
            scale (float): Width of Gaussian

        Returns:
            np.array
        """
        return scale * np.sqrt(2 * np.pi) * norm.pdf(x, loc=loc, scale=scale)

    def peak_events(self, start_event, period, event_length):
        """Generate peak events with fixed period.

        Args:
            start_event (str): First peak event date
            period (int): Peak event period in days
            event_length (int): Length of peak event in days

        Returns:
            np.array
        """
        i = 0
        y = np.zeros(self.x_size)
        while i * period < self.x_size + event_length:
            event_index = (pd.to_datetime(start_event) - self.start).days + i * period
            y += self.gaussian(self.x, event_index, event_length)
            i += 1
        return y

    def create_df(self, y, random_start, n_random_range):
        """Create a time series dataframe.

        Args:
            y (np.array): Time series data
            random_start (bool): Whether to randomize start dates
            n_random_range (int): Number of initial days in which the random start date can occur

        Returns:
            pd.DataFrame
        """
        df = pd.DataFrame({self.date_col: self.x_ts, self.y_col: y}).set_index(self.date_col)
        if random_start:
            start = np.random.choice(self.x_ts[:n_random_range])
            return df.loc[start:]
        return df

    def generate_sample(self, random_start=True, n_random_range=182):
        """Yields sample data.

        Args:
            random_start (bool, optional): Whether to use a random start date for the samples
            n_random_range (int, optional):
                Number of initial days in which the random start date can occur

        Yields:
            pd.DataFrame
        """
        trend = self.scaler(*self.trend_scaling) * self.linear_trend()
        if self.exp_trend:
            trend = np.exp(trend)

        seasonal = (
            self.scaler(*self.weekly_seasonal_scaling) * self.seasonal(self.days_in_week)
            + self.scaler(*self.yearly_seasonal_scaling) * self.seasonal(self.days_in_year)
        )

        noise = self.gaussian_noise(self.noise_scaling)

        peaks = np.zeros(self.x_size)
        if self.peak_list:
            for p in self.peak_list:
                peaks += (
                    self.scaler(*self.peak_scaling) *
                    self.peak_events(p, self.days_in_year, self.peak_event_length)
                )

        y = trend + seasonal + noise + peaks

        yield self.create_df(y, random_start=random_start, n_random_range=n_random_range)
