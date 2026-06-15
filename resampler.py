import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Any


class TimeSeriesResampler:
    VALID_AGG_METHODS = [
        'mean', 'sum', 'max', 'min', 'last', 'first',
        'std', 'var', 'median', 'count', 'nunique',
        'ohlc'
    ]

    def __init__(
        self,
        time_col: Optional[str] = None,
        datetime_format: Optional[str] = None,
        timezone: Optional[str] = None
    ):
        self.time_col = time_col
        self.datetime_format = datetime_format
        self.timezone = timezone
        self._freq_map = {
            'second': 's',
            'minute': 'min',
            'hour': 'h',
            'day': 'D',
            'week': 'W',
            'month': 'ME',
            'quarter': 'QE',
            'year': 'YE'
        }
        self._unit_map = {
            's': 's',
            'sec': 's',
            'second': 's',
            'min': 'min',
            'minute': 'min',
            't': 'min',
            'h': 'h',
            'hour': 'h',
            'hr': 'h',
            'd': 'D',
            'day': 'D',
            'w': 'W',
            'week': 'W',
            'm': 'ME',
            'month': 'ME',
            'q': 'QE',
            'quarter': 'QE',
            'y': 'YE',
            'year': 'YE'
        }

    def _validate_and_prepare(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        if self.time_col is None:
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(
                    "DataFrame must have a DatetimeIndex when time_col is not specified, "
                    "or specify time_col parameter"
                )
            df = df.copy()
        else:
            if self.time_col not in df.columns:
                raise ValueError(f"Time column '{self.time_col}' not found in DataFrame")
            df = df.copy()
            df[self.time_col] = pd.to_datetime(
                df[self.time_col],
                format=self.datetime_format,
                errors='coerce'
            )
            if df[self.time_col].isnull().any():
                null_count = df[self.time_col].isnull().sum()
                raise ValueError(
                    f"Failed to parse {null_count} datetime values in column '{self.time_col}'"
                )
            df = df.set_index(self.time_col)

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex after processing")

        if df.index.hasnans:
            raise ValueError("Datetime index contains NaT values")

        if self.timezone:
            if df.index.tz is None:
                df.index = df.index.tz_localize(self.timezone)

        return df

    def _parse_freq(self, freq: Union[str, int]) -> str:
        if isinstance(freq, int):
            if freq <= 0:
                raise ValueError("Frequency integer must be positive")
            return f"{freq}s"
        
        freq_str = str(freq)
        freq_lower = freq_str.lower()
        
        if freq_lower in self._freq_map:
            return self._freq_map[freq_lower]
        
        if freq_lower in self._unit_map:
            return self._unit_map[freq_lower]
        
        import re
        match = re.match(r'^(\d+)([a-zA-Z]+)$', freq_str)
        if match:
            num = match.group(1)
            unit = match.group(2).lower()
            if unit in self._unit_map:
                unit = self._unit_map[unit]
            elif unit in self._freq_map:
                unit = self._freq_map[unit]
            return f"{num}{unit}"
        
        return freq_lower

    def _get_agg_funcs(
        self,
        agg_method: Union[str, List[str], Dict[str, Any]]
    ) -> Any:
        if isinstance(agg_method, str):
            if agg_method not in self.VALID_AGG_METHODS:
                raise ValueError(
                    f"Invalid aggregation method '{agg_method}'. "
                    f"Valid methods are: {', '.join(self.VALID_AGG_METHODS)}"
                )
            return agg_method
        
        if isinstance(agg_method, list):
            for method in agg_method:
                if isinstance(method, str) and method not in self.VALID_AGG_METHODS:
                    raise ValueError(
                        f"Invalid aggregation method '{method}' in list. "
                        f"Valid methods are: {', '.join(self.VALID_AGG_METHODS)}"
                    )
            return agg_method
        
        if isinstance(agg_method, dict):
            for col, methods in agg_method.items():
                if isinstance(methods, str):
                    if methods not in self.VALID_AGG_METHODS:
                        raise ValueError(
                            f"Invalid aggregation method '{methods}' for column '{col}'. "
                            f"Valid methods are: {', '.join(self.VALID_AGG_METHODS)}"
                        )
                elif isinstance(methods, list):
                    for method in methods:
                        if isinstance(method, str) and method not in self.VALID_AGG_METHODS:
                            raise ValueError(
                                f"Invalid aggregation method '{method}' for column '{col}'. "
                                f"Valid methods are: {', '.join(self.VALID_AGG_METHODS)}"
                            )
            return agg_method
        
        raise TypeError(
            "agg_method must be str, list of str, or dict, "
            f"got {type(agg_method)}"
        )

    def resample(
        self,
        df: pd.DataFrame,
        freq: Union[str, int],
        agg_method: Union[str, List[str], Dict[str, Any]] = 'mean',
        fill_method: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        df_prepared = self._validate_and_prepare(df)
        parsed_freq = self._parse_freq(freq)
        agg_funcs = self._get_agg_funcs(agg_method)

        original_tz = df_prepared.index.tz

        if self.timezone is not None:
            output_tz = self.timezone
        else:
            output_tz = original_tz

        if original_tz is not None:
            df_for_agg = df_prepared.tz_convert('UTC')
        else:
            df_for_agg = df_prepared

        resampler = df_for_agg.resample(parsed_freq, **kwargs)
        result = resampler.agg(agg_funcs)

        if fill_method is not None:
            if fill_method == 'ffill':
                result = result.ffill()
            elif fill_method == 'bfill':
                result = result.bfill()
            elif fill_method == 'interpolate':
                result = result.interpolate()
            else:
                raise ValueError(
                    f"Invalid fill_method '{fill_method}'. "
                    "Valid values are: 'ffill', 'bfill', 'interpolate'"
                )

        if isinstance(result.columns, pd.MultiIndex):
            result.columns = ['_'.join(col).strip() for col in result.columns.values]

        if output_tz is not None:
            result = result.tz_convert(output_tz)

        return result

    def downsample_to_minute(
        self,
        df: pd.DataFrame,
        minutes: int = 1,
        agg_method: Union[str, List[str], Dict[str, Any]] = 'mean',
        fill_method: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        freq = f"{minutes}min"
        return self.resample(df, freq, agg_method, fill_method, **kwargs)

    def downsample_to_hour(
        self,
        df: pd.DataFrame,
        hours: int = 1,
        agg_method: Union[str, List[str], Dict[str, Any]] = 'mean',
        fill_method: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        freq = f"{hours}h"
        return self.resample(df, freq, agg_method, fill_method, **kwargs)

    def downsample_to_day(
        self,
        df: pd.DataFrame,
        days: int = 1,
        agg_method: Union[str, List[str], Dict[str, Any]] = 'mean',
        fill_method: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        freq = f"{days}D"
        return self.resample(df, freq, agg_method, fill_method, **kwargs)

    def batch_resample(
        self,
        df: pd.DataFrame,
        freq_list: List[Union[str, int]],
        agg_method: Union[str, List[str], Dict[str, Any]] = 'mean',
        fill_method: Optional[str] = None,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        results = {}
        for freq in freq_list:
            parsed_freq = self._parse_freq(freq)
            result = self.resample(df, parsed_freq, agg_method, fill_method, **kwargs)
            results[parsed_freq] = result
        return results


def generate_sample_data(
    start_time: str = '2024-01-01 00:00:00',
    periods: int = 3600,
    freq: str = 's',
    columns: List[str] = ['value1', 'value2', 'value3'],
    seed: Optional[int] = 42
) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)
    
    index = pd.date_range(start=start_time, periods=periods, freq=freq)
    data = {}
    for col in columns:
        base_val = np.random.uniform(10, 100)
        trend = np.linspace(0, np.random.uniform(-10, 10), periods)
        noise = np.random.normal(0, 2, periods)
        seasonal = 5 * np.sin(np.linspace(0, 4 * np.pi, periods))
        data[col] = base_val + trend + noise + seasonal
    
    df = pd.DataFrame(data, index=index)
    df.index.name = 'timestamp'
    return df


def resample_pipeline(
    df: pd.DataFrame,
    time_col: Optional[str] = None,
    target_freqs: Optional[List[str]] = None,
    agg_method: Union[str, List[str], Dict[str, Any]] = 'mean'
) -> Dict[str, pd.DataFrame]:
    if target_freqs is None:
        target_freqs = ['min', 'h', 'D']
    
    resampler = TimeSeriesResampler(time_col=time_col)
    results = {}
    
    for freq in target_freqs:
        result = resampler.resample(df, freq, agg_method)
        results[freq] = result
    
    return results
