import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Any


class TimeSeriesResampler:
    VALID_AGG_METHODS = [
        'mean', 'sum', 'max', 'min', 'last', 'first',
        'std', 'var', 'median', 'count', 'nunique',
        'ohlc'
    ]

    AGG_METHOD_ALIASES = {
        'mean': 'mean',
        'avg': 'mean',
        'average': 'mean',
        '均值': 'mean',
        '平均值': 'mean',
        'sum': 'sum',
        'total': 'sum',
        '总和': 'sum',
        '累计': 'sum',
        'max': 'max',
        'maximum': 'max',
        '最大值': 'max',
        '最高': 'max',
        'min': 'min',
        'minimum': 'min',
        '最小值': 'min',
        '最低': 'min',
        'last': 'last',
        '最后': 'last',
        '最终': 'last',
        'first': 'first',
        '首先': 'first',
        '最初': 'first',
        'std': 'std',
        '标准差': 'std',
        'var': 'var',
        '方差': 'var',
        'median': 'median',
        '中位数': 'median',
        'count': 'count',
        '计数': 'count',
        '数量': 'count',
        'nunique': 'nunique',
        '去重计数': 'nunique',
        '唯一值': 'nunique',
        'ohlc': 'ohlc',
        '开高低收': 'ohlc'
    }

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

    def _resolve_agg_alias(self, method: str) -> str:
        method_lower = method.lower()
        if method_lower in self.AGG_METHOD_ALIASES:
            return self.AGG_METHOD_ALIASES[method_lower]
        if method in self.AGG_METHOD_ALIASES:
            return self.AGG_METHOD_ALIASES[method]
        return method

    def _get_agg_funcs(
        self,
        agg_method: Union[str, List[str], Dict[str, Any]]
    ) -> tuple:
        column_mapping = {}

        if isinstance(agg_method, str):
            resolved = self._resolve_agg_alias(agg_method)
            if resolved not in self.VALID_AGG_METHODS:
                raise ValueError(
                    f"Invalid aggregation method '{agg_method}'. "
                    f"Valid methods are: {', '.join(self.VALID_AGG_METHODS)}\n"
                    f"Aliases: {', '.join(sorted(self.AGG_METHOD_ALIASES.keys()))}"
                )
            return resolved, column_mapping
        
        if isinstance(agg_method, list):
            resolved_list = []
            for method in agg_method:
                if isinstance(method, str):
                    resolved = self._resolve_agg_alias(method)
                    if resolved not in self.VALID_AGG_METHODS:
                        raise ValueError(
                            f"Invalid aggregation method '{method}' in list. "
                            f"Valid methods are: {', '.join(self.VALID_AGG_METHODS)}\n"
                            f"Aliases: {', '.join(sorted(self.AGG_METHOD_ALIASES.keys()))}"
                        )
                    resolved_list.append(resolved)
                    if method != resolved:
                        column_mapping[resolved] = method
                else:
                    resolved_list.append(method)
            return resolved_list, column_mapping
        
        if isinstance(agg_method, dict):
            resolved_dict = {}
            for col, methods in agg_method.items():
                if isinstance(methods, str):
                    resolved = self._resolve_agg_alias(methods)
                    if resolved not in self.VALID_AGG_METHODS:
                        raise ValueError(
                            f"Invalid aggregation method '{methods}' for column '{col}'. "
                            f"Valid methods are: {', '.join(self.VALID_AGG_METHODS)}\n"
                            f"Aliases: {', '.join(sorted(self.AGG_METHOD_ALIASES.keys()))}"
                        )
                    resolved_dict[col] = resolved
                    if methods != resolved:
                        column_mapping[f"{col}_{resolved}"] = f"{col}_{methods}"
                elif isinstance(methods, list):
                    resolved_col_list = []
                    for method in methods:
                        if isinstance(method, str):
                            resolved = self._resolve_agg_alias(method)
                            if resolved not in self.VALID_AGG_METHODS:
                                raise ValueError(
                                    f"Invalid aggregation method '{method}' for column '{col}'. "
                                    f"Valid methods are: {', '.join(self.VALID_AGG_METHODS)}\n"
                                    f"Aliases: {', '.join(sorted(self.AGG_METHOD_ALIASES.keys()))}"
                                )
                            resolved_col_list.append(resolved)
                            if method != resolved:
                                column_mapping[f"{col}_{resolved}"] = f"{col}_{method}"
                        else:
                            resolved_col_list.append(method)
                    resolved_dict[col] = resolved_col_list
                else:
                    resolved_dict[col] = methods
            return resolved_dict, column_mapping
        
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
        agg_funcs, column_mapping = self._get_agg_funcs(agg_method)

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

        if column_mapping:
            new_columns = []
            for col in result.columns:
                if col in column_mapping:
                    new_columns.append(column_mapping[col])
                else:
                    found = False
                    for old_suffix, new_suffix in column_mapping.items():
                        if col.endswith(f'_{old_suffix}'):
                            base = col[:-len(f'_{old_suffix}')]
                            new_columns.append(f'{base}_{new_suffix}')
                            found = True
                            break
                    if not found:
                        new_columns.append(col)
            result.columns = new_columns

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

    def resample_mean(
        self,
        df: pd.DataFrame,
        freq: Union[str, int],
        fill_method: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        return self.resample(df, freq, agg_method='mean', fill_method=fill_method, **kwargs)

    def resample_sum(
        self,
        df: pd.DataFrame,
        freq: Union[str, int],
        fill_method: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        return self.resample(df, freq, agg_method='sum', fill_method=fill_method, **kwargs)

    def resample_max(
        self,
        df: pd.DataFrame,
        freq: Union[str, int],
        fill_method: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        return self.resample(df, freq, agg_method='max', fill_method=fill_method, **kwargs)

    def resample_min(
        self,
        df: pd.DataFrame,
        freq: Union[str, int],
        fill_method: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        return self.resample(df, freq, agg_method='min', fill_method=fill_method, **kwargs)

    def resample_std(
        self,
        df: pd.DataFrame,
        freq: Union[str, int],
        fill_method: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        return self.resample(df, freq, agg_method='std', fill_method=fill_method, **kwargs)

    def resample_median(
        self,
        df: pd.DataFrame,
        freq: Union[str, int],
        fill_method: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        return self.resample(df, freq, agg_method='median', fill_method=fill_method, **kwargs)

    def resample_count(
        self,
        df: pd.DataFrame,
        freq: Union[str, int],
        fill_method: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        return self.resample(df, freq, agg_method='count', fill_method=fill_method, **kwargs)

    def resample_ohlc(
        self,
        df: pd.DataFrame,
        freq: Union[str, int],
        fill_method: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        return self.resample(df, freq, agg_method='ohlc', fill_method=fill_method, **kwargs)

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
