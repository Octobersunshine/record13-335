import unittest
import pandas as pd
import numpy as np
from datetime import timedelta
from resampler import TimeSeriesResampler, generate_sample_data, resample_pipeline


class TestTimeSeriesResampler(unittest.TestCase):

    def setUp(self):
        self.df_1h_second = generate_sample_data(
            start_time='2024-01-01 00:00:00',
            periods=3600,
            freq='s',
            columns=['temperature', 'humidity', 'pressure'],
            seed=42
        )
        
        self.df_with_time_col = self.df_1h_second.reset_index()
        self.resampler = TimeSeriesResampler()
        self.resampler_with_col = TimeSeriesResampler(time_col='timestamp')

    def test_generate_sample_data(self):
        df = generate_sample_data(periods=100, freq='s', seed=42)
        self.assertEqual(len(df), 100)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(list(df.columns), ['value1', 'value2', 'value3'])
        self.assertEqual(df.index.freq, 's')

    def test_validate_and_prepare_with_datetime_index(self):
        df_prepared = self.resampler._validate_and_prepare(self.df_1h_second)
        self.assertIsInstance(df_prepared.index, pd.DatetimeIndex)
        self.assertEqual(len(df_prepared), 3600)

    def test_validate_and_prepare_with_time_col(self):
        df_prepared = self.resampler_with_col._validate_and_prepare(self.df_with_time_col)
        self.assertIsInstance(df_prepared.index, pd.DatetimeIndex)
        self.assertEqual(len(df_prepared), 3600)
        self.assertEqual(df_prepared.index.name, 'timestamp')

    def test_validate_empty_dataframe(self):
        df_empty = pd.DataFrame()
        with self.assertRaises(ValueError) as ctx:
            self.resampler._validate_and_prepare(df_empty)
        self.assertIn("empty", str(ctx.exception).lower())

    def test_validate_missing_time_col(self):
        df = pd.DataFrame({'value': [1, 2, 3]})
        resampler = TimeSeriesResampler(time_col='nonexistent')
        with self.assertRaises(ValueError) as ctx:
            resampler._validate_and_prepare(df)
        self.assertIn("not found", str(ctx.exception))

    def test_validate_no_datetime_index(self):
        df = pd.DataFrame({'value': [1, 2, 3]}, index=[0, 1, 2])
        with self.assertRaises(ValueError) as ctx:
            self.resampler._validate_and_prepare(df)
        self.assertIn("DatetimeIndex", str(ctx.exception))

    def test_parse_freq_string(self):
        self.assertEqual(self.resampler._parse_freq('minute'), 'min')
        self.assertEqual(self.resampler._parse_freq('hour'), 'h')
        self.assertEqual(self.resampler._parse_freq('day'), 'D')
        self.assertEqual(self.resampler._parse_freq('5min'), '5min')
        self.assertEqual(self.resampler._parse_freq('2H'), '2h')

    def test_parse_freq_integer(self):
        self.assertEqual(self.resampler._parse_freq(30), '30s')
        self.assertEqual(self.resampler._parse_freq(60), '60s')

    def test_parse_freq_invalid_integer(self):
        with self.assertRaises(ValueError):
            self.resampler._parse_freq(-1)

    def test_get_agg_funcs_string(self):
        self.assertEqual(self.resampler._get_agg_funcs('mean'), 'mean')
        self.assertEqual(self.resampler._get_agg_funcs('sum'), 'sum')

    def test_get_agg_funcs_invalid_string(self):
        with self.assertRaises(ValueError) as ctx:
            self.resampler._get_agg_funcs('invalid_method')
        self.assertIn("Invalid aggregation method", str(ctx.exception))

    def test_get_agg_funcs_list(self):
        methods = ['mean', 'sum', 'max']
        self.assertEqual(self.resampler._get_agg_funcs(methods), methods)

    def test_get_agg_funcs_dict(self):
        agg_dict = {
            'temperature': 'mean',
            'humidity': ['max', 'min'],
            'pressure': 'sum'
        }
        self.assertEqual(self.resampler._get_agg_funcs(agg_dict), agg_dict)

    def test_resample_second_to_minute(self):
        result = self.resampler.resample(self.df_1h_second, freq='min', agg_method='mean')
        self.assertEqual(len(result), 60)
        self.assertIsInstance(result.index, pd.DatetimeIndex)
        self.assertEqual(result.index.freq, 'min')
        self.assertEqual(list(result.columns), ['temperature', 'humidity', 'pressure'])

    def test_resample_second_to_hour(self):
        result = self.resampler.resample(self.df_1h_second, freq='h', agg_method='mean')
        self.assertEqual(len(result), 1)
        self.assertEqual(result.index.freq, 'h')

    def test_resample_second_to_day(self):
        result = self.resampler.resample(self.df_1h_second, freq='D', agg_method='mean')
        self.assertEqual(len(result), 1)
        self.assertEqual(result.index.freq, 'D')

    def test_resample_with_time_col(self):
        result = self.resampler_with_col.resample(
            self.df_with_time_col, freq='min', agg_method='mean'
        )
        self.assertEqual(len(result), 60)
        self.assertIsInstance(result.index, pd.DatetimeIndex)

    def test_resample_multiple_agg_methods(self):
        result = self.resampler.resample(
            self.df_1h_second, freq='5min', agg_method=['mean', 'std', 'max']
        )
        self.assertEqual(len(result), 12)
        expected_cols = [
            'temperature_mean', 'temperature_std', 'temperature_max',
            'humidity_mean', 'humidity_std', 'humidity_max',
            'pressure_mean', 'pressure_std', 'pressure_max'
        ]
        self.assertEqual(list(result.columns), expected_cols)

    def test_resample_dict_agg_methods(self):
        agg_dict = {
            'temperature': 'mean',
            'humidity': ['max', 'min'],
            'pressure': 'sum'
        }
        result = self.resampler.resample(
            self.df_1h_second, freq='10min', agg_method=agg_dict
        )
        self.assertEqual(len(result), 6)
        expected_cols = [
            'temperature_mean',
            'humidity_max', 'humidity_min',
            'pressure_sum'
        ]
        self.assertEqual(list(result.columns), expected_cols)

    def test_resample_ohlc(self):
        result = self.resampler.resample(
            self.df_1h_second, freq='15min', agg_method='ohlc'
        )
        self.assertEqual(len(result), 4)
        expected_cols = [
            'temperature_open', 'temperature_high', 'temperature_low', 'temperature_close',
            'humidity_open', 'humidity_high', 'humidity_low', 'humidity_close',
            'pressure_open', 'pressure_high', 'pressure_low', 'pressure_close'
        ]
        self.assertEqual(list(result.columns), expected_cols)

    def test_resample_with_fill_ffill(self):
        df_gap = generate_sample_data(periods=120, freq='s', seed=42)
        df_gap = df_gap.drop(df_gap.index[30:60])
        result = self.resampler.resample(
            df_gap, freq='10s', agg_method='mean', fill_method='ffill'
        )
        self.assertFalse(result.isnull().any().any())

    def test_resample_with_fill_bfill(self):
        df_gap = generate_sample_data(periods=120, freq='s', seed=42)
        df_gap = df_gap.drop(df_gap.index[30:60])
        result = self.resampler.resample(
            df_gap, freq='10s', agg_method='mean', fill_method='bfill'
        )
        self.assertFalse(result.isnull().any().any())

    def test_resample_with_fill_interpolate(self):
        df_gap = generate_sample_data(periods=120, freq='s', seed=42)
        df_gap = df_gap.drop(df_gap.index[30:60])
        result = self.resampler.resample(
            df_gap, freq='10s', agg_method='mean', fill_method='interpolate'
        )
        self.assertFalse(result.isnull().any().any())

    def test_resample_invalid_fill_method(self):
        with self.assertRaises(ValueError) as ctx:
            self.resampler.resample(
                self.df_1h_second, freq='min', agg_method='mean', fill_method='invalid'
            )
        self.assertIn("Invalid fill_method", str(ctx.exception))

    def test_downsample_to_minute(self):
        result = self.resampler.downsample_to_minute(
            self.df_1h_second, minutes=5, agg_method='mean'
        )
        self.assertEqual(len(result), 12)
        self.assertEqual(result.index.freq, '5min')

    def test_downsample_to_hour(self):
        df_24h = generate_sample_data(
            start_time='2024-01-01 00:00:00',
            periods=86400,
            freq='s',
            columns=['value'],
            seed=42
        )
        result = self.resampler.downsample_to_hour(
            df_24h, hours=2, agg_method='mean'
        )
        self.assertEqual(len(result), 12)
        self.assertEqual(result.index.freq, '2h')

    def test_downsample_to_day(self):
        df_7d = generate_sample_data(
            start_time='2024-01-01 00:00:00',
            periods=604800,
            freq='s',
            columns=['value'],
            seed=42
        )
        result = self.resampler.downsample_to_day(
            df_7d, days=1, agg_method='mean'
        )
        self.assertEqual(len(result), 7)
        self.assertEqual(result.index.freq, 'D')

    def test_batch_resample(self):
        freq_list = ['min', '5min', '15min', 'h']
        results = self.resampler.batch_resample(
            self.df_1h_second, freq_list=freq_list, agg_method='mean'
        )
        self.assertEqual(len(results), 4)
        self.assertIn('min', results)
        self.assertIn('5min', results)
        self.assertIn('15min', results)
        self.assertIn('h', results)
        self.assertEqual(len(results['min']), 60)
        self.assertEqual(len(results['5min']), 12)
        self.assertEqual(len(results['15min']), 4)
        self.assertEqual(len(results['h']), 1)

    def test_resample_pipeline(self):
        results = resample_pipeline(
            self.df_1h_second,
            target_freqs=['min', 'h', 'D'],
            agg_method='mean'
        )
        self.assertEqual(len(results), 3)
        self.assertIn('min', results)
        self.assertIn('h', results)
        self.assertIn('D', results)

    def test_resample_pipeline_with_time_col(self):
        results = resample_pipeline(
            self.df_with_time_col,
            time_col='timestamp',
            target_freqs=['min', 'h'],
            agg_method='mean'
        )
        self.assertEqual(len(results), 2)
        self.assertIn('min', results)
        self.assertIn('h', results)

    def test_resample_with_timezone(self):
        resampler_tz = TimeSeriesResampler(timezone='UTC')
        result = resampler_tz.resample(
            self.df_1h_second, freq='min', agg_method='mean'
        )
        self.assertIsNotNone(result.index.tz)
        tz_name = str(result.index.tz)
        self.assertIn('UTC', tz_name)

    def test_resample_aggregation_correctness(self):
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }, index=pd.date_range('2024-01-01', periods=10, freq='s'))
        
        result_sum = self.resampler.resample(df, freq='5s', agg_method='sum')
        self.assertEqual(result_sum['value'].iloc[0], 15)
        self.assertEqual(result_sum['value'].iloc[1], 40)

        result_mean = self.resampler.resample(df, freq='5s', agg_method='mean')
        self.assertEqual(result_mean['value'].iloc[0], 3.0)
        self.assertEqual(result_mean['value'].iloc[1], 8.0)

        result_max = self.resampler.resample(df, freq='5s', agg_method='max')
        self.assertEqual(result_max['value'].iloc[0], 5)
        self.assertEqual(result_max['value'].iloc[1], 10)

        result_min = self.resampler.resample(df, freq='5s', agg_method='min')
        self.assertEqual(result_min['value'].iloc[0], 1)
        self.assertEqual(result_min['value'].iloc[1], 6)

    def test_resample_count_aggregation(self):
        df_gap = generate_sample_data(periods=60, freq='s', seed=42)
        df_gap = df_gap.drop(df_gap.index[10:20])
        
        result = self.resampler.resample(df_gap, freq='30s', agg_method='count')
        self.assertEqual(result['value1'].iloc[0], 20)
        self.assertEqual(result['value1'].iloc[1], 30)


if __name__ == '__main__':
    unittest.main(verbosity=2)
