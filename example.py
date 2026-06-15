import pandas as pd
from resampler import TimeSeriesResampler, generate_sample_data, resample_pipeline


def example_basic_usage():
    print("=" * 60)
    print("示例 1: 基本用法 - 秒级数据下采样到分钟级")
    print("=" * 60)

    df_seconds = generate_sample_data(
        start_time='2024-01-01 00:00:00',
        periods=3600,
        freq='s',
        columns=['temperature', 'humidity', 'pressure'],
        seed=42
    )

    print(f"\n原始秒级数据 (前5行):")
    print(df_seconds.head())
    print(f"\n数据量: {len(df_seconds)} 行, 时间范围: {df_seconds.index[0]} ~ {df_seconds.index[-1]}")

    resampler = TimeSeriesResampler()
    df_minute = resampler.resample(df_seconds, freq='min', agg_method='mean')

    print(f"\n下采样到分钟级数据 (前5行):")
    print(df_minute.head())
    print(f"\n数据量: {len(df_minute)} 行, 时间范围: {df_minute.index[0]} ~ {df_minute.index[-1]}")


def example_multiple_aggregations():
    print("\n" + "=" * 60)
    print("示例 2: 多种聚合方法同时使用")
    print("=" * 60)

    df_seconds = generate_sample_data(
        start_time='2024-01-01 00:00:00',
        periods=1800,
        freq='s',
        columns=['sales', 'visitors'],
        seed=123
    )

    print(f"\n原始秒级数据 (前5行):")
    print(df_seconds.head())

    resampler = TimeSeriesResampler()
    df_5min = resampler.resample(
        df_seconds,
        freq='5min',
        agg_method=['mean', 'sum', 'max', 'min']
    )

    print(f"\n5分钟聚合 (mean, sum, max, min):")
    print(df_5min)


def example_column_specific_aggregation():
    print("\n" + "=" * 60)
    print("示例 3: 按列指定不同的聚合方法")
    print("=" * 60)

    df_seconds = generate_sample_data(
        start_time='2024-01-01 08:00:00',
        periods=7200,
        freq='s',
        columns=['price', 'volume', 'count'],
        seed=456
    )

    print(f"\n原始秒级数据 (前5行):")
    print(df_seconds.head())

    agg_dict = {
        'price': ['mean', 'max', 'min'],
        'volume': 'sum',
        'count': 'last'
    }

    resampler = TimeSeriesResampler()
    df_hourly = resampler.resample(df_seconds, freq='h', agg_method=agg_dict)

    print(f"\n小时级聚合 (price: mean/max/min, volume: sum, count: last):")
    print(df_hourly)


def example_ohlc_aggregation():
    print("\n" + "=" * 60)
    print("示例 4: OHLC (开高低收) 聚合 - 适用于金融数据")
    print("=" * 60)

    df_seconds = generate_sample_data(
        start_time='2024-01-01 09:30:00',
        periods=3600,
        freq='s',
        columns=['stock_price'],
        seed=789
    )

    print(f"\n原始秒级股价数据 (前10行):")
    print(df_seconds.head(10))

    resampler = TimeSeriesResampler()
    df_15min_ohlc = resampler.resample(
        df_seconds, freq='15min', agg_method='ohlc'
    )

    print(f"\n15分钟 K 线数据 (OHLC):")
    print(df_15min_ohlc)


def example_with_time_column():
    print("\n" + "=" * 60)
    print("示例 5: 数据包含独立的时间列")
    print("=" * 60)

    df = generate_sample_data(
        start_time='2024-01-01 00:00:00',
        periods=1800,
        freq='s',
        columns=['value1', 'value2'],
        seed=321
    ).reset_index()

    print(f"\n原始数据 (前5行, 包含独立时间列):")
    print(df.head())
    print(f"\n数据列名: {list(df.columns)}")

    resampler = TimeSeriesResampler(time_col='timestamp')
    df_hourly = resampler.downsample_to_hour(df, hours=1, agg_method='mean')

    print(f"\n小时级聚合结果:")
    print(df_hourly)


def example_missing_data_fill():
    print("\n" + "=" * 60)
    print("示例 6: 缺失数据填充处理")
    print("=" * 60)

    df = generate_sample_data(
        start_time='2024-01-01 00:00:00',
        periods=120,
        freq='s',
        columns=['metric'],
        seed=654
    )

    df_with_gap = df.drop(df.index[30:60])
    print(f"\n原始数据 (含缺失段, 30-59秒被删除):")
    print(df_with_gap)

    resampler = TimeSeriesResampler()

    result_no_fill = resampler.resample(
        df_with_gap, freq='10S', agg_method='mean'
    )
    print(f"\n10秒聚合 (不填充缺失值):")
    print(result_no_fill)

    result_ffill = resampler.resample(
        df_with_gap, freq='10S', agg_method='mean', fill_method='ffill'
    )
    print(f"\n10秒聚合 (前向填充 ffill):")
    print(result_ffill)

    result_interpolate = resampler.resample(
        df_with_gap, freq='10S', agg_method='mean', fill_method='interpolate'
    )
    print(f"\n10秒聚合 (线性插值 interpolate):")
    print(result_interpolate)


def example_convenience_methods():
    print("\n" + "=" * 60)
    print("示例 7: 使用便捷方法 (downsample_to_minute/hour/day)")
    print("=" * 60)

    df = generate_sample_data(
        start_time='2024-01-01 00:00:00',
        periods=86400,
        freq='s',
        columns=['energy_usage'],
        seed=111
    )

    print(f"原始秒级数据量: {len(df)} 行")

    resampler = TimeSeriesResampler()

    df_5min = resampler.downsample_to_minute(df, minutes=5, agg_method='mean')
    print(f"\n5分钟聚合: {len(df_5min)} 行")
    print(df_5min.head())

    df_hourly = resampler.downsample_to_hour(df, hours=1, agg_method='mean')
    print(f"\n1小时聚合: {len(df_hourly)} 行")
    print(df_hourly.head())

    df_daily = resampler.downsample_to_day(df, days=1, agg_method='mean')
    print(f"\n1天聚合: {len(df_daily)} 行")
    print(df_daily)


def example_batch_resample():
    print("\n" + "=" * 60)
    print("示例 8: 批量多频率下采样")
    print("=" * 60)

    df = generate_sample_data(
        start_time='2024-01-01 00:00:00',
        periods=3600,
        freq='s',
        columns=['temperature', 'humidity'],
        seed=222
    )

    print(f"原始数据量: {len(df)} 行")

    resampler = TimeSeriesResampler()
    freq_list = ['1min', '5min', '15min', '30min', 'H']
    
    results = resampler.batch_resample(
        df, freq_list=freq_list, agg_method='mean'
    )

    print("\n批量下采样结果:")
    for freq, result in results.items():
        print(f"  {freq}: {len(result)} 行, 时间范围: {result.index[0]} ~ {result.index[-1]}")

    print("\n15分钟聚合数据:")
    print(results['15min'])


def example_resample_pipeline():
    print("\n" + "=" * 60)
    print("示例 9: 使用 Pipeline 一键多频率聚合")
    print("=" * 60)

    df = generate_sample_data(
        start_time='2024-01-01 00:00:00',
        periods=172800,
        freq='s',
        columns=['cpu_usage', 'memory_usage', 'network_io'],
        seed=333
    )

    print(f"原始数据量: {len(df)} 行 (2天秒级数据)")

    results = resample_pipeline(
        df,
        target_freqs=['5min', '15min', 'H', 'D'],
        agg_method=['mean', 'max', 'min']
    )

    print("\nPipeline 聚合结果:")
    for freq, result in results.items():
        print(f"  {freq}: {len(result)} 行, 列数: {len(result.columns)}")

    print("\n日级聚合数据 (含 mean, max, min):")
    print(results['D'])


def example_with_timezone():
    print("\n" + "=" * 60)
    print("示例 10: 带时区处理")
    print("=" * 60)

    df = generate_sample_data(
        start_time='2024-01-01 00:00:00',
        periods=3600,
        freq='s',
        columns=['value'],
        seed=444
    )

    resampler_utc = TimeSeriesResampler(timezone='UTC')
    result_utc = resampler_utc.resample(df, freq='h', agg_method='mean')
    print(f"\nUTC 时区结果: {result_utc.index.tz}")
    print(result_utc)


def example_convenience_agg_methods():
    print("\n" + "=" * 60)
    print("示例 11: 使用便捷聚合方法 (mean/sum/max/min)")
    print("=" * 60)

    df = generate_sample_data(
        start_time='2024-01-01 00:00:00',
        periods=600,
        freq='s',
        columns=['sales', 'visitors'],
        seed=555
    )

    resampler = TimeSeriesResampler()

    print(f"\n原始秒级数据量: {len(df)} 行")

    df_mean = resampler.resample_mean(df, freq='1min')
    print("\n1分钟均值聚合 (resample_mean):")
    print(df_mean.head())

    df_sum = resampler.resample_sum(df, freq='1min')
    print("\n1分钟总和聚合 (resample_sum):")
    print(df_sum.head())

    df_max = resampler.resample_max(df, freq='1min')
    print("\n1分钟最大值聚合 (resample_max):")
    print(df_max.head())

    df_min = resampler.resample_min(df, freq='1min')
    print("\n1分钟最小值聚合 (resample_min):")
    print(df_min.head())

    df_std = resampler.resample_std(df, freq='1min')
    print("\n1分钟标准差聚合 (resample_std):")
    print(df_std.head())

    df_median = resampler.resample_median(df, freq='1min')
    print("\n1分钟中位数聚合 (resample_median):")
    print(df_median.head())


def example_chinese_aliases():
    print("\n" + "=" * 60)
    print("示例 12: 使用中文聚合方法别名")
    print("=" * 60)

    df = pd.DataFrame({
        'temperature': [20, 21, 23, 25, 24, 22, 20, 19, 18, 20],
        'humidity': [50, 52, 55, 58, 57, 54, 51, 49, 48, 50]
    }, index=pd.date_range('2024-01-01 00:00:00', periods=10, freq='s'))

    resampler = TimeSeriesResampler()

    print("\n原始秒级数据:")
    print(df)

    result_mean = resampler.resample(df, freq='5s', agg_method='均值')
    print("\n5秒均值聚合 (agg_method='均值'):")
    print(result_mean)

    result_sum = resampler.resample(df, freq='5s', agg_method='总和')
    print("\n5秒总和聚合 (agg_method='总和'):")
    print(result_sum)

    result_max = resampler.resample(df, freq='5s', agg_method='最大值')
    print("\n5秒最大值聚合 (agg_method='最大值'):")
    print(result_max)

    result_min = resampler.resample(df, freq='5s', agg_method='最小值')
    print("\n5秒最小值聚合 (agg_method='最小值'):")
    print(result_min)

    result_multi = resampler.resample(df, freq='5s', agg_method=['均值', '最大值', '最小值'])
    print("\n5秒多指标聚合 (agg_method=['均值', '最大值', '最小值']):")
    print(result_multi)


def example_agg_method_comparison():
    print("\n" + "=" * 60)
    print("示例 13: 不同聚合方法对比")
    print("=" * 60)

    df = generate_sample_data(
        start_time='2024-01-01 00:00:00',
        periods=300,
        freq='s',
        columns=['value'],
        seed=666
    )

    resampler = TimeSeriesResampler()

    agg_methods = ['mean', 'sum', 'max', 'min', 'std', 'median']
    chinese_names = ['均值', '总和', '最大值', '最小值', '标准差', '中位数']

    print(f"\n原始秒级数据统计:")
    print(f"  均值: {df['value'].mean():.2f}")
    print(f"  总和: {df['value'].sum():.2f}")
    print(f"  最大值: {df['value'].max():.2f}")
    print(f"  最小值: {df['value'].min():.2f}")

    print("\n1分钟聚合结果对比:")
    for method, name in zip(agg_methods, chinese_names):
        result = resampler.resample(df, freq='1min', agg_method=method)
        print(f"  {name} ({method}): {result['value'].iloc[0]:.2f}")

    agg_dict = {
        'value': ['mean', 'sum', 'max', 'min', 'std', 'median']
    }
    result_all = resampler.resample(df, freq='1min', agg_method=agg_dict)
    print("\n1分钟所有指标聚合结果:")
    print(result_all)


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)

    example_basic_usage()
    example_multiple_aggregations()
    example_column_specific_aggregation()
    example_ohlc_aggregation()
    example_with_time_column()
    example_missing_data_fill()
    example_convenience_methods()
    example_batch_resample()
    example_resample_pipeline()
    example_with_timezone()
    example_convenience_agg_methods()
    example_chinese_aliases()
    example_agg_method_comparison()

    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
