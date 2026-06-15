import pandas as pd
import numpy as np
from resampler import TimeSeriesResampler


def demo_timezone_bug():
    print("=" * 70)
    print("演示：时区转换导致的聚合错误问题")
    print("=" * 70)

    times_utc = pd.date_range(
        start='2024-03-10 07:00:00',
        end='2024-03-10 12:00:00',
        freq='h',
        tz='UTC'
    )

    data = {
        'value': [10, 20, 30, 40, 50, 60]
    }

    df_utc = pd.DataFrame(data, index=times_utc)
    print("\n原始 UTC 数据:")
    print(df_utc)

    print("\n" + "-" * 70)
    print("问题演示：美国东部时区 (EST/EDT) 夏令时切换")
    print("2024-03-10 02:00 EST 时钟向前拨 1 小时到 03:00 EDT")
    print("-" * 70)

    df_nyc = df_utc.tz_convert('America/New_York')
    print("\n转换为 America/New_York 时区后:")
    print(df_nyc)

    print("\n方法 1: 直接在 America/New_York 时区按天聚合 (当前实现):")
    resampler_nyc = TimeSeriesResampler(timezone='America/New_York')
    result_nyc = resampler_nyc.resample(df_utc, freq='D', agg_method='sum')
    print(result_nyc)
    print(f"总和: {result_nyc['value'].sum()}")

    print("\n方法 2: 先转 UTC 聚合，再转回目标时区 (正确做法):")
    resampler_utc = TimeSeriesResampler(timezone='UTC')
    result_utc = resampler_utc.resample(df_utc, freq='D', agg_method='sum')
    result_utc_to_nyc = result_utc.tz_convert('America/New_York')
    print(result_utc_to_nyc)
    print(f"总和: {result_utc_to_nyc['value'].sum()}")

    print("\n" + "-" * 70)
    print("问题演示：跨天边界的时区差异")
    print("-" * 70)

    times_utc2 = pd.date_range(
        start='2024-01-01 22:00:00',
        end='2024-01-02 10:00:00',
        freq='h',
        tz='UTC'
    )
    data2 = {'value': [1] * 13}
    df_utc2 = pd.DataFrame(data2, index=times_utc2)

    print("\n原始 UTC 数据 (2024-01-01 22:00 到 2024-01-02 10:00, 每小时 1):")
    print(df_utc2)

    print("\n在 Asia/Shanghai (UTC+8) 时区下按天聚合:")
    resampler_sh = TimeSeriesResampler(timezone='Asia/Shanghai')
    result_sh = resampler_sh.resample(df_utc2, freq='D', agg_method='sum')
    print(result_sh)

    df_sh = df_utc2.tz_convert('Asia/Shanghai')
    print("\n转换为 Asia/Shanghai 时间查看:")
    print(df_sh)

    print("\n" + "=" * 70)
    print("问题分析：")
    print("1. 不同时区的日期边界不同，直接聚合可能导致数据归属错误")
    print("2. 夏令时切换期间，可能出现重复或缺失的时间戳")
    print("3. 解决方案：统一转为 UTC 聚合，再转换回目标时区")
    print("=" * 70)


if __name__ == '__main__':
    demo_timezone_bug()
