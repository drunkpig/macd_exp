import numpy as np
import pandas as pd
from futu import *
from pandas import DataFrame
from itertools import groupby
from operator import itemgetter


class KL_Period(object):
    KL_60 = "KL_60"
    KL_30 = "KL_30"
    KL_15 = "KL_15"


K_LINE_TYPE = {
    KL_Period.KL_60: KLType.K_60M,
    KL_Period.KL_30: KLType.K_30M,
    KL_Period.KL_15: KLType.K_15M,
}


class WaveType(object):
    RED_TOP = 2  # 红柱高峰
    RED_BOTTOM = 1  # 红柱峰底

    GREEN_TOP = -2  # 绿柱波峰
    GREEN_BOTTOM = -1  # 绿柱波底，乳沟深V的尖


def ema(data, n=12, val_name="close"):
    import numpy as np
    '''
        指数平均数指标 Exponential Moving Average
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的数据
          n:int
                      移动平均线时长，时间单位根据data决定
          val_name:string
                      计算哪一列的列名，默认为 close 
        return
        -------
          EMA:numpy.ndarray<numpy.float64>
              指数平均数指标
    '''

    prices = []

    EMA = []

    for index, row in data.iterrows():
        if index == 0:
            past_ema = row[val_name]
            EMA.append(row[val_name])
        else:
            # Y=[2*X+(N-1)*Y’]/(N+1)
            today_ema = (2 * row[val_name] + (n - 1) * past_ema) / (n + 1)
            past_ema = today_ema

            EMA.append(today_ema)

    return np.asarray(EMA)


def macd(data, quick_n=12, slow_n=26, dem_n=9, val_name="close"):
    import numpy as np
    '''
        指数平滑异同平均线(MACD: Moving Average Convergence Divergence)
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的
          quick_n:int
                      DIFF差离值中快速移动天数
          slow_n:int
                      DIFF差离值中慢速移动天数
          dem_n:int
                      DEM讯号线的移动天数
          val_name:string
                      计算哪一列的列名，默认为 close 
        return
        -------
          OSC:numpy.ndarray<numpy.float64>
              MACD bar / OSC 差值柱形图 DIFF - DEM
          DIFF:numpy.ndarray<numpy.float64>
              差离值
          DEM:numpy.ndarray<numpy.float64>
              讯号线
    '''

    ema_quick = np.asarray(ema(data, quick_n, val_name))
    ema_slow = np.asarray(ema(data, slow_n, val_name))
    DIFF = ema_quick - ema_slow
    data["diff"] = DIFF
    DEM = ema(data, dem_n, "diff")
    BAR = (DIFF - DEM) * 2
    # data['dem'] = DEM
    # data['bar'] = BAR
    return DIFF, DEM, BAR


def ma(data, n=10, val_name="close"):
    '''
    移动平均线 Moving Average
    Parameters
    ------
      data:pandas.DataFrame
                  通过 get_h_data 取得的股票数据
      n:int
                  移动平均线时长，时间单位根据data决定
      val_name:string
                  计算哪一列的列名，默认为 close 收盘值
    return
    -------
      list
          移动平均线
    '''

    values = []
    MA = []

    for index, row in data.iterrows():
        values.append(row[val_name])
        if len(values) == n:
            del values[0]

        MA.append(np.average(values))

    return np.asarray(MA)


def find_successive_bar_area(raw_df:DataFrame, field='bar', min_area_width=6):
    """
    改进的寻找连续区域算法
    :param raw_df:
    :param field:
    :param min_area_width:
    :return:
    """
    df = raw_df.copy()
    successive_areas = []
    # 第一步：把连续的同一颜色区域的index都放入一个数组
    arrays = [df[df[field]>=0].index.array, df[df[field]<=0].index.array]
    for arr in arrays:
        successive_area = []
        for k, g in groupby(enumerate(arr), lambda iv : iv[0] - iv[1]):
            index_group = list(map(itemgetter(1), g))
            successive_area.append((min(index_group), max(index_group)))
        successive_areas.append(successive_area)

    return successive_areas[0], successive_areas[1] # 分别是红色和绿色的区间

#
# def find_successive_bar_area_(raw_df:DataFrame, field='bar', min_blue_area_width=6):
#     """
#     寻找连续的红，绿区域
#     :param df:
#     :param field:
#     :param min_blue_area_width:
#     :return: red_area_list, blue_area_list
#     """
#     df = raw_df.copy()
#     blue_index_range = []  # 存放2元tuple， [start:end]
#
#     # 第一步：先把全部blue bar index都放入一个数组
#     blue_bar_temp_arr = []
#     for i in range(0, df.index.size):  # 从第一个开始
#         diff_val = df.loc[i:i, [field]][field].values[0]
#         if diff_val < 0:
#             blue_bar_temp_arr.append(i)
#
#     # 第二步：扫描连续的绿柱子做成[start_index, end_index]二元组
#
#     i = 0
#     while i < len(blue_bar_temp_arr):
#         start_i = i
#         end_i = i + 1
#         while end_i < len(blue_bar_temp_arr) and blue_bar_temp_arr[end_i - 1] + 1 == blue_bar_temp_arr[end_i]:
#             end_i += 1
#
#         s = blue_bar_temp_arr[start_i]
#         e = blue_bar_temp_arr[end_i - 1]
#         if e - s >= min_blue_area_width:  # 绿色区域要宽度足够大
#             blue_index_range.append((s, e))
#
#         i = end_i
#
#     return blue_index_range


def today():
    """

    :return:
    """
    tm_now = datetime.now()
    td = tm_now.strftime("%Y-%m-%d")
    return td


def n_days_ago(n_days):
    """

    :param n_days:
    :return:
    """
    tm_now = datetime.now()
    delta = timedelta(days=n_days)
    tm_start = tm_now - delta
    ago = tm_start.strftime("%Y-%m-%d")
    return ago


def prepare_csv_data(code_list):
    """

    :param code_list: 股票列表
    :return:
    """
    quote_ctx = OpenQuoteContext(host='futuapi.mkmerich.com', port=54012)
    for code in code_list:
        for _, ktype in K_LINE_TYPE.items():
            ret, df, page_req_key = quote_ctx.request_history_kline(code, start=n_days_ago(20), end=today(),
                                                                    ktype=ktype,
                                                                    fields=[KL_FIELD.DATE_TIME, KL_FIELD.CLOSE],
                                                                    max_count=1000)
            csv_file_name = df_file_name(code, ktype)
            df.to_csv(csv_file_name)
            time.sleep(3.1)  # 频率限制


def df_file_name(stock_code, ktype):
    """

    :param stock_code:
    :param ktype:
    :return:
    """
    return f'data/{stock_code}_{ktype}.csv'


def compute_df_bar(code_list):
    """
    计算60,30,15分钟的指标，存盘
    :param df:
    :return:
    """
    for code in code_list:
        for k, ktype in K_LINE_TYPE.items():
            csv_file_name = df_file_name(code, ktype)
            df = pd.read_csv(csv_file_name, index=0)
            diff, dem, bar = macd(df)
            df['macd_bar'] = bar  # macd
            df['ma5'] = ma(df, 5)
            df['ma10'] = ma(df, 10)
            df['em_bar'] = (df['ma5'] - df['ma10']).apply(lambda val: round(val, 2))  # 均线
            df.to_csv()


def __do_bar_wave_tag(raw_df: DataFrame, field, successive_bar_area, moutain_min_width=5):
    """

    :param raw_df:
    :param field:
    :param successive_bar_area: 想同样色柱子区域, [tuple(start, end)]
    :param moutain_min_width: 作为一个山峰最小的宽度，否则忽略
    :return: 打了tag 的df副本
    """
    df = raw_df.copy()
    tag_field = f'_{field}'
    df[tag_field] = 0  # 初始化为0
    df[field] = df[field].abs()  # 变成正值处理
    for start, end in successive_bar_area:  # 找到s:e这一段里的所有波谷
        sub_area_list = [(start, end)]
        for s, e in sub_area_list:  # 产生的破碎的连续区间加入这个list里继续迭代直到为空
            if e - s + 1 < moutain_min_width:  # 山峰的宽度太窄，可以忽略
                continue
            # 找到最大柱子，在df上打标
            min_row_index = df.iloc[s:e + 1][field].idxmax(axis=0)  # 寻找规定的行范围的某列最小值的索引
            # 先不急于设置为波峰，因为还需要判断宽度是否符合要求
            # 从这根最大柱子向两侧扫描，直到波谷
            arr = df.iloc[s:e + 1][field].array  # 扫描这个数组
            # 从min_index先向左侧扫
            arr_min_index = min_row_index - s  # 映射到以0开始的数组下标上
            i = j = -1
            for i in range(arr_min_index, 1, -1):  # 向左侧扫描, 下标是[arr_min_index, 2]
                if arr[i] >= arr[i - 1] or arr[i] >= arr[i - 2]:
                    continue
                else:
                    # 处理边界条件
                    i = 0 if i == 2 else i
                    break  # i 就是左侧波谷

            # 从min_index向右侧扫描
            for j in range(arr_min_index, e - s - 1):  # 下标范围是[arr_min_index, len(arr)-2]

                if arr[j] <= arr[j + 1] or arr[j] <= arr[j + 2]:
                    continue
                else:
                    j = e - s if j == (e - s - 2) else j
                    break  # j 就是右侧波谷

            # =========================================================
            # 现在连续的波段被分成了3段[s, s+i][s+i, s+j][s+j, e]
            # min_row_index 为波峰；s+i为波谷；s+j为波谷；
            df.at[min_row_index, tag_field] = WaveType.RED_TOP  # 打tag

            # 在下一个阶段中评估波峰波谷的变化度（是否是深V？）
            # 一段连续的区间里可以产生多个波峰，但是波谷可能是重合的，这就要评估是否是深V，合并波峰
            df.at[s + i, tag_field] = WaveType.RED_BOTTOM
            df.at[s + j, tag_field] = WaveType.RED_BOTTOM

            # 剩下两段加入sub_area_list继续迭代
            sub_area_list.append((s, s + i))
            sub_area_list.append((s + j, e))

        # 这里是一个连续区间处理完毕
        # 还需要对波谷、波峰进行合并，如果不是深V那么就合并掉
        # TODO

    return df


def __bar_wave_field_tag(df, field):
    """
    扫描一个字段的波谷波峰
    """
    blue_bar_area = find_successive_bar_area(df, field)
    __do_bar_wave_tag(df, field, blue_bar_area)
    return df


# def __bar_wave_tag(df, field_list):
#     """
#     为df里的字段列表代表的波谷打标
#     :param df:
#     :param field_list:
#     :return:
#     """
#     for f in field_list:
#         blue_bar_area = scan_blue_index(df, f)
#         __do_bar_wave_tag(df, f, blue_bar_area)
#     return df


def __is_bar_divergence(df, field): # TODO 从哪个位置开始算背离？
    """
    field字段是否出现了底背离
    :param df:
    :param field:
    :return: 背离：1， 否则0
    """
    pass  # TODO


def __bar_wave_cnt(df, field='macd_bar'): # TODO 从哪个位置开始数浪？
    """
    在一段连续的绿柱子区间，当前的波峰是第几个
    :param df:
    :param field:
    :return:  波峰个数, 默认1
    """
    pass  # TODO


def __is_bar_multi_wave(df, field='ma_bar'):
    """
    2波段
    :param df:
    :param field:
    :return: 如果是第2个波段，或者2个以上返回1，否则返回0
    """
    wave_cnt = __bar_wave_cnt(df, field)
    rtn = 1 if wave_cnt >= 2 else 0
    return rtn


def __is_macd_bar_reduce(df:DataFrame, field='macd_bar'):
    """
    macd 绿柱子第一根减少出现，不能减少太剧烈，前面的绿色柱子不能太少
    :param df:
    :param field:
    :return:
    """
    cur_bar_len = df.iloc[-1][field]
    pre_bar_len = df.iloc[-2][field]

    is_reduce = cur_bar_len > pre_bar_len # TODO 这里还需要评估一下到底减少多少幅度/速度是最优的
    return is_reduce


def macd_strategy(code_list):
    """
    策略入口
    :return:
    """
    ok_code = {}
    for code in code_list:
        total_score = 0

        df60 = pd.read_csv(df_file_name(code, KL_Period.KL_60))
        if __is_macd_bar_reduce(df60, "macd_bar") == 1:  # 如果60分绿柱变短
            total_score += 1  # 60分绿柱变短分数+1

            bar_60_order = __bar_wave_cnt(df60, 'macd_bar')  # 60分macd波段第几波？
            total_score += (bar_60_order) * 1  # 多一波就多一分
            ma_60_2wave = __is_bar_multi_wave(df60, 'ma_bar')
            total_score += ma_60_2wave * 1  # 60分均线两波下跌

            df30 = pd.read_csv(df_file_name(code, KL_Period.KL_30))
            bar_30_divergence = __is_bar_divergence(df30, 'macd_bar')  # 30分macd背离
            total_score += bar_30_divergence

            ma_30_2wave = __is_bar_multi_wave(df30, 'ma_bar')
            total_score += (ma_30_2wave + ma_60_2wave * ma_30_2wave) * 2

            df15 = pd.read_csv(df_file_name(code, KL_Period.KL_15))
            bar_15_divergence = __is_bar_divergence(df15, 'macd_bar')
            total_score += bar_15_divergence

            ma_15_2wave = __is_bar_multi_wave(df15, 'ma_bar')  # 15分钟2个波段
            total_score += (ma_15_2wave + ma_30_2wave * ma_15_2wave) * 2

            ok_code[code] = total_score

            return ok_code


if __name__ == '__main__':
    """
    df -> df 格式化统一 -> macd_bar, em5, em10 -> macd_bar, em_bar -> 
    macd_bar 判别, macd_wave_scan em_bar_wave_scan -> 按权重评分 
    """
    pass
