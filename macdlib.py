from itertools import groupby
from operator import itemgetter

import talib
from futu import *
from pandas import DataFrame
from tushare.util.dateu import trade_cal

import config


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


def MA(df, window, field, new_field):
    """

    :param df:
    :param window:
    :param field:
    :param new_field:
    :return:
    """
    df[new_field] = df[field].rolling(window=window).mean()
    return df


def MACD(df, field_name='close', quick_n=12, slow_n=26, dem_n=9):
    """
    
    :param df:
    :param field_name:
    :param quick_n:
    :param slow_n:
    :param dem_n:
    :return:
    """
    diff, macdsignal, macd_bar = talib.MACD(df[field_name], fastperiod=quick_n, slowperiod=slow_n, signalperiod=dem_n)
    return diff, macdsignal, macd_bar


def find_successive_bar_areas(df: DataFrame, field='bar'):
    """
    这个地方不管宽度，只管找连续的区域
    改进的寻找连续区域算法；
    还有一种算法思路，由于红色柱子>0, 绿色柱子<0, 只要找到 x[n]*x[n+1]<0的点然后做分组即可。
    :param raw_df:
    :param field:
    :return:
    """
    successive_areas = []
    # 第一步：把连续的同一颜色区域的index都放入一个数组
    arrays = [df[df[field] >= 0].index.array, df[df[field] <= 0].index.array]
    for arr in arrays:
        successive_area = []
        for k, g in groupby(enumerate(arr), lambda iv: iv[0] - iv[1]):
            index_group = list(map(itemgetter(1), g))
            successive_area.append((min(index_group), max(index_group)))
        successive_areas.append(successive_area)

    return successive_areas[0], successive_areas[1]  # 分别是红色和绿色的区间


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


def n_trade_days_ago(n_trade_days, end_dt=today()):
    """

    :param n_trade_days: 从start_dt开始往前面推几个交易日。
    :param start_dt: 往前推算交易日的开始日期，格式类似"2019-02-02"
    :return:
    """
    trade_days = trade_cal()
    last_idx = trade_days[trade_days.calendarDate == end_dt].index.values[0]

    df = trade_days[trade_days.isOpen == 1]
    start_date = df[df.index <= last_idx].tail(n_trade_days).head(1).iat[0, 0]
    return start_date


def prepare_csv_data(code_list, n_days=config.n_days_bar):
    """

    :param code_list: 股票列表
    :return:
    """
    quote_ctx = OpenQuoteContext(host=config.futuapi_address, port=config.futuapi_port)
    for code in code_list:
        for _, ktype in K_LINE_TYPE.items():
            ret, df, page_req_key = quote_ctx.request_history_kline(code, start=n_days_ago(n_days), end=today(),
                                                                    ktype=ktype,
                                                                    fields=[KL_FIELD.DATE_TIME, KL_FIELD.CLOSE,
                                                                            KL_FIELD.HIGH, KL_FIELD.LOW],
                                                                    max_count=1000)
            csv_file_name = df_file_name(code, ktype)
            df.to_csv(csv_file_name)
            time.sleep(3.1)  # 频率限制

    quote_ctx.close()


def get_df_of_code(code, ktype=K_LINE_TYPE[KL_Period.KL_60], n_days=config.n_days_bar):
    """

    :param code:
    :param ktype:
    :param n_days:
    :return:
    """
    quote_ctx = OpenQuoteContext(host=config.futuapi_address, port=config.futuapi_port)
    ret, df, page_req_key = quote_ctx.request_history_kline(code, start=n_days_ago(n_days), end=today(),
                                                            ktype=ktype,
                                                            fields=[KL_FIELD.DATE_TIME, KL_FIELD.CLOSE, KL_FIELD.HIGH,
                                                                    KL_FIELD.LOW],
                                                            max_count=1000)
    quote_ctx.close()
    return df


def df_file_name(stock_code, ktype):
    """

    :param stock_code:
    :param ktype:
    :return:
    """
    prefix = config.g_config.DEV_MODEL
    return f'data/{prefix}{stock_code}_{ktype}.csv'


def compute_df_bar(code_list):
    """
    计算60,30,15分钟的指标，存盘
    :param df:
    :return:
    """
    for code in code_list:
        for k, ktype in K_LINE_TYPE.items():
            csv_file_name = df_file_name(code, ktype)
            df = pd.read_csv(csv_file_name, index_col=0)
            diff, dem, bar = MACD(df)
            df['macd_bar'] = bar  # macd
            df = MA(df, 5, 'close', 'ma5')
            df = MA(df, 10, 'close', 'ma10')
            df['em_bar'] = (df['ma5'] - df['ma10']).apply(lambda val: round(val, 2))  # 均线
            df.to_csv(csv_file_name)


def __ext_field(field_name):
    """

    :param field_name:
    :return:
    """
    return f'_{field_name}_tag'


def do_bar_wave_tag(raw_df: DataFrame, field, successive_bar_area, moutain_min_width=5):
    """
    这里找波峰和波谷，找谷底的目的是为了测量波峰/谷的斜率
    # TODO 试一下FFT寻找波谷波峰

    :param raw_df:
    :param field:
    :param successive_bar_area: 想同样色柱子区域, [tuple(start, end)]
    :param moutain_min_width: 作为一个山峰最小的宽度，否则忽略
    :return: 打了tag 的df副本
    """
    df = raw_df[[field]].copy()
    tag_field = __ext_field(field)
    df[tag_field] = 0  # 初始化为0
    df[field] = df[field].abs()  # 变成正值处理
    for start, end in successive_bar_area:  # 找到s:e这一段里的所有波峰
        sub_area_list = [(start, end)]

        for s, e in sub_area_list:  # 产生的破碎的连续区间加入这个list里继续迭代直到为空
            if e - s + 1 < moutain_min_width:  # 山峰的宽度太窄，可以忽略
                continue
            # 找到最大柱子，在df上打标
            max_row_index = df.iloc[s:e + 1][field].idxmax(axis=0)  # 寻找规定的行范围的某列最大值的索引
            # 先不急于设置为波峰，因为还需要判断宽度是否符合要求
            # 从这根最大柱子向两侧扫描，直到波谷
            # max_row_index先向左侧扫描
            i, j = s, e
            for i in range(max_row_index, s + 1, -1):  # 向左侧扫描, 下标是(s,s+1,   [s+2, max_row_index])
                if df.at[i, field] >= df.at[i - 1, field] or df.at[i, field] >= df.at[i - 2, field]:
                    if i == s + 2:
                        i = s
                        break
                    else:
                        continue
                else:
                    break  # i 就是左侧波谷

            # 从min_index向右侧扫描
            for j in range(max_row_index, e - 1):  # 下标范围是[arr_min_index, len(arr)-2]
                if df.at[j, field] >= df.at[j + 1, field] or df.at[j, field] >= df.at[j + 2, field]:
                    if j == e - 2:
                        j = e
                    else:
                        continue
                else:
                    break  # j 就是右侧波谷

            # =========================================================
            # 现在连续的波段被分成了3段[s, i][i, j][j, e]
            # max_row_index 为波峰；i为波谷；j为波谷；'
            if j - i + 1 >= moutain_min_width:
                df.at[max_row_index, tag_field] = WaveType.RED_TOP  # 打tag

                # 在下一个阶段中评估波峰波谷的变化度（是否是深V？）
                # 一段连续的区间里可以产生多个波峰，但是波谷可能是重合的，这就要评估是否是深V，合并波峰
                df.at[i, tag_field] = WaveType.RED_BOTTOM
                df.at[j, tag_field] = WaveType.RED_BOTTOM

            # 剩下两段加入sub_area_list继续迭代
            if i - s + 1 >= moutain_min_width:
                sub_area_list.append((s, i))
            if e - j + 1 >= moutain_min_width:  # j为啥不能为0呢？如果为0 说明循环进不去,由此推倒出极值点位于最左侧开始的2个位置，这个宽度不足以参与下一个遍历。
                sub_area_list.append((j, e))

        # 这里是一个连续区间处理完毕
        # 还需要对波谷、波峰进行合并，如果不是深V那么就合并掉
        # TODO

    return df


# def do_bar_wave_tag2(raw_df: DataFrame, field, successive_bar_area, moutain_min_width=5):
#     """
#     寻找波峰波谷的快速算法
#     #  L=X(n)-X(n-1)| X(n)=0 when n<0 else X(n); 然后扫描连续的正值和负值连续区间
#     :param raw_df:
#     :param field:
#     :param successive_bar_area:
#     :param moutain_min_width:
#     :return:
#     """
#     df = raw_df[[field]].copy()
#     tag_field = f'_{field}_tag'
#     df[tag_field] = 0  # 初始化为0
#     df[field] = df[field].abs()  # 变成正值处理
#
#     skiped_diff = np.zeros(df.shape[0], dtype=np.bool)  # 全部False,方便后面的或操作
#     for i in range(1, config.wave_scan_max_gap + 1):
#         for j in range(i, df.shape[0]):
#             skiped_diff[j] = (df.at(j, field) - df[j - 1, field]) > 0 | skiped_diff[j]


def is_bar_bottom_divergence(df: DataFrame, field, value_field):
    """
    field字段是否出现了底背离
    :param df:
    :param field: bar的field名字
    :param value_field:  价格
    :return: 背离：1， 否则0
    """
    field_tag_name = __ext_field(field)
    last_idx = df[df[field_tag_name] > 0].tail(1).index[0]
    dftemp = df[(df[field_tag_name] != 0) & (df.index > last_idx) & (
                df[field_tag_name] == WaveType.GREEN_TOP)].copy().reset_index(drop=True)
    
    wave_cnt = dftemp.shape[0]
    if wave_cnt >= 3:  # 如果多于3波，那么只看最后一波，暂时这么干 TODO 优化多重背离
        dftemp = dftemp.tail(2)
    if wave_cnt == 2:
        bar_val_before = abs(dftemp.head(1).at[0, field])  # TODO 优化为矢量计算
        bar_val_now = abs(dftemp.tail(1).at[0, field])
        value_before = dftemp.head(1).at[0, value_field]
        value_now = dftemp.tail(1).at[0, value_field]
        if bar_val_now <= bar_val_before and value_now < value_before:
            return True

    return False


def __bar_wave_cnt(df: DataFrame, field='macd_bar'):
    """
    在一段连续的绿柱子区间，当前的波峰是第几个
    方法是：从当前时间开始找到前面第一段连续绿柱，然后计算绿柱区间有几个波峰；
    如果当前是红柱但是没超过设置的最大宽度，可以忽略这段红柱
    :param df:
    :param field:
    :return:  波峰个数, 默认1
    """
    field_tag_name = __ext_field(field)
    last_idx = df[df[field_tag_name] > 0].tail(1).index[0]
    wave_cnt = df[(df[field_tag_name] != 0) & (df.index > last_idx) & (df[field_tag_name] == WaveType.GREEN_TOP)].shape[
        0]
    return wave_cnt


def is_bar_mult_wave(df, field='ma_bar'):
    """
    2波段
    :param df:
    :param field:
    :return: 如果是第2个波段，或者2个以上返回1，否则返回0
    """
    wave_cnt = __bar_wave_cnt(df, field)
    rtn = 1 if wave_cnt >= 2 else 0
    return rtn


def is_macd_bar_reduce(df: DataFrame, field='macd_bar'):
    """
    macd 绿柱子第一根减少出现，不能减少太剧烈，前面的绿色柱子不能太少
    :param df:
    :param field:
    :return:
    """
    cur_bar_len = df.iloc[-1][field]
    pre_bar_1_len = df.iloc[-2][field]
    pre_bar_2_len = df.iloc[-3][field]

    is_reduce = cur_bar_len > pre_bar_1_len and cur_bar_len > pre_bar_2_len
    # TODO 这里还需要评估一下到底减少多少幅度/速度是最优的
    return is_reduce


if __name__ == '__main__':
    """
    df -> df 格式化统一 -> macd_bar, em5, em10 -> macd_bar, em_bar -> 
    macd_bar 判别, macd_wave_scan em_bar_wave_scan -> 按权重评分 
    """
    # STOCK_CODE = 'SZ.002405'
    # prepare_csv_data([STOCK_CODE], n_days=90)
    # compute_df_bar([STOCK_CODE])
    # fname = df_file_name(STOCK_CODE, KLType.K_60M)
    # df60 = pd.read_csv(fname, index_col=0)
    # red_areas, blue_areas = find_successive_bar_areas(df60, 'macd_bar')
    # df_new = do_bar_wave_tag(df60, 'macd_bar', red_areas)
    # print(df_new.index)
    print(n_trade_days_ago(3))
