import pandas as pd
import numpy as np
from futu import  *
from datetime import datetime

class KL_Period(object):
    KL_60 = "KL_60"
    KL_30 = "KL_30"
    KL_15 = "KL_15"


K_LINE_TYPE={
    KL_Period.KL_60: KLType.K_60M,
    KL_Period.KL_30: KLType.K_30M,
    KL_Period.KL_15: KLType.K_15M,
}

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
    BAR = (DIFF - DEM)*2
    #data['dem'] = DEM
    #data['bar'] = BAR
    return  DIFF, DEM, BAR

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


def scan_blue_index(df, field='bar', min_blue_area_width=6):
    """

    :param df:
    :param field:
    :param min_blue_area_width:
    :return:
    """
    blue_index_range= [] #存放2元tuple， [start:end]

    # 第一步：先把全部blue bar index都放入一个数组
    blue_bar_temp_arr = []
    for i in range(0, df.index.size): #从第一个开始
        diff_val = df.loc[i:i,[field]][field].values[0]
        if diff_val<0:
            blue_bar_temp_arr.append(i)

    # 第二步：扫描连续的绿柱子做成[start_index, end_index]二元组

    # for i in range(0,len(blue_bar_temp_arr)):
    i=0
    while i < len(blue_bar_temp_arr):
        start_i = i
        end_i = i+1
        while end_i < len(blue_bar_temp_arr) and blue_bar_temp_arr[end_i-1]+1 == blue_bar_temp_arr[end_i]:
            end_i += 1

        s = blue_bar_temp_arr[i]
        e = blue_bar_temp_arr[end_i-1]
        if e-s >= min_blue_area_width: #绿色区域要宽度足够大
            blue_index_range.append((s, e))

        i = end_i

    return blue_index_range


def scan_bar_wave(df, field='bar'):
    """
    扫描bar波浪，返回波浪的高、低点
    """

    pass


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
    tm_start = tm_now-delta
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
            time.sleep(3.1)  #频率限制


def df_file_name(stock_code, ktype):
    """

    :param stock_code:
    :param ktype:
    :return:
    """
    return f'data/{stock_code}_{ktype}.csv'


def compute_df_bar(code_list):
    """
    计算60,30,15分钟的指标，存盘  TODO
    :param df:
    :return:
    """
    for code in code_list:
        for k, ktype in K_LINE_TYPE.items():
            csv_file_name = df_file_name(code, ktype)
            df = pd.read_csv(csv_file_name, index=0)
            diff, dem, bar = macd(df)
            df['macd_bar'] = bar
            df['ma5'] = ma(df, 5)
            df['ma10'] = ma(df, 10)
            df['em_bar'] = (df['ma5'] - df['ma10']).apply(lambda val: round(val, 2))
            df.to_csv()


def __bar_wave_tag(df, field_list):
    """
    为df里的字段代表的波谷打标
    :param df:
    :param field_list:
    :return:
    """
    pass #TODO  先完成这个，后面的可以复用，通过扫描这个函数里的标记减少工作量


def __is_bar_divergence(df,  field):
    """
    field字段是否出现了底背离
    :param df:
    :param field:
    :return: 背离：1， 否则0
    """
    pass #TODO


def __bar_2wave(df, field='ma_bar'):
    """
    2波段
    :param df:
    :param field:
    :return:
    """
    pass #TODO


def __is_macd_bar_reduce(df, field='macd_bar'):
    """
    macd 绿柱子第一根减少出现
    :param df:
    :param field:
    :return:
    """
    pass #TODO


def __bar_wave_cnt(df, field='macd_bar'):
    """
    当前的波峰是第几个
    :param df:
    :param field:
    :return:  波峰个数, 默认1
    """
    pass  #TODO


def macd_strategy(code_list):
    """
    策略入口
    :return:
    """
    ok_code = {}
    for code in code_list:
        total_score = 0
        df60 = pd.read_csv(df_file_name(code, KL_Period.KL_60))
        if __is_macd_bar_reduce(df60, "macd_bar")==1:# 如果60分绿柱变短
            total_score += 1 # 60分绿柱变短分数+1
            bar_60_order = __bar_wave_cnt(df60, 'macd_bar') # 60分macd波段第几波？
            total_score += (bar_60_order)*1  #多一波就多一分

            ma_60_2wave = __bar_2wave(df60, 'ma_bar')
            total_score += ma_60_2wave*1# 60分两波下跌

            df30 = pd.read_csv(df_file_name(code, KL_Period.KL_30))
            bar_30_divergence = __is_bar_divergence(df30, 'macd_bar') # 30分macd背离
            total_score += bar_30_divergence

            ma_30_2wave = __bar_2wave(df30, 'ma_bar')
            total_score += (ma_30_2wave+ma_60_2wave*ma_30_2wave)*2

            df15 = pd.read_csv(df_file_name(code, KL_Period.KL_15))
            bar_15_divergence = __is_bar_divergence(df15, 'macd_bar')
            total_score += bar_15_divergence

            ma_15_2wave = __bar_2wave(df15, 'ma_bar') # 15分钟2个波段
            total_score += (ma_15_2wave+ma_30_2wave*ma_15_2wave)*2

            ok_code[code] = total_score

            return ok_code


if __name__=='__main__':
    """
    df -> df 格式化统一 -> macd_bar, em5, em10 -> macd_bar, em_bar -> 
    macd_bar 判别, macd_wave_scan em_bar_wave_scan -> 按权重评分 
    """
    pass


