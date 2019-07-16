import pandas as pd

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
    #data["diff"] = DIFF
    DEM = ema(data, dem_n, "diff")
    BAR = (DIFF - DEM)*2
    #data['dem'] = DEM
    #data['bar'] = BAR
    return  DIFF, DEM, BAR


def scan_blue_index(df):
    blue_index_range= [] #存放2元tuple， [start:end]

    # 第一步：先把全部blue bar index都放入一个数组
    blue_bar_temp_arr = []
    for i in range(0, df.index.size): #从第一个开始
        diff_val = df.loc[i:i,['bar']]['bar'].values[0]
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

        blue_index_range.append((blue_bar_temp_arr[i], blue_bar_temp_arr[end_i-1]))
        i = end_i

    return blue_index_range
