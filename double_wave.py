from macdlib import *


def __compute_score(df60, df30, df15):
    """

    :param df60:
    :param df30:
    :param df15:
    :return: 总得分和每一项得分的统计信息(用于可视化）
    """
    pass

    return None, None #TODO


def double_wave(code_list):
    """

    :param code_list:
    :return:
    """
    results = []
    for code in code_list:
        result = {}
        df60 = pd.read_csv(df_file_name(code, KL_Period.KL_60), index_col=0)
        df30 = pd.read_csv(df_file_name(code, KL_Period.KL_30), index_col=0)
        df15 = pd.read_csv(df_file_name(code, KL_Period.KL_15), index_col=0)
        total_score, score_table = __compute_score(df60, df30, df15)
        result[code] = {"score":total_score, "score_table":score_table}
        results.append(result)


def double_breast__(code_list):
    """
    策略入口
    :return:
    """
    ok_code = {}
    for code in code_list:
        total_score = 0

        df60 = pd.read_csv(df_file_name(code, KL_Period.KL_60), index_col=0)
        if is_macd_bar_reduce(df60, "macd_bar") == 1:  # 如果60分绿柱变短
            total_score += 1  # 60分绿柱变短分数+1

            bar_60_order = bar_wave_cnt(df60, 'macd_bar')  # 60分macd波段第几波？
            total_score += bar_60_order * 1  # 多一波就多一分
            ma_60_2wave = bar_wave_cnt(df60, 'ma_bar')
            total_score += ma_60_2wave * 1  # 60分均线波段下跌

            df30 = pd.read_csv(df_file_name(code, KL_Period.KL_30), index_col=0)
            bar_30_divergence = __is_bar_divergence(df30, 'macd_bar')  # 30分macd背离
            total_score += bar_30_divergence

            ma_30_2wave = __is_bar_2wave(df30, 'ma_bar')
            total_score += (ma_30_2wave + ma_60_2wave * ma_30_2wave) * 2

            df15 = pd.read_csv(df_file_name(code, KL_Period.KL_15), index_col=0)
            bar_15_divergence = __is_bar_divergence(df15, 'macd_bar')
            total_score += bar_15_divergence

            ma_15_2wave = __is_bar_2wave(df15, 'ma_bar')  # 15分钟2个波段
            total_score += (ma_15_2wave + ma_30_2wave * ma_15_2wave) * 2

            ok_code[code] = total_score

            return ok_code
