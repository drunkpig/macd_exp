from macdlib import *
import pandas as pd
import numpy as np

def test():
    code = "SZ.002405"
    quote_ctx = OpenQuoteContext(host=config.futuapi_address, port=config.futuapi_port)
    ret, df, pk = quote_ctx.request_history_kline(code, start=n_days_ago(365 * 2 - 1), end=today(),
                                                          ktype=K_LINE_TYPE[KL_Period.KL_30],
                                                          fields=[KL_FIELD.DATE_TIME, KL_FIELD.CLOSE],
                                                          max_count=500)

    ret, df , pk = quote_ctx.request_history_kline(code = code, ktype=K_LINE_TYPE[KL_Period.KL_30],
                                                   fields=[KL_FIELD.DATE_TIME, KL_FIELD.CLOSE],
                                                    max_count=500, page_req_key=pk)

    print(pk)




def get_all_backtest_data(code):
    quote_ctx = OpenQuoteContext(host=config.futuapi_address, port=config.futuapi_port)

    for _, ktype in K_LINE_TYPE.items():
        mydf = None
        ret, df, pk = quote_ctx.request_history_kline(code, start=n_trade_days_ago( 365*2-1 ), end=today(),
                                                                ktype=ktype,
                                                                fields=[KL_FIELD.DATE_TIME, KL_FIELD.CLOSE, KL_FIELD.HIGH, KL_FIELD.LOW],
                                                                max_count=500)
        while pk is not None or df is not None:
            if mydf is None:
                mydf = df
            else:
                mydf = mydf.append(df, ignore_index=True, sort=False)

            if pk is None:
                break
            ret, df, pk = quote_ctx.request_history_kline(code, start=n_trade_days_ago(365 * 2 - 1), end=today(),
                                                                    ktype=ktype,
                                                                    fields=[KL_FIELD.DATE_TIME, KL_FIELD.CLOSE, KL_FIELD.HIGH, KL_FIELD.LOW],
                                                                    max_count=500, page_req_key=pk)

        csv_file_name = df_file_name(code, ktype)
        mydf.to_csv(csv_file_name)
        time.sleep(3.1)  # 频率限制


def test_one_prob(code):
    compute_df_bar([code])

    fname = df_file_name(code, KLType.K_60M)
    df60 = pd.read_csv(fname, index_col=0)
    red_areas, blue_areas = find_successive_bar_areas(df60, 'macd_bar')
    df_blue = do_bar_wave_tag(df60, 'macd_bar', blue_areas)
    df_blue['_macd_bar_tag'] *= -1
    df_buy = df_blue[df_blue._macd_bar_tag == -2]

    idx = df_buy.index.array
    for i in idx:
        # 统计一下
        pass
    return df_buy



if __name__=="__main__":
    config.g_config.DEV_MODEL = "backtest_"
    STOCK_CODE = 'SZ.002405'
    #get_all_backtest_data('SZ.002405')
    compute_df_bar([STOCK_CODE])
