import time
from futu import *

class CurKlineTest(CurKlineHandlerBase):
    def on_recv_rsp(self, rsp_str):
        ret_code, data = super(CurKlineTest,self).on_recv_rsp(rsp_str)
        if ret_code != RET_OK:
            print("CurKlineTest: error, msg: %s" % data)
            return RET_ERROR, data

        print("CurKlineTest ", data) # CurKlineTest自己的处理逻辑

        return RET_OK, data

quote_ctx = OpenQuoteContext(host='180.165.129.103', port=54012)
handler = CurKlineTest()
quote_ctx.set_handler(handler)
quote_ctx.subscribe(['SH.601127'], [SubType.K_15M])
time.sleep(15000)
quote_ctx.close()