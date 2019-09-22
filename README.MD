## useful api

### 交易日期
```python
from tushare.util.dateu import trade_cal, is_holiday
trade_cal()

      calendarDate  isOpen
0       1990-12-19       1
1       1990-12-20       1
2       1990-12-21       1
3       1990-12-22       0
4       1990-12-23       0
5       1990-12-24       1
6       1990-12-25       1
7       1990-12-26       1
```

```python
is_holiday("2019-09-22") # True， 假期非交易日期
```

