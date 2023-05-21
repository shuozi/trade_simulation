#from https://www.vnpy.com/docs/cn/datafeed.html#k

from datetime import datetime
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.datafeed import get_datafeed
from vnpy.trader.object import HistoryRequest

# 获取数据服务实例
datafeed = get_datafeed()
#获取k线级别的历史数据

req = HistoryRequest(
    # 合约代码（示例cu888为米筐连续合约代码，仅用于示范，具体合约代码请根据需求查询数据服务提供商）
    symbol="cu888",
    # 合约所在交易所
    exchange=Exchange.SHFE,
    # 历史数据开始时间
    start=datetime(2019, 1, 1),
    # 历史数据结束时间
    end=datetime(2021, 1, 20),
    # 数据时间粒度，默认可选分钟级、小时级和日级，具体选择需要结合该数据服务的权限和需求自行选择
    interval=Interval.DAILY
)

# 获取k线历史数据
data = datafeed.query_bar_history(req)

#获取tick级别的历史数据
#由于tick数据量较大，下载前请先参考上文确认数据服务是否提供tick数据的下载服务

req = HistoryRequest(
    # 合约代码（示例cu888为米筐连续合约代码，仅用于示范，具体合约代码请根据需求查询数据服务提供商）
    symbol="cu888",
    # 合约所在交易所
    exchange=Exchange.SHFE,
    # 历史数据开始时间
    start=datetime(2019, 1, 1),
    # 历史数据结束时间
    end=datetime(2021, 1, 20),
    # 数据时间粒度，为tick级别
    interval=Interval.TICK
)

# 获取tick历史数据
data = datafeed.query_tick_history(req)