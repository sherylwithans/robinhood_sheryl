import math

import requests
from bs4 import BeautifulSoup

# import yahoo_fin.stock_info as si
# import yahoo_fin.options as ops

import talib as ta
from talib import BBANDS

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# import sqlite3
# from sqlite3 import Error

from robinhood_sheryl.rs_db import *

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

US_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())

logging.disable(level=logging.INFO)


# token = login()


def ws_get_nasdaq_futures_info(info=None):
    url = "https://liveindex.org/nasdaq-futures/"
    response = requests.get(url)
    if response.status_code == 200:
        results_page = BeautifulSoup(response.content, 'lxml')
        table = results_page.find(
            'table', {'class': "index_table indexes_single"})
        d = {}
        d['latest_price'] = table.find('td', {'title': 'Last Trade Price (NASDAQ 100 FUTURES)'}).text
        d['idx_change'] = table.find('td', {'title': 'Change (NASDAQ 100 FUTURES)'}).text
        d['pct_change'] = table.find('td', {'title': 'Change in % (NASDAQ 100 FUTURES)'}).text
        d['high'] = table.find('td', {'class': 'index-high'}).text
        d['low'] = table.find('td', {'class': 'index-low'}).text
    else:
        return 'FAILURE'

    return d[info] if info else d


def yf_get_latest_price(ticker):
    df = getLatestRowData(equitiesTable, ticker)
    latest_price = float(df['close'])
    return latest_price


def yf_get_prev_close_price(ticker):
    yesterday = (datetime.today() - US_BUSINESS_DAY).strftime('%Y-%m-%d')
    if ticker in ['^N225']:
        timezone = 'JST'
        start_time = ' 14:55'
        end_time = ' 15:00'
    elif ticker in ['^FTSE']:
        timezone = 'WET'
        start_time = ' 16:25'
        end_time = ' 16:30'
    else:
        timezone = 'US/Eastern'
        start_time = ' 15:59'
        end_time = ' 16:00'

    df = getData(equitiesTable, {'ticker': ticker}, start_date=yesterday + start_time,
                 end_date=yesterday + end_time, extended_hours=True, timezone=timezone)

    if df.empty:
        df = getData(equitiesTable, {'ticker': ticker}, start_date=yesterday + ' 09:30',
                     end_date=yesterday + end_time, extended_hours=False, timezone=timezone)

    if df.empty:
        # yesterday data not available
        return yf_get_latest_price(ticker)

    df = df['close'].resample('B').last()
    prev_close_price = df.loc[yesterday]
    return prev_close_price


def yf_get_pct_change(ticker):
    latest_price = yf_get_latest_price(ticker)
    prev_close_price = yf_get_prev_close_price(ticker)
    pct_change = latest_price / prev_close_price - 1
    return pct_change


def convert_period(period):
    period_dict = {'d': 1, 'mo': 20, 'y': 52 * 5, 'ytd': 0, 'max': 0}
    num = ''
    period_type = ''
    for c in period:
        try:
            i = int(c)
            num += c
        except:
            period_type += c
    num = int(num) if num else 0
    period = period_dict[period_type]
    days = num * period
    # period_start_date = datetime.today()-BDay(days)
    period_start_date = datetime.today() - US_BUSINESS_DAY * days
    #     period_start_date = period_start_date.replace(tzinfo=pytz.timezone('US/Eastern'))
    return period_start_date


def convert_interval(interval, window):
    interval_dict = {'m': 1 / 8 / 60, 'h': 1 / 8, 'd': 1, 'wk': 5, 'mo': 20}
    #     interval_dict = {'1m':7,'2m':60,'5m':60,'15m':60,'60m':73,'90m':60,'1h':73,1d, 5d, 1wk, 1mo, 3mo}
    num = ''
    interval_type = ''
    for c in interval:
        try:
            i = int(c)
            num += c
        except:
            interval_type += c
    num = int(num) if num else 0
    interval = interval_dict[interval_type]
    days = num * interval * window
    days = math.ceil(days)
    return days


def yf_get_moving_average(ticker=None, interval='1d', period='3mo', price_type='close', windows=[20, 50],
                          signals={'ema': (20, 50)}, latest=False, extended_hours=False, df=None):
    period_start_date = convert_period(period)

    if ticker:
        interval_days = convert_interval(interval, max(windows + [200]))
        start_date = period_start_date - BDay(interval_days)
        if ticker in INDEXES:
            ticker = INDEXES[ticker]
        df = getData(equitiesTable, {'ticker': ticker}, start_date=start_date, extended_hours=extended_hours)

    resample_interval = interval.replace('m', 'Min').replace('d', 'B')
    if 'h' in interval:
        # base=0.5
        offset = '0.5h'
    else:
        # base=0
        offset = '0s'
    # volume_agg = df['volume'].resample(resample_interval,base=base).sum()
    volume_agg = df['volume'].resample(resample_interval, offset=offset).sum()
    # df = df[['open', 'high', 'low', 'close', 'dividends', 'stock_splits']].resample(
    # resample_interval,base=base).last()
    df = df[['open', 'high', 'low', 'close', 'dividends', 'stock_splits']].resample(
        resample_interval, offset=offset).last()

    df['volume'] = volume_agg
    df = df.dropna()
    if df.empty:
        return df
    #     df_period = stock.history(interval=interval,period=period)
    #     df['volume'] = df['volume'].replace(to_replace=0, method='ffill')
    #     df['return'] = df[price_type].pct_change()
    for window in windows:
        #         df[f'pd_sma_{window}'] = df[price_type].rolling(window=window).mean()
        df[f'sma_{window}'] = ta.SMA(df[price_type], window)
        df[f'ema_{window}'] = ta.EMA(df[price_type], window)
        df[f'vol_sma_{window}'] = ta.SMA(df['volume'], window)

        df[f'upperband_{window}'], df[f'middleband_{window}'], df[f'lowerband_{window}'] = BBANDS(df[price_type],
                                                                                                  timeperiod=window,
                                                                                                  nbdevup=2, nbdevdn=2,
                                                                                                  matype=0)

        df[f'max_{window}'] = df[price_type].rolling(window=window).max() == df[price_type]
        df[f'min_{window}'] = df[price_type].rolling(window=window).min() == df[price_type]

    # 200 day moving average
    df[f'ema_200'] = ta.EMA(df[price_type], 200)

    for s, (v1, v2) in signals.items():
        df[f'{s}_signal_{v1}_{v2}'] = np.where(df[f'{s}_{v1}'] > df[f'{s}_{v2}'], 1, 0)
        df[f'{s}_position_{v1}_{v2}'] = df[f'{s}_signal_{v1}_{v2}'].diff()
        order_dict = {1: 'buy', -1: 'sell', 0: np.nan}
        df[f'{s}_execute_order_{v1}_{v2}'] = df[f'{s}_position_{v1}_{v2}'].map(
            lambda x: order_dict[x], na_action='ignore').replace(to_replace=np.nan, method='ffill')

        df[f'{s}_execute_time_{v1}_{v2}'] = np.where(
            df[f'{s}_position_{v1}_{v2}'] != 0, df.index.tz_localize(None), np.datetime64('NaT'))
        df[f'{s}_execute_time_{v1}_{v2}'] = df[
            f'{s}_execute_time_{v1}_{v2}'].replace(to_replace=np.datetime64('NaT'), method='ffill')

        df[f'{s}_execute_price_{v1}_{v2}'] = np.where(
            df[f'{s}_position_{v1}_{v2}'] != 0, df[price_type], np.nan)
        df[f'{s}_execute_price_{v1}_{v2}'] = df[
            f'{s}_execute_price_{v1}_{v2}'].replace(to_replace=np.nan, method='ffill')

    if df.index[0].tzinfo is None:
        selected_df = df[df.index >= period_start_date]
    else:
        selected_df = df[df.index >= period_start_date.replace(tzinfo=pytz.timezone('US/Eastern'))]

    if latest:
        latest_dt = df.index.max()
        return df.loc[latest_dt]
    return selected_df


def yf_backtest(ticker=None, interval='1d', period='3mo', price_type='close',
                windows=[20, 50], signals={'ema': (20, 50)}, long_only=False, results=False, extended_hours=False,
                df=None):
    if ticker:
        df = yf_get_moving_average(ticker,
                                   interval, period, price_type, windows, signals, latest=False,
                                   extended_hours=extended_hours)

    if df.empty:
        return df
    cols = df.columns
    start_price = df[price_type][0]
    #     print(start_price)
    df[f'hold_{period}_gain'] = df[price_type] - start_price
    df[f'hold_{period}_return'] = df[f'hold_{period}_gain'] / start_price

    for s, (v1, v2) in signals.items():
        df[f'{s}_ls_num_shares_outstanding_{v1}_{v2}'] = df[f'{s}_position_{v1}_{v2}'].cumsum()

        # start from first long
        start_long_idx = df[f'{s}_position_{v1}_{v2}'].gt(0).idxmax()
        #         check if any long over time period
        if df[f'{s}_position_{v1}_{v2}'].loc[start_long_idx] == 0:
            start_long_idx = None
            l_start_price = 0
        else:
            l_start_price = df.loc[start_long_idx][price_type]
        if long_only:
            df[f'{s}_l_position_{v1}_{v2}'] = np.where(
                df[f'{s}_position_{v1}_{v2}'] >= 0, df[f'{s}_position_{v1}_{v2}'], 0)
        else:
            df[f'{s}_l_flag_{v1}_{v2}'] = df.index.to_series().apply(
                lambda x: 1 if start_long_idx and x >= start_long_idx else 0)
            df[f'{s}_l_position_{v1}_{v2}'] = np.where(
                df[f'{s}_l_flag_{v1}_{v2}'] == 1, df[f'{s}_position_{v1}_{v2}'], 0)

        # This will be truely long only if first move is -1, then cumsum will always <=0, so buy when buy signal but never sell
        #         df[f'{s}_lo_position_{v1}_{v2}'] = np.where(
        #             df[f'{s}_ls_num_shares_outstanding_{v1}_{v2}']>=0,df[f'{s}_position_{v1}_{v2}'],0)

        df[f'{s}_l_num_shares_outstanding_{v1}_{v2}'] = df[f'{s}_l_position_{v1}_{v2}'].cumsum()
        df[f'{s}_l_cash_flow_{v1}_{v2}'] = df[f'{s}_l_position_{v1}_{v2}'] * df[price_type]
        df[f'{s}_l_cumu_cash_flow_{v1}_{v2}'] = df[f'{s}_l_cash_flow_{v1}_{v2}'].cumsum()

        df[f'{s}_l_cumu_gain_{v1}_{v2}'] = df[price_type] * df[
            f'{s}_l_num_shares_outstanding_{v1}_{v2}'] - df[f'{s}_l_cumu_cash_flow_{v1}_{v2}']

        # add start price to keep graph on same scale, gains don't account for increase in capital investment
        df[f'{s}_l_cumu_wealth_{v1}_{v2}'] = df[f'{s}_l_cumu_gain_{v1}_{v2}'] + l_start_price

        if long_only:
            # fillna(0) to account for cases where no long occurs throughout selected investment period
            df[f'{s}_l_cumu_return_{v1}_{v2}'] = (df[
                                                      f'{s}_l_cumu_gain_{v1}_{v2}'] / df[
                                                      f'{s}_l_cumu_cash_flow_{v1}_{v2}']).fillna(0)
        else:
            df[f'{s}_l_cumu_return_{v1}_{v2}'] = (df[f'{s}_l_cumu_gain_{v1}_{v2}'] / l_start_price).fillna(0)

        df['strategy_ratio'] = (df[f'{s}_l_cumu_return_{v1}_{v2}'] -
                                df[f'hold_{period}_return']) / abs(df[f'hold_{period}_return'])

    if results:
        return df[['close'] + [x for x in df.columns if any(
            keyword in x for keyword in
            ['execute', 'wealth', 'gain', 'return', 'hold', 'cumu_cash', 'strategy'])]].iloc[-1:, :]

    return df


def yf_backtest_wrapper(tickers, interval='5m', period='1mo', price_type='close', windows=[20, 50],
                        signals={'ema': (20, 50)}, extended_hours=False, long_only=False):
    #     if ticker in INDEXES:
    #         ticker = INDEXES[ticker]
    period_start_date = convert_period(period)
    interval_days = convert_interval(interval, max(windows + [200]))
    start_date = period_start_date - BDay(interval_days)
    resample_interval = interval.replace('m', 'Min').replace('d', 'B')

    chunks = []
    n = 25
    for i in range(0, len(tickers), n):
        chunks.append(tickers[i:i + n])

    df_list = []
    for tickers_list in chunks:
        df = getData(equitiesTable, tickers=tickers_list, start_date=start_date,
                     extended_hours=extended_hours)
        df = df.groupby('ticker').apply(lambda x: yf_get_moving_average(df=x,
                                                                        interval=interval, period=period,
                                                                        price_type=price_type, windows=windows,
                                                                        signals=signals, latest=False,
                                                                        extended_hours=extended_hours))
        df = df.groupby('ticker').apply(lambda x: yf_backtest(df=x.droplevel(0),
                                                              interval=interval, period=period,
                                                              price_type=price_type, windows=windows, signals=signals,
                                                              long_only=long_only, results=True,
                                                              extended_hours=extended_hours))
        df_list.append(df)

    df_agg = pd.concat(df_list)

    return df_agg


def yf_plot_moving_average(ticker, interval='1d', period='3mo', price_type='close', windows=[20, 50],
                           secondary_axis_type='volume', bband=True, signals={'sma': (20, 50)}, extended_hours=False):
    df = yf_get_moving_average(ticker, interval, period, price_type, windows, signals=signals, latest=False,
                               extended_hours=extended_hours)

    marker_colors = [{'up': 'limegreen', 'down': 'crimson'},
                     {'up': 'deepskyblue', 'down': 'orange'},
                     ]
    marker_styles = []
    for color in marker_colors:
        marker_up = dict(color=color['up'], size=10, symbol=5, line=dict(color='darkgreen', width=2))
        marker_down = dict(color=color['down'], size=10, symbol=6, line=dict(color='maroon', width=2))
        marker_styles.append({'up': marker_up, 'down': marker_down})

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces

    fig.add_trace(
        go.Scatter(x=df.index, y=df[price_type], name=price_type.lower()),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df['ema_200'], name='ema_200'),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(x=df.index, y=df[secondary_axis_type], name=secondary_axis_type.lower(),
               marker_color='darkred',
               ),
        secondary_y=True,
    )

    for window in windows:
        #         fig.add_trace(
        #             go.Scatter(x=df.index, y=df[f'sma_{window}'], name=f'sma_{window}'),
        #             secondary_y=False,
        #         )
        fig.add_trace(
            go.Scatter(x=df.index, y=df[f'ema_{window}'], name=f'ema_{window}'),
            secondary_y=False,
        )
        #         fig.add_trace(
        #             go.Scatter(x=df.index, y=df[f'vol_sma_{window}'], name=f'vol_sma_{window}',
        # #                       marker_color='lightsalmon',
        #                       ),
        #             secondary_y=True,
        #         )
        fig.add_trace(
            go.Scatter(x=df.index, y=np.where(df[f'max_{window}'] == True, df[price_type], np.nan),
                       name=f'max_{window}',
                       mode='markers',
                       marker=dict(color='lightcoral', size=5, symbol=49, line=dict(color='darkgreen', width=1)),
                       ),

            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=df.index, y=np.where(df[f'min_{window}'] == True, df[price_type], np.nan),
                       name=f'min_{window}',
                       mode='markers',
                       marker=dict(color='lightcoral', size=5, symbol=50, line=dict(color='darkgreen', width=1)),
                       ),

            secondary_y=False,
        )
        if bband:
            for band in ['upperband', 'middleband', 'lowerband']:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[f'{band}_{window}'], name=f'{band}_{window}'),
                    secondary_y=False,
                )

    i = 0
    for s, (v1, v2) in signals.items():
        fig.add_trace(
            go.Scatter(
                x=df.index, y=np.where(df[f'{s}_position_{v1}_{v2}'] == 1, df[price_type], np.nan),
                name=f'{s}_signal_{v1}_{v2}',
                mode='markers',
                #                 marker_symbol=5,
                #                 marker_line_color="midnightblue", marker_color="lightskyblue",
                #                 marker_line_width=2, marker_size=15,
                marker=marker_styles[i]['up'],
                #                 showlegend=False,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=np.where(df[f'{s}_position_{v1}_{v2}'] == -1, df[price_type], np.nan),
                name=f'{s}_signal_{v1}_{v2}',
                mode='markers',
                marker=marker_styles[i]['down']
                #                 showlegend=False,
            ),
            secondary_y=False,
        )
        i += 1

    # Add figure title
    fig.update_layout(
        title_text=f"<b>{ticker}</b> {price_type} and {secondary_axis_type} Time Series Plot"
    )

    rangebreaks = [
        dict(bounds=["sat", "mon"]),  # hide weekends
        #             dict(bounds=[16, 9.5], pattern="hour"),
        #         dict(values=["2015-12-25", "2016-01-01"])  # hide Christmas and New Year's
        #             dict(values=list(df.index),dvalue=1000*60*60*9.5),  # hide until 9:30am
    ]
    if any(x == interval[-1] for x in 'hm'):
        if extended_hours:
            rangebreaks += [dict(bounds=[20, 4], pattern="hour"), ]
        else:
            rangebreaks += [dict(bounds=[16, 9.5], pattern="hour"), ]

    fig.update_xaxes(
        title_text="Datetime",
        rangebreaks=rangebreaks,
    )

    # Set x-axis title
    #     fig.update_xaxes(title_text="Datetime")

    # Set y-axes titles
    fig.update_yaxes(title_text=f"<b>primary</b> {price_type}", secondary_y=False)
    fig.update_yaxes(title_text=f"<b>secondary</b> {secondary_axis_type}", secondary_y=True)

    #     fig.show()
    return fig


def yf_plot_backtest(ticker, interval='1d', period='3mo', price_type='close',
                     secondary_axis_type='return', windows=[20, 50], signals={'sma': (20, 50)},
                     long_only=False, extended_hours=False):
    df = yf_backtest(ticker=ticker, interval=interval, period=period, price_type=price_type, windows=windows,
                     signals=signals, long_only=long_only, extended_hours=extended_hours)

    marker_colors = [{'up': 'limegreen', 'down': 'crimson'},
                     {'up': 'deepskyblue', 'down': 'orange'},
                     ]
    marker_styles = []
    for color in marker_colors:
        marker_up = dict(color=color['up'], size=10, symbol=5, line=dict(color='darkgreen', width=2))
        marker_down = dict(color=color['down'], size=10, symbol=6, line=dict(color='maroon', width=2))
        marker_styles.append({'up': marker_up, 'down': marker_down})

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    benchmark = f'hold_{period}_return' if secondary_axis_type == 'return' else price_type

    # Add traces
    fig.add_trace(
        go.Scatter(x=df.index, y=df[price_type], name=price_type.lower()),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df['ema_200'], name='ema_200'),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df[benchmark], name=benchmark.lower()),
        secondary_y=True if secondary_axis_type == 'return' else False,
    )

    i = 0
    for s, (v1, v2) in signals.items():
        strategy = (f'{s}_l_cumu_return_{v1}_{v2}' if
                    secondary_axis_type == 'return' else f'{s}_l_cumu_wealth_{v1}_{v2}')
        fig.add_trace(
            go.Scatter(x=df.index, y=df[strategy], name=strategy.lower(),
                       marker_color='darkred',
                       ),
            secondary_y=True if secondary_axis_type == 'return' else False,
        )

        fig.add_trace(
            go.Scatter(x=df.index, y=df[f'{s}_{v1}'], name=f'{s}_{v1}'),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df[f'{s}_{v2}'], name=f'{s}_{v2}'),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=df.index, y=np.where(df[f'{s}_position_{v1}_{v2}'] == 1, df[price_type], np.nan),
                name=f'{s}_signal_{v1}_{v2}',
                mode='markers',
                marker=marker_styles[i]['up'],
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=np.where(df[f'{s}_position_{v1}_{v2}'] == -1, df[price_type], np.nan),
                name=f'{s}_signal_{v1}_{v2}',
                mode='markers',
                marker=marker_styles[i]['down']
                #                 showlegend=False,
            ),
            secondary_y=False,
        )
        i += 1

    # Add figure title
    fig.update_layout(
        title_text=f"<b>{ticker}</b> {price_type} and {secondary_axis_type} Time Series Plot"
    )

    rangebreaks = [
        dict(bounds=["sat", "mon"]),  # hide weekends
        #             dict(bounds=[16, 9.5], pattern="hour"),
        #         dict(values=["2015-12-25", "2016-01-01"])  # hide Christmas and New Year's
        #             dict(values=list(df.index),dvalue=1000*60*60*9.5),  # hide until 9:30am
    ]
    if any(x == interval[-1] for x in 'hm'):
        if extended_hours:
            rangebreaks += [dict(bounds=[20, 4], pattern="hour"), ]
        else:
            rangebreaks += [dict(bounds=[16, 9.5], pattern="hour"), ]

    fig.update_xaxes(
        title_text="Datetime",
        rangebreaks=rangebreaks,
    )

    # Set x-axis title
    #     fig.update_xaxes(title_text="Datetime")

    # Set y-axes titles
    fig.update_yaxes(title_text=f"<b>primary</b> {price_type}", secondary_y=False)
    fig.update_yaxes(title_text=f"<b>secondary</b> {secondary_axis_type}", secondary_y=True)

    #     fig.show()
    return fig


def rs_get_crypto_portfolio():
    return getLatestData(cryptoTable)


def rs_get_portfolio_equity(info=None):
    df = getLatestRowData(portfolioSummaryTable, value='sheryl', key='username')

    df['crypto_gain'] = df['crypto_equity'] - df['crypto_equity_prev_close']
    df['crypto_pct_change'] = df['crypto_gain'] / df['crypto_equity_prev_close']

    df['equity_gain'] = df['equity_latest'] - df['equity_prev_close']
    df['equity_pct_change'] = df['equity_latest'] / df['equity_prev_close'] - 1

    df['total_equity'] = df['crypto_equity'] + df['equity_latest']
    df['total_gain'] = df['equity_gain'] + df['crypto_gain']
    df['total_pct_change'] = df['total_gain'] / df['total_equity']

    return df[info].loc[0] if info else df


def rs_get_portfolio():
    df = getLatestDataNamed(portfolioTable)
    return df


def rs_get_option_portfolio():
    df = getLatestData(optionsTable, key='ticker', time_zone='US/Eastern', view_expired_options=False)
    df['name'] = df[['ticker', 'strike_price', 'option_type', 'exp_date']].apply(
        lambda x: f"{x[0]} {x[1]} {x[2]} exp {x[3]}", axis=1)
    return df


def rs_calc_portfolio(df, option=False, info=None):
    quantity_multiplier = 100 if option else 1
    df['pct_change'] = df['latest_price'] / df['prev_close_price'] - 1

    df['mkt_value'] = df['quantity'] * df['latest_price'] * quantity_multiplier
    df['weight'] = (df['mkt_value'] / df['mkt_value'].sum()).map(lambda x: "{:.2%}".format(x))

    df['day_gain'] = (df['latest_price'] - df['prev_close_price']) * df['quantity'] * quantity_multiplier
    df['total_gain'] = (df['latest_price'] - df['average_buy_price']) * df['quantity'] * quantity_multiplier
    df['total_pct_gain'] = df['latest_price'] / df['average_buy_price'] - 1

    return df[info].loc[0] if info else df


def rs_calc_agg_portfolio():
    agg_cols = ['name', 'ticker', 'average_buy_price', 'quantity', 'prev_close_price', 'latest_price', ]

    df = rs_get_portfolio()
    if df is False:
        return df
    else:
        df = df[agg_cols]
    df = rs_calc_portfolio(df)

    o_df = rs_get_option_portfolio()[agg_cols]
    o_df = rs_calc_portfolio(o_df, option=True)

    c_df = rs_get_crypto_portfolio()[agg_cols]
    c_df = rs_calc_portfolio(c_df)

    agg_df = pd.concat([o_df, c_df, df], keys=['options', 'crypto', 'equities'],
                       names=['Series name', 'Row ID'])

    #     agg_df['weight'] = (agg_df['mkt_value']/agg_df['mkt_value'].sum()).map(lambda x:"{:.2%}".format(x))
    agg_df['weight'] = agg_df['mkt_value'] / agg_df['mkt_value'].sum()
    #     agg_df['total_gain'] = agg_df['total_gain'].apply(lambda x:"{:.2f}".format(x))
    #     agg_df['mkt_value'] = agg_df['mkt_value'].apply(lambda x:"{:.2f}".format(x))
    #     agg_df['average_buy_price'] = agg_df['average_buy_price'].apply(lambda x:"{:.2f}".format(x))

    return agg_df[['name', 'ticker', 'quantity', 'average_buy_price', 'mkt_value',
                   'weight',
                   'day_gain', 'pct_change',
                   'total_gain', 'total_pct_gain', 'latest_price', ]]


## graph portfolio


def rs_calculations(df,price_type,period):
    start_price = df[price_type][0]
    df[f'hold_{period}_gain'] = df[price_type]-start_price
    df[f'hold_{period}_return'] = df[f'hold_{period}_gain']/start_price
    return df


def resample_backfill_calc(df, interval='5m', period='1mo', price_type='', extended_hours=False):
    start_date = str(convert_period(period).date())

    # reindex to fill in values for missing time periods
    resample_interval = interval.replace('m', 'Min').replace('d', 'B')
    if 'h' in interval:
        # base = 0.5
        offset = '0.5h'
        freq = '30min'
    else:
        # base = 0
        offset = '0'
        freq = resample_interval
    if extended_hours:
        interval_start_time = '04:00'
        interval_end_time = '19:59'
    else:
        interval_start_time = '09:30'
        interval_end_time = '15:59'

    # df = df.resample(resample_interval,base=base).last()
    df = df.resample(resample_interval, offset=offset).last()
    current_end_time = df.index[-1]
    #     .between_time('9:31', '15:59')

    idx = pd.date_range(start_date, str(datetime.today()), freq=freq, tz='US/Eastern')
    df = df.reindex(idx, fill_value=np.NaN)
    df = df.replace(to_replace=np.NaN, method='ffill')
    df = df.between_time(interval_start_time, interval_end_time)
    df.index.name = 'datetime'
    if price_type:
        df = rs_calculations(df, price_type, period)

    return df[:current_end_time]


def rs_plot(df, interval='5m', period='1mo', primary_axis_type='total_equity', secondary_axis_type='',
            secondary_axis_bar=False, extended_hours=True, title=''):
    start_date = str(convert_period(period).date())

    df = resample_backfill_calc(df,
                                interval=interval,
                                period=period,
                                price_type=primary_axis_type,
                                extended_hours=extended_hours)

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces: primary_axis_type
    fig.add_trace(
        go.Scatter(x=df.index,
                   y=df[primary_axis_type],
                   name=primary_axis_type.lower(),
                   ),
        secondary_y=False,
    )

    fig.update_yaxes(title_text=f"<b>primary</b> {primary_axis_type}", secondary_y=False)

    # Add figure title
    if title:
        title = title + " Time Series Plot"
    else:
        title = f"<b>Portfolio</b> {primary_axis_type} Time Series Plot"
    fig.update_layout(
        title_text=title
    )

    if secondary_axis_type:
        if secondary_axis_bar:
            fig.add_trace(
                go.Bar(x=df.index, y=df[secondary_axis_type], name=secondary_axis_type.lower(),
                       marker_color='darkred',
                       ),
                secondary_y=True,
            )
        else:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[secondary_axis_type], name=secondary_axis_type.lower(),
                           marker_color='darkred',
                           ),
                secondary_y=True,
            )
        fig.update_yaxes(title_text=f"<b>secondary</b> {secondary_axis_type}", secondary_y=True)

        if not title:
            fig.update_layout(
                title_text=f"<b>Portfolio</b> {primary_axis_type} and {secondary_axis_type} Time Series Plot"
            )

    rangebreaks = [
        dict(bounds=["sat", "mon"]),  # hide weekends
    ]
    if any(x == interval[-1] for x in 'hm'):
        if extended_hours:
            rangebreaks += [dict(bounds=[20, 4], pattern="hour"), ]
        else:
            rangebreaks += [dict(bounds=[16, 9.5], pattern="hour"), ]

    fig.update_xaxes(
        title_text="Datetime",
        rangebreaks=rangebreaks,
    )

    return fig


def rs_plot_portfolio(interval='5m', period='1mo',
                      primary_axis_type='total_equity', secondary_axis_type='', extended_hours=True):
    start_date = str(convert_period(period).date())
    df = getData(portfolioSummaryTable, start_date=start_date, extended_hours=extended_hours)

    df['crypto_gain'] = df['crypto_equity'] - df['crypto_equity_prev_close']
    # df['crypto_pct_change'] = df['crypto_gain']/df['crypto_equity_prev_close']

    df['equity_gain'] = df['equity_latest'] - df['equity_prev_close']
    # df['equity_pct_change'] = df['equity_latest']/df['equity_prev_close']-1

    df['total_equity'] = df['crypto_equity'] + df['equity_latest']
    df['total_gain'] = df['equity_gain'] + df['crypto_gain']
    df['total_pct_change'] = df['total_gain'] / df['total_equity']

    return rs_plot(df, interval=interval, period=period,
                   primary_axis_type=primary_axis_type, secondary_axis_type=secondary_axis_type,
                   extended_hours=extended_hours)


def rs_plot_selection(tickers: [], interval='5m', period='1mo',
                      primary_axis_type='portfolio_close', secondary_axis_type='', extended_hours=False):
    start_date = str(convert_period(period).date())

    # get latest quantity from portfolio table
    df_quantity = getLatestData(portfolioTable, column='ticker,quantity,latest_price', tickers=tickers)
    df_quantity = df_quantity.set_index('ticker')
    quantity_dict = df_quantity['quantity'].to_dict()

    # get price history from equities table (more stable)
    df = getData(equitiesTable, tickers=tickers, start_date=start_date, extended_hours=extended_hours)
    df['quantity'] = df['ticker'].apply(lambda x: quantity_dict[x])
    df['portfolio_close'] = df['close'] * df['quantity']
    df_agg = df.groupby('ticker').apply(lambda x: resample_backfill_calc(x,
                                                                         interval,
                                                                         period,
                                                                         price_type=primary_axis_type,
                                                                         extended_hours=extended_hours))
    df_sum = df_agg.drop('ticker', axis=1).reset_index().groupby('datetime').sum()
    # recalculate hold returns instead of simple sum
    df_sum = rs_calculations(df_sum, price_type=primary_axis_type, period=period)

    col_name = f"hold_{period}_return"
    title = f'<b>Total {df_sum[col_name][-1]:.2%}:</b> '
    df_quantity['weight'] = df_quantity[
                                'quantity'] * df_quantity['latest_price'] / sum(
        df_quantity['quantity'] * df_quantity['latest_price'])
    for ticker in df_quantity.index:
        weight = df_quantity.loc[ticker]['weight']
        title += f'{ticker} ({weight:.2%}),'

    fig = rs_plot(df_sum, interval=interval, period=period,
                  primary_axis_type=primary_axis_type, secondary_axis_type=secondary_axis_type,
                  secondary_axis_bar=False,
                  extended_hours=extended_hours,
                  title=title)

    annotations = []
    for ticker in df_quantity.index:
        df_ticker = df_agg.loc[ticker]
        # Add traces
        fig.add_trace(
            go.Scatter(x=df_ticker.index,
                       y=df_ticker[col_name],
                       name=f'{ticker} {period} {df_ticker[col_name][-1]:.2%}',
                       ),
            secondary_y=True,
        )

    fig.update_layout(annotations=annotations)
    return fig

