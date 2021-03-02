from robinhood_sheryl.rs_db import *


if __name__ == "__main__":

    login()  # location of robinhood login pickle file
    # # insert_yf_data()
    # insert_orders_data()
    # print(getLatestData(ordersTable))
    # portfolio_df = get_portfolio_data()
    # df = getData(tickersTable)
    update_portfolio_ticker_symbols('SBE','CHPT')

    # print(df[df['ticker']=='CHPT'])
    # initTables()

    # insert_portfolio_data()
    # print(pd.isnull(None))
