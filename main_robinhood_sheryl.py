from robinhood_sheryl.rs_db import *

if __name__ == "__main__":
    login()  # location of robinhood login pickle file
    # insert_yf_data()
    # print(getLatestData(equitiesTable))
    insert_portfolio_data()
    # print(pd.isnull(None))