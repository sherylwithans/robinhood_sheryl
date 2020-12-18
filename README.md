# robinhood_sheryl

robinhood_sheryl is a dashboard that provides a comprehensive overview of an investor's portfolio holdings extracted from the robinhood api. It also incorporates investment suggestions based on simple statistics such as exponential moving averages and momentum indicators. Data is stored in postgres on google cloud instance, with updates in 5-min intervals scheduled in Airflow. Options analytics function in progress.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install sqlalchemy for database import and robin_stocks for robinhood api.

```bash
# db installs
pip install sqlalchemy
pip install robin_stocks

# analytics installs
pip install jupyter_dash
pip install plotly
pip install dash
pip install yfinance
pip install requests
```

## Usage

```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```

## Contributing
For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[tbdlicense](https://nolicenseyet.com/licenses/butiwantone/)
