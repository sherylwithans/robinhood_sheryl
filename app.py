import dash
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output

import dash_table
import dash_table.FormatTemplate as FormatTemplate
from dash_table.Format import Format, Sign
import dash_bootstrap_components as dbc

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

holdings_df = rs_calc_agg_portfolio()

def yf_backtest_wrapper(ticker):
    s = 'ema'
    (w1,w2) = (20,50)
    interval = '5m'
    period = '1mo'
    # print(ticker)
    df = yf_backtest(ticker,interval=interval,period=period,signals={s:(w1,w2)},results=True,long_only=False)
    df.columns = [x.replace(f'_{w1}_{w2}','') for x in df.columns]
    cols =['close', f'{s}_execute_order', f'{s}_execute_price',
       f'{s}_execute_time', f'{s}_l_cumu_return', 'strategy_ratio',
       f'hold_{period}_return']
    if df.empty:
#         cols = ['close',f'{s}_execute_order', f'{s}_execute_time',
#                f'hold_{period}_return', f'{s}_l_cumu_return']
        return pd.Series([np.nan]*len(cols),index=cols)
    df = df[cols]
    return pd.Series(df.values[0],index=df.columns)

print('generating initial portfolio analytics df...')
pa_df = pd.concat([holdings_df,holdings_df['ticker'].apply(lambda x: yf_backtest_wrapper(x))],axis=1)


def format_columns(df):
    columns = df.columns
    type_dict = dict(df.iloc[0].map(lambda x: 'text' if isinstance(x,str) else 'numeric'))
    col_styles = [{"name": i, "id": i, "type": type_dict[i]} for i in columns]
    format_dict = {'price':FormatTemplate.money(0), 'pct':FormatTemplate.percentage(1).sign(Sign.positive)}
    for i in col_styles:
        col_name = i['name']
        if any([x in col_name for x in ('pct','return')]):
            i['format'] = FormatTemplate.percentage(2).sign(Sign.positive)
        elif any([x in col_name for x in ('price','value','gain','wealth','cash')]):
            i['format'] = FormatTemplate.money(2)
        elif 'weight' in col_name:
            i['format'] = FormatTemplate.percentage(2)
        elif 'quantity' in col_name:
            i['format'] = Format(
                precision=4,
#                 scheme=Scheme.fixed,
#                 symbol=Symbol.yes,
#                 symbol_suffix=u'˚F'
            )

    return col_styles


def format_red_green(columns):
    styles = []
    for c in columns:
        if any([x in c for x in ('pct', 'gain')]):
            styles.append({
                    'if': {
                        'filter_query': f'{{{c}}} < 0',
        #                 'filter_query': '{pct_change} > 0 && {Humidity} < 41',
                        'column_id': c
                    },
                    'color': 'tomato',
                    'fontWeight': 'bold'
                })

            styles.append({
                    'if': {
                        'filter_query': f'{{{c}}} >= 0',
                        'column_id': c
                    },
                    'color': '#3D9970',
                    'fontWeight': 'bold'
                })

        if 'pct' in c:
            styles.append({
                    'if': {
                        'filter_query': f'{{{c}}} >= 0.05',
                        'column_id': c
                    },
#                     'color': 'white',
                    'backgroundColor': '#D7F2CE'
                })

        if c=='latest_price':
            styles.append({
                    'if': {
                        'filter_query': f'{{{c}}} <= {{average_buy_price}}',
                        'column_id': c
                    },
#                     'color': 'white',
                    'backgroundColor': '#B8DFE6'
                })


    return styles

def format_red_green_text(columns):
    styles = []
    for c in columns:
        if any([x in c for x in ['execute_order']]):
            styles.append({
                    'if': {
                        'filter_query': f'{{{c}}} ="buy"',
                        'column_id': c
                    },
                    'color': '#3D9970',
                    'fontWeight': 'bold'
                })
            styles.append({
                    'if': {
                        'filter_query': f'{{{c}}} ="buy"'+
                            ' && {latest_price} <= {ema_execute_price}'+
                            ' && !({name} contains "call") && !({name} contains "put")',
                        'column_id': c
                    },
                    'color': 'white',
                    'backgroundColor': '#3D9970'
                })
            styles.append({
                    'if': {
                        'filter_query': f'{{{c}}} ="sell"',
                        'column_id': c
                    },
                    'color': 'tomato',
                    'fontWeight': 'bold'
                })
            styles.append({
                    'if': {
                        'filter_query': f'{{{c}}} ="sell"'+
                        ' && {latest_price} >= {ema_execute_price}'+
                        ' && !({name} contains "call") && !({name} contains "put")',
                        'column_id': c
                    },
                    'color': 'white',
                    'backgroundColor': 'tomato'
                })

    return styles

def format_bold(columns):
    styles = []
    for c in columns:
        if any([x in c for x in ['latest']]):
            styles.append({
                'if': {
                        'column_id': c
                    },
                    'fontWeight': 'bold'
                })

    return styles


def data_bars(df, column):
    n_bins = 100
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    ranges = [
        ((df[column].max() - df[column].min()) * i) + df[column].min()
        for i in bounds
    ]
    styles = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        max_bound_percentage = bounds[i] * 100
        styles.append({
            'if': {
                'filter_query': (
                    '{{{column}}} >= {min_bound}' +
                    (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                'column_id': column
            },
            'background': (
                """
                    linear-gradient(90deg,
                    #0074D9 0%,
                    #0074D9 {max_bound_percentage}%,
                    white {max_bound_percentage}%,
                    white 100%)
                """.format(max_bound_percentage=max_bound_percentage)
            ),
            'paddingBottom': 2,
            'paddingTop': 2,
        })

    return styles

def format_bg_color():
    styles = []
    styles.append({
        'if': {'row_index': 'odd'},
        'backgroundColor': 'rgb(248, 248, 248)'
    })
    return styles


def data_bars_diverging(df, column, color_above='#3D9970', color_below='#FF4136'):
    n_bins = 100
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]

#     col_max = df[column].max()
#     col_min = df[column].min()
    col_max = max(abs(df[column].max()),abs(df[column].min()))
    col_min = -col_max
    ranges = [
        ((col_max - col_min) * i) + col_min
        for i in bounds
    ]
    midpoint = (col_max + col_min) / 2.
#     midpoint = 0

    styles = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        min_bound_percentage = bounds[i - 1] * 100
        max_bound_percentage = bounds[i] * 100

        style = {
            'if': {
                'filter_query': (
                    '{{{column}}} >= {min_bound}' +
                    (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                'column_id': 'name'
            },
            'paddingBottom': 2,
            'paddingTop': 2
        }
        if max_bound > midpoint:
            background = (
                """
                    linear-gradient(90deg,
                    white 0%,
                    white 50%,
                    {color_above} 50%,
                    {color_above} {max_bound_percentage}%,
                    white {max_bound_percentage}%,
                    white 100%)
                """.format(
                    max_bound_percentage=max_bound_percentage,
                    color_above=color_above
                )
            )
        else:
            background = (
                """
                    linear-gradient(90deg,
                    white 0%,
                    white {min_bound_percentage}%,
                    {color_below} {min_bound_percentage}%,
                    {color_below} 50%,
                    white 50%,
                    white 100%)
                """.format(
                    min_bound_percentage=min_bound_percentage,
                    color_below=color_below
                )
            )
        style['background'] = background
        styles.append(style)

    return styles


# the style arguments for the main content page.
CONTENT_STYLE = {
    'margin-left': '1.5%',
    'margin-right': '1.5%',
    'padding': '1px 0.5p'
}

TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#191970'
}

CARD_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#0074D9',
    'font-size': '15px'
}



def format_cards(card_title_id, card_title_text, card_subtitle_id, card_text_id, button_id=None):
    elements = [
                html.H2(id=card_title_id, children=[card_title_text], className='card-title',
                        style=CARD_TEXT_STYLE),
                html.H4(id=card_subtitle_id, children=['Sample subtitle'], className="card-subtitle",
                        style=CARD_TEXT_STYLE),
                html.P(id=card_text_id, children=['Sample text.'], style=CARD_TEXT_STYLE),
                ]
    if button_id:
        button = dbc.Button("Go somewhere", id=button_id, color="primary", size="sm", className="mt-auto")
        elements += [button]
    card = dbc.Card(
                [dbc.CardBody(elements)]
            )
    return card

def format_graphs(graph_id):
    return dcc.Graph(id=graph_id,style={'height': '500px'})

def format_radio_buttons(radio_id,values_dict):
    options = []
    for k,v in values_dict.items():
        options.append({'label':v,'value':k})
    return dcc.RadioItems(id = radio_id,
                options=options,
                    value=list(values_dict.keys())[0],
                    labelStyle={'display': 'block'},
                )

# content format
def format_content():

    content_first_row = dbc.Row([
        dbc.CardGroup(
            [
                format_cards('futures_title_1','Nasdaq Futures','futures_subtitle_1','futures_text_1'),
            ]
        ),
        dbc.CardGroup(
            [
                format_cards('index_title_1','Nasdaq','index_subtitle_1','index_text_1'),
                format_cards('index_title_2','S&P500','index_subtitle_2','index_text_2'),
                format_cards('index_title_3','Dow 30','index_subtitle_3','index_text_3'),
                format_cards('index_title_4','Vix','index_subtitle_4','index_text_4'),
                format_cards('index_title_5','Russell 2000','index_subtitle_5','index_text_5'),
                format_cards('index_title_6','FTSE','index_subtitle_6','index_text_6'),
                format_cards('index_title_7','Nikkei 255','index_subtitle_7','index_text_7'),
            ]
        ),
        dbc.CardGroup(
            [
                format_cards('portfolio_title_1','Portfolio','portfolio_subtitle_1','portfolio_text_1'),
                format_cards('portfolio_title_2','Bitcoin','portfolio_subtitle_2','portfolio_text_2'),
            ]
        ),

    ])

    content_second_row = dbc.Row([
        dbc.Col(
            format_graphs('graph_1'),
            width=11),
        dbc.Col([
            dbc.Row(format_radio_buttons('radio_1',{'1mo':'1 month','5d': '1 week', '1d':'1 day'}),
#             width=1,
                align='center'),
            dbc.Row(format_radio_buttons('radio_2',
                                         {'5m':'5 minutes',
                                          '15m':'15 minutes',
                                          '30m':'30 minutes',
                                          '1h':'1 hour',
                                         }),
                align='center'),
            dbc.Row(format_radio_buttons('radio_3',
                                         {'portfolio':'portfolio',
                                          'backtest':'backtest',
                                          'moving_average':'historical',

                                         }),
                align='center'),
            ],
            align='center',
        ),
#         dbc.Col(
#             format_graphs('graph_3'),
#             md=4)
    ])


    content = html.Div(
        [
    #         html.H2('Analytics Dashboard Template', style=TEXT_STYLE),
            html.Hr(),
            content_first_row,
            content_second_row,
    #         content_third_row,
    #         content_fourth_row
        ],
        style=CONTENT_STYLE
    )
    return content


def format_table(df):
    columns = df.columns
    table = dash_table.DataTable(
        id='table',
        # allows wrap lines
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
            'maxWidth': '30px'
        },
        columns=format_columns(df),
        data=df.to_dict('records'),
        fixed_rows={'headers': True, 'data': 0},
#         fixed_columns={'headers': True,'data': 2},
#         style_table={'minWidth': '100%'}, #needed after fix row/column
        style_data_conditional=
            data_bars(df,'weight') +
            data_bars_diverging(df,'day_gain') +
            format_bg_color() +
            format_red_green(columns) +
            format_red_green_text(columns) +
            format_bold(columns) +
            [{'if': {'column_id': 'name'},'width': '10%'}]
            ,
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
            },
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable="single",
        row_selectable="single",
        selected_rows=[],
        page_action='native',
    )
    return table


app.layout = html.Div([
    html.H4(id='welcome_msg'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),
    
    html.Div(id='nasdaq_futures'),
    
    format_content(),
    format_table(pa_df),
    
    dcc.Interval(
            id='interval-component',
            interval=120*1000, # in milliseconds
            n_intervals=0,   
        )
    ],

    style = CONTENT_STYLE
)

@app.callback(
    Output('table','data'),
    [Input('interval-component', 'n_intervals')])
def generate_table(n):
    print("generating holdings df...")
    holdings_df = rs_calc_agg_portfolio()
#     cols_list = ['average_buy_price','quantity','day_gain','pct_change','total_gain','total_pct_gain']
#     holdings_df[cols_list] = holdings_df[cols_list].apply(lambda x: random.uniform(-2,2)*x ,axis=1)
#     pa_df = pd.concat([holdings_df,holdings_df.apply(yf_backtest_wrapper,axis=1)],axis=1)
    print("generating portfolio analytics df...")
    pa_df = pd.concat([holdings_df,holdings_df['ticker'].apply(lambda x: yf_backtest_wrapper(x))],axis=1)
    
    return pa_df.to_dict('records')

@app.callback(
    Output('welcome_msg','children'),
    [Input('interval-component', 'n_intervals')])
def generate_today(n):
    today = datetime.today().strftime('%a %H:%M:%S %Y-%m-%d')
    welcome_msg = f"Hello Sheryl, it's {today} today"
    return welcome_msg


# card 1
@app.callback(
    [Output('futures_subtitle_1','children'),
    Output('futures_text_1','children')],
    [Input('interval-component', 'n_intervals')])
def generate_card_1(n):
    d = ws_get_nasdaq_futures_info()
    result = d['latest_price']
    pct = d['pct_change']
    return result,pct

# card 2-8 values
index_output_list = []
index_items_list = list(enumerate(INDEXES.items()))
for i,idx in index_items_list:
    index_output_list += [Output(f'index_subtitle_{i+1}','children'),Output(f'index_text_{i+1}','children')]

# card 2-8 function
@app.callback(
    index_output_list,
    [Input('interval-component', 'n_intervals')])
def generate_cards(n):
    results=[]
    for i,(k,v) in index_items_list:
        result = yf_get_latest_price(v)
        pct = yf_get_pct_change(v)
        sign = '+' if pct>=0 else ''
        results.extend([f"{result:,.2f}",f"{sign}{pct:.2%}"])
    return(results)


# card 9
@app.callback(
    [Output('portfolio_subtitle_1','children'),
    Output('portfolio_text_1','children')],
    [Input('interval-component', 'n_intervals')])
def generate_card_5(n):
    result = rs_get_portfolio_equity('total_equity')
    pct = rs_get_portfolio_equity('total_pct_change')
    gain = rs_get_portfolio_equity('total_gain')
    sign = '+' if pct>=0 else ''
    return f"{result:,.2f}",f"{sign}{pct:.2%} {sign}{gain:.2f}"


# card 10
@app.callback(
    [Output('portfolio_subtitle_2','children'),
    Output('portfolio_text_2','children')],
    [Input('interval-component', 'n_intervals')])
def generate_card_6(n):
    c_df = rs_get_crypto_portfolio()
    result = rs_calc_portfolio(c_df,info='latest_price')
    pct = rs_calc_portfolio(c_df,info='pct_change')
    gain = rs_calc_portfolio(c_df,info='day_gain')
    sign = '+' if pct>=0 else ''
    return f"{result:,.2f}",f"{sign}{pct:.2%} {sign}{gain:.2f}"

# # button
# @app.callback(
#     Output("button_5", "figure"), [Input("button_5", "n_clicks")]
# )
# def on_button_click(n):
#     if n is None:
#         print("Not clicked.")
#     else:
#         print(f"Clicked {n} times.")

# graph 1
@app.callback(
    Output('graph_1', 'figure'),
    [
        Input('table', 'derived_virtual_data'),
        Input('table', 'derived_virtual_selected_rows'),  # derived view of filtered data
        Input('radio_1','value'),
        Input('radio_2','value'),
        Input('radio_3','value'),
#         Input("button_9", "n_clicks"),
    ]
)
def update_graphs(rows,derived_virtual_selected_rows,radio_period,radio_interval,radio_type):
#     if n is None:
#         print("Not clicked.")
#     else:
#         print(f"Clicked {n} times.")
#         n=None
    if derived_virtual_selected_rows is None or derived_virtual_selected_rows == []:
        derived_virtual_selected_rows = [0]
#         radio_type='portfolio'
#     print(derived_virtual_selected_rows)
    
    dff = pa_df if rows is None else pd.DataFrame(rows)

    ticker = dff['ticker'].iloc[derived_virtual_selected_rows[0]]
#     print(derived_virtual_selected_rows)
#     print(ticker)
    extended_hours = True if datetime.now().hour>=16 else False
    if radio_type=='portfolio':
        fig = rs_plot_portfolio(interval=radio_interval,period=radio_period,
                                primary_axis_type='total_equity', secondary_axis_type='',extended_hours=extended_hours)
    elif extended_hours or radio_type=='moving_average':
        fig = yf_plot_moving_average(ticker,interval=radio_interval,period=radio_period,windows=[20,50],
                                     signals={'ema':(20,50)},bband=False,extended_hours=extended_hours)
    else:
        fig = yf_plot_backtest(ticker,interval=radio_interval,period=radio_period,
                           signals={'ema':(20,50)},secondary_axis_type='return',long_only=False)
    return fig

# style
@app.callback(
    Output('table', 'style_data_conditional'),
    [Input('table', 'derived_virtual_data'),
     Input('table', 'derived_virtual_selected_rows')]
)
def update_styles(rows,derived_virtual_selected_rows):
    if derived_virtual_selected_rows is None or derived_virtual_selected_rows == []:
        derived_virtual_selected_rows = [0]
    dff = pa_df if rows is None else pd.DataFrame(rows)
    columns = dff.columns
        
    style = [{
        'if': { 'row_index': i },
        'background_color': '#D2F3FF'
    } for i in derived_virtual_selected_rows]
    
    style_data_conditional= \
            data_bars(dff,'weight') + \
            data_bars_diverging(dff,'day_gain') +\
            style +\
            format_bg_color() +\
            format_red_green(columns) + \
            format_red_green_text(columns) +\
            format_bold(columns) + \
            [{'if': {'column_id': 'name'},'width': '10%'}]
    
    return style_data_conditional


if __name__ == '__main__':
#    holdings_df = rs_calc_agg_portfolio()
#    pa_df = pd.concat([holdings_df,holdings_df['ticker'].apply(lambda x: yf_backtest_wrapper(x))],axis=1)
    app.run_server(debug=True,port=8050)

