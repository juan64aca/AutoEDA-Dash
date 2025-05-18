import base64
import io
import statistics

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from scipy import stats

from dash.exceptions import PreventUpdate
from dash_extensions.enrich import (
    DashProxy,
    Output,
    Input,
    State,
    Serverside,
    html,
    dcc,
    ServersideOutputTransform,
    dash_table,
)

from exceptions import InvalidFileException

# Initialize the Dash app
app = DashProxy(transforms=[ServersideOutputTransform()])
app.config.suppress_callback_exceptions = True

# Styles
style_tab = {'background-color': 'darkslategray', 'color': 'white', 'border': '1px solid #ccc', 'border-radius': '5px'}
style_tab_selected = {'font-weight': 'bold'}
style_missing_load = {'background-color': '#ffe4e1', 'padding': '20px', 'border': '1px solid #ff69b4',
                      'border-radius': '5px', 'text-align': 'center', 'margin': '20px auto', 'width': '50%'}
font = "'Open Sans', sans-serif"
style_output = {'border': '1px solid gray', 'color': 'black', 'background': 'lightgray', 'border-radius': '2px',
                'margin': '5px', 'paddingLeft': '10px'}

missing_load = html.Div([
    html.H1('Load a file first.', style=style_missing_load),
    html.Center(
        html.Img(src='assets/eda.png')
    )
])

# Layout of the app
app.layout = html.Div(
    style={'fontFamily': font},
    children=[
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=False
        ),
        dcc.Tabs(
            id='tabs',
            children=[
                # Data summary
                dcc.Tab(
                    id='summary-tab',
                    label='Data Summary',
                    children=[missing_load],
                    style=style_tab,
                    selected_style=style_tab_selected
                ),
                # Data analysis
                dcc.Tab(
                    label='Data Analysis',
                    children=[
                        html.Div(id='analysis-variable', children=[missing_load]),
                        html.Div(id='analysis-dropdown-container'),
                        html.Div(id='analysis-structure'),
                        html.Div(id='analysis-structure-cont'),
                        html.Div(id='analysis-summary'),
                        html.Div(id='analysis-summary-cont'),
                        html.Div(id='analysis-detail'),
                        html.Div(id='analysis-detail-cont')
                    ],
                    style=style_tab,
                    selected_style=style_tab_selected
                ),
                # Important plots
                dcc.Tab(
                    id='plots-tab',
                    label='Important Plots',
                    children=[missing_load],
                    style=style_tab,
                    selected_style=style_tab_selected
                ),
                # Statistics
                dcc.Tab(
                    id='stats-tab',
                    label='Statistics',
                    children=[missing_load],
                    style=style_tab,
                    selected_style=style_tab_selected
                ),
                # Correlation Matrix
                dcc.Tab(
                    id='corr-tab',
                    label='Correlation Matrix',
                    children=[missing_load],
                    style=style_tab,
                    selected_style=style_tab_selected
                ),
                # Regression
                dcc.Tab(
                    id='reg-tab',
                    label='Regression',
                    children=[
                        html.Div(id='regression-tab', children=[missing_load])
                    ],
                    style=style_tab,
                    selected_style=style_tab_selected
                ),
            ]
        ),
        dcc.Loading(dcc.Store(id="store"), fullscreen=True, type="dot")
    ]
)


@app.callback(
    Output('store', 'data'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')],
    prevent_initial_call=True
)
def read_file(contents, filename):
    if contents is None:
        raise PreventUpdate

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), sep=None, engine='python')
        elif 'xls' in filename:
            # Assume that the user uploaded an Excel file
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            raise InvalidFileException(f'File should be an xlsx or a csv file')

        return Serverside(df)
    except Exception:
        raise InvalidFileException(f'Error reading the uploaded file')


@app.callback(
    Output('summary-tab', 'children'),
    Input('store', 'data'),
    prevent_initial_call=True)
def create_resumen(data):
    if data is None:
        raise PreventUpdate

    num_vars = data.select_dtypes(include=['number']).columns.tolist()
    non_num_vars = data.select_dtypes(exclude=['number']).columns.tolist()

    num_list = '\n'.join(num_vars)
    non_num_list = '\n'.join(non_num_vars)

    component_table = dash_table.DataTable(
        data=data.to_dict('records'),
        page_size=20,
        sort_action='native',
        style_table={
            'overflowX': 'auto',  # Use 'auto' for horizontal overflow to handle content properly
            'overflowY': 'auto',  # Use 'auto' for vertical overflow to handle content properly
            'maxHeight': '500px',  # Set a maximum height for the table
            'border': 'thin lightgrey solid',  # Add a light grey border to the table
        },
        style_cell={
            'padding': '10px',  # Add padding to cells
            'fontFamily': font,  # Set the font family
            'textAlign': 'left',  # Align text to the left
            'minWidth': '80px', 'width': '150px', 'maxWidth': '200px',  # Set cell width
            'whiteSpace': 'normal',  # Allow text to wrap in cells
        },
        style_header={
            'backgroundColor': 'lightgrey',  # Set the background color of the header
            'fontWeight': 'bold',  # Make header text bold
            'textAlign': 'center',  # Center align header text
            'border': 'thin lightgrey solid',  # Add a border to the header
        },
        style_data={
            'border': 'thin lightgrey solid',  # Add a border to data cells
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},  # Apply styling to odd rows
                'backgroundColor': 'rgb(248, 248, 248)',  # Set background color for odd rows
            }
        ],
        style_as_list_view=True,  # Display the table in list view style
    )

    component = html.Div([
        html.H4(html.Strong('Resumen de datos')),
        html.P(
            'En esta pestaña se muestra un resumen de los datos cargados, incluyendo información sobre las variables numéricas y no numéricas.'),
        html.H4(html.Strong('Listado de variables numéricas:')),
        html.Div(
            html.Pre(num_list),
            style=style_output
        ),
        html.H4(html.Strong('Listado de variables no numéricas:')),
        html.Div(
            html.Pre(non_num_list),
            style=style_output
        ),
        html.H4(html.Strong('Tabla de datos:')),
        component_table
    ])

    return component


@app.callback(
    [Output('analysis-variable', 'children'),
     Output('analysis-dropdown-container', 'children')],
    Input('store', 'data'),
    prevent_initial_call=True)
def create_dropdown_analisis(data):
    if data is None:
        raise PreventUpdate

    columnas = data.columns
    columna = columnas[0]

    variable_component = html.H4(html.Strong('Variable:')),

    dropdown_component = dcc.Dropdown(
        id='analisis-dropdown',
        options=[{'label': col, 'value': col} for col in columnas],
        value=columna
    )

    return variable_component, dropdown_component


@app.callback(
    [Output('analysis-structure', 'children'),
     Output('analysis-structure-cont', 'children'),
     Output('analysis-summary', 'children'),
     Output('analysis-summary-cont', 'children'),
     Output('analysis-detail', 'children'),
     Output('analysis-detail-cont', 'children')],
    [Input('store', 'data'),
     Input('analisis-dropdown', 'value')],
    prevent_initial_call=True
)
def update_analysis(data, selected_column):
    if data is None:
        raise PreventUpdate

    if selected_column is None:
        raise PreventUpdate

    var_repr = repr(data[selected_column].head(10))
    describe_repr = repr(data[selected_column].describe())
    buffer = io.StringIO()
    data[selected_column].info(buf=buffer)
    info_repr = buffer.getvalue()

    analisis_estructura = html.H4("Estructura de la variable:")
    analisis_estructura_cont = html.Div(
        html.Pre(var_repr),
        style=style_output
    )
    analisis_resumen = html.H4('Resumen de la variable:')
    analisis_resumen_cont = html.Div(
        html.Pre(describe_repr),
        style=style_output
    )
    analisis_detalle = html.H4('Resumen detallado de la variable:')
    analisis_detalle_cont = html.Div(
        html.Pre(info_repr),
        style=style_output
    )

    return analisis_estructura, analisis_estructura_cont, analisis_resumen, analisis_resumen_cont, \
        analisis_detalle, analisis_detalle_cont


@app.callback(
    Output('plots-tab', 'children'),
    Input('store', 'data'),
    prevent_initial_call=True)
def create_tab_plots(data):
    if data is None:
        raise PreventUpdate

    num_vars = data.select_dtypes(include=['number']).columns.tolist()
    non_num_vars = data.select_dtypes(exclude=['number']).columns.tolist()

    plots_component = html.Div([
        html.H4(html.Strong('Gráficos:')),
        html.P(
            "En esta pestaña se generan gráficos para explorar las variables numéricas y no numéricas del conjunto de datos."),
        dcc.Tabs(children=[
            # Variables numericas
            dcc.Tab(label='Variables numéricas', children=[
                dcc.Dropdown(
                    id='plots-dropdown-numericas',
                    options=[{'label': col, 'value': col} for col in num_vars],
                    value=num_vars[0]
                ),
                dcc.Loading(
                    type="circle",
                    children=html.Div(id='num-graph')
                )
            ]),
            dcc.Tab(label='Variables no numéricas', children=[
                dcc.Dropdown(
                    id='plots-dropdown-no-numericas',
                    options=[{'label': col, 'value': col} for col in non_num_vars],
                    value=non_num_vars[0]
                ),
                dcc.Loading(
                    type="circle",
                    children=html.Div(id='non-num-graph')
                )
            ])
        ])
    ])

    return plots_component


@app.callback(
    Output('num-graph', 'children'),
    [Input('store', 'data'),
     Input('plots-dropdown-numericas', 'value')]
)
def update_num_graph(data, selected_column):
    if data is None:
        raise PreventUpdate

    if selected_column is None:
        raise PreventUpdate

    chart_data = data[selected_column]
    fig_hist = px.histogram(chart_data)
    fig_box = px.box(x=chart_data)

    component = html.Div([
        dcc.Graph(figure=fig_hist),
        dcc.Graph(figure=fig_box)
    ])

    return component


@app.callback(
    Output('non-num-graph', 'children'),
    [Input('store', 'data'),
     Input('plots-dropdown-no-numericas', 'value')]
)
def update_non_num_graph(data, selected_column):
    if data is None:
        raise PreventUpdate

    if selected_column is None:
        raise PreventUpdate

    chart_data = data[selected_column]
    pie_data = chart_data.value_counts().reset_index()
    pie_data.columns = ['Category', 'Frequency']

    fig_hist = px.histogram(chart_data)
    fig_pie = px.pie(pie_data, values='Frequency', names='Category', title='Pie Chart of Category Frequencies')

    component = html.Div([
        dcc.Graph(figure=fig_hist),
        dcc.Graph(figure=fig_pie)
    ])

    return component


@app.callback(
    Output('stats-tab', 'children'),
    Input('store', 'data'),
    prevent_initial_call=True)
def create_tab_stats(data):
    if data is None:
        raise PreventUpdate

    num_vars = data.select_dtypes(include=['number']).columns.tolist()

    component = html.Div([
        html.H4(html.Strong('Statistics')),
        html.P(
            "En esta pestaña se calculan y muestran diferentes medidas estádisticas de las variables númericas."),
        dcc.Tabs(children=[
            # Tendencia central
            dcc.Tab(label='Medidas de tendencia central', children=[
                html.H5(html.Strong('Media')),
                html.P(
                    "La media es el promedio de todos los valores en una variable numérica. Representa el valor central de la distribución de los datos."),
                html.H5(html.Strong('Mediana')),
                html.P(
                    "La mediana es el valor que se encuentra en el medio de una distribución de datos ordenados de menor a mayor. Es útil para describir la tendencia central en datos sesgados."),
                html.H5(html.Strong('Moda')),
                html.P(
                    "La moda es el valor que aparece con mayor frecuencia en un conjunto de datos."),
                html.H5(html.Strong('Seleccione una variable:')),
                dcc.Dropdown(
                    id='medidas-tendencia-dropdown',
                    options=[{'label': col, 'value': col} for col in num_vars],
                    value=num_vars[0]
                ),
                dcc.Loading(
                    type="circle",
                    children=html.Div(id='medidas-tendencia')
                )
            ]),
            # Dispersión
            dcc.Tab(label='Medidas de dispersión', children=[
                html.H5(html.Strong('Varianza')),
                html.P(
                    "La varianza es la esperanza del cuadrado de la desviación estándar de una variable respecto a su media."),
                html.H5(html.Strong('Desviación Estándar')),
                html.P(
                    "La desviación estándar muestra cuánto varían los datos con respecto a la media. Indica la cantidad de dispersión o propagación de los datos."),
                html.H5(html.Strong('Rango')),
                html.P(
                    "El rango es la diferencia entre el valor máximo y el valor mínimo en una variable numérica. Proporciona una medida de la amplitud total de los datos."),
                html.H5(html.Strong('IQR')),
                html.P(
                    "El IQR o rango intercuartílico es la diferencia entre el tercer y el primer cuartil de una distribución. Es una medida de la dispersión estadística."),
                html.H5(html.Strong('Seleccione una variable:')),
                dcc.Dropdown(
                    id='medidas-dispersion-dropdown',
                    options=[{'label': col, 'value': col} for col in num_vars],
                    value=num_vars[0]
                ),
                dcc.Loading(
                    type="circle",
                    children=html.Div(id='medidas-dispersion')
                )
            ]),
            # Asimetría
            dcc.Tab(label='Medidas de asimetría', children=[
                html.H5(html.Strong('Asimetría')),
                html.P(
                    "La asimetría es una medida de la simetría de la distribución de los datos. Puede ser positiva (sesgo a la derecha), negativa (sesgo a la izquierda) o cero (simetría)."),
                html.H5(html.Strong('Seleccione una variable:')),
                dcc.Dropdown(
                    id='medidas-asimetria-dropdown',
                    options=[{'label': col, 'value': col} for col in num_vars],
                    value=num_vars[0]
                ),
                dcc.Loading(
                    type="circle",
                    children=html.Div(id='medidas-asimetria')
                )
            ])
        ])
    ])

    return component


@app.callback(
    Output('medidas-tendencia', 'children'),
    [Input('store', 'data'),
     Input('medidas-tendencia-dropdown', 'value')]
)
def update_medidas_tendencia(data, selected_column):
    if data is None:
        raise PreventUpdate

    if selected_column is None:
        raise PreventUpdate

    filtered_data = data[selected_column]
    media = round(np.mean(filtered_data), 4)
    mediana = round(np.median(filtered_data), 4)
    moda = stats.mode(filtered_data)

    component = html.Div([
        html.Div(
            children=[
                html.P("mean:"),
                html.P(media)],
            style=style_output
        ),
        html.Div(
            children=[
                html.P("median:"),
                html.P(mediana)],
            style=style_output
        ),
        html.Div(
            children=[
                html.P("mode:"),
                html.P(moda)],
            style=style_output
        )
    ])

    return component


@app.callback(
    Output('medidas-dispersion', 'children'),
    [Input('store', 'data'),
     Input('medidas-dispersion-dropdown', 'value')]
)
def update_medidas_dispersion(data, selected_column):
    if data is None:
        raise PreventUpdate

    if selected_column is None:
        raise PreventUpdate

    filtered_data = data[selected_column]

    varianza = round(statistics.variance(filtered_data), 4)
    desv_est = round(statistics.stdev(filtered_data), 4)
    rango = round(np.ptp(filtered_data), 4)
    iqr = round(stats.iqr(filtered_data), 4)

    component = html.Div([
        html.Div(
            children=[
                html.P("variance:"),
                html.P(varianza)],
            style=style_output
        ),
        html.Div(
            children=[
                html.P("standard deviation:"),
                html.P(desv_est)],
            style=style_output
        ),
        html.Div(
            children=[
                html.P("range:"),
                html.P(rango)],
            style=style_output
        ),
        html.Div(
            children=[
                html.P("IQR"),
                html.P(iqr)],
            style=style_output
        )
    ])

    return component


@app.callback(
    Output('medidas-asimetria', 'children'),
    [Input('store', 'data'),
     Input('medidas-asimetria-dropdown', 'value')]
)
def update_medidas_asimetria(data, selected_column):
    if data is None:
        raise PreventUpdate

    if selected_column is None:
        raise PreventUpdate

    filtered_data = data[selected_column]

    asimetria = stats.skew(filtered_data)

    component = html.Div(
            children=[
                html.P("asimetria:"),
                html.P(asimetria)],
            style=style_output
        )

    return component


@app.callback(
    Output('corr-tab', 'children'),
    Input('store', 'data'),
    prevent_initial_call=True)
def create_tab_correlacion(data):
    if data is None:
        raise PreventUpdate

    correlation = data.corr(numeric_only=True)
    correlation = round(correlation, 2)

    fig_corr = px.imshow(correlation, text_auto=True)

    component = html.Div([
        html.H4(html.Strong("Matriz de Correlación")),
        html.P("En esta pestaña se calcula y muestra la matriz de correlación entre las variables numéricas."),
        html.P(
            "La correlación es una medida estadística que indica la relación entre dos variables. Puede variar entre -1 y 1. Un valor cercano a 1 indica una correlación positiva, un valor cercano a -1 indica una correlación negativa y un valor cercano a 0 indica una correlación débil o nula."),
        dcc.Graph(figure=fig_corr, style={'height': '800px', 'width': '100%'})
    ])

    component = dcc.Loading(
        type="circle",
        children=component
    )

    return component


@app.callback(
    Output('regression-tab', 'children'),
    Input('store', 'data'),
    prevent_initial_call=True)
def create_dropdown_regression(data):
    if data is None:
        raise PreventUpdate

    num_vars = data.select_dtypes(include=['number']).columns.tolist()
    columna = num_vars[0]

    variable_component = html.H4(html.Strong('Análisis de regresión:'))

    desc_component = html.P("Seleccione las variables numéricas para el análisis de regresión.")

    dropdown_x = dcc.Dropdown(
        id='regression-dropdown-x',
        options=[{'label': col, 'value': col} for col in num_vars],
        value=columna,
        style={'width': '50%'}
    )

    dropdown_y = dcc.Dropdown(
        id='regression-dropdown-y',
        options=[{'label': col, 'value': col} for col in num_vars],
        value=columna,
        style={'width': '50%'}
    )

    component = html.Div([
        variable_component,
        desc_component,
        html.Div(
            children=[
                dropdown_x,
                dropdown_y
            ],
            style={'display': 'flex'}
        ),
        html.Div(id="regression")
    ])

    return component


@app.callback(
    Output('regression', 'children'),
    [Input('store', 'data'),
     Input('regression-dropdown-x', 'value'),
     Input('regression-dropdown-y', 'value')]
)
def update_num_graph(data, selected_column_x, selected_column_y):
    if data is None:
        raise PreventUpdate

    if selected_column_x is None:
        raise PreventUpdate

    if selected_column_y is None:
        raise PreventUpdate

    x = data[selected_column_x]
    y = data[selected_column_y]

    x = sm.add_constant(x)

    model = sm.OLS(y, x).fit()

    summary = model.summary()
    repr_summary = repr(summary)

    fig_scatter = px.scatter(data, x=selected_column_x, y=selected_column_y, trendline="ols", trendline_color_override="red")

    component_summary = html.Pre(repr_summary)
    component_graph = dcc.Graph(figure=fig_scatter)

    component = dcc.Loading(
        type="circle",
        children=html.Div([
            html.Div(
                children=[
                    html.P(component_summary)],
                style=style_output
            ),
            component_graph
        ])
    )

    return component


if __name__ == '__main__':
    # app.run_server(debug=True)
    app.run_server(debug=False, dev_tools_ui=False)
