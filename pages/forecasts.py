import os
from pathlib import Path

import dash
import dash_mantine_components as dmc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash_extensions.enrich import (Input, Output, Serverside, callback,
                                    clientside_callback, ctx, dcc, html,
                                    no_update)
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)
from sktime.forecasting.chronos import ChronosForecaster
from sktime.forecasting.compose import BaggingForecaster
from sktime.transformations.bootstrap import STLBootstrapTransformer
from statsforecast import StatsForecast
from statsforecast.models import AutoCES

from modules.helpers import *

load_dotenv()

HORIZON = 60

pio.templates.default = "plotly_white"
colors = px.colors.qualitative.Pastel
colors = [*colors * 100]

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

dash.register_page(
    __name__,
    path="/forecasts",
    name="Forecasts")


######################################################################
# Stats Forecast
######################################################################

@callback(
    Output("stats-forecasts-div", "children"),
    Output("stats-mean", "children"),
    Output("stats-mae", "children"),
    Output("stats-mape", "children"),
    Output("stats-rmse", "children"),
    Input("data-store", "data"),
    Input("stats-forecast-btn", "n_clicks"),
    Input("type-select", "value"),
    running=[(Output("stats-forecast-btn", "loading"), True, False)]
)
def get_scatter(df:pd.DataFrame, stats_clicks, type_select):
    df = df[[type_select, "Date"]].copy()
    df["unique_id"] = 42
    df = df.rename(columns={
        type_select: "y",
        "Date": "ds",
        })

    df_train = df[:-HORIZON].copy()
    df_test = df[-HORIZON:].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_train.ds,
            y=df_train.y,
            mode='lines',
            name="History",
            marker_color=colors[1],
            )
        )
    fig.add_trace(
        go.Scatter(
            x=df_test.ds,
            y=df_test.y,
            mode='lines',
            name="Observed",
            marker_color=colors[0],
            )
        )
    mean = f"{df_test.y.mean():,.0f}"
    mae = "run model first"
    mape = "run model first"
    rmse = "run model first"
    
    if ctx.triggered_id == "stats-forecast-btn":
        ces_path = Path(f"data/ces_{type_select}.csv")
        if not ces_path.exists():
            season_length = 365
            horizon = HORIZON

            sf = StatsForecast(
                models=[AutoCES(season_length=season_length)],
                freq="1d", 
                n_jobs=1,
            )

            sf.fit(df_train)
            df_pred = sf.predict(h=horizon, level=[95])
            df_pred.to_csv(ces_path)
        else:
            df_pred = pd.read_csv(ces_path)

        model = "CES"
        fig.add_trace(
            go.Scatter(
                x=df_pred.ds,
                y=df_pred[f"{model}-hi-95"],
                mode='lines',
                name=f"{model}-hi-95",
                marker_color=colors[2],
                showlegend=False,
                line_width=0,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=df_pred.ds,
                y=df_pred[f"{model}"],
                mode='lines',
                name=f"{model}-95-ci",
                fill="tonexty",
                marker_color=colors[2],
                )
            )
        fig.add_trace(
            go.Scatter(
                x=df_pred.ds,
                y=df_pred[f"{model}-lo-95"],
                mode='lines',
                name=f"{model}-lo-95",
                marker_color=colors[2],
                fill="tonexty",
                showlegend=False,
                line_width=0,
                )
            )
        fig.update_xaxes(range=["2024-05-24", "2024-11-11"])
        mae = f"{mean_absolute_error(df_test.y, df_pred[model]):,.3f}"
        mape = f"{mean_absolute_percentage_error(df_test.y, df_pred[model]) * 100:,.3f} %"
        rmse = f"{mean_squared_error(df_test.y, df_pred[model]) ** 0.5:,.3f}"
    return dcc.Graph(figure=fig, style={"width": "100%"}), mean, mae, mape, rmse

######################################################################
# Neural Forecast
######################################################################

@callback(
    Output("chronos-forecasts-div", "children"),
    Output("chronos-mean", "children"),
    Output("chronos-mae", "children"),
    Output("chronos-mape", "children"),
    Output("chronos-rmse", "children"),
    Input("data-store", "data"),
    Input("chronos-forecast-btn", "n_clicks"),
    Input("type-select", "value"),
    running=[(Output("chronos-forecast-btn", "loading"), True, False)]
)
def get_scatter(df:pd.DataFrame, ttm_clicks, type_select):
    df = df[[type_select, "Date"]].copy()
    df["unique_id"] = 42
    df = df.rename(columns={
        type_select: "y",
        "Date": "ds",
        })
    
    df_train = df[:-HORIZON].copy()
    df_test = df[-HORIZON:].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_train.ds,
            y=df_train.y,
            mode='lines',
            name="History",
            marker_color=colors[1],
            )
        )
    fig.add_trace(
        go.Scatter(
            x=df_test.ds,
            y=df_test.y,
            mode='lines',
            name="Observed",
            marker_color=colors[0],
            )
        )
    mean = f"{df_test.y.mean():,.0f}"
    mae = "run model first"
    mape = "run model first"
    rmse = "run model first"
    
    if ctx.triggered_id == "chronos-forecast-btn":
        chronos_path = Path(f"data/chronos_{type_select}.csv")
        if not chronos_path.exists():
            base = ChronosForecaster(model_path="amazon/chronos-t5-tiny")
            model = BaggingForecaster(STLBootstrapTransformer(), base)

            y = df_train.y.values

            base.fit(y=y, fh=range(1, HORIZON+1))
            model.fit(y=y, fh=range(1, HORIZON+1))

            df_test["preds"] = base.predict(fh=range(1, HORIZON+1)).ravel()
            interval = model.predict_interval(fh=range(1, HORIZON+1), coverage=0.95)
            df_test["upper"] = interval[0][0.95]["upper"]
            df_test["lower"] = interval[0][0.95]["lower"]

            df_test.to_csv(chronos_path)
        else:
            df_test = pd.read_csv(chronos_path)

        model_name = "chronos"
        fig.add_trace(
            go.Scatter(
                x=df_test.ds,
                y=df_test.upper,
                mode='lines',
                name=f"{model_name}-hi-95",
                marker_color=colors[2],
                showlegend=False,
                line_width=0,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=df_test.ds,
                y=df_test.preds,
                mode='lines',
                name=f"{model_name}-95-ci",
                fill="tonexty",
                marker_color=colors[2],
                )
            )
        fig.add_trace(
            go.Scatter(
                x=df_test.ds,
                y=df_test.lower,
                mode='lines',
                name=f"{model_name}-lo-95",
                marker_color=colors[2],
                fill="tonexty",
                showlegend=False,
                line_width=0,
                )
            )
        fig.update_xaxes(range=["2024-05-24", "2024-11-11"])
        mae = f"{mean_absolute_error(df_test.y, df_test.preds):,.3f}"
        mape = f"{mean_absolute_percentage_error(df_test.y, df_test.preds) * 100:,.3f} %"
        rmse = f"{mean_squared_error(df_test.y, df_test.preds) ** 0.5:,.3f}"
    return dcc.Graph(figure=fig, style={"width": "100%"}), mean, mae, mape, rmse


def get_forecast_metric_cards(model):
    return dmc.Flex([
        dmc.Card([
            dmc.Stack(
                [
                    dmc.Badge("Data", color="pink"),
                    dmc.Text("Mean Daily Ridership", fw=500),
                    dmc.Text(
                        id=f"{model}-mean",
                        size="sm",
                        c="dimmed",
                        ta="left",
                    ),
                ],
                align="flex-start",
                justify="space-between",
                mt="md",
                mb="xs",
            ),
        ], w="100%"),
        dmc.Card([
            dmc.Stack(
                [
                    dmc.Badge("MAE", color="pink"),
                    dmc.Text("Mean Absolute Error", fw=500),
                    dmc.Text(
                        id=f"{model}-mae",
                        size="sm",
                        c="dimmed",
                        ta="left",
                    ),
                ],
                align="flex-start",
                justify="space-between",
                mt="md",
                mb="xs",
            ),
        ], w="100%"),
        dmc.Card([
            dmc.Stack(
                [
                    dmc.Badge("MAPE", color="pink"),
                    dmc.Text("Mean Absolute Percentage Error", fw=500),
                    dmc.Text(
                        id=f"{model}-mape",
                        size="sm",
                        c="dimmed",
                        ta="left",
                    ),
                ],
                align="flex-start",
                justify="space-between",
                mt="md",
                mb="xs",
            ),
        ], w="100%"),
        dmc.Card([
            dmc.Stack(
                [
                    dmc.Badge("rmse", color="pink"),
                    dmc.Text("Root Mean Squared Error", fw=500),
                    dmc.Text(
                        id=f"{model}-rmse",
                        size="sm",
                        c="dimmed",
                        ta="left",
                    ),
                ],
                align="flex-start",
                justify="space-between",
                mt="md",
                mb="xs",
            ),
        ], w="100%"),
        ],
        direction={"base": "column", "sm": "row"},
        gap={"base": "sm", "sm": "lg"},
        justify={"sm": "space-between"},
        )

######################################################################
# Layout
######################################################################

layout = html.Div([
    dmc.Flex([
        dmc.Select(
            label="Select a means of transportation",
            placeholder="Select one",
            id="type-select",
            value="Subways: Total Estimated Ridership",
            comboboxProps={"position": "right", "middlewares": {"flip": False, "shift": True}},
            data=[
                {"value": "Subways: Total Estimated Ridership", "label": "Subway"},
                {"value": "Buses: Total Estimated Ridership", "label": "Buses"},
                {"value": "LIRR: Total Estimated Ridership", "label": "LIRR"},
                {"value": "Metro-North: Total Estimated Ridership", "label": "Metro-North"},
                {"value": "Access-A-Ride: Total Scheduled Trips", "label": "Access-A-Ride"},
                {"value": "Bridges and Tunnels: Total Traffic", "label": "Bridges and Tunnels"},
                {"value": "Staten Island Railway: Total Estimated Ridership", "label": "Staten Island Railway"},
            ],
        ),
    ],
    gap="md",
    justify="flex-start",
    align="flex-end",
    ),
    dmc.Space(h=20),

    # stats
    html.Div(id="stats-forecasts-div"),
    dmc.Space(h=20),
    get_forecast_metric_cards("stats"),
    dmc.Space(h=20),
    dmc.Button("Complex Exponential Smoothing", id="stats-forecast-btn", n_clicks=0),

    dmc.Space(h=20),
    html.Hr(),
    dmc.Space(h=20),

    # neural
    dmc.Button("Chronos", id="chronos-forecast-btn", n_clicks=0),
    dmc.Space(h=20),
    get_forecast_metric_cards("chronos"),
    dmc.Space(h=20),
    html.Div(id="chronos-forecasts-div"),
])
