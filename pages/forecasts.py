import os

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash_extensions.enrich import (Input, Output, Serverside, callback,
                                    clientside_callback, ctx, dcc, html,
                                    no_update)
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsforecast import StatsForecast
from statsforecast.models import AutoCES

from modules.helpers import *

load_dotenv()

pio.templates.default = "plotly_white"
colors = px.colors.qualitative.Pastel
colors = [*colors * 100]

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

dash.register_page(
    __name__,
    path="/forecasts",
    name="Forecasts")


######################################################################
# Pies
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

    df_train = df[(df.ds > "2020-04-30") & (df.ds < "2024-04-01")].copy()
    df_test = df[df.ds >= "2024-04-01"].copy()

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

        season_length = 365
        horizon = len(df_test)

        sf = StatsForecast(
            models=[AutoCES(season_length=season_length)],
            freq="1d", 
            n_jobs=1,
        )

        sf.fit(df_train)
        df_pred = sf.predict(h=horizon, level=[95])

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
                name=f"{model}-hi-95",
                marker_color=colors[2],
                fill="tonexty",
                showlegend=False,
                line_width=0,
                )
            )
        fig.update_xaxes(range=["2024-03-24", "2024-11-11"])
        mae = f"{mean_absolute_error(df_test.y, df_pred[model]):,.3f}"
        mape = f"{mean_absolute_percentage_error(df_test.y, df_pred[model]) * 100:,.3f} %"
        rmse = f"{mean_squared_error(df_test.y, df_pred[model]) ** 0.5:,.3f}"
    return dcc.Graph(figure=fig, style={"width": "100%"}), mean, mae, mape, rmse

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
        dmc.Button("Complex Exponential Smoothing", id="stats-forecast-btn", n_clicks=0),
    ],
    gap="md",
    justify="flex-start",
    align="flex-end",
    ),
    dmc.Space(h=20),
    html.Div(id="stats-forecasts-div"),
    dmc.Space(h=20),
    dmc.Group([
        dmc.Card([
            dmc.Group(
                [
                    dmc.Text("Mean Daily Ridership", fw=500),
                ],
                justify="flex-start",
                mt="md",
                mb="xs",
            ),
            dmc.Text(
                id="stats-mean",
                size="sm",
                c="dimmed",
                ta="left",
            ),
        ]),
        dmc.Card([
            dmc.Group(
                [
                    dmc.Text("Mean Absolute Error", fw=500),
                    dmc.Badge("MAE", color="pink"),
                ],
                justify="flex-start",
                mt="md",
                mb="xs",
            ),
            dmc.Text(
                id="stats-mae",
                size="sm",
                c="dimmed",
                ta="left",
            ),
        ]),
        dmc.Card([
            dmc.Group(
                [
                    dmc.Text("Mean Absolute Percentage Error", fw=500),
                    dmc.Badge("MAPE", color="pink"),
                ],
                justify="flex-start",
                mt="md",
                mb="xs",
            ),
            dmc.Text(
                id="stats-mape",
                size="sm",
                c="dimmed",
                ta="left",
            ),
        ]),
        dmc.Card([
            dmc.Group(
                [
                    dmc.Text("Root Mean Squared Error", fw=500),
                    dmc.Badge("rmse", color="pink"),
                ],
                justify="flex-start",
                mt="md",
                mb="xs",
            ),
            dmc.Text(
                id="stats-rmse",
                size="sm",
                c="dimmed",
                ta="left",
            ),
        ]),
    ],
    gap="md",
    align="flex-start",
    justify="space-between",
    grow=True
    )
])
