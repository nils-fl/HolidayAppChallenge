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
from statsforecast.models import AutoARIMA, AutoCES, AutoETS, AutoTheta

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
    Input("data-store", "data"),
    Input("stats-forecast-btn", "n_clicks"),
    running=[(Output("stats-forecast-btn", "loading"), True, False)]
)
def get_scatter(df:pd.DataFrame, stats_clicks):
    df = df[["Subways: Total Estimated Ridership", "Date"]].copy()
    df["unique_id"] = 42
    df = df.rename(columns={
        "Subways: Total Estimated Ridership": "y",
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
    
    if ctx.triggered_id == "stats-forecast-btn":

        season_length = 365
        horizon = len(df_test)

        models = [
            # AutoETS(season_length=season_length),
            # AutoTheta(season_length=season_length),
            AutoCES(season_length=season_length),
        ]

        sf = StatsForecast(
            models=models,
            freq="1d", 
            n_jobs=1,
        )

        sf.fit(df_train)
        df_pred = sf.predict(h=horizon, level=[95])

        models = [
            # "AutoETS",
            # "AutoTheta",
            "CES"
            ]
        for i,model in enumerate(models):
            fig.add_trace(
                go.Scatter(
                    x=df_pred.ds,
                    y=df_pred[f"{model}-hi-95"],
                    mode='lines',
                    name=f"{model}-hi-95",
                    marker_color=colors[i+2],
                    showlegend=False,
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=df_pred.ds,
                    y=df_pred[f"{model}"],
                    mode='lines',
                    name=f"{model}-95-ci",
                    fill="tonexty",
                    marker_color=colors[i+2],
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=df_pred.ds,
                    y=df_pred[f"{model}-lo-95"],
                    mode='lines',
                    name=f"{model}-hi-95",
                    marker_color=colors[i+2],
                    fill="tonexty",
                    showlegend=False,
                    )
                )
            fig.update_xaxes(range=["2024-03-24", "2024-11-11"])
            print(mean_squared_error(df_test.y, df_pred[f"{model}"]) ** 0.5)
            print(mean_absolute_error(df_test.y, df_pred[f"{model}"]))
            print(mean_absolute_percentage_error(df_test.y, df_pred[f"{model}"]))
    return dcc.Graph(figure=fig, style={"width": "100%"})

######################################################################
# Layout
######################################################################

layout = html.Div([
    dmc.Flex([
        dmc.Button("Complex Exponential Smoothing", id="stats-forecast-btn", n_clicks=0),
    ],
    gap="md",
    justify="flex-start",
    align="flex-end",
    ),
    dmc.Space(h=20),
    html.Div(id="stats-forecasts-div"),
])
