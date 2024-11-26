import os

import dash
import dash_mantine_components as dmc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash_extensions.enrich import (Input, Output, Serverside, callback,
                                    clientside_callback, ctx, dcc, html,
                                    no_update)
from dash_iconify import DashIconify
from dotenv import load_dotenv
from plotly.subplots import make_subplots

from modules.helpers import *

load_dotenv()

pio.templates.default = "plotly_white"
colors = px.colors.qualitative.Pastel
colors = [*colors * 100]

dash.register_page(
    __name__,
    path="/",
    name="Map")


######################################################################
# Cards
######################################################################

@callback(
    Output("home-stats", "children"),
    Input("data-store", "data")
    )
def update_analytics_graph(df:pd.DataFrame):
    total = df[[c for c in df.columns if "Total" in c]].sum().sum()

    total_subway = df["Subways: Total Estimated Ridership"].sum()
    min_subway = df["Subways: % of Comparable Pre-Pandemic Day"].min()
    max_subway = df[-200:]["Subways: % of Comparable Pre-Pandemic Day"].max()
    
    total_bus = df["Buses: Total Estimated Ridership"].sum()
    min_bus = df["Buses: % of Comparable Pre-Pandemic Day"].min()
    max_bus = df[-200:]["Buses: % of Comparable Pre-Pandemic Day"].max()

    total_lirr = df["LIRR: Total Estimated Ridership"].sum()
    min_lirr = df["LIRR: % of Comparable Pre-Pandemic Day"].min()
    max_lirr = df[-200:]["LIRR: % of Comparable Pre-Pandemic Day"].max()

    total_metro = df["Metro-North: Total Estimated Ridership"].sum()
    min_metro = df["Metro-North: % of Comparable Pre-Pandemic Day"].min()
    max_metro = df[-200:]["Metro-North: % of Comparable Pre-Pandemic Day"].max()

    total_aride = df["Access-A-Ride: Total Scheduled Trips"].sum()
    min_aride = df["Access-A-Ride: % of Comparable Pre-Pandemic Day"].min()
    max_aride = df[-200:]["Access-A-Ride: % of Comparable Pre-Pandemic Day"].max()

    total_bridges = df["Bridges and Tunnels: Total Traffic"].sum()
    min_bridges = df["Bridges and Tunnels: % of Comparable Pre-Pandemic Day"].min()
    max_bridges = df[-200:]["Bridges and Tunnels: % of Comparable Pre-Pandemic Day"].max()

    total_rail = df["Staten Island Railway: Total Estimated Ridership"].sum()
    min_rail = df["Staten Island Railway: % of Comparable Pre-Pandemic Day"].min()
    max_rail = df[-200:]["Staten Island Railway: % of Comparable Pre-Pandemic Day"].max()

    docs = {
        "Total": [total, "traffic-light"],
        "Subways": [total_subway, "train", min_subway, max_subway],
        "Buses": [total_bus, "bus", min_bus, max_bus],
        "LIRR": [total_lirr, "train", min_lirr, max_lirr],
        "Metro-North": [total_metro, "metro", min_metro, max_metro],
        "A-RIDE": [total_aride, "car", min_aride, max_aride],
        "Bridges & Tunnels": [total_bridges, "bridge", min_bridges, max_bridges],
        "Staten Island Railway": [total_rail, "train-car", min_rail, max_rail],
    }
    docs = {k: v for k, v in sorted(docs.items(), key=lambda x: x[1][0], reverse=True)}
    cards = dmc.Flex([
                dmc.Card(
                    children=[
                        dmc.Center(DashIconify(icon=f"mdi:{docs[k][1]}", height=30)),
                        dmc.Text(k, fw=700, h=15),
                        html.Hr(className="stats-card-hr"),
                        dmc.Text(f"Total: {docs[k][0]:,.0f}", fw=500, h=15),
                        dmc.Space(h=10),
                        dmc.Text(f"Min: {docs[k][2] * 100:,.2f} %" if i>0 else "", fw=500, h=15),
                        dmc.Space(h=10),
                        dmc.Text(f"Max: {docs[k][3] * 100:,.2f} %" if i>0 else "", fw=500, h=15),
                        ],
                    className="stats-card"
                ) for i,k in enumerate(docs.keys())
            ],
            direction={"base": "column", "lg": "row"},
            gap={"base": "sm", "sm": "md"},
            justify={"sm": "space-between"},
            )
    return cards

######################################################################
# Scatter
######################################################################

@callback(
    Output("scatter-div", "children"),
    Input("data-store", "data"),
    Input("home-daily-btn", "n_clicks"),
    Input("home-weekly-btn", "n_clicks"),
    Input("home-monthly-btn", "n_clicks"),
    Input("home-quarterly-btn", "n_clicks"),
    Input("home-yearly-btn", "n_clicks"),
)
def get_scatter(df:pd.DataFrame, n_daily, n_weekly, n_monthly, n_quarterly, n_yearly):
    if ctx.triggered_id == "home-daily-btn":
        df = df.set_index("Date").resample("D").sum().reset_index()
    elif ctx.triggered_id == "home-weekly-btn":
        df = df.set_index("Date").resample("W").sum().reset_index()
    elif ctx.triggered_id == "home-monthly-btn":
        df = df.set_index("Date").resample("ME").sum().reset_index()
    elif ctx.triggered_id == "home-quarterly-btn":
        df = df.set_index("Date").resample("QE").sum().reset_index()
    elif ctx.triggered_id == "home-yearly-btn":
        df = df.set_index("Date").resample("YE").sum().reset_index()
    else:
        pass
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "Total Separate",
            "Total Stacked",
            "Total Percent Compared to Pre-Pandemic",
        ]
    )
    mean_y = df["Subways: Total Estimated Ridership"].mean()
    df_events = pd.DataFrame([
        {
            "Date": "2020-03-11",
            "y": mean_y,
            "desc": ("After more than 118,000 cases in 114 countries and 4,291 deaths,<br>"
                     "the WHO declares COVID-19 a pandemic."),
        },
        {
            "Date": "2020-03-15",
            "y": mean_y,
            "desc": ("States begin to implement shutdowns in order to prevent the spread<br>"
                     "of COVID-19. The New York City public school system— the largest<br>"
                     "school system in the U.S., with 1.1 million students— shuts down,<br>"
                     "while Ohio calls for restaurants and bars to close."),
        },
        {
            "Date": "2020-04-04",
            "y": mean_y,
            "desc": ("More than 1 million cases of COVID-19 had been confirmed worldwide,<br>"
                     "a more than ten-fold increase in less than a month."),
        },
    ])
    total_cols =  [c for c in df.columns if "Total" in c]
    pct_cols =  [c for c in df.columns if "Comparable" in c]
    for i,col in enumerate(total_cols):
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df[col],
                mode='lines',
                legendgroup=col.split(":")[0],
                showlegend=True,
                name=col.split(":")[0],
                marker_color=colors[i],
                ), row=1, col=1
            )
    for i,col in enumerate(total_cols):
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df[col],
                mode='lines',
                legendgroup=col.split(":")[0],
                showlegend=False,
                name=col.split(":")[0],
                marker_color=colors[i],
                stackgroup="one",
                ), row=2, col=1
            )
    for i,col in enumerate(pct_cols):
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df[col],
                mode='lines',
                legendgroup=col.split(":")[0],
                showlegend=False,
                name=col.split(":")[0],
                marker_color=colors[i],
                ), row=3, col=1
            )
    fig.add_trace(
        go.Scatter(
            x=df_events['Date'],
            y=df_events['y'],
            mode='markers',
            showlegend=True,
            customdata=df_events['desc'],
            hovertemplate='<b>%{x}</b><br>%{customdata}',
            name="Selected Moments",
            ), row=1, col=1
        )
    fig.update_xaxes(title="date", row=3, col=1)
    fig.update_yaxes(title="total ridership/trips", row=1, col=1)
    fig.update_yaxes(title="total ridership/trips", row=2, col=1)
    fig.update_yaxes(title="% of comparable pre-pandemic day", row=3, col=1)
    fig.update_layout(
        height=1200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="right",
            x=1
        ),
        hovermode="x unified",
    )
    return dcc.Graph(figure=fig, style={"width": "100%"})


######################################################################
# Layout
######################################################################

layout = html.Div([
    html.Div(id="home-stats"),
    dmc.Space(h=20),
    dmc.Flex([
        dmc.Button("Daily", id="home-daily-btn", n_clicks=0),
        dmc.Button("Weekly", id="home-weekly-btn", n_clicks=0),
        dmc.Button("Monthly", id="home-monthly-btn", n_clicks=0),
        dmc.Button("Quarterly", id="home-quarterly-btn", n_clicks=0),
        dmc.Button("Yearly", id="home-yearly-btn", n_clicks=0),
    ],
    gap="md",
    justify="flex-start",
    align="flex-end",
    ),
    dmc.Space(h=20),
    html.Div(id="scatter-div"),
    dmc.Space(h=20),
])
