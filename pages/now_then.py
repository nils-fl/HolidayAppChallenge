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
from groq import Groq

from modules.helpers import *

load_dotenv()

pio.templates.default = "plotly_white"
colors = px.colors.qualitative.Pastel
colors = [*colors * 100]

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

dash.register_page(
    __name__,
    path="/now_then",
    name="Now & Then")


######################################################################
# Pies
######################################################################

@callback(
    Output("now-then-pies", "children"),
    Input("data-store", "data"),
)
def get_scatter(df:pd.DataFrame):
    df_then = df[df["Date"] < "2020-03-08"].copy()
    df_now = df[(df["Date"] > "2024-09-02") & (df.Date < "2024-10-27")].copy()

    total_cols = [c for c in df_then.columns if "Total" in c]
    
    total_then = df_then[total_cols].sum().sum()
    total_now = df_now[total_cols].sum().sum()

    labels = ["Subways", "Buses", "LIRR", "Metro-North", "A-RIDE", "Bridges & Tunnels", "Staten Island Railway"]

    pie_then = go.Figure()
    pie_now = go.Figure()

    pie_then.add_trace(
        go.Pie(
            labels=labels,
            values=[df_then[c].sum() for c in df_then.columns if "Total" in c],
            hole=0.6,
            direction="clockwise",
            sort=True,
            showlegend=True,
            marker_colors=colors,
        )
    )
    pie_now.add_trace(
        go.Pie(
            labels=labels,
            values=[df_now[c].sum() for c in df_now.columns if "Total" in c],
            hole=0.6,
            direction="clockwise",
            sort=True,
            showlegend=True,
            marker_colors=colors,
        )
    )

    pie_then.update_layout(
        title="Pre-Pandemic Rides and Trips",
        title_x=0.5,
        title_y=0.95,
    )
    pie_now.update_layout(
        title="Post-Pandemic Rides and Trips",
        title_x=0.5,
        title_y=0.95,
    )

    dots = go.Figure()
    dots.add_trace(
        go.Scatter(
            x=[df_then[c].sum() / total_then for c in df_then.columns if "Total" in c],
            y=labels,
            mode="markers",
            marker_color=colors[0],
            marker_size=15,
            name="Pre-Pandemic",
        )
    )
    dots.add_trace(
        go.Scatter(
            x=[df_now[c].sum() / total_now for c in df_now.columns if "Total" in c],
            y=labels,
            mode="markers",
            marker_color=colors[1],
            marker_size=15,
            name="Post-Pandemic",
        )
    )
    dots.update_xaxes(title="percentage")
    dots.update_layout(
        title="Share of Rides and Trips",
        title_x=0.5,
        title_y=0.95,
    )

    flex = dmc.Flex([
            dcc.Graph(figure=pie_then, style={"width": "100%"}),
            dmc.Space(w=20),
            dcc.Graph(figure=pie_now, style={"width": "100%"}),
            dmc.Space(w=20),
            dcc.Graph(figure=dots, style={"width": "100%"}),
        ],
        gap="md",
        align="flex-start",
        justify="flex-start",
        direction="row"
    )
    return flex


@callback(
    Output("now-then-text", "children"),
    Input("data-store", "data"),
)
def get_scatter(df:pd.DataFrame):
    df_then = df[df["Date"] < "2020-03-08"].copy()
    df_now = df[(df["Date"] > "2024-09-02") & (df.Date < "2024-10-27")].copy()

    total_cols = [c for c in df_then.columns if "Total" in c]
    
    total_then = df_then[total_cols].sum().sum()
    total_now = df_now[total_cols].sum().sum()

    labels = ["Subways", "Buses", "LIRR", "Metro-North", "A-RIDE", "Bridges & Tunnels", "Staten Island Railway"]
    
    totals_then = [df_then[c].sum() for c in df_then.columns if "Total" in c]
    totals_now = [df_now[c].sum() for c in df_now.columns if "Total" in c]

    pcts_then = [df_then[c].sum() / total_then for c in df_then.columns if "Total" in c]
    pcts_now = [df_now[c].sum() / total_now for c in df_now.columns if "Total" in c]

    chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": (f"Please tell me some insights about the pre versus post-pandemic situation regarding traffic in NY."
                                f"Here is an array of the labels: {labels}"
                                f"And here is an array with the pre-pandemic totals: {totals_then}"
                                f"And here is an array with the post-pandemic totals: {totals_now}"
                                f"And here is an array with the pre-pandemic percentages: {pcts_then}"
                                f"And here is an array with the post-pandemic percentages: {pcts_now}"
                                "Please format the answer in Markdown")
                }
            ],
            model="llama3-8b-8192",
            temperature=0.5,
            seed=42,
        )
    answer = chat_completion.choices[0].message.content
    answer = dcc.Markdown(answer)
    return dmc.Card(answer)

######################################################################
# Layout
######################################################################

layout = html.Div([
    html.Div(id="now-then-pies"),
    dmc.Space(h=20),
    html.Div(id="now-then-text"),
])
