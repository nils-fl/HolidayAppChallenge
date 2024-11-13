import dash_mantine_components as dmc
import pandas as pd
from dash import _dash_renderer
from dash_extensions.enrich import (DashProxy, Input, Output, Serverside,
                                    ServersideOutputTransform, dcc, html,
                                    page_container)
from dash_iconify import DashIconify

from modules.helpers import *

_dash_renderer._set_react_version("18.2.0")

app = DashProxy(
    __name__,
    update_title="Holiday Season App Challenge",
    use_pages=True,
    external_stylesheets=dmc.styles.ALL,
    transforms=[ServersideOutputTransform()]
    )

app.config.suppress_callback_exceptions = True
server = app.server

def get_nav_content():
    return [
        dmc.Image(
            src="https://cdn-icons-png.flaticon.com/512/439/439882.png",
            className="brand-icon",
        ),
        dmc.NavLink(
            href="/",
            leftSection=DashIconify(icon="mdi:analytics", height=16), 
            label="Home",
        ),
        dmc.NavLink(
            href="/now_then",
            leftSection=DashIconify(icon="mdi:clock-outline", height=16), 
            label="Now & Then",
        ),
        dmc.NavLink(
            href="/forecasts",
            leftSection=DashIconify(icon="mdi:brain", height=16), 
            label="Forecasts",
        ),
    ]

@app.callback(
    Output("data-store", "data"),
    Input("url", "search"),
    )
def display_page(url):
    df = pd.read_csv(DATA_PATH)
    df.Date = pd.to_datetime(df.Date)
    return Serverside(df)


app.layout = dmc.MantineProvider(
    id="m2d-mantine-provider",
    forceColorScheme="light",
    children=[
        dcc.Location(id="url", refresh="callback-nav"),
        dmc.AppShell(
            children=[
                dmc.AppShellNavbar(get_nav_content(), zIndex=2000, w="12em", className="nav-left"),
                dmc.AppShellMain(children=page_container),
                dcc.Store(id="data-store"),
            ]
        ),
    ])

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8071, debug=True, use_reloader=True)
