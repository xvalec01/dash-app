import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

CC_DATASET = Path("./cc_dataset_processed.json")

# Initialize Dash app
app = Dash(__name__)


def create_mongo_client():
    username = "admin"
    password = "password"

    MongoClient(
        "mongodb://localhost:27017/",
        username=username,
        password=password,
        authSource="admin",
        authMechanism="SCRAM-SHA-256",
    )


def get_database(name: str, client: MongoClient) -> Database:
    return client[name]


def get_collection(name: str, db: Database) -> Collection:
    return db[name]


def fetch_data(source: Collection | Path) -> pd.DataFrame:
    if isinstance(source, Path):
        return pd.read_json(source)

    cursor = source.find({})
    data = list(cursor)
    df = pd.DataFrame(data)
    return df


df = fetch_data(CC_DATASET)


categories = df["category"].unique()
categories = ["All"] + sorted(categories)

app.layout = html.Div(
    children=[
        html.H1(children="Certificates Visualization Dashboard"),
        # Dropdown for filtering by category (used for pie chart and bar chart)
        html.Label("Select Category for Pie Chart and Bar Chart:"),
        dcc.Dropdown(
            id="category-dropdown",
            options=[{"label": cat, "value": cat} for cat in categories],
            value="All",  # Default value is 'All'
            multi=False,
        ),
        dcc.Dropdown(
            id="year-dropdown",
            options=[
                {"label": year, "value": year} for year in df["year_from"].unique()
            ],
            value="All",
            multi=False,
        ),
        # Pie chart for category distribution
        dcc.Graph(id="category-pie-chart"),
        # Bar chart for category distribution per year
        dcc.Graph(id="category-year-bar-chart"),
        dcc.Graph(id="certificate-validity-boxplot"),
    ]
)


# Callback to update pie chart based on dropdown selection
@app.callback(
    Output("category-pie-chart", "figure"), [Input("category-dropdown", "value")]
)
def update_pie_chart(selected_category):
    # Fetch updated data
    df = fetch_data(CC_DATASET)

    # Filter data based on the selected category
    if selected_category != "All":
        df = df[df["category"] == selected_category]

    # Group data by category and count occurrences
    category_counts = df["category"].value_counts()

    # Create pie chart
    figure = {
        "data": [
            go.Pie(
                labels=category_counts.index,
                values=category_counts.values,
                hole=0.3,  # Optional: this makes it a donut chart
            )
        ],
        "layout": go.Layout(
            title=f"Certificates Grouped by Category: {selected_category}"
        ),
    }

    return figure


# Callback to generate the bar chart for category per year
@app.callback(
    Output("category-year-bar-chart", "figure"), [Input("category-dropdown", "value")]
)
def update_bar_chart(selected_category):
    # Fetch updated data
    df = fetch_data(CC_DATASET)

    # Optionally filter data by category for the bar chart as well
    if selected_category != "All":
        df = df[df["category"] == selected_category]

    # Generate the category per year DataFrame
    category_per_year = (
        df.groupby(["year_from", "category"]).size().unstack(fill_value=0)
    )

    # Create a stacked bar chart
    data = []
    for category in category_per_year.columns:
        data.append(
            go.Bar(
                name=category, x=category_per_year.index, y=category_per_year[category]
            )
        )

    # Layout configuration for the bar chart
    figure = {
        "data": data,
        "layout": go.Layout(
            title="Certificates Grouped by Category and Year",
            barmode="stack",
            xaxis={"title": "Year"},
            yaxis={"title": "Number of Certificates"},
        ),
    }

    return figure


# Callback to generate the boxplot for certificate validity periods
@app.callback(
    Output("certificate-validity-boxplot", "figure"), [Input("year-dropdown", "value")]
)
def update_boxplot(selected_year):
    # Fetch updated data
    df = fetch_data(CC_DATASET)

    # Convert timestamps to datetime
    df["not_valid_before"] = pd.to_datetime(df["not_valid_before"], unit="ms")
    df["not_valid_after"] = pd.to_datetime(df["not_valid_after"], unit="ms")

    df["validity"] = (df["not_valid_after"] - df["not_valid_before"]).dt.days / 365.25

    df["year_from"] = df["year_from"].astype(str)

    # Create the boxplot
    figure = px.box(
        df,
        x="year_from",
        y="validity",
        title="Boxplot of Certificate Validity Periods",
        labels={
            "validity": "Lifetime of certificates (in years)",
            "year_from": "Year of certification",
        },
    )
    return figure


# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
