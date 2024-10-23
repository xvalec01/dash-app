import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

if platform.system() == "Windows":
    SVGS_DIR = Path("D:/diplomka/seccerts-data/svgs")
    SVGS_DIR.mkdir(parents=True, exist_ok=True)
    CC_DATASET = Path("D:/diplomka/seccerts-data/cc/processed_dataset.json")
    DF_CVES = Path("D:/diplomka/sec-certs/notebooks/cc/results/exploded_cves.csv")
    DF_VALIDITY = Path("D:/diplomka/sec-certs/notebooks/cc/results/df_validity.csv")
    DF_AVG_EAL = Path("D:/diplomka/sec-certs/notebooks/cc/results/avg_eal.csv")
    DF_INTERESTING_SCHEMAS = Path(
        "D:/diplomka/sec-certs/notebooks/cc/results/interesting_schemes.csv"
    )
    DF_POPULAR_CATEGORIES = Path(
        "D:/diplomka/sec-certs/notebooks/cc/results/popular_categories.csv"
    )
else:
    SVGS_DIR = Path("/mnt/d/diplomka/seccerts-data/svgs")
    SVGS_DIR.mkdir(parents=True, exist_ok=True)
    CC_DATASET = Path("/mnt/d/diplomka/seccerts-data/cc/processed_dataset.json")
    DF_CVES = Path("/mnt/d/diplomka/sec-certs/notebooks/cc/results/exploded_cves.csv")
    DF_VALIDITY = Path("/mnt/d/diplomka/sec-certs/notebooks/cc/results/df_validity.csv")
    DF_AVG_EAL = Path("/mnt/d/diplomka/sec-certs/notebooks/cc/results/avg_eal.csv")
    DF_INTERESTING_SCHEMAS = Path(
        "/mnt/d/diplomka/sec-certs/notebooks/cc/results/interesting_schemes.csv"
    )
    DF_POPULAR_CATEGORIES = Path(
        "/mnt/d/diplomka/sec-certs/notebooks/cc/results/popular_categories.csv"
    )

df_cves = pd.read_csv(DF_CVES)
df_validity = pd.read_csv(DF_VALIDITY)
df_avg_levels = pd.read_csv(DF_AVG_EAL)
df_interesting_schemes = pd.read_csv(DF_INTERESTING_SCHEMAS)
df_popular_categories = pd.read_csv(DF_POPULAR_CATEGORIES)

color_palette = px.colors.qualitative.T10

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
        if source.suffix == ".csv":
            return pd.read_csv(source)
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
        html.H1(children="Category Distribution"),
        # Pie chart for category distribution
        dcc.Graph(id="category-pie-chart"),
        # Bar chart for category distribution per year
        html.H1(children="Category Distribution per Year"),
        dcc.Graph(id="category-year-bar-chart"),
        html.H1(children="Certificate Validity Periods"),
        html.Button("Save as SVG", id="save-svg-btn"),
        dcc.Graph(id="certificate-validity-boxplot"),
        html.H1(children="Evolution of Average EAL Over Time"),
        dcc.Graph(id="eal-line-chart"),
        html.H1(children="Interesting Schemes Evolution"),
        dcc.Graph(id="schemes-line-chart"),
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
            title=f"Certificates Grouped by Category: {selected_category}",
            margin=dict(t=20, l=20, r=20, b=20),  # Tight layout
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
    for idx, category in enumerate(category_per_year.columns):
        data.append(
            go.Bar(
                name=category,
                x=category_per_year.index,
                y=category_per_year[category],
                marker=dict(
                    color=color_palette[idx % len(color_palette)]
                ),  # Apply predefined color palette
            )
        )

    # Layout configuration for the bar chart
    figure = {
        "data": data,
        "layout": go.Layout(
            title="Certificates Grouped by Category and Year",
            barmode="relative",
            xaxis={"title": "Year"},
            yaxis={"title": "Number of Certificates"},
            margin=dict(t=40, l=40, r=40, b=40),  # Tight layout
            height=1000,
        ),
    }

    return figure


@app.callback(
    Output("save-svg-btn", "children"),
    [
        Input("certificate-validity-boxplot", "figure"),
        Input("save-svg-btn", "n_clicks"),
    ],
)
def save_chart_as_svg(fig, n_clicks):
    if n_clicks is not None:
        # Convert the figure dict back to a plotly figure object
        fig_obj = go.Figure(fig)

        # Save the figure as an SVG
        svg_path = SVGS_DIR / f"boxplot_chart_{n_clicks}.svg"
        pio.write_image(fig_obj, svg_path, format="svg")

        return f"Saved as SVG {n_clicks}"
    return "Save as SVG"


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
    sorted_years = sorted(df["year_from"].unique())

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
        category_orders={"year_from": sorted_years},
        color_discrete_sequence=color_palette,
        width=1400,
    )
    return figure


def prepare_data_for_eal_line_chart() -> pd.DataFrame:
    """Prepare data for the EAL line chart."""
    df_avg_levels = fetch_data(DF_AVG_EAL)
    df_avg_levels["smartcard_category"] = df_avg_levels.category.map(
        lambda x: x if x == "ICs, Smartcards" else "Other 14 categories"
    )

    df_other_categories = df_avg_levels[
        df_avg_levels["smartcard_category"] == "Other 14 categories"
    ]
    df_other_categories_grouped = df_other_categories.groupby(
        ["year_from", "smartcard_category"], as_index=False
    ).agg({"eal_number": "mean"})
    print(df_other_categories_grouped["smartcard_category"].value_counts())

    return df_avg_levels, df_other_categories_grouped


@app.callback(Output("eal-line-chart", "figure"), [Input("eal-line-chart", "id")])
def update_line_chart(_):
    """
    Update the EAL line chart with the latest data.
    """
    df = fetch_data(CC_DATASET)

    eal_to_num_mapping = {
        eal: index
        for index, eal in enumerate(df["eal"].astype("category").cat.categories)
    }

    df_avg_levels, df_other_categories_grouped = prepare_data_for_eal_line_chart()

    fig = go.Figure()

    df_ics_smartcards = df_avg_levels[
        df_avg_levels["smartcard_category"] == "ICs, Smartcards"
    ]
    fig.add_trace(
        go.Scatter(
            x=df_ics_smartcards["year_from"],
            y=df_ics_smartcards["eal_number"],
            mode="lines+markers",
            name="ICs, Smartcards",
            marker=dict(symbol="circle", size=5, color="purple", line=dict(width=2)),
            line=dict(dash="dash", color="orange", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_other_categories_grouped["year_from"],
            y=df_other_categories_grouped["eal_number"],
            mode="lines+markers",
            name="Other 14 categories",
            marker=dict(symbol="circle", size=2, color="green"),
            line=dict(dash="solid", color="green", width=2),
        )
    )

    ymin = 1
    ymax = 9
    ylabels = [
        x if "+" in x else x for x in list(eal_to_num_mapping.keys())[ymin : ymax + 1]
    ]
    fig.update_layout(
        title="Average EAL Over Time for Smartcards and Other Categories",
        xaxis_title="Year",
        yaxis_title="Average EAL Number",
        legend_title="Smartcard Category",
        xaxis=dict(tickmode="linear", tick0=1998, dtick=1),
        yaxis=dict(tickvals=np.arange(1, 10, 1), ticktext=ylabels),
        margin=dict(t=20, l=20, r=20, b=20),
        width=1800,
        height=800,
        showlegend=True,
    )

    return fig


@app.callback(
    Output(
        "schemes-line-chart", "figure"
    ),  # Output is the figure in the 'schemes-line-chart'
    [Input("schemes-line-chart", "id")],  # Trigger callback on loading
)
def update_schemes_graph(_):
    # Create the line plot using Plotly Express
    df_interesting_schemes = fetch_data(DF_INTERESTING_SCHEMAS)
    fig = px.line(
        df_interesting_schemes,
        x="year_from",
        y="size",
        color="scheme",  # Scheme contains the country codes
        markers=True,
        labels={
            "year_from": "Year",
            "size": "Size",
            "scheme": "Country Code",  # Display country code in the legend
        },
        title="Evolution of Interesting Schemes",
    )

    # Update the layout to match the style from the PDF
    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=[1998, 2003, 2008, 2013, 2018, 2023],  # Set x-axis ticks manually
            range=[1997, 2024],  # Set the x-axis range
        ),
        yaxis=dict(
            tickvals=list(range(0, 90, 20)),  # Set y-axis tick values
            range=[0, 80],  # Set the y-axis range
        ),
        legend_title="Country Code",  # Set legend title to Country Code
        width=1000,
        height=600,
        margin=dict(t=20, l=20, r=20, b=20),  # Tight layout
        showlegend=True,
    )

    return fig


# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
