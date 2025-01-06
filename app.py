import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from sec_certs.dataset import CCDataset

if platform.system() == "Windows":
    print("Running on Windows")
    SVGS_DIR = Path("D:/diplomka/seccerts-data/svgs")
    SVGS_DIR.mkdir(parents=True, exist_ok=True)
    CC_DATASET_PATH = Path("D:/diplomka/seccerts-data/cc/processed_dataset.json")
    CC_UNPROCESSED_DATASET_PATH = Path("D:/diplomka/seccerts-data/cc/dataset.json")
    DF_CVES_PATH = Path("D:/diplomka/sec-certs/notebooks/cc/results/exploded_cves.csv")
    DF_VALIDITY_PATH = Path(
        "D:/diplomka/sec-certs/notebooks/cc/results/df_validity.csv"
    )
    DF_AVG_EAL_PATH = Path("D:/diplomka/sec-certs/notebooks/cc/results/avg_eal.csv")
    DF_INTERESTING_SCHEMAS_PATH = Path(
        "D:/diplomka/sec-certs/notebooks/cc/results/interesting_schemes.csv"
    )
    DF_POPULAR_CATEGORIES_PATH = Path(
        "D:/diplomka/sec-certs/notebooks/cc/results/popular_categories.csv"
    )
else:
    SVGS_DIR = Path("/mnt/d/diplomka/seccerts-data/svgs")
    CC_DATASET_PATH = Path("/mnt/d/diplomka/seccerts-data/cc/processed_dataset.json")
    CC_UNPROCESSED_DATASET_PATH = Path("/mnt/d/diplomka/seccerts-data/cc/dataset.json")
    DF_CVES_PATH = Path(
        "/mnt/d/diplomka/sec-certs/notebooks/cc/results/exploded_cves.csv"
    )
    DF_VALIDITY_PATH = Path(
        "/mnt/d/diplomka/sec-certs/notebooks/cc/results/df_validity.csv"
    )
    DF_AVG_EAL_PATH = Path("/mnt/d/diplomka/sec-certs/notebooks/cc/results/avg_eal.csv")
    DF_INTERESTING_SCHEMAS_PATH = Path(
        "/mnt/d/diplomka/sec-certs/notebooks/cc/results/interesting_schemes.csv"
    )
    DF_POPULAR_CATEGORIES_PATH = Path(
        "/mnt/d/diplomka/sec-certs/notebooks/cc/results/popular_categories.csv"
    )


df_cves = pd.read_csv(DF_CVES_PATH)
df_validity = pd.read_csv(DF_VALIDITY_PATH)
df_avg_levels = pd.read_csv(DF_AVG_EAL_PATH)
df_interesting_schemes = pd.read_csv(DF_INTERESTING_SCHEMAS_PATH)
df_popular_categories = pd.read_csv(DF_POPULAR_CATEGORIES_PATH)

color_palette = px.colors.qualitative.T10

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


def fetch_data(source: Collection | Path | CCDataset) -> pd.DataFrame:
    if isinstance(source, Path):
        return get_data_from_file(source)

    if isinstance(source, CCDataset):
        return source.to_pandas()

    cursor = source.find({})
    data = list(cursor)
    df = pd.DataFrame(data)
    return df


def get_data_from_file(file_path: Path) -> pd.DataFrame:
    if file_path.suffix == ".csv":
        return pd.read_csv(file_path)
    return pd.read_json(file_path)


def get_cc_dataset() -> CCDataset:
    return CCDataset.from_json(CC_UNPROCESSED_DATASET_PATH)


CC_DATASET = get_cc_dataset()
df = fetch_data(CC_DATASET)

categories = df["category"].unique()
categories = ["All"] + sorted(categories)


def create_svg_download_config(chart_name: str) -> dict:
    return {
        "toImageButtonOptions": {
            "format": "svg",
            "filename": f"{chart_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        }
    }


def set_font_size(
    fig: go.Figure,
    font_size_axis: int,
    font_size_title: int,
    font_size_legend: int,
    tickfont_size_axis: int,
) -> go.Figure:
    """
    Set the font size of the title, x-axis, and y-axis labels.
    """
    kwargs = {
        "title": {"font": {"size": font_size_title}},
        "xaxis": {
            "title": {"font": {"size": font_size_axis}},
            "tickfont": {"size": tickfont_size_axis},
        },
        "yaxis": {
            "title": {"font": {"size": font_size_axis}},
            "tickfont": {"size": tickfont_size_axis},
        },
        "legend": {"font": {"size": font_size_legend}},
    }
    fig.update_layout(**kwargs)
    return fig


# set height and width of the plot
def set_figure_size(
    fig: go.Figure, height: Optional[int] = None, width: Optional[int] = None
) -> go.Figure:
    if height:
        fig.update_layout(height=height)
    if width:
        fig.update_layout(width=width)
    return fig


app.layout = html.Div(
    children=[
        html.H1(children="Certificates Visualization Dashboard"),
        html.Label("Select Category for Pie Chart and Bar Chart:"),
        dcc.Dropdown(
            id="category-dropdown",
            options=[{"label": cat, "value": cat} for cat in categories],
            value="All",
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
        dcc.Graph(
            id="category-pie-chart",
            config=create_svg_download_config("category-pie-chart"),
        ),
        html.H1(children="Category Distribution per Year"),
        dcc.Graph(
            id="category-year-bar-chart",
            config=create_svg_download_config("category-year-bar-chart"),
        ),
        html.H1(children="Certificate Validity Periods"),
        dcc.Graph(
            id="certificate-validity-boxplot",
            config=create_svg_download_config("certificate-validity-boxplot"),
        ),
        html.H1(children="Evolution of Average EAL Over Time"),
        dcc.Graph(
            id="eal-line-chart", config=create_svg_download_config("eal-line-chart")
        ),
        html.H1(children="Interesting Schemes Evolution"),
        dcc.Graph(
            id="schemes-line-chart",
            config=create_svg_download_config("schemes-line-chart"),
        ),
        html.H1(children="Cert Labs Category Over Time"),
        dcc.Graph(
            id="cert-labs-category-stacked-bar",
            config=create_svg_download_config("cert-labs-category-stacked-bar"),
        ),
        html.H1(children="Cert Labs Over Time"),
        dcc.Graph(
            id="cert-labs-graph",
            config=create_svg_download_config("cert-labs-graph"),
        ),
        html.Div(id="save-message"),
    ]
)


@app.callback(
    Output("category-pie-chart", "figure"), [Input("category-dropdown", "value")]
)
def update_pie_chart(selected_category):
    df = fetch_data(CC_DATASET_PATH)

    if selected_category != "All":
        df = df[df["category"] == selected_category]

    category_counts = df["category"].value_counts()

    fig = go.Figure(
        data=[
            go.Pie(
                labels=category_counts.index,
                values=category_counts.values,
                hole=0.3,
                textfont=dict(size=18),
            )
        ]
    )

    # Update the layout with layout properties
    fig.update_layout(
        title="Number of issued certificates in different categories",
        margin=dict(t=80, l=40, r=40, b=40),
    )
    set_figure_size(fig, height=1000)
    set_font_size(
        fig,
        font_size_axis=22,
        font_size_title=26,
        font_size_legend=24,
        tickfont_size_axis=16,
    )

    return fig


@app.callback(
    Output("category-year-bar-chart", "figure"), [Input("category-dropdown", "value")]
)
def update_bar_chart(selected_category):
    df = fetch_data(CC_DATASET_PATH)

    if selected_category != "All":
        df = df[df["category"] == selected_category]

    category_per_year = (
        df.groupby(["year_from", "category"]).size().unstack(fill_value=0)
    )

    fig = go.Figure()

    # Add each category as a bar trace
    for idx, category in enumerate(category_per_year.columns):
        fig.add_trace(
            go.Bar(
                name=category,
                x=category_per_year.index,
                y=category_per_year[category],
                marker=dict(color=color_palette[idx % len(color_palette)]),
            )
        )

    # Update the layout with layout properties
    fig.update_layout(
        title="Certificates grouped by category and year",
        barmode="relative",
        xaxis=dict(title="Year"),
        yaxis=dict(title="Number of Certificates"),
        margin=dict(t=80, l=40, r=40, b=40),
    )
    set_figure_size(fig, height=1000)
    set_font_size(
        fig,
        font_size_axis=22,
        font_size_title=26,
        font_size_legend=24,
        tickfont_size_axis=16,
    )
    return fig


@app.callback(
    Output("certificate-validity-boxplot", "figure"), [Input("year-dropdown", "value")]
)
def update_boxplot(selected_year):
    df = fetch_data(CC_DATASET_PATH)

    df["not_valid_before"] = pd.to_datetime(df["not_valid_before"], unit="ms")
    df["not_valid_after"] = pd.to_datetime(df["not_valid_after"], unit="ms")

    df["validity"] = (df["not_valid_after"] - df["not_valid_before"]).dt.days / 365.25

    df["year_from"] = df["year_from"].astype(str)
    sorted_years = sorted(df["year_from"].unique())

    fig = px.box(
        df,
        x="year_from",
        y="validity",
        title="Variance of certificate validity per years for which a certificate is valid",
        labels={
            "validity": "Lifetime of certificates (in years)",
            "year_from": "Year of certification",
        },
        category_orders={"year_from": sorted_years},
        color_discrete_sequence=color_palette,
    )

    set_figure_size(fig, height=800, width=2200)
    set_font_size(
        fig,
        font_size_axis=22,
        font_size_title=26,
        font_size_legend=24,
        tickfont_size_axis=16,
    )
    return fig


def prepare_data_for_eal_line_chart() -> pd.DataFrame:
    """Prepare data for the EAL line chart."""
    df_avg_levels = fetch_data(DF_AVG_EAL_PATH)
    df_avg_levels["smartcard_category"] = df_avg_levels.category.map(
        lambda x: x if x == "ICs, Smartcards" else "Other 14 categories"
    )

    df_other_categories = df_avg_levels[
        df_avg_levels["smartcard_category"] == "Other 14 categories"
    ]
    df_other_categories_grouped = df_other_categories.groupby(
        ["year_from", "smartcard_category"], as_index=False
    ).agg({"eal_number": "mean"})

    return df_avg_levels, df_other_categories_grouped


@app.callback(Output("eal-line-chart", "figure"), [Input("eal-line-chart", "id")])
def update_line_chart(_):
    """
    Update the EAL line chart with the latest data.
    """
    df = fetch_data(CC_DATASET_PATH)

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
        title=dict(
            text="Average EAL over time for smartcards and other categories",
        ),
        xaxis_title="Year",
        yaxis_title="Average EAL Number",
        legend_title="Smartcard Category",
        xaxis=dict(
            tickmode="linear",
            tick0=1998,
            dtick=1,
        ),
        yaxis=dict(
            tickvals=np.arange(1, 10, 1),
            ticktext=ylabels,
        ),
        margin=dict(t=80, l=40, r=40, b=40),
        showlegend=True,
    )
    set_figure_size(fig, height=800, width=1800)
    set_font_size(
        fig,
        font_size_axis=22,
        font_size_title=26,
        font_size_legend=24,
        tickfont_size_axis=16,
    )

    return fig


@app.callback(
    Output("schemes-line-chart", "figure"),
    Input("schemes-line-chart", "id"),
)
def update_schemes_graph(_):
    df_interesting_schemes = fetch_data(
        DF_INTERESTING_SCHEMAS_PATH
    )  # Replace with your actual data
    fig = px.line(
        df_interesting_schemes,
        x="year_from",
        y="size",
        color="scheme",
        markers=True,
        labels={
            "year_from": "Year",
            "size": "Size",
            "scheme": "Country Code",
        },
        title="The number of issued certificates for selected schemes",
    )

    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=[1998, 2003, 2008, 2013, 2018, 2023],
            range=[1997, 2024],
        ),
        legend_title="Country Code",
        margin=dict(t=80, l=40, r=40, b=40),
        showlegend=True,
    )
    set_figure_size(fig, height=800, width=1800)
    set_font_size(
        fig,
        font_size_axis=22,
        font_size_title=26,
        font_size_legend=24,
        tickfont_size_axis=16,
    )
    return fig


def get_cert_labs_eval_per_month():
    df = fetch_data(CC_DATASET)

    df = df[df["cert_lab"].notna()].copy()
    df["not_valid_before"] = pd.to_datetime(df["not_valid_before"], unit="ms")

    cert_lab_counts = (
        df.groupby(df["not_valid_before"].dt.to_period("M"))["cert_lab"]
        .value_counts()
        .reset_index(name="count")
        .rename(columns={"not_valid_before": "month_year"})
        .sort_values("month_year")
    )

    cert_lab_counts["month_year"] = cert_lab_counts["month_year"].dt.to_timestamp()
    return cert_lab_counts


@app.callback(
    Output("cert-labs-category-stacked-bar", "figure"),
    Input("cert-labs-category-stacked-bar", "id"),
)
def cert_labs_category_stacked_bar(_):
    cert_lab_counts = get_cert_labs_eval_per_month()
    fig = px.bar(
        cert_lab_counts,
        x="month_year",
        y="count",
        color="cert_lab",
        title="Certification Labs Distribution Over Time",
        labels={
            "month_year": "Month",
            "count": "Number of Certifications",
            "cert_lab": "Cert Lab",
        },
    )

    fig.update_layout(
        xaxis_tickformat="%b %Y",
        barmode="stack",
        margin=dict(t=40, l=40, r=40, b=40),
    )
    set_figure_size(fig, height=1000, width=2400)
    set_font_size(
        fig,
        font_size_axis=22,
        font_size_title=26,
        font_size_legend=24,
        tickfont_size_axis=20,
    )

    return fig


def get_cert_labs_per_month():
    df = fetch_data(CC_DATASET)

    df = df[df["cert_lab"].notna()].copy()
    df["not_valid_before"] = pd.to_datetime(df["not_valid_before"])

    # Group by month and sum the counts
    cert_lab_counts = (
        df.groupby(df["not_valid_before"].dt.to_period("M"))["cert_lab"]
        .value_counts()
        .reset_index(name="count")
        .rename(columns={"not_valid_before": "month_year"})
        .sort_values("month_year")
        .groupby("month_year")["count"]
        .sum()
        .reset_index()
    )

    # Convert period to timestamp in one step
    cert_lab_counts["month_year"] = cert_lab_counts["month_year"].dt.to_timestamp()
    return cert_lab_counts


@app.callback(
    Output("cert-labs-graph", "figure"),
    Input("cert-labs-graph", "id"),
)
def cert_labs_graph(_):
    cert_lab_counts = get_cert_labs_per_month()

    fig = px.line(
        cert_lab_counts,
        x="month_year",
        y="count",
        color="cert_lab",
        title="Total Certification Labs Distribution Over Time",
        labels={
            "month_year": "Month",
            "count": "Number of Certifications",
            "cert_lab": "Cert Lab",
        },
    )
    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=[2000, 2004, 2008, 2012, 2016, 2020, 2024],
            range=[1997, 2025],
        ),
        legend_title="Cert Lab",
        margin=dict(t=80, l=40, r=40, b=40),
        showlegend=True,
    )

    set_figure_size(fig, height=1000, width=2000)
    set_font_size(
        fig,
        font_size_axis=22,
        font_size_title=26,
        font_size_legend=24,
        tickfont_size_axis=20,
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True)
