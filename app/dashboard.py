import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# BACKEND IMPORTS
# ============================================================

from app.pipeline import NetworkPipeline
from app.world_simulation import WorldModelSimulator
from app.config import WORLD_SCORES_PATH


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Network Defence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# BACKEND LOADERS
# ============================================================

@st.cache_resource
def load_pipeline():
    return NetworkPipeline()


@st.cache_resource
def load_simulator():
    return WorldModelSimulator()


@st.cache_data
def load_timeline():
    """
    Load the precomputed world-model evaluation results.

    This avoids rerunning the complete 550-state inference
    every time the dashboard loads.
    """

    df = pd.read_csv(
        WORLD_SCORES_PATH
    )

    df["TimeWindow"] = pd.to_datetime(
        df["TimeWindow"]
    )

    df = (
        df
        .sort_values("TimeWindow")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# LOAD BACKEND
# ============================================================

pipeline = load_pipeline()
simulator = load_simulator()
timeline = load_timeline()


# ============================================================
# GLOBAL TITLE
# ============================================================

st.title(
    "AI NETWORK DEFENCE"
)

st.caption(
    "World Model · Predictive Threat Analysis · Offline"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Network Analysis"
)


if timeline.empty:

    st.error(
        "No world-model timeline data was found."
    )

    st.stop()


timestamps = timeline[
    "TimeWindow"
].tolist()


# ------------------------------------------------------------
# Default timestamp = 09:51
# ------------------------------------------------------------

default_index = 0

for i, timestamp in enumerate(
    timestamps
):

    if timestamp.strftime(
        "%H:%M"
    ) == "09:51":

        default_index = i
        break


selected_timestamp = st.sidebar.select_slider(
    "Analysis timestamp",

    options=timestamps,

    value=timestamps[
        default_index
    ],

    format_func=lambda x: x.strftime(
        "%H:%M"
    ),
)


st.sidebar.divider()


# ------------------------------------------------------------
# Model information
# ------------------------------------------------------------

st.sidebar.subheader(
    "Model"
)

st.sidebar.write(
    "**World Model:** LSTM"
)

st.sidebar.write(
    "**State Features:** 79"
)

st.sidebar.write(
    "**Input Window:** 10 minutes"
)

st.sidebar.write(
    "**Simulation:** K-step"
)

st.sidebar.write(
    "**Mode:** Offline"
)


st.sidebar.divider()


# ------------------------------------------------------------
# Dataset information
# ------------------------------------------------------------

st.sidebar.subheader(
    "Dataset"
)

st.sidebar.write(
    "**Source:** CIC-IDS-2018"
)

st.sidebar.write(
    "**State Resolution:** 1 minute"
)

st.sidebar.write(
    "**Network States:** 570"
)

st.sidebar.write(
    "**Evaluated States:** 550"
)


# ============================================================
# CURRENT SNAPSHOT
# ============================================================

selected_timestamp = pd.Timestamp(
    selected_timestamp
)


with st.spinner(
    "Running network analysis..."
):

    snapshot = pipeline.get_snapshot(
        selected_timestamp
    )


if snapshot is None:

    st.error(
        "Unable to generate a snapshot for the selected timestamp."
    )

    st.stop()


# ============================================================
# CURRENT NETWORK STATE
# ============================================================

st.header(
    "Current Network State"
)

st.caption(
    "Output from the temporal world model and forecast engine."
)


m1, m2, m3, m4 = st.columns(
    4
)


# ------------------------------------------------------------
# Prediction MSE
# ------------------------------------------------------------

with m1:

    st.metric(
        label="Prediction MSE",

        value=f"{snapshot['prediction_mse']:.3f}",
    )


# ------------------------------------------------------------
# Forecast Score
# ------------------------------------------------------------

with m2:

    st.metric(
        label="Forecast Score",

        value=f"{snapshot['forecast_score']:.1f}/10",
    )


# ------------------------------------------------------------
# Forecast State
# ------------------------------------------------------------

with m3:

    st.metric(
        label="Forecast State",

        value=snapshot[
            "forecast_state"
        ],
    )


# ------------------------------------------------------------
# Observed State
# ------------------------------------------------------------

with m4:

    observed_state = (
        "INFILTRATION"
        if snapshot[
            "is_infiltration"
        ]
        else "NORMAL"
    )

    st.metric(
        label="Observed State",

        value=observed_state,
    )


# ============================================================
# FORWARD THREAT FORECAST
# ============================================================

st.header(
    "Forward Threat Forecast"
)

st.caption(
    "Rule-based interpretation of the world-model anomaly signal."
)


f1, f2, f3 = st.columns(
    3
)


# ------------------------------------------------------------
# 5 MIN
# ------------------------------------------------------------

with f1:

    st.metric(
        label="5 MIN",

        value=snapshot[
            "forecast_5m"
        ],

        delta=(
            f"Score "
            f"{snapshot['risk_5m']:.2f}"
        ),

        delta_color="off",
    )


# ------------------------------------------------------------
# 10 MIN
# ------------------------------------------------------------

with f2:

    st.metric(
        label="10 MIN",

        value=snapshot[
            "forecast_10m"
        ],

        delta=(
            f"Score "
            f"{snapshot['risk_10m']:.2f}"
        ),

        delta_color="off",
    )


# ------------------------------------------------------------
# 15 MIN
# ------------------------------------------------------------

with f3:

    st.metric(
        label="15 MIN",

        value=snapshot[
            "forecast_15m"
        ],

        delta=(
            f"Score "
            f"{snapshot['risk_15m']:.2f}"
        ),

        delta_color="off",
    )


st.info(
    f"**Forecast reasoning:** "
    f"{snapshot['forecast_reason']}"
)


# ============================================================
# ATTACK STAGE INTERPRETATION
# ============================================================

st.header(
    "Attack Stage Interpretation"
)

st.caption(
    "ATT&CK-oriented interpretation of the current telemetry."
)


stage = snapshot[
    "attack_stage"
]


stage_left, stage_right = st.columns(
    [1, 2]
)


with stage_left:

    st.subheader(
        stage["stage"]
    )

    st.write(
        f"**Confidence:** "
        f"{stage['confidence']}"
    )


with stage_right:

    st.write(
        stage["explanation"]
    )


st.subheader(
    "Evidence"
)


for evidence in stage.get(
    "evidence",
    []
):

    st.write(
        f"- {evidence}"
    )


st.subheader(
    "MITRE Context"
)

st.info(
    stage["mitre_context"]
)


# ============================================================
# DRIVING FEATURE ERRORS
# ============================================================

st.header(
    "Driving Feature Errors"
)

st.caption(
    "Features contributing most to the current prediction error."
)


features = snapshot.get(
    "top_features",
    []
)


if features:

    feature_names = []

    feature_values = []


    for item in features[:10]:

        # ----------------------------------------------------
        # Dictionary format
        # ----------------------------------------------------

        if isinstance(
            item,
            dict
        ):

            name = item.get(
                "feature",

                item.get(
                    "name",
                    "Unknown"
                )
            )

            value = item.get(
                "error",

                item.get(
                    "value",
                    0
                )
            )

        # ----------------------------------------------------
        # Tuple / list format
        # ----------------------------------------------------

        elif isinstance(
            item,
            (
                tuple,
                list
            )
        ):

            name = str(
                item[0]
            )

            value = float(
                item[1]
            )

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        else:

            name = str(
                item
            )

            value = 0


        feature_names.append(
            name
        )

        feature_values.append(
            float(value)
        )


    # Reverse so the largest values appear at the top
    feature_names = (
        feature_names[::-1]
    )

    feature_values = (
        feature_values[::-1]
    )


    fig = go.Figure()


    fig.add_trace(
        go.Bar(

            x=feature_values,

            y=feature_names,

            orientation="h",

            name="Feature Error",
        )
    )


    fig.update_layout(

        height=420,

        margin=dict(
            l=10,
            r=10,
            t=20,
            b=20,
        ),

        xaxis=dict(
            title="Prediction Error",

            gridcolor="#333333",
        ),

        yaxis=dict(
            title="",

            gridcolor="#222222",
        ),

        paper_bgcolor=(
            "rgba(0,0,0,0)"
        ),

        plot_bgcolor=(
            "rgba(0,0,0,0)"
        ),

        font=dict(
            color="white"
        ),

        showlegend=False,
    )


    st.plotly_chart(
        fig,

        width="stretch",

        config={
            "displayModeBar": False
        },
    )


else:

    st.info(
        "No feature-level explanation available."
    )


# ============================================================
# NETWORK ANOMALY TIMELINE
# ============================================================

st.header(
    "Network Anomaly Timeline"
)

st.caption(
    "World-model next-state prediction error over time."
)


start_time = (
    selected_timestamp
    - pd.Timedelta(
        minutes=60
    )
)


recent = timeline[
    (
        timeline[
            "TimeWindow"
        ]
        >= start_time
    )
    &
    (
        timeline[
            "TimeWindow"
        ]
        <= selected_timestamp
    )
].copy()


if not recent.empty:

    fig = go.Figure()


    # --------------------------------------------------------
    # MSE line
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(

            x=recent[
                "TimeWindow"
            ],

            y=recent[
                "Prediction_MSE"
            ],

            mode="lines",

            name="Prediction MSE",

            line=dict(
                width=2
            ),
        )
    )


    # --------------------------------------------------------
    # Selected point
    # --------------------------------------------------------

    selected_row = recent[
        recent[
            "TimeWindow"
        ]
        == selected_timestamp
    ]


    if not selected_row.empty:

        fig.add_trace(
            go.Scatter(

                x=selected_row[
                    "TimeWindow"
                ],

                y=selected_row[
                    "Prediction_MSE"
                ],

                mode="markers",

                marker=dict(
                    size=12
                ),

                name="Selected State",
            )
        )


    fig.update_layout(

        height=380,

        margin=dict(
            l=10,
            r=10,
            t=20,
            b=20,
        ),

        xaxis=dict(

            title="Time",

            gridcolor="#333333",
        ),

        yaxis=dict(

            title="Prediction MSE",

            gridcolor="#333333",
        ),

        paper_bgcolor=(
            "rgba(0,0,0,0)"
        ),

        plot_bgcolor=(
            "rgba(0,0,0,0)"
        ),

        font=dict(
            color="white"
        ),
    )


    st.plotly_chart(
        fig,

        width="stretch",

        config={
            "displayModeBar": False
        },
    )


else:

    st.info(
        "No timeline data available."
    )


# ============================================================
# K-STEP WORLD MODEL SIMULATION
# ============================================================

st.header(
    "K-Step World Model Simulation"
)

st.caption(
    "Recursive simulation of future network-state dynamics."
)


simulation_steps = st.slider(

    "Simulation horizon",

    min_value=3,

    max_value=15,

    value=5,

    step=1,
)


try:

    simulation = simulator.simulate(

        selected_timestamp,

        steps=simulation_steps,
    )


    # IMPORTANT:
    # simulate() returns a DataFrame.
    # Never use `if simulation:` here.

    if (
        simulation is not None
        and not simulation.empty
    ):

        sim_df = simulation.copy()


        # ----------------------------------------------------
        # Columns to display
        # ----------------------------------------------------

        display_columns = [

            "Future_Time",

            "Transition_Shock",

            "Future_State_Deviation",

            "Predicted_State_Change",

            "Future_Stability",

            "Simulation_Risk",
        ]


        display_columns = [

            column

            for column in display_columns

            if column in sim_df.columns
        ]


        formatted_df = sim_df[
            display_columns
        ].copy()


        # ----------------------------------------------------
        # Round numeric values
        # ----------------------------------------------------

        numeric_columns = [

            "Transition_Shock",

            "Future_State_Deviation",

            "Predicted_State_Change",

            "Future_Stability",

            "Simulation_Risk",
        ]


        for column in numeric_columns:

            if column in formatted_df.columns:

                formatted_df[
                    column
                ] = formatted_df[
                    column
                ].round(4)


        # ----------------------------------------------------
        # Simulation table
        # ----------------------------------------------------

        st.dataframe(

            formatted_df,

            width="stretch",

            hide_index=True,
        )


        # ====================================================
        # SIMULATION RISK GRAPH
        # ====================================================

        if (
            "Simulation_Risk"
            in sim_df.columns
        ):

            fig = go.Figure()


            fig.add_trace(
                go.Scatter(

                    x=sim_df[
                        "Future_Time"
                    ],

                    y=sim_df[
                        "Simulation_Risk"
                    ],

                    mode=(
                        "lines+markers"
                    ),

                    name=(
                        "Simulation Risk"
                    ),
                )
            )


            fig.update_layout(

                height=320,

                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=20,
                ),

                xaxis=dict(

                    title="Future Time",

                    gridcolor="#333333",
                ),

                yaxis=dict(

                    title="Simulation Risk",

                    gridcolor="#333333",
                ),

                paper_bgcolor=(
                    "rgba(0,0,0,0)"
                ),

                plot_bgcolor=(
                    "rgba(0,0,0,0)"
                ),

                font=dict(
                    color="white"
                ),

                showlegend=False,
            )


            st.plotly_chart(

                fig,

                width="stretch",

                config={
                    "displayModeBar": False
                },
            )


        st.caption(
            "Simulation Risk is a relative "
            "world-model signal, not a calibrated "
            "attack probability."
        )


    else:

        st.info(
            "No valid simulation available "
            "for this timestamp."
        )


except Exception as exc:

    st.error(
        f"World-model simulation failed: {exc}"
    )


# ============================================================
# RECENT NETWORK ACTIVITY
# ============================================================

st.header(
    "Recent Network Activity"
)

st.caption(
    "Recent model outputs surrounding the selected state."
)


recent_table = timeline[

    (
        timeline[
            "TimeWindow"
        ]

        >= (

            selected_timestamp

            - pd.Timedelta(
                minutes=10
            )
        )
    )

    &

    (
        timeline[
            "TimeWindow"
        ]

        <= selected_timestamp
    )

].copy()


if not recent_table.empty:

    available_columns = [

        column

        for column in [

            "TimeWindow",

            "Prediction_MSE",

        ]

        if column
        in recent_table.columns
    ]


    display_df = recent_table[
        available_columns
    ].copy()


    display_df[
        "TimeWindow"
    ] = (

        display_df[
            "TimeWindow"
        ]

        .dt.strftime(
            "%H:%M"
        )
    )


    display_df = display_df.rename(

        columns={

            "TimeWindow": "Time",

            "Prediction_MSE":
                "Prediction MSE",
        }
    )


    st.dataframe(

        display_df,

        width="stretch",

        hide_index=True,
    )


else:

    st.info(
        "No recent activity available."
    )


# ============================================================
# TECHNICAL DETAILS
# ============================================================

with st.expander(
    "Technical Details"
):

    st.json(

        {

            "timestamp": str(
                snapshot[
                    "timestamp"
                ]
            ),

            "prediction_mse": snapshot[
                "prediction_mse"
            ],

            "prediction_mae": snapshot[
                "prediction_mae"
            ],

            "forecast_score": snapshot[
                "forecast_score"
            ],

            "forecast_state": snapshot[
                "forecast_state"
            ],

            "infiltration_ratio": snapshot[
                "infiltration_ratio"
            ],

            "ground_truth": snapshot[
                "ground_truth"
            ],

            "world_model_features": 79,

            "history_window": 10,

            "forecast_horizons": [

                "5m",

                "10m",

                "15m",
            ],

            "simulation_horizon":
                simulation_steps,
        }
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Network Defence · Offline inference · LSTM World Model"
)