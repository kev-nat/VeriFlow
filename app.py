import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import random # <-- Import the random library
from datetime import date, timedelta
import plotly.graph_objects as go
        
# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="VeriFlow",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MOCK DATA CREATION ---
def create_mock_data(num_rows=100):
    """Generates a DataFrame with mock supply chain and AI analysis data."""
    start_date = date(2024, 1, 1)
    data = {
        'shipment_id': [f'SHP-{1000 + i}' for i in range(num_rows)],
        'timestamp': [start_date + timedelta(days=np.random.randint(0, 180)) for _ in range(num_rows)],
        'vehicle_gps_latitude': np.random.uniform(30, 50, num_rows),
        'vehicle_gps_longitude': np.random.uniform(-120, -70, num_rows),
        'risk_classification': np.random.choice(['Low Risk', 'Moderate Risk', 'High Risk'], num_rows, p=[0.6, 0.3, 0.1]),
    }
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Simulate AI analysis results
    risk_levels = ['LOW', 'MODERATE', 'HIGH']
    df['risk_level'] = np.random.choice(risk_levels, num_rows, p=[0.5, 0.3, 0.2])
    
    score_map = {'LOW': (0.1, 0.4), 'MODERATE': (0.4, 0.7), 'HIGH': (0.7, 1.0)}
    df['global_score'] = df['risk_level'].apply(lambda x: np.random.uniform(score_map[x][0], score_map[x][1]))

    agents = [
        'Geolocation', 'Fuel/Environment', 'Logistics',
        'Supplier', 'Cargo', 'Human Factors'
    ]
    df['anomaly_source'] = [np.random.choice(agents + [None], p=[0.1, 0.05, 0.2, 0.15, 0.1, 0.25, 0.15]) if risk != 'LOW' else None for risk in df['risk_level']]
    
    # Simulate recommendations for high-risk shipments
    recommendations_templates = [
        ["IMMEDIATE: Contact driver to address critical behavior score.", "Notify recipient of high delay probability."],
        ["IMMEDIATE: Address critical warehouse inventory levels.", "Assess route safety due to high risk level."],
        ["IMMEDIATE: Halt shipment to assess driver fatigue.", "Activate contingency plans for disruption likelihood."]
    ]
    # FIX: Use random.choice instead of np.random.choice for list of lists
    df['recommendations'] = df.apply(lambda row: random.choice(recommendations_templates) if row['risk_level'] == 'HIGH' else [], axis=1)

    return df

mock_df = create_mock_data(250)

# --- SIDEBAR / FILTERS ---
with st.sidebar:
    st.title("Dashboard Controls")
    st.markdown("---")

    date_range = st.date_input(
        "Select Date Range",
        (mock_df['timestamp'].min().date(), mock_df['timestamp'].max().date()),
        min_value=mock_df['timestamp'].min().date(),
        max_value=mock_df['timestamp'].max().date(),
    )
    
    granularity = st.selectbox(
    "Chart View",
    ["Monthly", "Quarterly", "Yearly"]
    )
        
    risk_level_filter = st.selectbox(
        "Risk Level (AI)",
        options=['All'] + sorted(list(mock_df['risk_level'].unique())),
        index=0
    )

    manual_class_filter = st.selectbox(
        "Manual Risk Level",
        options=['All'] + sorted(list(mock_df['risk_classification'].unique())),
        index=0
    )
    
    anomaly_agent_filter = st.selectbox(
    "Anomaly Source",
    options=['All'] + sorted([agent for agent in mock_df['anomaly_source'].unique() if agent is not None]),
    index=0
    )

# --- FILTERING DATA ---
filtered_df = mock_df.copy()
if date_range and len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[(filtered_df['timestamp'].dt.date >= start_date) & (filtered_df['timestamp'].dt.date <= end_date)]

if risk_level_filter != 'All':
    filtered_df = filtered_df[filtered_df['risk_level'] == risk_level_filter]

if anomaly_agent_filter != 'All':
    filtered_df = filtered_df[filtered_df['anomaly_source'] == anomaly_agent_filter]

if manual_class_filter != 'All':
    filtered_df = filtered_df[filtered_df['risk_classification'] == manual_class_filter]


# --- MAIN DASHBOARD LAYOUT ---
st.title("🚚 VeriFlow")
st.markdown("Real-time anomaly detection and risk assessment using hierarchical multi-agent system.")
st.markdown("---")


# --- KPI CARDS ---
col1, col2, col3, col4 = st.columns(4)

total_shipments = len(filtered_df)
col1.metric("Total Shipments Analyzed", f"{total_shipments}")

shipments_at_risk = len(filtered_df[filtered_df['risk_level'].isin(['MODERATE', 'HIGH'])])
delta_value = f"{((shipments_at_risk/total_shipments)*100):.1f}% of total" if total_shipments > 0 else "0%"
col2.metric("Shipments at Risk", f"{shipments_at_risk}", delta=delta_value, delta_color="inverse")

if not filtered_df['anomaly_source'].isnull().all():
    top_anomaly_source = filtered_df['anomaly_source'].mode()[0]
else:
    top_anomaly_source = "None Detected"
col3.metric("Top Anomaly Source", top_anomaly_source)

# Update: Format score as a percentage
avg_risk_score = filtered_df['global_score'].mean() if not filtered_df.empty else 0.0
col4.metric("Fleet Average Risk Score", f"{avg_risk_score:.1%}")


st.markdown("---")

# --- MAIN CHARTS & MAP ---
chart_col, map_col = st.columns([2, 2])

with chart_col:
      
    if not filtered_df.empty:
        risk_over_time = filtered_df.copy()
        
        if granularity == "Monthly":
            risk_over_time['period'] = risk_over_time['timestamp'].dt.to_period('M').astype(str)
        elif granularity == "Quarterly":
            risk_over_time['period'] = risk_over_time['timestamp'].dt.to_period('Q').astype(str)
        else:  # Yearly
            risk_over_time['period'] = risk_over_time['timestamp'].dt.to_period('Y').astype(str)

        risk_counts = risk_over_time.groupby(['period', 'risk_level']).size().reset_index(name='count')
        
        fig_risk_time = px.bar(
            risk_counts,
            x='period',
            y='count',
            color='risk_level',
            labels={'period': '', 'count': 'Total Shipments'},
            color_discrete_map={
                'LOW': '#D6FFF6',
                'MODERATE': '#4DCCBD',
                'HIGH': '#2374AB'
            },
            category_orders={'risk_level': ['LOW', 'MODERATE', 'HIGH']}
        )

        # Layout updates
        fig_risk_time.update_layout(
            title='Risk Levels Trend',
            barmode='stack',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.3,
                xanchor='center',
                x=0.5,
                font=dict(size=12),
                title=None
            )
        )

        fig_risk_time.update_traces(hovertemplate='%{y}<extra></extra>')

        st.plotly_chart(fig_risk_time, use_container_width=True, config={'displayModeBar': False})

    else:
        st.info("No data to display for 'Risk Levels Over Time' chart.")

        # --- Live Risk Map ---

    map_df = filtered_df.copy()
    map_df.dropna(subset=['vehicle_gps_latitude', 'vehicle_gps_longitude'], inplace=True)

    if not map_df.empty:
        risk_color_map = {'LOW': '#61D095', 'MODERATE': '#FFCAB1', 'HIGH': '#D90368'}
        map_df['color'] = map_df['risk_level'].map(risk_color_map)

        fig_scatter_map = px.scatter(
            map_df,
            x="vehicle_gps_longitude",
            y="vehicle_gps_latitude",
            color="risk_level",
            color_discrete_map=risk_color_map,
            category_orders={"risk_level": ["LOW", "MODERATE", "HIGH"]},
            labels={
                "vehicle_gps_longitude": "Longitude",
                "vehicle_gps_latitude": "Latitude",
            },
            hover_data={
                "shipment_id": True,
                "global_score": ':.2%',
                "vehicle_gps_latitude": False,
                "vehicle_gps_longitude": False,
                "risk_level": False,
                "color": False
            },
        )
        fig_scatter_map.update_traces(marker=dict(size=10, opacity=0.7), hovertemplate='<br>'.join([
            "Shipment: %{customdata[0]}",
            "Global Score: %{customdata[1]:.2%}"
        ]))
        fig_scatter_map.update_layout(
            title="Shipment Risk Distribution Map",
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            legend_title_text=None,
            legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5)
        )

        st.plotly_chart(fig_scatter_map, use_container_width=True, config={'displayModeBar': False})
    else:
        st.warning("No data to display on the map for the selected filters.")

# --- RIGHT COLUMN ---
with map_col:
    
    if not filtered_df['anomaly_source'].isnull().all():
        anomaly_counts = filtered_df['anomaly_source'].value_counts().reset_index()
        anomaly_counts.columns = ['agent', 'count']

        fig_anomaly_breakdown = px.bar(
            anomaly_counts.sort_values('count'), x='agent', y='count',
            labels={'agent': ' ', 'count': 'Total Anomalies'},
            color='count', color_continuous_scale=px.colors.sequential.Blues
        )
        fig_anomaly_breakdown.update_traces(hovertemplate='%{y}<extra></extra>')

        fig_anomaly_breakdown.update_layout(
            title='Anomalies Detected per Agent',
            xaxis={'categoryorder': 'total ascending'},
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_anomaly_breakdown, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("No anomaly data available for 'Anomaly Breakdown' chart.")

    # --- UNDERNEATH: Live AI Recommendations ---
    st.subheader("Live AI Recommendations")

    high_risk_shipments = filtered_df[filtered_df['risk_level'] == 'HIGH'].sort_values('global_score', ascending=False).head(3)

    if not high_risk_shipments.empty:
        for index, row in high_risk_shipments.iterrows():
            with st.container(border=True):
                st.markdown(f"**Shipment ID:** `{row['shipment_id']}` | **Risk Score:** `{row['global_score']:.1%}`")
                for rec in row['recommendations']:
                    if "IMMEDIATE:" in rec:
                        st.error(f"🚨 {rec}")
                    else:
                        st.warning(f"🔸 {rec}")
    else:
        st.success("No high-risk shipments detected for the selected filters.")


# --- DATA TABLE ---
st.markdown("---")
st.subheader("Shipment Data Explorer")
st.dataframe(filtered_df[['timestamp', 'shipment_id', 'risk_level', 'global_score', 'anomaly_source', 'risk_classification']].reset_index(drop=True))
