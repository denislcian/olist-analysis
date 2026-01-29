import streamlit as st
import pandas as pd
import os
import plotly.express as px
from src.engine import AnalyticsEngine

# Configuración de UI
st.set_page_config(page_title="Olist Customer Analytics", layout="wide", page_icon="📊")
st.title("📊 Olist Customer Intelligence Portal")

# Definición de rutas de datos
DATA_DIR = "data"
REQUIRED_FILES = {
    "orders": os.path.join(DATA_DIR, "olist_orders_dataset.csv"),
    "items": os.path.join(DATA_DIR, "olist_order_items_dataset.csv"),
    "customers": os.path.join(DATA_DIR, "olist_customers_dataset.csv")
}

@st.cache_data
def run_pipeline(k):
    engine = AnalyticsEngine(REQUIRED_FILES["orders"], REQUIRED_FILES["items"], REQUIRED_FILES["customers"])
    rfm_df = engine.get_rfm_metrics()
    final_df = engine.execute_segmentation(rfm_df, k)
    stats = engine.get_cluster_profiles(final_df)
    return final_df, stats

# Sidebar de control
st.sidebar.header("Control de Algoritmo")
k_clusters = st.sidebar.slider("Número de Segmentos (K)", 3, 7, 5)

if all(os.path.exists(f) for f in REQUIRED_FILES.values()):
    if st.sidebar.button("Ejecutar Análisis"):
        data, profiles = run_pipeline(k_clusters)
        
        # KPIs Principales
        m1, m2, m3 = st.columns(3)
        m1.metric("Base de Clientes", f"{len(data):,}")
        m2.metric("Ticket Promedio", f"R$ {data['monetario'].mean():.2f}")
        m3.metric("Recencia Media", f"{data['recencia'].mean():.0f} días")

        # Visualización 3D
        st.subheader("Segmentación Vectorial de Clientes (RFM)")
        fig = px.scatter_3d(data, x='recencia', y='frecuencia', z='monetario', 
                            color='cluster', opacity=0.7, height=700, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # Perfilamiento de Clusters
        st.subheader("Perfiles Estadísticos por Segmento")
        st.dataframe(profiles, use_container_width=True)
        
        # Exportación
        st.download_button(
            "Descargar Segmentación Completa (CSV)",
            data.to_csv(index=False).encode('utf-8'),
            "olist_segments.csv",
            "text/csv"
        )
else:
    st.error("Error: Datasets no detectados en la carpeta /data.")