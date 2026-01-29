import pandas as pd
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class AnalyticsEngine:
    """
    Motor de procesamiento para segmentación RFM y Clustering.
    Gestiona la ingesta de datos vía SQL y la ejecución de modelos de ML.
    """
    def __init__(self, orders_path, items_path, customers_path):
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        
        # Ingesta eficiente de datos locales
        pd.read_csv(orders_path).to_sql('orders', self.conn, index=False)
        pd.read_csv(items_path).to_sql('order_items', self.conn, index=False)
        pd.read_csv(customers_path).to_sql('customers', self.conn, index=False)

    def get_rfm_metrics(self):
        """Calcula métricas de Recencia, Frecuencia y Monetario mediante SQL."""
        query = """
        SELECT 
            c.customer_unique_id,
            CAST((julianday((SELECT MAX(order_purchase_timestamp) FROM orders)) - 
                 julianday(MAX(o.order_purchase_timestamp))) AS INT) AS recencia,
            COUNT(o.order_id) AS frecuencia,
            SUM(i.price) AS monetario
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN order_items i ON o.order_id = i.order_id
        WHERE o.order_status = 'delivered'
        GROUP BY 1
        """
        return pd.read_sql_query(query, self.conn)

    def execute_segmentation(self, df, k=5):
        """Aplica normalización Z-Score y algoritmo K-Means."""
        scaler = StandardScaler()
        features = ['recencia', 'frecuencia', 'monetario']
        scaled_data = scaler.fit_transform(df[features])
        
        model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        df['cluster'] = model.fit_predict(scaled_data)
        return df

    def get_cluster_profiles(self, df):
        """Genera el perfil estadístico de cada segmento."""
        return df.groupby('cluster').agg({
            'recencia': 'mean',
            'frecuencia': 'mean',
            'monetario': ['mean', 'count']
        }).reset_index()