# 📊 Olist Customer Analysis & Segmentation 

Este repositorio contiene un framework integral de **Inteligencia de Clientes** aplicado al dataset de Olist E-commerce. El proyecto utiliza técnicas de **Machine Learning No Supervisado** para segmentar la base de usuarios y automatizar la toma de decisiones estratégicas.



## 🎯 Objetivos del Proyecto
* **Análisis RFM:** Cálculo de Recencia, Frecuencia y valor Monetario para +90k clientes únicos.
* **Clustering:** Implementación del algoritmo K-Means para la identificación de 5 perfiles de comportamiento.
* **Productización:** Despliegue de una interfaz analítica interactiva mediante Streamlit.

## 🛠️ Stack Tecnológico
* **Backend:** Python, SQLite (In-memory processing).
* **Machine Learning:** Scikit-Learn (StandardScaler, KMeans).
* **Frontend:** Streamlit, Plotly (Interactive 3D Visuals).
* **Análisis:** Pandas, NumPy.

## 🏗️ Arquitectura
El proyecto está desacoplado mediante un motor lógico (`src/engine.py`) que gestiona las operaciones pesadas de datos, permitiendo que la interfaz (`app.py`) se mantenga ligera y eficiente mediante el uso de **caching**.

El análisis completo se realiza en el notebook, (`notebooks/olist-segmentation.py`).

## 🚀 Instalación y Uso

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/denislcian/olist-analysis.git](https://github.com/denislcian/olist-analysis.git)
   cd olist-analysis
2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
3. **Ejecutar la aplicación:**
   ```bash
   streamlit run app.py
---
🧠 Modelado Matemático

Se aplicó normalización Z-Score para garantizar la convergencia del algoritmo:
$$z = \frac{(x - \mu)}{\sigma}$$
La determinación de clusters óptimos se realizó mediante el Método del Codo (Elbow Method), optimizando la Suma de Cuadrados Intra-Cluster (WCSS).

Desarrollado por Denis Lucian con IA Generativa (Gemini)

### 📦 4. Dependencias: `requirements.txt`

```text
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
streamlit>=1.25.0
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
nbformat>=4.2.0
