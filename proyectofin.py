import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import re
import unicodedata
import datetime
import base64
import io
from nltk.corpus import stopwords
import nltk
from PIL import Image
from fpdf import FPDF
import streamlit.components.v1 as components

# Descargar stopwords si es necesario
try:
    stopwords.words('spanish')
except LookupError:
    nltk.download('stopwords')

# ============================
# CONFIGURACIÓN DE LA PÁGINA
# ============================
st.set_page_config(
    page_title="Categorizador de Reparaciones - Motos Casa Tuning", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar Session State para Facturación de forma global
if 'items_factura' not in st.session_state:
    st.session_state.items_factura = []  
if 'descargar_pdf' not in st.session_state:
    st.session_state.descargar_pdf = None
if 'trigger_download' not in st.session_state:
    st.session_state.trigger_download = False
if 'form_reset' not in st.session_state:
    st.session_state.form_reset = 0
if 'item_reset' not in st.session_state:
    st.session_state.item_reset = 0
if 'ventas_diarias' not in st.session_state:
    st.session_state.ventas_diarias = []

# ============================
# ESTILOS CSS CUSTOM
# ============================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a0b2e 0%, #2d1b4e 100%);
        color: #e8d9ff;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f051f 0%, #1f0a3a 100%);
    }

    [data-testid="stSidebar"] * {
        color: #e8d9ff !important;
    }

    h1, h2, h3, h4, h5, h6, label, .stMarkdown {
        color: #c9a5ff !important;
    }

    .stCard, .stExpander, .stAlert, [data-testid="stForm"] {
        background-color: #2a1a3e !important;
        border-radius: 14px !important;
        padding: 1rem !important;
        box-shadow: 0 10px 24px rgba(0,0,0,0.3) !important;
        border: 1px solid #4a2a6e !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #6f42c1 0%, #8e5edd 100%) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #8e5edd 0%, #a87cf5 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(110, 66, 193, 0.4);
    }

    .stDataFrame {
        border: 1px solid #6f42c1 !important;
        border-radius: 10px !important;
        background-color: #1e0f30 !important;
    }
    
    .stDataFrame thead th {
        background-color: #2d1b4e !important;
        color: #c9a5ff !important;
    }
    
    .stDataFrame tbody td {
        background-color: #1e0f30 !important;
        color: #e8d9ff !important;
    }
    
    .stTextArea textarea, .stTextInput input, .stNumberInput input, .stSelectbox select {
        background-color: #1e0f30 !important;
        color: #e8d9ff !important;
        border-color: #4a2a6e !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #8e5edd !important;
        box-shadow: 0 0 5px #8e5edd !important;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        background-color: #1e0f30 !important;
        border-color: #4a2a6e !important;
    }
    
    .stProgress > div > div {
        background-color: #6f42c1 !important;
    }
    
    .stAlert {
        background-color: #2a1a3e !important;
        border-left: 4px solid #6f42c1 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a0b2e !important;
        padding: 6px !important;
        border-radius: 12px !important;
    }

    .stTabs [data-baseweb="tab-list"] button {
        background-color: transparent !important;
        color: #c9a5ff !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #6f42c1 !important;
        color: white !important;
        box-shadow: 0px 4px 12px rgba(111, 66, 193, 0.4) !important;
    }
    
    [data-testid="stMetric"] {
        background-color: #2a1a3e !important;
        border-radius: 10px !important;
        padding: 10px !important;
        border: 1px solid #4a2a6e !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# TÍTULO CON LOGO A LA IZQUIERDA
# ============================
col_logo, col_title = st.columns([1, 5])

with col_logo:
    try:
        logo = Image.open("logotuning_120526.png")
        st.image(logo, width=150)
    except:
        st.image("https://img.icons8.com/ios-filled/100/4B0082/motorcycle.png", width=60)

with col_title:
    st.markdown("""
    <h1 style='margin: 0; color: White;'> Sistema Inteligente de Operaciones</h1>
    <p style='margin: 0; font-size: 24px; color: #c9a5ff;'>Motos Casa Tuning </p>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================
# 1. CARGAR DATOS ESTÁTICOS (CSV)
# ============================
@st.cache_data
def load_static_data():
    try:
        df = pd.read_csv("reparaciones_5000.csv")
        return df
    except FileNotFoundError:
        return None

df_static = load_static_data()

# ============================
# 2. CARGAR MODELO Y VECTORIZADOR
# ============================
@st.cache_resource
def load_model_and_vectorizer():
    try:
        model = joblib.load("modelo_predictor_categorias.pkl")
        vectorizer = joblib.load("vectorizador.pkl")
        return model, vectorizer
    except:
        return None, None

model, vectorizer = load_model_and_vectorizer()

# ============================
# 3. FUNCIONES DE PREPROCESAMIENTO Y PREDICCIÓN
# ============================
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    
    # Normalización básica para quitar acentos y pasar a minúsculas
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^a-z\s]', '', text)
    
    words = text.split()
    stop_words = set(stopwords.words('spanish'))
    filtered_words = [word for word in words if word not in stop_words]
    
    # CORRECCIÓN DE ERROR 5000 REGISTROS: Si se queda vacío tras filtros, usar texto plano base limpio
    if len(filtered_words) == 0:
        words_backup = [w for w in words if w.strip() != ""]
        if len(words_backup) == 0:
            return "reparacion general" # Categoría comodín por defecto para evitar roturas de TF-IDF
        return ' '.join(words_backup)
        
    return ' '.join(filtered_words)

def predict_category(descripcion, model, vectorizer):
    try:
        if not descripcion or not isinstance(descripcion, str) or descripcion.strip() == "":
            return "Mantenimiento Rutinario", None
        
        texto_procesado = preprocess_text(descripcion)
        texto_vectorizado = vectorizer.transform([texto_procesado])
        prediccion = model.predict(texto_vectorizado)[0]
        
        probabilidades = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilidades = model.predict_proba(texto_vectorizado)[0]
            except:
                probabilidades = None
        
        return prediccion, probabilidades
    except Exception as e:
        return "Mantenimiento Rutinario", None

def descarga_automatica(data, filename):
    b64 = base64.b64encode(data).decode()
    html = f"""
    <html><body>
        <a id="download" href="data:application/pdf;base64,{b64}" download="{filename}"></a>
        <script>
            setTimeout(function() {{
                document.getElementById("download").click();
            }}, 300);
        </script>
    </body></html>
    """
    components.html(html, height=0, width=0)

def crear_pdf_factura(cliente, moto, placa, metodo, items, total, fecha, hora):
    pdf = FPDF()
    pdf.add_page()
    
    # Encabezado Comercial Profesional
    pdf.set_fill_color(42, 26, 62) # Color morado oscuro de la app
    pdf.rect(0, 0, 210, 40, 'F')
    
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 12, "TALLER DE MOTOS CASA TUNING", ln=True, align="L")
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 5, "Del INEP una cuadra al sur - Matagalpa  |  Whatsapp: 8832-9893", ln=True, align="L")
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(15)
    
    # Bloque Informativo Cliente
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 6, "DATOS DE LA FACTURA Y CLIENTE", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    col_w = 45
    pdf.cell(col_w, 6, f"Cliente: {cliente}")
    pdf.cell(col_w, 6, f"Moto: {moto}")
    pdf.cell(col_w, 6, f"Placa: {placa}")
    pdf.cell(col_w, 6, f"Fecha: {fecha}", ln=True)
    pdf.cell(col_w, 6, f"Método de Pago: {metodo}", ln=True)
    pdf.ln(6)
    
    # Tabla de Servicios Prestados
    pdf.set_font("Arial", "B", 10)
    pdf.set_fill_color(111, 66, 193) # Encabezado tabla morado vivo
    pdf.set_text_color(255, 255, 255)
    
    pdf.cell(50, 8, " Servicio / Categoria", border=1, fill=True)
    pdf.cell(75, 8, " Descripcion", border=1, fill=True)
    pdf.cell(20, 8, " Cant.", border=1, fill=True, align="C")
    pdf.cell(25, 8, " P. Unit", border=1, fill=True, align="R")
    pdf.cell(20, 8, " Total", border=1, fill=True, align="R", ln=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=9)
    
    for item in items:
        # Codificación segura para evitar errores de caracteres especiales en FPDF
        serv = str(item['Producto/Servicio']).encode("latin-1", "replace").decode("latin-1")
        desc = str(item['Descripción']).encode("latin-1", "replace").decode("latin-1")
        
        pdf.cell(50, 7, f" {serv}", border=1)
        pdf.cell(75, 7, f" {desc}", border=1)
        pdf.cell(20, 7, f" {item['Cantidad']}", border=1, align="C")
        pdf.cell(25, 7, f" ${item['Precio Unitario']:.2f}", border=1, align="R")
        pdf.cell(20, 7, f" ${item['Subtotal']:.2f}", border=1, align="R", ln=True)
        
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(145, 8, "")
    pdf.cell(45, 8, f"TOTAL GENERAL: ${total:.2f}", border=1, align="C", ln=True)
    
    # Pie de página institucional
    pdf.ln(15)
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(111, 66, 193)
    pdf.cell(0, 10, '"Todo se reduce a no rendirse nunca"', ln=True, align="C")
    
    return io.BytesIO(pdf.output(dest="S").encode("latin-1"))

# ============================
# INTERFAZ DE PESTAÑAS (TABS)
# ============================
tab_dash, tab_factura, tab_masiva, tab_manual = st.tabs([
    "📊 Dashboard Base 5000", 
    "🧾 Facturación Profesional e Inteligente", 
    "📂 Clasificación Masiva CSV", 
    "✏️ Entrada Manual"
])

# ============================
# PESTAÑA 1: DASHBOARD ESTÁTICO
# ============================
with tab_dash:
    if df_static is not None and model is not None:
        st.header("Análisis de Reparaciones Históricas")
        
        if 'Categoria_Predicha' not in df_static.columns:
            with st.spinner("Procesando histórico de base estática..."):
                predicciones = []
                total = len(df_static)
                progress_bar = st.progress(0)
                
                for idx, desc in enumerate(df_static['Descripcion']):
                    if idx % 250 == 0:
                        progress_bar.progress(idx / total)
                    cat, _ = predict_category(desc, model, vectorizer)
                    predicciones.append(cat)
                    
                progress_bar.progress(1.0)
                df_static['Categoria_Predicha'] = predicciones
                st.success("¡Datos estáticos listos y procesados sin fallas de vacíos!")
        
        # Métricas principales en el Sidebar Dinámico
        if 'Categoria_Predicha' in df_static.columns:
            df_filtered = df_static[df_static['Categoria_Predicha'] != "Error"]
            counts = df_filtered['Categoria_Predicha'].value_counts()
            
            st.sidebar.metric("Total Registros Históricos", len(df_static))
            if st.session_state.ventas_diarias:
                df_v = pd.DataFrame(st.session_state.ventas_diarias)
                st.sidebar.metric("Ventas de Hoy Facturadas", f"${df_v['total'].sum():.2f}")
            
            col1, col2 = st.columns(2)
            with col1:
                fig_pie = px.pie(values=counts.values, names=counts.index, title="Distribución de Reparaciones", color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_pie.update_layout(paper_bgcolor='#2a1a3e', plot_bgcolor='#2a1a3e', font_color="#ECE7EF")
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                fig_bar = px.bar(x=counts.index, y=counts.values, title="Volumen por Categoría", labels={'x':'Categoría','y':'Total'}, color=counts.index, color_discrete_sequence=px.colors.qualitative.Bold)
                fig_bar.update_layout(paper_bgcolor='#2a1a3e', plot_bgcolor='#2a1a3e', font_color="#F4F1F7")
                st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("⚠️ Sube o coloca el archivo 'reparaciones_5000.csv' en la raíz del proyecto.")

# ============================
# PESTAÑA 2: FACTURACIÓN CON MODELO ML INTEGRADO
# ============================
with tab_factura:
    st.header("Generador de Facturas con Clasificación Automática")
    
    # Disparador del componente de descarga automática
    if st.session_state.trigger_download and st.session_state.descargar_pdf:
        st.success("✅ ¡Factura Procesada con Éxito y enviada a descargas del equipo!")
        descarga_automatica(st.session_state.descargar_pdf["data"], st.session_state.descargar_pdf["filename"])
        st.download_button("📥 Re-descargar Factura Manualmente", data=st.session_state.descargar_pdf["data"], file_name=st.session_state.descargar_pdf["filename"], mime="application/pdf")
        st.session_state.trigger_download = False
        st.markdown("---")

    col_f1, col_f2 = st.columns([2, 1])
    
    with col_f2:
        st.markdown("### Resumen del Día")
        if st.session_state.ventas_diarias:
            df_v = pd.DataFrame(st.session_state.ventas_diarias)
            st.dataframe(df_v[['cliente', 'producto', 'total']], use_container_width=True)
            st.metric("Caja Chica Hoy", f"${df_v['total'].sum():.2f}")
        else:
            st.info("No se han emitido facturas en esta sesión.")

    with col_f1:
        st.markdown("### Datos del Cliente")
        bloquear_datos = len(st.session_state.items_factura) > 0
        
        c_1, c_2, c_3 = st.columns(3)
        with c_1:
            fact_cliente = st.text_input("Nombre Completo", key=f"f_cli_{st.session_state.form_reset}", disabled=bloquear_datos)
        with c_2:
            fact_moto = st.text_input("Modelo de Moto", key=f"f_mot_{st.session_state.form_reset}", disabled=bloquear_datos)
        with c_3:
            fact_placa = st.text_input("Número Placa", key=f"f_pla_{st.session_state.form_reset}", disabled=bloquear_datos)
            
        fact_pago = st.selectbox("Forma de pago", ["Efectivo", "Transferencia"], key=f"f_pag_{st.session_state.form_reset}", disabled=bloquear_datos)
        
        st.markdown("---")
        st.markdown("### Detalles de la Reparación / Venta")
        
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            prod_desc = st.text_area("Descripción de la falla/servicio realizado", placeholder="Ej: Cambio de balineras delanteras y ajuste de cadena", key=f"i_desc_{st.session_state.item_reset}")
        with col_i2:
            prod_cant = st.number_input("Cantidad", min_value=1, value=1, key=f"i_cant_{st.session_state.item_reset}")
            prod_prec = st.number_input("Precio de Operación ($)", min_value=0.0, value=0.0, key=f"i_prec_{st.session_state.item_reset}")
            
        if st.button("Añadir Concepto a Factura", use_container_width=True):
            if not fact_cliente:
                st.error("Escribe el nombre del cliente antes de registrar servicios.")
            elif prod_prec <= 0:
                st.error("El precio unitario debe ser mayor a 0.")
            elif prod_desc.strip() == "":
                st.error("Ingresa el detalle físico de la reparación.")
            else:
                # El modelo clasifica automáticamente la línea de texto ingresada en caliente
                categoria_sugerida, _ = predict_category(prod_desc, model, vectorizer)
                
                nuevo_item = {
                    "Producto/Servicio": categoria_sugerida,
                    "Descripción": prod_desc,
                    "Cantidad": prod_cant,
                    "Precio Unitario": prod_prec,
                    "Subtotal": prod_cant * prod_prec
                }
                st.session_state.items_factura.append(nuevo_item)
                st.toast("🔧 Item Clasificado y añadido")
                st.session_state.item_reset += 1
                st.rerun()

        st.markdown("### Líneas Registradas")
        if st.session_state.items_factura:
            df_fac = pd.DataFrame(st.session_state.items_factura)
            df_fac_ed = st.data_editor(df_fac, use_container_width=True, num_rows="dynamic")
            
            if not df_fac_ed.equals(df_fac):
                st.session_state.items_factura = df_fac_ed.to_dict(orient="records")
                st.rerun()
                
            total_f = df_fac_ed["Subtotal"].sum() if not df_fac_ed.empty else 0.0
            st.markdown(f"#### **Total de Operación: ${total_f:.2f}**")
            
            if st.button("✅ Confirmar y Descargar Factura", use_container_width=True):
                fecha_act = datetime.datetime.now().strftime("%d/%m/%Y")
                hora_act = datetime.datetime.now().strftime("%H:%M")
                
                pdf_generado = crear_pdf_factura(fact_cliente, fact_moto, fact_placa, fact_pago, st.session_state.items_factura, total_f, fecha_act, hora_act)
                name_file = f"Factura_{fact_cliente.replace(' ', '_')}_{datetime.datetime.now().strftime('%d%m%Y')}.pdf"
                
                st.session_state.descargar_pdf = {"data": pdf_generado.getvalue(), "filename": name_file}
                
                # Sincronizar ítems de la factura procesados al panel analítico del día
                for lines in st.session_state.items_factura:
                    st.session_state.ventas_diarias.append({
                        "producto": lines["Producto/Servicio"],
                        "cliente": fact_cliente,
                        "total": lines["Subtotal"]
                    })
                    
                st.session_state.items_factura = []
                st.session_state.form_reset += 1
                st.session_state.item_reset += 1
                st.session_state.trigger_download = True
                st.rerun()
        else:
            st.info("La tabla de la factura está vacía.")

# ============================
# PESTAÑA 3: CLASIFICACIÓN MASIVA CSV
# ============================
with tab_masiva:
    st.header("Categorización Masiva de Archivos")
    uploaded_file = st.file_uploader("Sube tu archivo de reparaciones (.csv)", type=['csv'])

    if uploaded_file and model is not None:
        try:
            df = pd.read_csv(uploaded_file)
            col_desc = None
            for col in df.columns:
                if col.lower() in ['descripcion', 'descripción', 'problema', 'reparacion']:
                    col_desc = col
                    break
            
            if col_desc is None:
                st.error("No se localizó ninguna columna con formato de descripción.")
            else:
                with st.spinner("Clasificando registros..."):
                    predicciones = []
                    for desc in df[col_desc]:
                        cat, _ = predict_category(desc, model, vectorizer)
                        predicciones.append(cat)
                    df['Categoria_Predicha'] = predicciones
                
                st.success("¡Documento indexado con éxito!")
                st.dataframe(df.head(15), use_container_width=True)
                
                csv_file = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button("📥 Descargar Base Categorizada", data=csv_file, file_name="reparaciones_procesadas.csv", mime="text/csv", use_container_width=True)
        except Exception as e:
            st.error(f"Error procesando documento masivo: {e}")

# ============================
# PESTAÑA 4: CATEGORIZACIÓN MANUAL
# ============================
with tab_manual:
    st.header("Prueba Rápida del Modelo")
    manual_input = st.text_area("Ingresa la descripción individual:", placeholder="Ej: Rectificación de cilindro y cambio de pistón")
    
    if st.button("Procesar Texto", type="primary") and manual_input:
        cat, probs = predict_category(manual_input, model, vectorizer)
        st.success(f"### Categoría Asignada por Inteligencia Artificial: **{cat}**")

# ============================
# BARRA LATERAL (INFORMACIÓN)
# ============================
with st.sidebar.expander("💡 Ejemplos Rápidos"):
    st.markdown("""
    * `cambio de llanta trasera`
    * `ajuste de frenos y cable`
    * `limpieza de carburador`
    * `corto circuito en pidevias`
    """)

if model is not None:
    st.sidebar.markdown("---")
    st.sidebar.info(f"""
    **Estatus de Algoritmo:**
    * Motor: SVM + TF-IDF 
    * Estado de Datos: Corrección por vacíos activa
    """)
