# app.py
# Streamlit dashboard demo: Mantenimiento predictivo (datos ficticios)
# Requiere: streamlit, pandas, numpy, plotly, openpyxl
#
# Estructura esperada del Excel:
# - Hoja "Fact_Unificada_Dashboard" (principal)
# - (Opcional) Hoja "Fact_Tickets" (para Pareto/SLA/costos)
#
# Ejecutar:
#   streamlit run app.py

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Mantenimiento Predictivo - Demo", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
YES_VALUES = {"si", "sí", "sì", "yes", "y", "true", "1"}

def to_yesno(x) -> str:
    if pd.isna(x):
        return "No"
    s = str(x).strip().lower()
    return "Sí" if s in YES_VALUES else "No"

def safe_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns

def month_period(series_date: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series_date, errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp()

@st.cache_data(show_spinner=False)
def load_data(xlsx_path: str):
    xlsx_path = str(xlsx_path)
    df = pd.read_excel(xlsx_path, sheet_name="Fact_Unificada_Dashboard", engine="openpyxl")
    # Normalize core columns
    if safe_col(df, "fecha"):
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    for c in ["cumple_pauta_general", "se_genero_ticket", "falla_en_7_dias", "falla_en_30_dias"]:
        if safe_col(df, c):
            df[c] = df[c].apply(to_yesno)

    # Optional tickets sheet
    tickets = None
    try:
        tickets = pd.read_excel(xlsx_path, sheet_name="Fact_Tickets", engine="openpyxl")
        if safe_col(tickets, "fecha_ticket"):
            tickets["fecha_ticket"] = pd.to_datetime(tickets["fecha_ticket"], errors="coerce")
        if safe_col(tickets, "resuelto_en_plazo"):
            tickets["resuelto_en_plazo"] = tickets["resuelto_en_plazo"].apply(to_yesno)
    except Exception:
        tickets = None

    return df, tickets

def kpi_card(label: str, value):
    st.metric(label, value)

# -----------------------------
# Sidebar: data source + filters
# -----------------------------
st.sidebar.header("Fuente de datos")

default_file = "BD_Mantenimiento_Predictivo_Demo_SENAPRED.xlsx"
uploaded = st.sidebar.file_uploader("Sube tu Excel (o usa el del repo)", type=["xlsx"])

if uploaded is not None:
    data_path = uploaded
    df, tickets = load_data(data_path)
else:
    # Use local file if present (recommended for GitHub deployment)
    local_path = Path(default_file)
    if not local_path.exists():
        st.error(
            f"No se encontró '{default_file}' en el repo. "
            "Sube el archivo en la barra lateral o súbelo al repositorio."
        )
        st.stop()
    df, tickets = load_data(str(local_path))

st.sidebar.divider()
st.sidebar.header("Filtros")

# Date filter
if safe_col(df, "fecha") and df["fecha"].notna().any():
    min_date = df["fecha"].min().date()
    max_date = df["fecha"].max().date()
    date_range = st.sidebar.date_input(
        "Rango de fecha",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        d0, d1 = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    else:
        d0, d1 = pd.to_datetime(min_date), pd.to_datetime(max_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
else:
    d0, d1 = None, None

def multiselect_filter(df_in: pd.DataFrame, col: str, label: str):
    if not safe_col(df_in, col):
        return df_in
    vals = sorted([v for v in df_in[col].dropna().unique()])
    sel = st.sidebar.multiselect(label, vals, default=vals)
    return df_in[df_in[col].isin(sel)] if sel else df_in.iloc[0:0]

df_f = df.copy()
if d0 is not None and safe_col(df_f, "fecha"):
    df_f = df_f[(df_f["fecha"] >= d0) & (df_f["fecha"] <= d1)]

df_f = multiselect_filter(df_f, "tipo_activo", "Tipo activo")
df_f = multiselect_filter(df_f, "region", "Región")
df_f = multiselect_filter(df_f, "sede", "Sede")
df_f = multiselect_filter(df_f, "criticidad", "Criticidad")
df_f = multiselect_filter(df_f, "tipo_mantenimiento", "Tipo mantenimiento")
df_f = multiselect_filter(df_f, "severidad", "Severidad")

# -----------------------------
# Header
# -----------------------------
st.title("Demo: Mantenimiento Predictivo (desde checklist hacia datos accionables)")
st.caption(
    "Este tablero muestra cómo pasar de una pauta Sí/No a mediciones numéricas y señales de riesgo, "
    "vinculadas a tickets/fallas para anticipación."
)

if df_f.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()

# -----------------------------
# KPIs
# -----------------------------
c1, c2, c3, c4, c5, c6 = st.columns(6)

n_eventos = df_f["id_evento_mant"].nunique() if safe_col(df_f, "id_evento_mant") else len(df_f)
n_activos = df_f["id_activo"].nunique() if safe_col(df_f, "id_activo") else np.nan

cumple_pct = np.nan
if safe_col(df_f, "cumple_pauta_general"):
    cumple_pct = 100 * (df_f["cumple_pauta_general"] == "Sí").mean()

n_tickets = (df_f["se_genero_ticket"] == "Sí").sum() if safe_col(df_f, "se_genero_ticket") else np.nan
falla30_pct = 100 * (df_f["falla_en_30_dias"] == "Sí").mean() if safe_col(df_f, "falla_en_30_dias") else np.nan
falla7_pct = 100 * (df_f["falla_en_7_dias"] == "Sí").mean() if safe_col(df_f, "falla_en_7_dias") else np.nan

with c1: kpi_card("# Eventos", f"{n_eventos:,}")
with c2: kpi_card("# Activos", f"{n_activos:,}" if pd.notna(n_activos) else "—")
with c3: kpi_card("% Cumple pauta", f"{cumple_pct:,.1f}%" if pd.notna(cumple_pct) else "—")
with c4: kpi_card("# Tickets", f"{int(n_tickets):,}" if pd.notna(n_tickets) else "—")
with c5: kpi_card("% Falla 7 días", f"{falla7_pct:,.1f}%" if pd.notna(falla7_pct) else "—")
with c6: kpi_card("% Falla 30 días", f"{falla30_pct:,.1f}%" if pd.notna(falla30_pct) else "—")

st.divider()

# -----------------------------
# Row 1: trend + severity vs future failure
# -----------------------------
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Tendencia de condición (puntaje de pauta)")
    if safe_col(df_f, "fecha") and safe_col(df_f, "puntaje_pauta"):
        tmp = df_f.dropna(subset=["fecha"]).copy()
        tmp["mes"] = month_period(tmp["fecha"])
        g = tmp.groupby("mes", as_index=False).agg(
            puntaje_prom=("puntaje_pauta", "mean"),
            eventos=("id_evento_mant", "nunique") if safe_col(tmp, "id_evento_mant") else ("puntaje_pauta", "size"),
        )
        fig = px.line(g, x="mes", y="puntaje_prom", markers=True, hover_data=["eventos"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No se encontró 'fecha' y/o 'puntaje_pauta' en los datos filtrados.")

with right:
    st.subheader("Severidad vs prob. de falla (30 días)")
    if safe_col(df_f, "severidad") and safe_col(df_f, "falla_en_30_dias"):
        g = df_f.groupby(["severidad", "falla_en_30_dias"], as_index=False).size()
        fig = px.bar(
            g, x="severidad", y="size", color="falla_en_30_dias",
            barmode="group", labels={"size": "# Eventos"}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No se encontró 'severidad' y/o 'falla_en_30_dias'.")

st.divider()

# -----------------------------
# Row 2: Scatter (predictive) + score buckets
# -----------------------------
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Relación entre variables predictivas y falla futura")
    mode = st.radio("Vista", ["Generadores (arranque vs temperatura)", "UPS (batería vs temperatura)"], horizontal=True)

    if mode.startswith("Generadores"):
        need = ["arranque_segundos", "temp_motor_max_c", "falla_en_30_dias", "tipo_activo"]
        if all(safe_col(df_f, c) for c in need):
            d = df_f[df_f["tipo_activo"].astype(str).str.lower().str.contains("generador")].copy()
            d = d.dropna(subset=["arranque_segundos", "temp_motor_max_c"])
            if d.empty:
                st.info("No hay filas de Generador con arranque/temperatura en los filtros.")
            else:
                fig = px.scatter(
                    d,
                    x="arranque_segundos", y="temp_motor_max_c",
                    color="falla_en_30_dias",
                    hover_data=[c for c in ["id_activo","region","puntaje_pauta","presion_aceite_bar"] if safe_col(d, c)],
                    labels={"arranque_segundos":"Arranque (s)", "temp_motor_max_c":"Temp. motor máx (°C)"}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Faltan columnas para esta vista (arranque_segundos, temp_motor_max_c, falla_en_30_dias, tipo_activo).")

    else:
        need = ["bateria_salud_pct", "temperatura_ups_c", "alarmas_ups", "falla_en_30_dias", "tipo_activo"]
        if all(safe_col(df_f, c) for c in need):
            d = df_f[df_f["tipo_activo"].astype(str).str.lower().str.contains("ups")].copy()
            d = d.dropna(subset=["bateria_salud_pct", "temperatura_ups_c"])
            if d.empty:
                st.info("No hay filas de UPS con batería/temperatura en los filtros.")
            else:
                fig = px.scatter(
                    d,
                    x="bateria_salud_pct", y="temperatura_ups_c",
                    color="alarmas_ups",
                    symbol="falla_en_30_dias",
                    hover_data=[c for c in ["id_activo","region","puntaje_pauta","voltaje_salida_v"] if safe_col(d, c)],
                    labels={"bateria_salud_pct":"Salud batería (%)", "temperatura_ups_c":"Temp. UPS (°C)"}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Faltan columnas para esta vista (bateria_salud_pct, temperatura_ups_c, alarmas_ups, falla_en_30_dias, tipo_activo).")

with right:
    st.subheader("Distribución por rangos de puntaje")
    if safe_col(df_f, "puntaje_pauta") and safe_col(df_f, "se_genero_ticket"):
        d = df_f.dropna(subset=["puntaje_pauta"]).copy()
        bins = [0, 60, 75, 90, 101]
        labels = ["0–59", "60–74", "75–89", "90–100"]
        d["bucket_puntaje"] = pd.cut(d["puntaje_pauta"], bins=bins, labels=labels, right=False, include_lowest=True)
        g = d.groupby(["bucket_puntaje", "se_genero_ticket"], as_index=False).size()
        fig = px.bar(g, x="bucket_puntaje", y="size", color="se_genero_ticket", barmode="group", labels={"size":"# Eventos"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No se encontró 'puntaje_pauta' y/o 'se_genero_ticket'.")

st.divider()

# -----------------------------
# Row 3: Tickets Pareto + SLA (if tickets available)
# -----------------------------
st.subheader("Gestión correctiva (tickets)")

if tickets is None or tickets.empty:
    st.info("No se encontró la hoja 'Fact_Tickets' (opcional). Se omite sección de Pareto/SLA/costos.")
else:
    # Apply same filters by date/region/type if possible (join with df_f via id_evento_mant or id_activo)
    # We'll filter tickets to those linked to current filtered fact rows.
    if safe_col(df_f, "id_evento_mant") and safe_col(tickets, "id_evento_mant"):
        tickets_f = tickets[tickets["id_evento_mant"].isin(df_f["id_evento_mant"].dropna().unique())].copy()
    elif safe_col(df_f, "id_activo") and safe_col(tickets, "id_activo"):
        tickets_f = tickets[tickets["id_activo"].isin(df_f["id_activo"].dropna().unique())].copy()
    else:
        tickets_f = tickets.copy()

    t1, t2, t3 = st.columns([1, 1, 1])

    with t1:
        st.markdown("**Pareto por tipo de falla**")
        if safe_col(tickets_f, "tipo_falla"):
            g = tickets_f.groupby("tipo_falla", as_index=False).agg(
                tickets=("id_ticket", "nunique") if safe_col(tickets_f, "id_ticket") else ("tipo_falla", "size"),
                costo=("costo_reparacion_clp", "sum") if safe_col(tickets_f, "costo_reparacion_clp") else ("tipo_falla", "size"),
            ).sort_values("tickets", ascending=False)
            fig = px.bar(g, y="tipo_falla", x="tickets", orientation="h", hover_data=["costo"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Falta columna 'tipo_falla' en tickets.")

    with t2:
        st.markdown("**Cumplimiento de SLA (resuelto en plazo)**")
        if safe_col(tickets_f, "resuelto_en_plazo"):
            g = tickets_f.groupby("resuelto_en_plazo", as_index=False).size()
            fig = px.pie(g, names="resuelto_en_plazo", values="size")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Falta columna 'resuelto_en_plazo' en tickets.")

    with t3:
        st.markdown("**Costo y downtime**")
        cost = tickets_f["costo_reparacion_clp"].sum() if safe_col(tickets_f, "costo_reparacion_clp") else np.nan
        down = tickets_f["tiempo_fuera_servicio_h"].sum() if safe_col(tickets_f, "tiempo_fuera_servicio_h") else np.nan
        st.metric("Costo total (CLP)", f"{int(cost):,}" if pd.notna(cost) else "—")
        st.metric("Horas fuera de servicio", f"{down:,.1f}" if pd.notna(down) else "—")

st.divider()

# -----------------------------
# Row 4: Ranking (priorización)
# -----------------------------
st.subheader("Ranking de activos para priorización")

rank_cols_needed = ["id_activo", "tipo_activo", "region", "criticidad", "anios_servicio", "puntaje_pauta", "severidad", "se_genero_ticket", "falla_en_30_dias"]
if all(safe_col(df_f, c) for c in rank_cols_needed):
    d = df_f.copy()

    # KPIs by asset
    def pct(cond: pd.Series) -> float:
        return 100.0 * cond.mean() if len(cond) else np.nan

    g = d.groupby("id_activo", as_index=False).agg(
        tipo_activo=("tipo_activo", "first"),
        region=("region", "first"),
        sede=("sede", "first") if safe_col(d, "sede") else ("id_activo", "first"),
        criticidad=("criticidad", "first"),
        anios_servicio=("anios_servicio", "first"),
        puntaje_prom=("puntaje_pauta", "mean"),
        eventos=("id_evento_mant", "nunique") if safe_col(d, "id_evento_mant") else ("puntaje_pauta", "size"),
    )

    # Add derived rates (computed on original rows)
    rates = d.groupby("id_activo").apply(
        lambda x: pd.Series({
            "%_AltaCrit": pct(x["severidad"].isin(["Alta","Crítica"])),
            "%_Falla30": pct(x["falla_en_30_dias"] == "Sí"),
            "#_Tickets": int((x["se_genero_ticket"] == "Sí").sum()),
        })
    ).reset_index()

    g = g.merge(rates, on="id_activo", how="left")

    # Priority score (simple heuristic)
    crit_weight = {"Alta": 3, "Media": 2, "Baja": 1}
    g["w_criticidad"] = g["criticidad"].map(crit_weight).fillna(1)

    # Lower score = worse, higher risk rates = worse, more tickets = worse
    g["score_prioridad"] = (
        (101 - g["puntaje_prom"].fillna(0)) * 0.6
        + g["%_Falla30"].fillna(0) * 0.3
        + g["#_Tickets"].fillna(0) * 4.0
    ) * g["w_criticidad"]

    g = g.sort_values("score_prioridad", ascending=False)

    top_n = st.slider("Mostrar Top N activos", min_value=5, max_value=min(30, len(g)), value=min(12, len(g)))
    st.dataframe(
        g.head(top_n)[
            ["id_activo","tipo_activo","region","sede","criticidad","anios_servicio","eventos","puntaje_prom","%_AltaCrit","%_Falla30","#_Tickets","score_prioridad"]
        ].style.format({
            "puntaje_prom":"{:.1f}",
            "%_AltaCrit":"{:.1f}",
            "%_Falla30":"{:.1f}",
            "score_prioridad":"{:.1f}",
        }),
        use_container_width=True,
        height=420
    )
else:
    st.info("Faltan columnas para construir el ranking (requiere id_activo, puntaje_pauta, severidad, falla_en_30_dias, etc.).")

st.caption(
    "Sugerencia: para GitHub/Streamlit Cloud, sube el Excel al repo o usa un enlace de descarga y cárgalo en tiempo de ejecución."
)
