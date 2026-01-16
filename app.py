# app.py
# Streamlit dashboard demo: Mantenimiento predictivo (datos ficticios)
# Tabs: Tablero Semáforo / Análisis / Ranking
#
# requirements.txt:
#   streamlit
#   pandas
#   numpy
#   plotly
#   openpyxl

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

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

def pct(cond: pd.Series) -> float:
    return 100.0 * cond.mean() if len(cond) else np.nan

# -----------------------------
# Robust loaders (Cloud-friendly)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data_from_bytes(xlsx_bytes: bytes):
    bio = io.BytesIO(xlsx_bytes)
    df = pd.read_excel(bio, sheet_name="Fact_Unificada_Dashboard", engine="openpyxl")
    if safe_col(df, "fecha"):
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    for c in ["cumple_pauta_general", "se_genero_ticket", "falla_en_7_dias", "falla_en_30_dias"]:
        if safe_col(df, c):
            df[c] = df[c].apply(to_yesno)

    tickets = None
    try:
        bio2 = io.BytesIO(xlsx_bytes)
        tickets = pd.read_excel(bio2, sheet_name="Fact_Tickets", engine="openpyxl")
        if safe_col(tickets, "fecha_ticket"):
            tickets["fecha_ticket"] = pd.to_datetime(tickets["fecha_ticket"], errors="coerce")
        if safe_col(tickets, "resuelto_en_plazo"):
            tickets["resuelto_en_plazo"] = tickets["resuelto_en_plazo"].apply(to_yesno)
    except Exception:
        tickets = None

    return df, tickets

@st.cache_data(show_spinner=False)
def load_data_from_path(path: str):
    df = pd.read_excel(path, sheet_name="Fact_Unificada_Dashboard", engine="openpyxl")
    if safe_col(df, "fecha"):
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    for c in ["cumple_pauta_general", "se_genero_ticket", "falla_en_7_dias", "falla_en_30_dias"]:
        if safe_col(df, c):
            df[c] = df[c].apply(to_yesno)

    tickets = None
    try:
        tickets = pd.read_excel(path, sheet_name="Fact_Tickets", engine="openpyxl")
        if safe_col(tickets, "fecha_ticket"):
            tickets["fecha_ticket"] = pd.to_datetime(tickets["fecha_ticket"], errors="coerce")
        if safe_col(tickets, "resuelto_en_plazo"):
            tickets["resuelto_en_plazo"] = tickets["resuelto_en_plazo"].apply(to_yesno)
    except Exception:
        tickets = None

    return df, tickets

# -----------------------------
# Sidebar: data source + rules
# -----------------------------
st.sidebar.header("Fuente de datos")

DEFAULT_FILE = "BD_Mantenimiento_Predictivo_Demo_SENAPRED.xlsx"
uploaded = st.sidebar.file_uploader("Sube tu Excel (opcional)", type=["xlsx"])

if uploaded is not None:
    df, tickets = load_data_from_bytes(uploaded.getvalue())
    st.sidebar.success(f"Datos cargados desde upload: {len(df):,} filas")
else:
    local_path = Path(DEFAULT_FILE)
    if not local_path.exists():
        st.sidebar.warning(
            f"No se encontró '{DEFAULT_FILE}' en el repo. "
            "Súbelo (misma carpeta que app.py) o usa el uploader."
        )
        st.stop()
    df, tickets = load_data_from_path(str(local_path))
    st.sidebar.success(f"Datos cargados desde repo: {len(df):,} filas")

st.sidebar.divider()
st.sidebar.header("Reglas del semáforo (ajustables)")

# Umbrales recomendados (puedes cambiarlos en vivo)
yellow_score_threshold = st.sidebar.slider(
    "Umbral puntaje para AMARILLO (si el último puntaje es menor a este valor)",
    min_value=50, max_value=100, value=90
)
downtime_red_threshold = st.sidebar.slider(
    "Downtime (h) para marcar ROJO (si hay ticket reciente y downtime >= umbral)",
    min_value=0.0, max_value=24.0, value=3.0, step=0.5
)
recent_days = st.sidebar.slider(
    "Ventana 'reciente' (días) para considerar tickets/estado",
    min_value=1, max_value=60, value=14
)

st.sidebar.divider()
st.sidebar.header("Filtros")

# Date filter
if safe_col(df, "fecha") and df["fecha"].notna().any():
    min_date = df["fecha"].min().date()
    max_date = df["fecha"].max().date()
    date_range = st.sidebar.date_input(
        "Rango de fecha (eventos)",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        d0 = pd.to_datetime(date_range[0])
        d1 = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    else:
        d0 = pd.to_datetime(min_date)
        d1 = pd.to_datetime(max_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
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

if df_f.empty:
    st.title("Demo: Mantenimiento Predictivo")
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()

# -----------------------------
# Build per-asset 'latest state' table for the semaphore
# -----------------------------
df_for_state = df_f.dropna(subset=["id_activo"]).copy()
if safe_col(df_for_state, "fecha"):
    df_for_state = df_for_state.sort_values(["id_activo", "fecha"])
else:
    df_for_state["fecha"] = pd.NaT

# latest event per asset (within current filters)
latest = df_for_state.groupby("id_activo", as_index=False).tail(1).copy()

# Bring ticket features in a robust way (tickets are optional)
today_ref = (df_for_state["fecha"].max() if safe_col(df_for_state, "fecha") else pd.Timestamp.today())
if pd.isna(today_ref):
    today_ref = pd.Timestamp.today()

recent_cut = pd.Timestamp(today_ref) - pd.Timedelta(days=int(recent_days))

ticket_agg = None
if tickets is not None and len(tickets) > 0 and safe_col(tickets, "id_activo") and safe_col(tickets, "fecha_ticket"):
    t = tickets.copy()
    # Filter tickets to assets within df_f (respecting filters)
    t = t[t["id_activo"].isin(df_f["id_activo"].dropna().unique())].copy()

    t_recent = t[t["fecha_ticket"] >= recent_cut].copy()
    # Aggregate per asset (recent)
    ticket_agg = t_recent.groupby("id_activo", as_index=False).agg(
        tickets_recientes=("id_ticket", "nunique") if safe_col(t_recent, "id_ticket") else ("fecha_ticket", "size"),
        downtime_h=("tiempo_fuera_servicio_h", "sum") if safe_col(t_recent, "tiempo_fuera_servicio_h") else ("fecha_ticket", "size"),
        costo_clp=("costo_reparacion_clp", "sum") if safe_col(t_recent, "costo_reparacion_clp") else ("fecha_ticket", "size"),
        ultima_fecha_ticket=("fecha_ticket", "max"),
    )
else:
    ticket_agg = pd.DataFrame(columns=["id_activo","tickets_recientes","downtime_h","costo_clp","ultima_fecha_ticket"])

state = latest.merge(ticket_agg, on="id_activo", how="left")
for c in ["tickets_recientes","downtime_h","costo_clp"]:
    if safe_col(state, c):
        state[c] = state[c].fillna(0)
if safe_col(state, "ultima_fecha_ticket"):
    state["ultima_fecha_ticket"] = pd.to_datetime(state["ultima_fecha_ticket"], errors="coerce")

# Define semaphore rules (simple, explainable)
# ROJO:
#   - Severidad Crítica OR
#   - (tickets recientes > 0 AND downtime_h >= threshold)
# AMARILLO:
#   - Severidad Media/Alta OR
#   - (puntaje_pauta < yellow_threshold) OR
#   - falla_en_30_dias == Sí
# VERDE:
#   - Otherwise
def compute_status(row) -> str:
    sev = str(row.get("severidad", "")).strip()
    score = row.get("puntaje_pauta", np.nan)
    falla30 = str(row.get("falla_en_30_dias", "No")).strip()
    tcount = float(row.get("tickets_recientes", 0) or 0)
    down = float(row.get("downtime_h", 0) or 0)

    is_red = (sev == "Crítica") or (tcount > 0 and down >= float(downtime_red_threshold))
    if is_red:
        return "Rojo"

    is_yellow = (sev in ["Media", "Alta"]) or (pd.notna(score) and float(score) < float(yellow_score_threshold)) or (falla30 == "Sí")
    if is_yellow:
        return "Amarillo"

    return "Verde"

state["estado_semaforo"] = state.apply(compute_status, axis=1)

# Friendly label + sort key
state["estado_icono"] = state["estado_semaforo"].map({"Verde": "🟢 Verde", "Amarillo": "🟡 Amarillo", "Rojo": "🔴 Rojo"})
state["estado_rank"] = state["estado_semaforo"].map({"Rojo": 1, "Amarillo": 2, "Verde": 3}).fillna(9)

# -----------------------------
# App header + Tabs
# -----------------------------
st.title("Mantenimiento Predictivo — Tablero demo (Streamlit)")
st.caption(
    "Incluye un tablero semáforo por activo (verde/amarillo/rojo) basado en la última medición y tickets recientes."
)

tab1, tab2, tab3 = st.tabs(["Tablero Semáforo", "Análisis Predictivo", "Activos Prioritarios"])

# -----------------------------
# TAB 1: Semaphore Board
# -----------------------------
with tab1:
    st.subheader("Tablero Semáforo por activo")

    # Summary KPIs for the semaphore
    k1, k2, k3, k4 = st.columns(4)
    n_assets = state["id_activo"].nunique()
    n_green = int((state["estado_semaforo"] == "Verde").sum())
    n_yellow = int((state["estado_semaforo"] == "Amarillo").sum())
    n_red = int((state["estado_semaforo"] == "Rojo").sum())
    with k1: st.metric("Activos (con estado)", f"{n_assets:,}")
    with k2: st.metric("🟢 Verdes", f"{n_green:,}")
    with k3: st.metric("🟡 Amarillos", f"{n_yellow:,}")
    with k4: st.metric("🔴 Rojos", f"{n_red:,}")

    st.markdown("**Distribución de estado**")
    g = state.groupby("estado_semaforo", as_index=False).size().sort_values("estado_semaforo")
    fig = px.bar(g, x="estado_semaforo", y="size", labels={"size": "# Activos"})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Listado de activos (ordenado por criticidad y estado)**")
    display_cols = [
        "estado_icono",
        "id_activo",
        "tipo_activo",
        "region",
        "sede",
        "criticidad",
        "anios_servicio",
        "fecha",
        "puntaje_pauta",
        "severidad",
        "falla_en_30_dias",
        "tickets_recientes",
        "downtime_h",
        "costo_clp",
        "ultima_fecha_ticket",
        "observaciones",
    ]
    display_cols = [c for c in display_cols if safe_col(state, c)]

    # Sorting: red first, then yellow, then green; within that, criticidad and low score
    crit_rank = {"Alta": 1, "Media": 2, "Baja": 3}
    state["_crit_rank"] = state["criticidad"].map(crit_rank).fillna(9)
    state["_score_sort"] = state["puntaje_pauta"].fillna(9999)

    table = state.sort_values(["estado_rank", "_crit_rank", "_score_sort"], ascending=[True, True, True])[display_cols].copy()

    st.dataframe(
        table.style.format({
            "puntaje_pauta": "{:.0f}",
            "downtime_h": "{:.1f}",
            "costo_clp": "{:,.0f}",
        }),
        use_container_width=True,
        height=520
    )

    st.info(
        "Reglas actuales del semáforo:\n"
        f"- 🔴 Rojo: Severidad = Crítica, o tickets recientes y downtime >= {downtime_red_threshold}h.\n"
        f"- 🟡 Amarillo: Severidad Media/Alta, o puntaje < {yellow_score_threshold}, o falla_en_30_dias = Sí.\n"
        "- 🟢 Verde: resto de casos.\n"
        "Puedes ajustar umbrales en la barra lateral."
    )

# -----------------------------
# TAB 2: Predictive analysis (trend + scatter + severity/failure)
# -----------------------------
with tab2:
    st.subheader("KPIs generales (eventos filtrados)")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    n_eventos = df_f["id_evento_mant"].nunique() if safe_col(df_f, "id_evento_mant") else len(df_f)
    n_activos = df_f["id_activo"].nunique() if safe_col(df_f, "id_activo") else np.nan
    cumple_pct = 100 * (df_f["cumple_pauta_general"] == "Sí").mean() if safe_col(df_f, "cumple_pauta_general") else np.nan
    n_tickets = (df_f["se_genero_ticket"] == "Sí").sum() if safe_col(df_f, "se_genero_ticket") else np.nan
    falla30_pct = 100 * (df_f["falla_en_30_dias"] == "Sí").mean() if safe_col(df_f, "falla_en_30_dias") else np.nan
    falla7_pct = 100 * (df_f["falla_en_7_dias"] == "Sí").mean() if safe_col(df_f, "falla_en_7_dias") else np.nan

    with c1: st.metric("# Eventos", f"{n_eventos:,}")
    with c2: st.metric("# Activos", f"{n_activos:,}" if pd.notna(n_activos) else "—")
    with c3: st.metric("% Cumple pauta", f"{cumple_pct:,.1f}%" if pd.notna(cumple_pct) else "—")
    with c4: st.metric("# Tickets (flag)", f"{int(n_tickets):,}" if pd.notna(n_tickets) else "—")
    with c5: st.metric("% Falla 7 días", f"{falla7_pct:,.1f}%" if pd.notna(falla7_pct) else "—")
    with c6: st.metric("% Falla 30 días", f"{falla30_pct:,.1f}%" if pd.notna(falla30_pct) else "—")

    st.divider()

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
            st.info("No se encontró 'fecha' y/o 'puntaje_pauta'.")

    with right:
        st.subheader("Severidad vs falla futura (30 días)")
        if safe_col(df_f, "severidad") and safe_col(df_f, "falla_en_30_dias"):
            g = df_f.groupby(["severidad", "falla_en_30_dias"], as_index=False).size()
            fig = px.bar(g, x="severidad", y="size", color="falla_en_30_dias", barmode="group", labels={"size": "# Eventos"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No se encontró 'severidad' y/o 'falla_en_30_dias'.")

    st.divider()

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Variables predictivas vs falla futura")
        mode = st.radio("Vista", ["Generadores (arranque vs temperatura)", "UPS (batería vs temperatura)"], horizontal=True)

        if mode.startswith("Generadores"):
            need = ["arranque_segundos", "temp_motor_max_c", "falla_en_30_dias", "tipo_activo"]
            if all(safe_col(df_f, c) for c in need):
                d = df_f[df_f["tipo_activo"].astype(str).str.lower().str.contains("generador")].copy()
                d = d.dropna(subset=["arranque_segundos", "temp_motor_max_c"])
                if d.empty:
                    st.info("No hay filas de Generador con arranque/temperatura según filtros.")
                else:
                    fig = px.scatter(
                        d, x="arranque_segundos", y="temp_motor_max_c", color="falla_en_30_dias",
                        hover_data=[c for c in ["id_activo", "region", "puntaje_pauta", "presion_aceite_bar"] if safe_col(d, c)],
                        labels={"arranque_segundos": "Arranque (s)", "temp_motor_max_c": "Temp. motor máx (°C)"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Faltan columnas (arranque_segundos, temp_motor_max_c, falla_en_30_dias, tipo_activo).")
        else:
            need = ["bateria_salud_pct", "temperatura_ups_c", "alarmas_ups", "falla_en_30_dias", "tipo_activo"]
            if all(safe_col(df_f, c) for c in need):
                d = df_f[df_f["tipo_activo"].astype(str).str.lower().str.contains("ups")].copy()
                d = d.dropna(subset=["bateria_salud_pct", "temperatura_ups_c"])
                if d.empty:
                    st.info("No hay filas de UPS con batería/temperatura según filtros.")
                else:
                    fig = px.scatter(
                        d, x="bateria_salud_pct", y="temperatura_ups_c", color="alarmas_ups", symbol="falla_en_30_dias",
                        hover_data=[c for c in ["id_activo", "region", "puntaje_pauta", "voltaje_salida_v"] if safe_col(d, c)],
                        labels={"bateria_salud_pct": "Salud batería (%)", "temperatura_ups_c": "Temp. UPS (°C)"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Faltan columnas (bateria_salud_pct, temperatura_ups_c, alarmas_ups, falla_en_30_dias, tipo_activo).")

    with right:
        st.subheader("Distribución por rangos de puntaje")
        if safe_col(df_f, "puntaje_pauta") and safe_col(df_f, "se_genero_ticket"):
            d = df_f.dropna(subset=["puntaje_pauta"]).copy()
            bins = [0, 60, 75, 90, 101]
            labels = ["0–59", "60–74", "75–89", "90–100"]
            d["bucket_puntaje"] = pd.cut(d["puntaje_pauta"], bins=bins, labels=labels, right=False, include_lowest=True)
            g = d.groupby(["bucket_puntaje", "se_genero_ticket"], as_index=False).size()
            fig = px.bar(g, x="bucket_puntaje", y="size", color="se_genero_ticket", barmode="group", labels={"size": "# Eventos"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No se encontró 'puntaje_pauta' y/o 'se_genero_ticket'.")

# -----------------------------
# TAB 3: Ranking
# -----------------------------
with tab3:
    st.subheader("Ranking de activos (priorización)")

    needed = ["id_activo", "tipo_activo", "region", "criticidad", "anios_servicio", "puntaje_pauta", "severidad", "se_genero_ticket", "falla_en_30_dias"]
    if not all(safe_col(df_f, c) for c in needed):
        st.info("Faltan columnas para construir el ranking.")
    else:
        d = df_f.copy()

        g = d.groupby("id_activo", as_index=False).agg(
            tipo_activo=("tipo_activo", "first"),
            region=("region", "first"),
            sede=("sede", "first") if safe_col(d, "sede") else ("id_activo", "first"),
            criticidad=("criticidad", "first"),
            anios_servicio=("anios_servicio", "first"),
            puntaje_prom=("puntaje_pauta", "mean"),
            eventos=("id_evento_mant", "nunique") if safe_col(d, "id_evento_mant") else ("puntaje_pauta", "size"),
        )

        rates = d.groupby("id_activo").apply(
            lambda x: pd.Series({
                "%_AltaCrit": pct(x["severidad"].isin(["Alta", "Crítica"])),
                "%_Falla30": pct(x["falla_en_30_dias"] == "Sí"),
                "#_Tickets": int((x["se_genero_ticket"] == "Sí").sum()),
            })
        ).reset_index()

        g = g.merge(rates, on="id_activo", how="left")

        crit_weight = {"Alta": 3, "Media": 2, "Baja": 1}
        g["w_criticidad"] = g["criticidad"].map(crit_weight).fillna(1)

        g["score_prioridad"] = (
            (101 - g["puntaje_prom"].fillna(0)) * 0.6
            + g["%_Falla30"].fillna(0) * 0.3
            + g["#_Tickets"].fillna(0) * 4.0
        ) * g["w_criticidad"]

        g = g.sort_values("score_prioridad", ascending=False)

        top_n = st.slider("Mostrar Top N activos", min_value=5, max_value=min(30, len(g)), value=min(12, len(g)))
        st.dataframe(
            g.head(top_n)[
                ["id_activo", "tipo_activo", "region", "sede", "criticidad", "anios_servicio",
                 "eventos", "puntaje_prom", "%_AltaCrit", "%_Falla30", "#_Tickets", "score_prioridad"]
            ].style.format({
                "puntaje_prom": "{:.1f}",
                "%_AltaCrit": "{:.1f}",
                "%_Falla30": "{:.1f}",
                "score_prioridad": "{:.1f}",
            }),
            use_container_width=True,
            height=520
        )

st.caption(
    "Nota: el semáforo es una regla simple y explicable. En un siguiente paso, puedes reemplazarlo por un modelo "
    "predictivo (p.ej., probabilidad de falla en 30 días) manteniendo el mismo tablero."
)
