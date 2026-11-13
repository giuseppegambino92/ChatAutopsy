"""
ChatAutopsy — componenti grafici condivisi della dashboard.

Stile unico per tutti i grafici (legenda sotto il grafico, titoli che non collidono),
etichette e colonne in italiano, riquadri "Come leggerlo", callout di verdetto,
tabelle formattate e grafico a "tiro alla fune" per la pagella dell'equilibrio.
"""
from __future__ import annotations

import html
import math
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from chatautopsy.i18n import decimal_sep, get_lang, t, thousands_sep, tv

PLOT_TEMPLATE = "plotly_dark"
# Colori persone: alto contrasto, distinguibili anche da chi ha deficit rosso/verde
PERSON_COLORS = ["#4FC3F7", "#FF8A65", "#BA68C8", "#FFD54F", "#81C784", "#F06292", "#90A4AE", "#A1887F",
                 "#4DB6AC", "#E57373", "#9575CD", "#AED581"]
NEUTRAL = "#9aa3b8"
# Colori fissi per tipo di contenuto (distinti dai colori delle persone)
MEDIA_COLORS = {"Foto": "#ffb74d", "Vocali / audio": "#ba68c8", "Video": "#4db6ac", "Sticker": "#f06292",
                "GIF": "#aed581", "Documenti": "#90a4ae", "Inoltri": "#7986cb", "Posizioni": "#a1887f",
                "Contatti": "#e0e0e0", "Sondaggi": "#fff176", "Chiamate": "#4dd0e1", "Eliminati": "#e57373",
                "Altri media": "#757575", "Testo": "#cfd8dc"}
GOOD = "#5ad1a0"
WARN = "#ffcc66"
BAD = "#ff6b81"

CSS = """
<style>
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #1b2133 0%, #0e1117 55%); }
.block-container { padding-top: 2.2rem; max-width: 1500px; }
.kpi-card { background: linear-gradient(135deg, #171b26 0%, #1e2433 100%); border: 1px solid #2a3145;
            border-radius: 14px; padding: 14px 16px; height: 100%; min-height: 118px; }
.kpi-label { color: #9aa3b8; font-size: 0.74rem; text-transform: uppercase; letter-spacing: .06em; }
.kpi-value { color: #f2f3f7; font-size: 1.45rem; font-weight: 700; margin-top: 4px; line-height: 1.2;
             overflow-wrap: anywhere; }
.kpi-sub { color: #b9c0d4; font-size: 0.78rem; margin-top: 4px; }
.callout { border-radius: 10px; padding: 10px 14px; margin: 4px 0 10px 0; font-size: 0.92rem; line-height: 1.45;
           border: 1px solid; }
.callout.info { background: rgba(79,195,247,.08); border-color: rgba(79,195,247,.35); }
.callout.good { background: rgba(90,209,160,.08); border-color: rgba(90,209,160,.35); }
.callout.warn { background: rgba(255,204,102,.08); border-color: rgba(255,204,102,.40); }
.callout.bad  { background: rgba(255,107,129,.08); border-color: rgba(255,107,129,.40); }
.section-intro { color: #c3c9d8; font-size: 0.93rem; margin: -4px 0 10px 0; line-height: 1.5; }
.badge { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }
.badge.eq { background: rgba(90,209,160,.15); color: #5ad1a0; }
.badge.mild { background: rgba(255,204,102,.15); color: #ffcc66; }
.badge.strong { background: rgba(255,107,129,.15); color: #ff6b81; }
.badge.na { background: rgba(154,163,184,.15); color: #9aa3b8; }
.transcript { background: #11151f; border: 1px solid #262c3d; border-radius: 10px; padding: 12px 14px;
              font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem; white-space: pre-wrap;
              max-height: 520px; overflow-y: auto; color: #d7dae3; }
.flag { color: #ff6b81; font-weight: 600; }
div[data-testid="stExpander"] details summary p { font-size: 0.9rem; }
</style>
"""

# Nomi di colonna/assi in italiano (usati da table() e dai grafici)
LABELS: Dict[str, str] = {
    "sender": "Persona", "responder": "Chi risponde", "writer": "A chi risponde", "author": "Autore",
    "month": "Mese", "period": "Periodo", "messages": "Messaggi", "initiations": "Avvii", "pct": "%",
    "chains": "Turni", "double_text_events": "Catene ≥2 tentativi", "triple_text_events": "Catene ≥3 tentativi",
    "double_ratio_pct": "% turni con ≥2 tentativi", "triple_ratio_pct": "% turni con ≥3 tentativi",
    "max_attempts": "Max tentativi di fila", "text_messages": "Messaggi di testo", "total_chars": "Caratteri totali",
    "chars_share_pct": "% caratteri", "avg_chars": "Caratteri medi", "median_chars": "Caratteri mediani",
    "avg_words": "Parole medie", "questions": "Domande", "questions_per_100_msgs": "Domande /100 msg",
    "media_sent": "Media inviati", "emoji_sent": "Emoji inviate", "walls_of_text": "Muri di testo (>350 car.)",
    "replies": "Risposte", "median_minutes": "Mediana (min)", "mean_minutes": "Media (min)", "p90_minutes": "90° percentile (min)",
    "devaluation_index": "Indice di svalutazione", "interpretation": "Lettura",
    "stonewalling_events": "Episodi", "cold_replies": "Risposte glaciali", "long_silences": "Silenzi > 4h",
    "long_messages_received": "Messaggi lunghi ricevuti", "stonewalling_rate_pct": "% episodi su ricevuti",
    "timestamp": "Quando", "long_chars": "Caratteri", "long_text": "Messaggio lungo", "reply_text": "Risposta",
    "silence_minutes": "Silenzio (min)", "type": "Tipo", "session_id": "Sessione",
    "apologies": "Scuse", "apology_msgs": "Msg con scuse", "apologies_per_1000": "Scuse /1000 msg",
    "blame": "Colpe/accuse", "blame_msgs": "Msg con accuse", "blame_per_1000": "Accuse /1000 msg",
    "apology_blame_ratio": "Rapporto scuse/accuse", "repairs_initiated": "Paci avviate", "repair_share_pct": "% paci",
    "with_apology": "Con scuse", "median_hours_before_repair": "Ore di silenzio (mediana)",
    "conflict_start": "Inizio conflitto", "conflict_score": "Tensione", "peacemaker": "Chi ha fatto pace",
    "repair_time": "Quando", "hours_of_silence": "Ore di silenzio", "repair_text": "Messaggio di pace",
    "start_time": "Inizio", "end_time": "Fine", "message_count": "Messaggi", "participants": "Partecipanti",
    "initiator": "Ha iniziato", "emotional_intensity_score": "Intensità", "wall_of_text_count": "Muri di testo",
    "cold_reply_count": "Risposte fredde", "accusation_lexicon_hits": "Accuse",
    "term": "Termine", "count": "Occorrenze", "score": "Punteggio", "exclusivity_pct": "Esclusività %",
    "total": "Totale", "first_seen": "Prima volta", "last_seen": "Ultima volta", "peak_month": "Mese di picco",
    "peak_count": "Picco", "detected_from_corpus": "Scoperto nel corpus",
    "conversations_started": "Conversazioni avviate", "avg_messages": "Messaggi medi per conversazione",
    "median_messages": "Messaggi mediani", "avg_participants": "Partecipanti medi", "avg_duration_min": "Durata media (min)",
    "catalyst_score": "Punteggio catalizzatore", "turns_evaluated": "Turni valutati",
    "conversations_killed": "Conversazioni spente", "killer_rate_pct": "% turni seguiti da silenzio",
    "night_messages": "Messaggi notturni", "vampire_index_pct": "% notturni", "weekday": "Giorno",
    "mean_conflict": "Tensione media", "high_conflict_sessions": "Sessioni tese", "sessions": "Sessioni",
    "words": "Parole", "we": "Noi", "i": "Io", "you": "Tu/Voi", "you_absolute": "\"Tu sempre/mai\"",
    "we_index_pct": "Indice Noi %", "you_index_pct": "Indice Tu %", "i_index_pct": "Indice Io %",
    "you_absolute_per_1000_words": "\"Tu sempre/mai\" /1000 parole", "closings": "Chiusure",
    "within_1m_pct": "Entro 1 min %", "within_5m_pct": "Entro 5 min %", "over_30m_pct": "Oltre 30 min %",
    "over_2h_pct": "Oltre 2 h %", "fascia": "Fascia", "tipo": "Tipo", "emoji": "Emoji", "emoji_per_msg": "Emoji per messaggio",
    "share_pct": "Quota messaggi %", "chars_share_pct_m": "Quota caratteri %",
    "affection_per_1000": "Affetto /1000", "conflict_per_1000": "Conflitto /1000", "apology_per_1000": "Scuse /1000",
    "laughter_pct": "Risate %", "affection_conflict_ratio": "Affetto ÷ conflitto",
    "hour": "Ora", "probability": "Probabilità", "source": "Da", "target": "A", "median_latency_s": "Latenza mediana (s)",
}

LB_LABELS = {
    "msgs_per_day": "Messaggi al giorno", "median_latency_min": "Latenza mediana (min)",
    "affection_per_100": "Affetto /100 msg", "hearts_per_100": "Cuori /100 msg", "avg_chars": "Caratteri medi",
}


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def color_map(participants: List[str]) -> Dict[str, str]:
    cmap = {p: PERSON_COLORS[i % len(PERSON_COLORS)] for i, p in enumerate(participants)}
    cmap["Totale"] = "#e8eaf0"
    cmap["Total"] = "#e8eaf0"
    return cmap


def short_name(name: str) -> str:
    """Nome breve per assi e legende ("Mario Rossi" -> "Mario")."""
    parts = str(name).split()
    return parts[0] if parts else str(name)


# ------------------------------------------------------------------ testo
def section(title: str, intro: str = "") -> None:
    st.subheader(title)
    if intro:
        st.markdown(f"<div class='section-intro'>{intro}</div>", unsafe_allow_html=True)


def callout(text: str, kind: str = "info") -> None:
    """Riquadro colorato. text supporta **grassetto** markdown base."""
    import re

    body = html.escape(text)
    body = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", body)
    st.markdown(f"<div class='callout {kind}'>{body}</div>", unsafe_allow_html=True)


def explain(what: str, how: str, method: str = "", limits: str = "", label: Optional[str] = None) -> None:
    """Riquadro "Come leggerlo". I testi arrivano già tradotti da t() nel chiamante."""
    with st.expander(label or t("📖 Come leggerlo")):
        st.markdown(f"**{t('Cosa misura')}** — {what}")
        st.markdown(f"**{t('Come leggerlo')}** — {how}")
        if method:
            st.markdown(f"**{t('Metodo')}** — {method}")
        if limits:
            st.markdown(f"**{t('Limiti')}** — {limits}")


def kpi(col, label: str, value: Any, sub: str = "") -> None:
    col.markdown(
        f"<div class='kpi-card'><div class='kpi-label'>{html.escape(str(label))}</div>"
        f"<div class='kpi-value'>{html.escape(str(value))}</div>"
        f"<div class='kpi-sub'>{html.escape(str(sub))}</div></div>",
        unsafe_allow_html=True,
    )


def badge_html(verdict: str) -> str:
    v = verdict or ""
    cls = "eq" if v.startswith("Equilibrato") else "mild" if v.startswith("Lieve") else "strong" if v.startswith("Forte") else "na"
    return f"<span class='badge {cls}'>{html.escape(v)}</span>"


# ---------------------------------------------------------------- numeri
def fmt_num(v: Any, digits: int = 1) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return "—"
    if abs(f) >= 1000:
        return f"{f:,.0f}".replace(",", thousands_sep())
    if float(f).is_integer():
        return f"{int(f)}"
    return f"{f:.{digits}f}".replace(".", decimal_sep())


def fmt_minutes(m: Optional[float]) -> str:
    if m is None or (isinstance(m, float) and not np.isfinite(m)):
        return "—"
    if m < 1:
        return f"{m * 60:.0f} s"
    if m < 60:
        return f"{m:.1f} min".replace(".", decimal_sep())
    return f"{m / 60:.1f} h".replace(".", decimal_sep())


# ---------------------------------------------------------------- grafici
def style(fig: go.Figure, height: int = 380, legend: bool = True, title: Optional[str] = None,
          x_title: Optional[str] = None, y_title: Optional[str] = None) -> go.Figure:
    """Stile uniforme: titolo in alto a sinistra, legenda orizzontale SOTTO il grafico (mai sopra il titolo)."""
    if title is not None:
        fig.update_layout(title=title)
    fig.update_layout(
        template=PLOT_TEMPLATE, height=height,
        margin=dict(l=10, r=10, t=56 if (title or fig.layout.title.text) else 16, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(x=0.0, xanchor="left", y=0.98, yanchor="top", font=dict(size=15)),
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0, title_text="",
                    font=dict(size=12)),
        showlegend=legend if legend is not None else True,
        hoverlabel=dict(font_size=12),
        font=dict(size=12),
    )
    if x_title is not None:
        fig.update_xaxes(title_text=x_title)
    # con un titolo sull'asse X la legenda scende sotto il titolo (niente sovrapposizioni)
    xt = getattr(getattr(fig.layout.xaxis, "title", None), "text", None)
    if xt:
        fig.update_layout(legend_y=-0.24)
    if y_title is not None:
        fig.update_yaxes(title_text=y_title)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.12)")
    # facet: "term=amore" -> "amore"
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=", 1)[-1]) if a.text and "=" in a.text else None)
    return fig


def show(fig: go.Figure, height: int = 380, key: Optional[str] = None, **kwargs) -> None:
    st.plotly_chart(style(fig, height, **kwargs), width="stretch", key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def tug_of_war(indicators: List[Dict[str, Any]], a: str, b: str, cmap: Dict[str, str]) -> go.Figure:
    """Tutti gli indicatori in un solo grafico, raggruppati per area: quota di A sul totale A+B.
    50% = parità; la barra si allunga verso chi prevale (colore della persona)."""
    rows = []
    for ind in indicators:
        v = ind.get("valori") or {}
        va, vb = v.get(a), v.get(b)
        if va is None or vb is None or (va + vb) <= 0:
            continue
        share_a = va / (va + vb) * 100
        rows.append((t(ind["categoria"]), t(ind["indicatore"]), share_a, va, vb, t(ind.get("unita", "")), tv(ind.get("verdetto", ""))))
    fig = go.Figure()
    if not rows:
        return fig
    rows = rows[::-1]
    cats = [r[0] for r in rows]
    names = [r[1] for r in rows]
    shares = [r[2] for r in rows]
    colors = [cmap.get(a) if s_ > 50 else cmap.get(b) for s_ in shares]
    hover = [f"<b>{r[1]}</b><br>{short_name(a)}: {fmt_num(r[3], 2)} {r[5]}<br>{short_name(b)}: {fmt_num(r[4], 2)} {r[5]}"
             f"<br>{r[6]}" for r in rows]
    fig.add_bar(y=[cats, names], x=[s_ - 50 for s_ in shares], base=[50] * len(shares), orientation="h",
                marker_color=colors, hovertext=hover, hoverinfo="text", showlegend=False,
                text=[f"{s_:.0f}% · {100 - s_:.0f}%" for s_ in shares],
                # barre lunghe: etichetta dentro (fuori finirebbe sopra i nomi degli indicatori)
                textposition=["inside" if abs(s_ - 50) > 30 else "outside" for s_ in shares],
                insidetextanchor="middle", cliponaxis=False, textfont=dict(size=11))
    fig.add_vrect(x0=50 - 5.6, x1=50 + 5.6, fillcolor="rgba(90,209,160,0.10)", line_width=0)
    fig.add_vline(x=50, line_color="#e8eaf0", line_width=1)
    fig.update_xaxes(range=[0, 100], tickvals=[0, 25, 50, 75, 100], ticktext=["100%", "75%", t("parità"), "75%", "100%"],
                     side="top")
    fig.update_yaxes(tickfont=dict(size=12), dividercolor="rgba(255,255,255,0.25)", dividerwidth=1)
    fig.add_annotation(x=0, y=1.0, xref="paper", yref="paper", yanchor="bottom", showarrow=False, yshift=26,
                       text=t("◀ prevale <b>{p}</b>", p=html.escape(short_name(b))), font=dict(color=cmap.get(b), size=13), xanchor="left")
    fig.add_annotation(x=1, y=1.0, xref="paper", yref="paper", yanchor="bottom", showarrow=False, yshift=26,
                       text=t("prevale <b>{p}</b> ▶", p=html.escape(short_name(a))), font=dict(color=cmap.get(a), size=13), xanchor="right")
    fig.update_layout(height=110 + 30 * len(rows), bargap=0.3, margin=dict(t=70))
    return fig


def gauge(value: Optional[float], title: str, height: int = 240) -> go.Figure:
    v = float(value) if value is not None else 0.0
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=v, number=dict(suffix="/100", font=dict(size=34)),
        title=dict(text=title, font=dict(size=14)),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#e8eaf0", thickness=0.25),
                   steps=[dict(range=[0, 60], color="rgba(255,107,129,.35)"),
                          dict(range=[60, 80], color="rgba(255,204,102,.35)"),
                          dict(range=[80, 100], color="rgba(90,209,160,.35)")]),
    ))
    fig.update_layout(height=height, margin=dict(l=36, r=36, t=56, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      template=PLOT_TEMPLATE)
    return fig


def calendar_heatmap(daily: pd.DataFrame, year: int, value_col: str = "total") -> go.Figure:
    """Calendario stile GitHub: colonne = settimane, righe = giorni della settimana."""
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"].dt.year == year]
    full = pd.DataFrame({"date": pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")})
    d = full.merge(d[["date", value_col]], on="date", how="left")
    d["week"] = ((d["date"] - pd.Timestamp(f"{year}-01-01")).dt.days + pd.Timestamp(f"{year}-01-01").weekday()) // 7
    d["wd"] = d["date"].dt.day_of_week
    z = d.pivot(index="wd", columns="week", values=value_col)
    txt = d.pivot(index="wd", columns="week", values="date")
    days = (["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] if get_lang() == "en"
            else ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"])
    fig = go.Figure(go.Heatmap(
        z=z.values, x=list(z.columns), y=days, colorscale="Viridis", xgap=2, ygap=2,
        customdata=[[d_.strftime("%Y-%m-%d" if get_lang() == "en" else "%d/%m/%Y") if isinstance(d_, pd.Timestamp) else ""
                     for d_ in row] for row in txt.values],
        hovertemplate="%{customdata}: %{z} " + t("messaggi") + "<extra></extra>", colorbar=dict(title="msg", thickness=10),
    ))
    month_starts = d[d["date"].dt.day == 1]
    mesi = (["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] if get_lang() == "en"
            else ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu", "Lug", "Ago", "Set", "Ott", "Nov", "Dic"])
    fig.update_xaxes(tickvals=month_starts["week"].tolist(), ticktext=[mesi[m - 1] for m in month_starts["date"].dt.month],
                     showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


# ---------------------------------------------------------------- tabelle
# Colonne i cui VALORI sono etichette dell'engine da tradurre (mai i testi dei messaggi)
TRANSLATE_VALUE_COLS = {"type", "interpretation", "tipo", "fascia", "giorno", "weekday", "status", "verdetto", "segnale"}


def table(df: pd.DataFrame, height: Optional[int] = None, key: Optional[str] = None, select: bool = False,
          rename: Optional[Dict[str, str]] = None, wide_text: Iterable[str] = (), **kwargs):
    """st.dataframe con intestazioni tradotte, numeri arrotondati e colonne di testo larghe."""
    if df is None:
        return None
    d = df.copy()
    labels = dict(LABELS)
    if rename:
        labels.update(rename)
    for c in d.columns:
        if len(d) and isinstance(d[c].iloc[0], (list, tuple)):
            d[c] = d[c].apply(lambda x: ", ".join(map(str, x)) if isinstance(x, (list, tuple)) else x)
        if c == "participants":
            d[c] = d[c].astype(str).str.replace(r'[\[\]"]', "", regex=True)
        if c in TRANSLATE_VALUE_COLS:
            d[c] = d[c].map(tv)
    labels = {k: t(v) for k, v in labels.items()}
    d = d.rename(columns={c: labels.get(c, c) for c in d.columns})
    cfg = {}
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            cfg[c] = st.column_config.NumberColumn(c, format="%.2f")
    for c in wide_text:
        lc = labels.get(c, c)
        if lc in d.columns:
            cfg[lc] = st.column_config.TextColumn(lc, width="large")
    extra = dict(on_select="rerun", selection_mode="single-row") if select else {}
    if height is not None:
        extra["height"] = int(height)
    return st.dataframe(d, hide_index=True, width="stretch", key=key, column_config=cfg, **extra, **kwargs)


def csv_button(df: pd.DataFrame, filename: str, label: Optional[str] = None, key: Optional[str] = None) -> None:
    st.download_button(label or t("⬇️ Scarica CSV"), df.to_csv(index=False).encode("utf-8"), file_name=filename,
                       mime="text/csv", key=key)
