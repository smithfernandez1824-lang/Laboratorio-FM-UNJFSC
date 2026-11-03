import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
from scipy.integrate import cumulative_trapezoid
from pathlib import Path

# ============================================================
# LABORATORIO VIRTUAL DE MODULACIÓN Y DEMODULACIÓN FM
# Universidad Nacional José Faustino Sánchez Carrión
# Autor: Deyvid Fernandez | Ingeniería Electrónica
# ============================================================

st.set_page_config(
    page_title="Laboratorio FM | UNJFSC",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Identidad institucional ----------
logo_path = Path(__file__).parent / "logo_unjfsc.png"
head_left, head_right = st.columns([1, 8], vertical_alignment="center")
with head_left:
    if logo_path.exists():
        st.image(str(logo_path), width=112)
    else:
        st.markdown("## 📡")
with head_right:
    st.markdown(
        """
        <div style="padding: 2px 0 0 0;">
          <div style="font-size:0.88rem; letter-spacing:1.2px; color:#D6B94A; font-weight:700;">
            UNIVERSIDAD NACIONAL JOSÉ FAUSTINO SÁNCHEZ CARRIÓN
          </div>
          <div style="font-size:1.85rem; line-height:1.18; font-weight:800; color:#F4F7FB;">
            LABORATORIO VIRTUAL DE MODULACIÓN Y DEMODULACIÓN FM
          </div>
          <div style="font-size:0.98rem; color:#AAB7C7; margin-top:7px;">
            Análisis espectral, transmisión y recuperación de señales no sinusoidales
            en canales de radiocomunicación
          </div>
          <div style="font-size:0.88rem; color:#D6B94A; margin-top:10px; font-weight:600;">
            DEYVID FERNANDEZ · INGENIERÍA ELECTRÓNICA · PROYECTO ACADÉMICO
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <style>
      .block-container {
    padding-top: 3.5rem !important;
    padding-bottom: 2rem !important;
}
      div[data-testid="stMetric"] {
        background: rgba(31, 41, 55, 0.55);
        border: 1px solid rgba(148,163,184,0.20);
        padding: 13px 15px;
        border-radius: 12px;
      }
      div[data-testid="stMetricLabel"] {color:#C5D0DE;}
      div[data-testid="stMetricValue"] {color:#F8FAFC;}
      section[data-testid="stSidebar"] {border-right: 1px solid rgba(148,163,184,.18);}
      .small-note {color:#9CA3AF; font-size:0.86rem;}
    </style>
    """,
    unsafe_allow_html=True
)
st.divider()

# ---------- Funciones auxiliares ----------
def normalize_message(x):
    """Centra y normaliza el mensaje a pico unitario."""
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    peak = np.max(np.abs(x))
    return x / peak if peak > 1e-12 else np.zeros_like(x)


def spectrum_db(x, fs):
    """Espectro unilateral normalizado en dB, con ventana Hann."""
    x = np.asarray(x, dtype=float)
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(len(x), d=1/fs)
    mag = np.abs(X)
    db = 20 * np.log10(mag / (np.max(mag) + 1e-15) + 1e-12)
    return f, db


def rayleigh_channel(n, fs, doppler_hz, rng):
    """
    Coeficiente complejo Rayleigh normalizado.
    Doppler=0: canal constante durante la observación.
    Doppler>0: proceso correlacionado temporalmente.
    """
    if doppler_hz <= 0:
        h0 = (rng.normal() + 1j * rng.normal()) / np.sqrt(2)
        return np.full(n, h0, dtype=complex)

    # Proceso complejo AR(1): aproximación eficiente de desvanecimiento
    # Rayleigh con correlación temporal. Evita convoluciones gaussianas
    # enormes cuando el Doppler es pequeño.
    alpha = float(np.exp(-2 * np.pi * doppler_hz / fs))
    alpha = min(max(alpha, 0.0), 0.999999999)
    innovation_scale = np.sqrt(max(1.0 - alpha ** 2, 0.0))
    innovations = (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)
    h_previous = (rng.normal() + 1j * rng.normal()) / np.sqrt(2)
    h, _ = signal.lfilter(
        [innovation_scale], [1.0, -alpha], innovations,
        zi=np.array([alpha * h_previous], dtype=complex)
    )
    power = np.mean(np.abs(h) ** 2)
    return h / np.sqrt(max(power, 1e-15))


def butter_sos(cutoff, fs, order, btype):
    """Diseña un Butterworth y protege el corte respecto de Nyquist."""
    safe_cutoff = float(np.clip(cutoff, 1.0, 0.45 * fs))
    return signal.butter(order, safe_cutoff, btype=btype, fs=fs, output="sos")


def safe_sosfiltfilt(sos, x):
    """Filtrado de fase cero, con respaldo para señales muy cortas."""
    try:
        return signal.sosfiltfilt(sos, x)
    except ValueError:
        return signal.sosfilt(sos, x)


def estimate_deviation_robust(frequency, message, valid_slice):
    """Estima Δf con regresión de medianas para reducir picos de transición."""
    x = np.asarray(message[valid_slice], dtype=float)
    y = np.asarray(frequency[valid_slice], dtype=float)
    if x.size < 20 or np.std(x) < 1e-10:
        return 0.0

    levels = np.unique(np.round(x, 4))
    x_centers, f_medians = [], []
    if levels.size <= 10:
        for level in levels:
            selected = np.abs(x - level) <= 5e-4
            if np.count_nonzero(selected) >= 5:
                x_centers.append(float(np.median(x[selected])))
                f_medians.append(float(np.median(y[selected])))
    else:
        edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, 31)))
        for low, high in zip(edges[:-1], edges[1:]):
            selected = (x >= low) & (x <= high)
            if np.count_nonzero(selected) >= max(10, int(0.001 * x.size)):
                x_centers.append(float(np.median(x[selected])))
                f_medians.append(float(np.median(y[selected])))

    if len(x_centers) >= 2 and np.ptp(x_centers) > 1e-8:
        slope = np.polyfit(x_centers, f_medians, 1)[0]
        return float(abs(slope))

    # Respaldo robusto cuando no hay suficientes niveles o intervalos válidos.
    center = float(np.mean(y))
    return float(np.percentile(np.abs(y - center), 99.5))


# ---------- Controles ----------
with st.sidebar:
    st.header("⚙️ Parámetros de simulación")
    forma = st.selectbox(
        "Señal moduladora",
        ["Cuadrada", "Triangular", "Diente de sierra", "Impulsiva", "Chirp"]
    )
    fm_label = (
        "Frecuencia de referencia para Δf y duración (Hz)"
        if forma == "Chirp" else "Frecuencia base del mensaje (Hz)"
    )
    fm = st.slider(fm_label, 20, 250, 50, 5)
    beta = st.slider("Índice de modulación nominal β", 0.1, 8.0, 2.5, 0.1)
    fc = st.slider("Frecuencia de portadora (Hz)", 2000, 10000, 4000, 250)
    amplitud = st.slider("Amplitud relativa del mensaje", 0.2, 2.0, 1.0, 0.1)

    duty = st.slider("Ciclo de trabajo del tren impulsivo (%)", 2, 40, 10, 1) if forma == "Impulsiva" else 10
    if forma == "Chirp":
        chirp_f0 = st.slider("Frecuencia inicial del chirp (Hz)", 5, 200, 25, 5)
        chirp_f1 = st.slider("Frecuencia final del chirp (Hz)", 20, 500, 200, 5)
        if chirp_f1 <= chirp_f0:
            st.warning("La frecuencia final debe ser mayor que la inicial.")
            chirp_f1 = min(500, chirp_f0 + 5)
    else:
        chirp_f0, chirp_f1 = 25, 200

    st.divider()
    st.subheader("Zoom y filtrado")
    zoom_ms = st.slider("Zoom temporal del mensaje (ms)", 10, 200, 80, 5)
    zoom_rf_ms = st.select_slider("Zoom de la portadora FM (ms)", [1, 2, 5, 10, 20], value=5)
    fs_preview = max(100000, int(fc * 40))
    cutoff_max = min(5000, int(fs_preview * 0.20))
    cutoff_default = int(min(cutoff_max, max(30 * fm, 1.5 * chirp_f1 if forma == "Chirp" else 0, 1000)))
    cutoff_default = max(50, cutoff_default)
    cutoff_lp = st.slider(
        "Corte del filtro pasabajos (Hz)",
        50, max(500, cutoff_max), min(max(50, cutoff_default), max(500, cutoff_max)), 10
    )
    filter_order = st.selectbox("Orden del filtro Butterworth", [2, 4, 6, 8], index=1)
    st.caption("Un corte mayor conserva más armónicos (útil para señales cuadradas e impulsivas); un corte menor suaviza más, pero puede redondear los pulsos.")

    st.divider()
    st.subheader("Canal de radiopropagación")
    channel = st.selectbox(
        "Modelo de canal",
        ["Ideal (sin pérdidas)", "Atenuación", "Desvanecimiento Rayleigh", "Rayleigh + multitrayecto"]
    )
    attenuation_db = st.slider("Atenuación del canal (dB)", 0, 40, 0, 1)
    doppler = st.slider("Frecuencia Doppler (Hz; 0 = canal estático)", 0, 50, 0, 1)
    paths = st.selectbox("Número de trayectorias", [1, 2, 3, 4], index=2)
    delay_ms = st.slider("Retardo máximo entre trayectorias (ms)", 0.05, 1.00, 0.30, 0.05)

    st.divider()
    st.subheader("Ruido del receptor")
    snr_db = st.slider("Relación señal/ruido SNR (dB)", -5, 40, 30, 1)
    noise_on = st.toggle("Activar ruido AWGN", value=True)
    seed = st.number_input("Semilla del ruido y canal", 0, 99999, 7, 1)
    st.caption("La SNR de referencia se define respecto de la potencia transmitida, antes de las pérdidas del canal.")

# ---------- Generación del mensaje ----------
fs = max(100000, int(fc * 40))
requested_duration = 8 / fm
# Mantiene exactamente la ventana solicitada (8 periodos de referencia).
# El mínimo de 2048 muestras evita registros demasiado cortos sin alargar
# artificialmente la duración en las frecuencias altas.
N = int(np.clip(round(fs * requested_duration), 2048, 240000))
duration = N / fs
t = np.arange(N) / fs
time_ms = t * 1000
rng = np.random.default_rng(int(seed))
phase_message = (t * fm) % 1.0

if forma == "Cuadrada":
    raw_message = signal.square(2 * np.pi * fm * t)
elif forma == "Triangular":
    raw_message = signal.sawtooth(2 * np.pi * fm * t, width=0.5)
elif forma == "Diente de sierra":
    raw_message = signal.sawtooth(2 * np.pi * fm * t)
elif forma == "Impulsiva":
    pulse = np.where(phase_message < duty / 100.0, 1.0, 0.0)
    raw_message = pulse - np.mean(pulse)
else:
    raw_message = signal.chirp(
        t, f0=chirp_f0, t1=duration, f1=chirp_f1, method="linear"
    )

message_norm = normalize_message(raw_message)
message = amplitud * message_norm

# Definición consistente: beta nominal se refiere a amplitud unitaria.
# La amplitud relativa escala la desviación efectiva.
beta_effective = beta * amplitud
deviation = beta_effective * fm
instantaneous_frequency_ideal = fc + deviation * message_norm

# La fase es la integral de la frecuencia instantánea: phi(t)=2*pi*integral(f_inst dt).
phase_modulation = 2 * np.pi * cumulative_trapezoid(
    deviation * message_norm, t, initial=0.0
)
phase_carrier = 2 * np.pi * fc * t
tx = np.cos(phase_carrier + phase_modulation)

# ---------- Canal ----------
tx_analytic = signal.hilbert(tx)
linear_gain = 10 ** (-attenuation_db / 20.0)
attenuation_applied = 0.0

if channel == "Ideal (sin pérdidas)":
    rx_analytic_channel = tx_analytic.copy()
elif channel == "Atenuación":
    rx_analytic_channel = linear_gain * tx_analytic
    if doppler > 0:
        rx_analytic_channel *= np.exp(1j * 2 * np.pi * doppler * t)
    attenuation_applied = float(attenuation_db)
elif channel == "Desvanecimiento Rayleigh":
    rx_analytic_channel = linear_gain * rayleigh_channel(N, fs, doppler, rng) * tx_analytic
    attenuation_applied = float(attenuation_db)
else:
    power_profile = np.array([1.0 / (k + 1) for k in range(paths)], dtype=float)
    power_profile /= np.sum(power_profile)
    rx_analytic_channel = np.zeros(N, dtype=complex)
    for k in range(paths):
        delay_samples = 0 if k == 0 else int(round(delay_ms * 1e-3 * fs * k / max(paths - 1, 1)))
        delayed = np.zeros(N, dtype=complex)
        if delay_samples == 0:
            delayed[:] = tx_analytic
        elif delay_samples < N:
            delayed[delay_samples:] = tx_analytic[:-delay_samples]
        fading = rayleigh_channel(N, fs, doppler, rng)
        rx_analytic_channel += np.sqrt(power_profile[k]) * fading * delayed
    rx_analytic_channel *= linear_gain
    attenuation_applied = float(attenuation_db)

rx_channel = np.real(rx_analytic_channel)

# ---------- Ruido AWGN ----------
tx_power = np.mean(tx ** 2)
noise_power = tx_power / (10 ** (snr_db / 10.0))
noise = rng.normal(0.0, np.sqrt(noise_power), N) if noise_on else np.zeros(N)
rx = rx_channel + noise
snr_measured = (
    10 * np.log10(max(np.mean(rx_channel ** 2), 1e-15) / max(np.mean(noise ** 2), 1e-15))
    if noise_on else np.inf
)

# ---------- Receptor FM ----------
# El filtro RF debe contener la desviación y margen para armónicos/bandas laterales.
message_band_est = max(10 * fm, 2.0 * chirp_f1 if forma == "Chirp" else 0)
rf_margin = max(2.5 * deviation + message_band_est, 12 * fm, 1.5 * chirp_f1 if forma == "Chirp" else 0)
f_low = max(20.0, fc - rf_margin)
f_high = min(0.48 * fs, fc + rf_margin)

if f_low < f_high:
    bp_sos = signal.butter(4, [f_low, f_high], btype="bandpass", fs=fs, output="sos")
    rx_bp = safe_sosfiltfilt(bp_sos, rx)
else:
    rx_bp = rx.copy()

analytic_rx = signal.hilbert(rx_bp)
phase_increment = np.angle(analytic_rx[1:] * np.conj(analytic_rx[:-1]))
frequency_estimated = np.empty(N)
frequency_estimated[0] = fc
frequency_estimated[1:] = phase_increment * fs / (2 * np.pi)

# Recuperación sin filtrar y recuperación con pasabajos ajustable.
message_demod_raw = (frequency_estimated - fc) / max(deviation, 1e-12)
message_demod_raw -= np.mean(message_demod_raw)

lp_sos = butter_sos(cutoff_lp, fs, filter_order, "lowpass")
message_demod_filtered = safe_sosfiltfilt(lp_sos, message_demod_raw)
message_demod_filtered -= np.mean(message_demod_filtered)

# ---------- Métricas ----------
edge = max(10, min(int(0.05 * N), int(1.0 * fs / fm)))
valid = slice(edge, N - edge)
original_eval = message_norm[valid] - np.mean(message_norm[valid])
recovered_eval = message_demod_filtered[valid] - np.mean(message_demod_filtered[valid])

correlation = (
    float(np.corrcoef(original_eval, recovered_eval)[0, 1])
    if np.std(original_eval) > 1e-10 and np.std(recovered_eval) > 1e-10 else 0.0
)
rmse = float(np.sqrt(np.mean((original_eval - recovered_eval) ** 2)))

# Ajuste robusto entre el mensaje normalizado y la frecuencia estimada.
# La pendiente estima Δf y el intercepto absorbe un posible desplazamiento Doppler.
deviation_estimated = estimate_deviation_robust(
    frequency_estimated, message_norm, valid
)

# Ancho ocupado 99% (OBW): percentiles 0.5% y 99.5% de TODA la
# potencia unilateral de la señal transmitida. No se renormaliza una banda
# recortada, porque eso sobreestimaría la fracción de potencia contenida.
window = np.hanning(N)
tx_fft = np.fft.rfft(tx * window)
freq_fft = np.fft.rfftfreq(N, 1 / fs)
power_tx_spectrum = np.abs(tx_fft) ** 2
if len(power_tx_spectrum) > 2:
    power_tx_spectrum[1:-1] *= 2.0  # espectro unilateral de potencia
power_total = float(np.sum(power_tx_spectrum))
if power_total > 1e-15:
    cumulative_power = np.cumsum(power_tx_spectrum) / power_total
    i_low = min(int(np.searchsorted(cumulative_power, 0.005)), len(freq_fft) - 1)
    i_high = min(int(np.searchsorted(cumulative_power, 0.995)), len(freq_fft) - 1)
    occupied_bw_99 = float(freq_fft[i_high] - freq_fft[i_low])
else:
    occupied_bw_99 = 0.0

# ---------- Indicadores ----------
st.subheader("Indicadores principales")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Índice de modulación efectivo β", f"{beta_effective:.2f}")
m2.metric("Desviación nominal Δf", f"{deviation:.1f} Hz")
m3.metric("SNR efectiva recibida", "∞ dB" if not noise_on else f"{snr_measured:.1f} dB")
m4.metric("Correlación de recuperación", f"{correlation:.3f}")

m5, m6, m7 = st.columns(3)
m5.metric("Desviación estimada (ajuste robusto)", f"{deviation_estimated:.1f} Hz")
m6.metric("RMSE normalizado", f"{rmse:.4f}")
m7.metric("Ancho ocupado 99% (TX)", f"{occupied_bw_99:.0f} Hz")

st.caption(
    f"Señal: {forma} | fc={fc} Hz | fm base={fm} Hz | β nominal={beta:.2f} | "
    f"β efectivo={beta_effective:.2f} | Δf={deviation:.1f} Hz | "
    f"Canal: {channel} | Atenuación: {attenuation_applied:.0f} dB | "
    f"SNR referencia: {snr_db} dB | "
    f"SNR medida: {'∞ dB' if not noise_on else f'{snr_measured:.1f} dB'} | "
    f"Pasabajos: {cutoff_lp} Hz, orden {filter_order}"
)

if deviation >= fc:
    st.warning(
        "La desviación nominal es igual o superior a la portadora. "
        "Esta configuración puede llevar la frecuencia instantánea ideal a cero o valores negativos; "
        "aumenta fc o reduce β/amplitud para una simulación FM convencional."
    )
if noise_on and snr_db <= 0:
    st.info("Con SNR cercana o inferior a 0 dB, el discriminador FM puede presentar errores grandes y saltos de fase.")
recommended_cutoff = chirp_f1 if forma == "Chirp" else fm
if cutoff_lp < recommended_cutoff:
    st.warning(
        f"El corte del pasabajos ({cutoff_lp} Hz) está por debajo de la frecuencia "
        f"de referencia del mensaje ({recommended_cutoff} Hz). La señal demodulada "
        "puede quedar atenuada. Aumenta el corte para evaluar una recuperación más fiel."
    )

# ---------- Espectros ----------
f_msg, s_msg = spectrum_db(message_norm, fs)
f_tx, s_tx = spectrum_db(tx, fs)
f_rx, s_rx = spectrum_db(rx, fs)

msg_limit = min(fs / 2, max(20 * fm, 2.5 * chirp_f1 if forma == "Chirp" else 20 * fm))
msg_mask = f_msg <= msg_limit
tx_mask = (f_tx >= max(0, fc - rf_margin)) & (f_tx <= min(fs / 2, fc + rf_margin))
rx_mask = (f_rx >= max(0, fc - rf_margin)) & (f_rx <= min(fs / 2, fc + rf_margin))

# ---------- Selección de puntos para gráficos ----------
step = max(1, N // 7000)
step_rf = max(1, int(fs // max(25 * fc, 25000)))
idx_message = np.where(time_ms <= zoom_ms)[0][::step]
idx_rf = np.where(time_ms <= zoom_rf_ms)[0][::step_rf]
idx_valid = np.arange(edge, N - edge, step)
if len(idx_message) < 2:
    idx_message = np.arange(min(N, 2))
if len(idx_rf) < 2:
    idx_rf = np.arange(min(N, 2))
if len(idx_valid) < 10:
    idx_valid = np.arange(N)[::step]

# ---------- Gráficas ----------
st.subheader("Visualización y análisis de señales")
fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=(
        "1. Señal moduladora original",
        "2. Espectro de la señal moduladora",
        "3. Señal FM transmitida",
        "4. Espectro de la señal FM transmitida",
        "5. Señal recibida y filtrada en RF",
        "6. Mensaje original y demodulado",
        "7. Espectro de la señal recibida",
        "8. Frecuencia instantánea ideal y estimada"
    ),
    vertical_spacing=0.085,
    horizontal_spacing=0.10
)

colors = {
    "message": "#60A5FA",
    "message_spectrum": "#A3E635",
    "fm": "#F97316",
    "fm_spectrum": "#C084FC",
    "channel": "#22D3EE",
    "received": "#F472B6",
    "filtered": "#FBBF24",
    "original": "#22D3EE",
    "demod": "#F472B6",
    "demod_lp": "#FBBF24",
    "ideal": "#A78BFA",
    "estimated": "#34D399"
}

def add_line(x, y, name, row, col, color, width=1.6, dash="solid"):
    fig.add_trace(
        go.Scatter(
            x=x, y=y, name=name, mode="lines",
            line=dict(color=color, width=width, dash=dash)
        ),
        row=row, col=col
    )

add_line(time_ms[idx_message], message[idx_message], "Mensaje original", 1, 1, colors["message"])
add_line(f_msg[msg_mask], s_msg[msg_mask], "Espectro del mensaje", 1, 2, colors["message_spectrum"])
add_line(time_ms[idx_rf], tx[idx_rf], "FM transmitida", 2, 1, colors["fm"])
add_line(f_tx[tx_mask], s_tx[tx_mask], "Espectro FM transmitida", 2, 2, colors["fm_spectrum"])
add_line(time_ms[idx_rf], rx_channel[idx_rf], "Salida del canal sin AWGN", 3, 1, colors["channel"], 1.1, "dot")
add_line(time_ms[idx_rf], rx[idx_rf], "Señal recibida", 3, 1, colors["received"], 1.0)
add_line(time_ms[idx_rf], rx_bp[idx_rf], "Después del filtro pasa banda", 3, 1, colors["filtered"], 1.5)
add_line(time_ms[idx_valid], message_norm[idx_valid], "Mensaje original normalizado", 3, 2, colors["original"], 1.8)
add_line(time_ms[idx_valid], message_demod_raw[idx_valid], "Demodulado sin pasabajos", 3, 2, colors["demod"], 1.0, "dot")
add_line(time_ms[idx_valid], message_demod_filtered[idx_valid], "Demodulado con pasabajos", 3, 2, colors["demod_lp"], 1.7)
add_line(f_rx[rx_mask], s_rx[rx_mask], "Espectro de la señal recibida", 4, 1, colors["message_spectrum"])
add_line(time_ms[idx_valid], instantaneous_frequency_ideal[idx_valid], "Frecuencia ideal", 4, 2, colors["ideal"], 1.8, "dash")
add_line(time_ms[idx_valid], frequency_estimated[idx_valid], "Frecuencia estimada", 4, 2, colors["estimated"], 1.2)

# Ejes y cuadrículas para lectura clara.
fig.update_xaxes(title_text="Tiempo (ms)", range=[0, zoom_ms], row=1, col=1)
fig.update_xaxes(title_text="Frecuencia (Hz)", row=1, col=2)
fig.update_xaxes(title_text="Tiempo (ms)", range=[0, zoom_rf_ms], row=2, col=1)
fig.update_xaxes(title_text="Frecuencia (Hz)", row=2, col=2)
fig.update_xaxes(title_text="Tiempo (ms)", range=[0, zoom_rf_ms], row=3, col=1)
fig.update_xaxes(title_text="Tiempo (ms)", range=[0, zoom_ms], row=3, col=2)
fig.update_xaxes(title_text="Frecuencia (Hz)", row=4, col=1)
fig.update_xaxes(title_text="Tiempo (ms)", range=[0, zoom_ms], row=4, col=2)

for r, c in [(1, 1), (2, 1), (3, 1)]:
    fig.update_yaxes(title_text="Amplitud", row=r, col=c)
fig.update_yaxes(title_text="Magnitud (dB)", row=1, col=2)
fig.update_yaxes(title_text="Magnitud (dB)", row=2, col=2)
fig.update_yaxes(title_text="Amplitud normalizada", row=3, col=2, range=[-1.6, 1.6])
fig.update_yaxes(title_text="Magnitud (dB)", row=4, col=1)
freq_span = max(1.6 * deviation, 100)
fig.update_yaxes(
    title_text="Frecuencia (Hz)",
    range=[max(0, fc - freq_span), fc + freq_span],
    row=4, col=2
)
fig.update_layout(
    height=1550,
    template="plotly_dark",
    paper_bgcolor="#0B1018",
    plot_bgcolor="#0F1722",
    font=dict(family="Arial, sans-serif", size=12, color="#E5E7EB"),
    legend=dict(orientation="h", yanchor="bottom", y=1.025, xanchor="left", x=0, font=dict(size=10)),
    margin=dict(l=55, r=30, t=125, b=55),
    hovermode="x unified"
)
fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)", zerolinecolor="rgba(148,163,184,0.35)")
fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)", zerolinecolor="rgba(148,163,184,0.35)")
st.plotly_chart(fig, use_container_width=True)

# ---------- Interpretación ----------
with st.expander("📘 Interpretación de resultados y fundamento teórico", expanded=False):
    st.markdown(
        f"""
        **Configuración actual**

        - Señal moduladora: **{forma}**.
        - Frecuencia base/de referencia para la desviación: **{fm} Hz**.
        - Portadora: **{fc} Hz**.
        - Índice nominal: **β = {beta:.2f}**.
        - Amplitud relativa: **{amplitud:.2f}**.
        - Índice efectivo: **β efectivo = {beta_effective:.2f}**.
        - Desviación nominal: **Δf = β efectivo × fm = {deviation:.1f} Hz**.
        - Canal: **{channel}**.
        - SNR de referencia: **{snr_db} dB**.
        - SNR efectiva medida: **{'infinita (ruido desactivado)' if not noise_on else f'{snr_measured:.2f} dB'}**.
        - Filtro pasabajos: **{cutoff_lp} Hz**, orden **{filter_order}**.

        **Fundamento de la modulación FM**

        La frecuencia instantánea se define como
        **fᵢ(t) = fc + Δf · mₙ(t)**, donde mₙ(t) es el mensaje normalizado.
        La fase de la portadora se obtiene integrando la frecuencia instantánea.
        En esta versión se utiliza integración trapezoidal para aproximar esa integral.

        **Demodulación**

        El receptor aplica un filtro pasa banda alrededor de la portadora,
        obtiene la señal analítica mediante la transformada de Hilbert y estima
        el incremento de fase entre muestras. De esa variación obtiene la frecuencia
        instantánea y resta la frecuencia de portadora. El filtro pasabajos Butterworth
        reduce componentes de alta frecuencia y fluctuaciones residuales.

        **SNR y canal**

        La SNR de referencia se calcula antes de las pérdidas del canal. El ruido
        AWGN se añade después del canal; por ello, la atenuación puede reducir la SNR
        efectiva recibida. Rayleigh y multitrayecto son modelos académicos simplificados.
        Con Doppler igual a cero, los coeficientes Rayleigh permanecen constantes
        durante la ventana de simulación. Para Doppler distinto de cero se utiliza
        un proceso complejo AR(1) como aproximación estadística eficiente del fading;
        no es una implementación exacta del espectro clásico de Jakes.

        **Indicadores**

        La correlación mide semejanza de forma entre el mensaje original y el recuperado.
        El RMSE cuantifica la diferencia entre las señales normalizadas. La desviación
        estimada usa el percentil 99.5 de la magnitud de la desviación estimada respecto
        de la portadora; es una estimación robusta y puede variar por ruido, filtrado
        y transitorios. El ancho ocupado al 99% se calcula sobre la señal FM transmitida,
        no sobre la señal recibida con AWGN.

        **Nota sobre señales no sinusoidales**

        Las señales cuadrada, triangular, diente de sierra e impulsiva contienen armónicos.
        Por ello, el ancho de banda real depende de su contenido espectral y no queda
        descrito completamente por una sola frecuencia fundamental. Para el chirp,
        la frecuencia de la propia señal moduladora cambia linealmente durante la ventana.
        """
    )

# ---------- Exportación ----------
data = pd.DataFrame({
    "tiempo_s": t,
    "mensaje_original": message,
    "mensaje_normalizado": message_norm,
    "senal_fm_transmitida": tx,
    "senal_salida_canal_sin_awgn": rx_channel,
    "ruido_awgn": noise,
    "senal_recibida": rx,
    "senal_filtrada_rf": rx_bp,
    "frecuencia_instantanea_ideal_hz": instantaneous_frequency_ideal,
    "frecuencia_estimada_hz": frequency_estimated,
    "mensaje_demodulado_sin_filtro": message_demod_raw,
    "mensaje_demodulado_filtrado": message_demod_filtered
})

comparison = data[[
    "tiempo_s",
    "mensaje_original",
    "mensaje_normalizado",
    "mensaje_demodulado_sin_filtro",
    "mensaje_demodulado_filtrado"
]]

st.divider()
st.subheader("Guardar resultados")
download1, download2 = st.columns(2)
filename_tag = forma.lower().replace(" ", "_")
with download1:
    st.download_button(
        "⬇️ Descargar simulación completa (CSV)",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=f"laboratorio_FM_{filename_tag}_completo.csv",
        mime="text/csv",
        use_container_width=True
    )
with download2:
    st.download_button(
        "⬇️ Guardar señal original y demodulada (CSV)",
        data=comparison.to_csv(index=False).encode("utf-8"),
        file_name=f"senal_original_y_demodulada_{filename_tag}.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown(
    """
    <div style="text-align:center; color:#9CA3AF; font-size:0.82rem; padding-top:15px;">
      UNIVERSIDAD NACIONAL JOSÉ FAUSTINO SÁNCHEZ CARRIÓN · INGENIERÍA ELECTRÓNICA<br>
      Laboratorio académico de modulación FM · Deyvid Fernandez
    </div>
    """,
    unsafe_allow_html=True
)
