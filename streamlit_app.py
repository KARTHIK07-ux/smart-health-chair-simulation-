import streamlit as st
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Health Chair — Simulation", layout="wide")

st.title("Smart Contactless Health Monitoring Chair — Virtual Prototype")
st.markdown("Professional dashboard simulation: signal synthesis, filtering, feature extraction, and ANN-based BP estimation.")

# Sidebar controls
fs = st.sidebar.number_input("Sampling rate (Hz)", value=100, min_value=20, max_value=1000, step=10)
hr_true = st.sidebar.slider("Simulated Heart Rate (BPM)", 50, 110, 75)
rr_true = st.sidebar.slider("Simulated Respiration Rate (BPM)", 8, 25, 18)
noise_level = st.sidebar.slider("Sensor Noise Level (0-1)", 0.0, 0.5, 0.05, step=0.01)

def synthesize_bcg(hr_bpm, rr_bpm, fs, duration_sec=8, noise=0.05):
    t = np.arange(0, duration_sec, 1/fs)
    hr_hz = hr_bpm/60.0
    rr_hz = rr_bpm/60.0
    heart = 0.8*np.sin(2*np.pi*hr_hz*t) + 0.2*np.sin(2*np.pi*2*hr_hz*t)
    respiration = 0.6*np.sin(2*np.pi*rr_hz*t)
    bcg = (1 + 0.2*respiration) * heart
    bcg += noise * np.random.normal(0,1,len(t))
    return t, bcg, respiration

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut/nyq
    high = highcut/nyq
    b,a = butter(order, [low, high], btype='band')
    return b,a

def bandpass_filter(data, lowcut, highcut, fs, order=3):
    b,a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b,a,data)
    return y

def generate_dataset(n_samples=300, fs=100):
    X, y = [], []
    for i in range(n_samples):
        hr = np.random.randint(55,95)
        rr = np.random.randint(10,22)
        sys_bp = 100 + 0.8*hr + np.random.normal(0,5)
        dia_bp = 60 + 0.5*hr + np.random.normal(0,4)
        t, bcg, resp = synthesize_bcg(hr, rr, fs, duration_sec=6, noise=0.05)
        try:
            filt = bandpass_filter(bcg, 0.5, 6, fs)
        except Exception:
            continue
        peaks, _ = find_peaks(filt, distance=fs*0.4, height=np.std(filt)*0.2)
        if len(peaks) < 3:
            continue
        ipd_mean = np.mean(np.diff(peaks))/fs
        peak_amp = np.mean(filt[peaks])
        resp_amp = np.std(resp)
        X.append([1/ipd_mean, peak_amp, resp_amp, hr, rr])
        y.append([sys_bp, dia_bp])
    return np.array(X), np.array(y)

@st.cache_resource
def train_model(fs):
    X, y = generate_dataset(400, fs=fs)
    scalerX = StandardScaler().fit(X)
    scalerY = StandardScaler().fit(y)
    Xs = scalerX.transform(X)
    Ys = scalerY.transform(y)
    model = MLPRegressor(hidden_layer_sizes=(16,8), max_iter=500, random_state=42)
    model.fit(Xs, Ys)
    return model, scalerX, scalerY

with st.spinner("Training ANN model..."):
    model, scalerX, scalerY = train_model(fs)

t, raw, resp = synthesize_bcg(hr_true, rr_true, fs, duration_sec=8, noise=noise_level)
filt = bandpass_filter(raw, 0.5, 6, fs)
peaks, _ = find_peaks(filt, distance=fs*0.4, height=np.std(filt)*0.2)
est_hr = int(60.0/np.mean(np.diff(peaks)/fs)) if len(peaks) >= 2 else 0
resp_zeros = np.where(np.diff(np.sign(resp)))[0]
est_rr = int(len(resp_zeros)/(2*(len(t)/fs/60.0))) if len(resp_zeros)>0 else 0

peak_amp = np.mean(filt[peaks]) if len(peaks)>0 else np.mean(filt)
resp_amp = np.std(resp)
feat = np.array([[est_hr, peak_amp, resp_amp, hr_true, rr_true]])
Xs = scalerX.transform(feat)
Ys_pred = model.predict(Xs)
bp_pred = scalerY.inverse_transform(Ys_pred)[0]

col1, col2, col3 = st.columns(3)
col1.metric("Heart Rate (BPM)", f"{est_hr}")
col2.metric("Respiration (BPM)", f"{est_rr}")
col3.metric("Estimated BP (SYS/DIA mmHg)", f"{int(bp_pred[0])}/{int(bp_pred[1])}")

st.subheader("Live Signal View")
fig, axes = plt.subplots(3,1, figsize=(10,6), sharex=True)
axes[0].plot(t, raw); axes[0].set_title("Raw Signal")
axes[1].plot(t, filt); axes[1].plot(t[peaks], filt[peaks], 'rx'); axes[1].set_title("Filtered Signal (0.5-6 Hz)")
axes[2].plot(t, resp); axes[2].set_title("Respiration Signal")
for ax in axes: ax.grid(True)
st.pyplot(fig)
st.info("Simulation uses synthetic signals. Replace with real-time sensor input for hardware testing.")
