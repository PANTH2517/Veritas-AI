import os
import numpy as np

try:
    import scipy.io.wavfile as wav
    import scipy.signal as signal
    _SCIPY_OK = True
except ImportError:
    wav = None
    signal = None
    _SCIPY_OK = False

WEIGHTS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "voice_model_weights.npz"
)

class DeepfakeVoiceDetector:
    def __init__(self):
        # Load pre-trained weights from numpy file
        if os.path.exists(WEIGHTS_FILE):
            self.weights = np.load(WEIGHTS_FILE)
            print(f"Loaded voice model weights from {WEIGHTS_FILE}")
        else:
            raise FileNotFoundError(f"Voice weights file not found: {WEIGHTS_FILE}")

    def run_mlp(self, x):
        # Input shape (6,)
        # Layer 0: w_0 shape (6, 12), b_0 (12,)
        # Layer 1: w_1 shape (12, 6), b_1 (6,)
        # Layer 2: w_2 shape (6, 1), b_2 (1,)
        
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)
            
        # Dense 0 + ReLU
        w0 = self.weights['w_0']
        b0 = self.weights['b_0']
        x = np.dot(x, w0) + b0
        x = np.maximum(0, x)
        
        # Dense 1 + ReLU
        w1 = self.weights['w_1']
        b1 = self.weights['b_1']
        x = np.dot(x, w1) + b1
        x = np.maximum(0, x)
        
        # Dense 2 + Sigmoid
        w2 = self.weights['w_2']
        b2 = self.weights['b_2']
        x = np.dot(x, w2) + b2
        x = 1.0 / (1.0 + np.exp(-x))
        
        return float(x[0, 0])

    def reload_weights(self):
        if os.path.exists(WEIGHTS_FILE):
            try:
                self.weights = np.load(WEIGHTS_FILE)
                print("Voice model weights reloaded successfully.")
                return True
            except Exception as e:
                print(f"Error reloading voice model weights: {e}")
        return False

    def train_on_data(self, features_matrix, labels):
        print("Training is not supported in CPU/serverless inference runtime.")
        return {}

    def extract_audio_features(self, rate, data):
        """Extract spectral features from raw audio data.
        Assumes rate is 16000 and data is a 1D numpy float or int array.
        """
        if not _SCIPY_OK:
            raise RuntimeError("SciPy is not installed. Audio features extraction is unavailable.")
        if data is None or len(data) == 0:
            return np.zeros(6, dtype=np.float32)
            
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
            
        f, t, Zxx = signal.stft(data, fs=rate, nperseg=512, noverlap=256)
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        num_frames = magnitude.shape[1]
        if num_frames < 2:
            return np.zeros(6, dtype=np.float32)
            
        # 1. High-Frequency Energy Ratio
        high_freq_indices = np.where((f >= 6000) & (f <= 8000))[0]
        mid_freq_indices = np.where((f >= 1000) & (f <= 4000))[0]
        
        high_energy = np.sum(magnitude[high_freq_indices, :], axis=0) if len(high_freq_indices) > 0 else np.zeros(num_frames)
        mid_energy = np.sum(magnitude[mid_freq_indices, :], axis=0) if len(mid_freq_indices) > 0 else np.ones(num_frames) * 1e-5
        hf_ratio = high_energy / (mid_energy + 1e-5)
        
        hf_mean = np.mean(hf_ratio)
        hf_std = np.std(hf_ratio)
        hf_feat = np.clip((hf_mean * 2.0 + hf_std * 5.0) / 0.8, 0.0, 1.0)
        
        # 2. Phase Inconsistency
        phase_diff = np.diff(phase, axis=1)
        phase_diff2 = np.diff(phase_diff, axis=1)
        phase_incoherence = np.var(phase_diff2)
        phase_feat = np.clip(phase_incoherence / 3.5, 0.0, 1.0)
        
        # 3. Spectral Flux
        norm_mag = magnitude / (np.sum(magnitude, axis=0) + 1e-6)
        flux = np.sqrt(np.sum(np.diff(norm_mag, axis=1)**2, axis=0))
        flux_mean = np.mean(flux)
        flux_var = np.var(flux)
        flux_feat = np.clip((abs(flux_mean - 0.15) * 3.0 + flux_var * 20.0), 0.0, 1.0)
        
        # 4. Spectral Centroid
        freq_grid = f[:, np.newaxis]
        centroid = np.sum(freq_grid * magnitude, axis=0) / (np.sum(magnitude, axis=0) + 1e-6)
        centroid_mean = np.mean(centroid)
        centroid_feat = np.clip(abs(centroid_mean - 2000.0) / 1200.0, 0.0, 1.0)
        
        # 5. Spectral Rolloff
        cumulative_energy = np.cumsum(magnitude, axis=0)
        total_energy = cumulative_energy[-1, :]
        rolloff_freqs = []
        for col in range(num_frames):
            rolloff_idx = np.where(cumulative_energy[:, col] >= 0.85 * (total_energy[col] + 1e-6))[0]
            if len(rolloff_idx) > 0:
                rolloff_freqs.append(f[rolloff_idx[0]])
            else:
                rolloff_freqs.append(f[-1])
        rolloff_mean = np.mean(rolloff_freqs)
        rolloff_feat = np.clip(abs(rolloff_mean - 4500.0) / 2500.0, 0.0, 1.0)
        
        # 6. Spectral Bandwidth
        centroid_grid = centroid[np.newaxis, :]
        bandwidth = np.sqrt(np.sum(((freq_grid - centroid_grid)**2) * magnitude, axis=0) / (np.sum(magnitude, axis=0) + 1e-6))
        bandwidth_mean = np.mean(bandwidth)
        bandwidth_feat = np.clip(abs(bandwidth_mean - 1800.0) / 1000.0, 0.0, 1.0)
        
        return np.array([
            hf_feat,
            phase_feat,
            flux_feat,
            centroid_feat,
            rolloff_feat,
            bandwidth_feat
        ], dtype=np.float32)

    def predict(self, rate, data):
        """Evaluate audio WAV data for deepfake cloning anomalies.
        Returns:
            score: float, deepfake probability [0.0, 1.0]
            feature_breakdown: dict, analysis results per feature
        """
        features = self.extract_audio_features(rate, data)
        score = self.run_mlp(features)
        
        breakdown = {
            "high_frequency_energy_anomaly": round(float(features[0]), 3),
            "phase_incoherence": round(float(features[1]), 3),
            "spectral_flux_deviation": round(float(features[2]), 3),
            "spectral_centroid_deviation": round(float(features[3]), 3),
            "spectral_rolloff_deviation": round(float(features[4]), 3),
            "spectral_bandwidth_anomaly": round(float(features[5]), 3),
        }
        
        return score, breakdown
