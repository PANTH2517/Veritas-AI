import os
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import tensorflow as tf

MODEL_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "voice_model.keras")

class DeepfakeVoiceDetector:
    def __init__(self):
        # Disable GPU logs for clean console
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        # Load or Build Keras Model
        if os.path.exists(MODEL_FILE):
            try:
                self.model = tf.keras.models.load_model(MODEL_FILE)
                print(f"Loaded voice model from {MODEL_FILE}")
            except Exception as e:
                print(f"Failed to load Keras model: {e}. Falling back to default initialization.")
                self.model = self._build_model()
                self._initialize_logical_weights()
        else:
            self.model = self._build_model()
            try:
                from app.database import load_voice_signatures_from_db
                features, labels = load_voice_signatures_from_db()
                if len(features) > 0:
                    print("Training voice model on SQLite voice signatures...")
                    self.train_on_data(features, labels)
                    print("Voice model trained and saved successfully.")
                else:
                    self._initialize_logical_weights()
            except Exception as e:
                print(f"Could not train voice model on startup: {e}. Using logical weights.")
                self._initialize_logical_weights()


    def _build_model(self):
        """Construct a Keras neural network that classifies audio frequency anomalies."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(6,)),
            tf.keras.layers.Dense(12, activation='relu'),
            tf.keras.layers.Dense(6, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _initialize_logical_weights(self):
        """Initialize weights of the model to act as a logical classifier for voice anomalies."""
        w1 = np.zeros((6, 12), dtype=np.float32)
        w1[0, :] = 0.45  # High-frequency ratio
        w1[1, :] = 0.40  # Phase inconsistency
        w1[2, :] = 0.25  # Spectral flux
        w1[3, :] = 0.20  # Centroid deviation
        w1[4, :] = 0.20  # Rolloff deviation
        w1[5, :] = 0.20  # Bandwidth anomaly
        
        b1 = np.full((12,), -0.9, dtype=np.float32)
        
        w2 = np.ones((12, 6), dtype=np.float32) * 0.3
        b2 = np.full((6,), -0.3, dtype=np.float32)
        
        w3 = np.ones((6, 1), dtype=np.float32) * 0.5
        b3 = np.array([-0.1], dtype=np.float32)
        
        # Break weight initialization symmetry
        np.random.seed(42)
        w1 = w1 + np.random.normal(0.0, 0.02, size=w1.shape).astype(np.float32)
        w2 = w2 + np.random.normal(0.0, 0.02, size=w2.shape).astype(np.float32)
        w3 = w3 + np.random.normal(0.0, 0.02, size=w3.shape).astype(np.float32)
        
        self.model.layers[0].set_weights([w1, b1])
        self.model.layers[1].set_weights([w2, b2])
        self.model.layers[2].set_weights([w3, b3])

    def reload_weights(self):
        """Reload the model weights from the saved Keras model file."""
        if os.path.exists(MODEL_FILE):
            try:
                tf.keras.backend.clear_session()
                self.model = tf.keras.models.load_model(MODEL_FILE)
                print("Voice model reloaded successfully.")
                return True
            except Exception as e:
                print(f"Error reloading voice model weights: {e}")
        return False

    def train_on_data(self, features_matrix, labels):
        """Train the neural network using backpropagation on the provided dataset."""
        x_train = np.array(features_matrix, dtype=np.float32)
        y_train = np.array(labels, dtype=np.float32)
        
        # Shuffle dataset before training to ensure proper validation split distribution
        indices = np.arange(x_train.shape[0])
        np.random.seed(42)
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]
        
        history = self.model.fit(
            x_train, 
            y_train, 
            epochs=15, 
            batch_size=32, 
            validation_split=0.2, 
            verbose=0
        )
        
        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        self.model.save(MODEL_FILE)
        return history.history

    def extract_audio_features(self, rate, data):
        """Extract spectral features from raw audio data.
        Assumes rate is 16000 and data is a 1D numpy float or int array.
        """
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
        input_tensor = np.expand_dims(features, axis=0)
        score = float(self.model.predict(input_tensor, verbose=0)[0][0])
        
        breakdown = {
            "high_frequency_energy_anomaly": round(float(features[0]), 3),
            "phase_incoherence": round(float(features[1]), 3),
            "spectral_flux_deviation": round(float(features[2]), 3),
            "spectral_centroid_deviation": round(float(features[3]), 3),
            "spectral_rolloff_deviation": round(float(features[4]), 3),
            "spectral_bandwidth_anomaly": round(float(features[5]), 3),
        }
        
        return score, breakdown
