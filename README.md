# 🛰️ Satellite Telemetry Anomaly Detection using LSTM Autoencoder

A deep-learning system for detecting abnormal satellite telemetry using a PyTorch LSTM Autoencoder. The model learns normal temporal patterns and flags anomalies using reconstruction-error thresholds.

---

## 📌 Project Overview

This project provides an end-to-end anomaly detection pipeline for high-frequency satellite telemetry (50,000+ rows, multi-feature).  
The system is trained on *normal* telemetry only and identifies unusual behavior using reconstruction error.

**Key features include:**

- Full preprocessing (cleaning, timestamp generation, normalization)  
- Resampling and interpolation of irregular telemetry  
- Sliding-window sequence generation (e.g., 60-second windows)  
- LSTM Autoencoder (encoder → latent space → decoder)  
- Threshold selection via validation error (99.5 percentile)  
- Visualization of anomaly separation  

---

## 🧠 Architecture

```
Raw Telemetry → Cleaning → Resampling → Windowing
                                        ↓
                              ┌────────────────────┐
                              │   LSTM Encoder     │
                              └────────────────────┘
                                          ↓
                                  Latent Representation
                                          ↓
                              ┌────────────────────┐
                              │   LSTM Decoder     │
                              └────────────────────┘
                                          ↓
                               Reconstruction Error
                                          ↓
                          Anomaly Flag (threshold-based)
```

---

## 🔧 Features

### Preprocessing Pipeline
- Timestamp synthesis  
- Normalization of all numeric features  
- Alignment of irregular intervals  
- Interpolation of missing values  

### Sequence Engineering
- Configurable sliding window  
- Train/validation/test splitting  
- Support for overlapping windows  

### LSTM Autoencoder
- Encoder compresses each time window  
- Decoder reconstructs the signal  
- Mean Squared Error (MSE) used as anomaly metric  

### Thresholding
- Automatically computed from validation reconstruction error  
- Outliers marked as anomalies  

### Visual Output
- Reconstruction error distribution plots  
- Normal vs anomaly scatter  
- Loss curve across epochs  

---

## 📊 Results Summary

The trained model provides clear separation between:

- **Normal windows** (low reconstruction error)  
- **Anomalous windows** (high reconstruction error)  

This makes it suitable for:

- Satellite health monitoring  
- Communication system diagnostics  
- Predictive maintenance  
- Federated learning use cases  

---


## 🛠 Tech Stack
- Python  
- PyTorch  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  

---

## 🌍 Use Cases
- Satellite constellation monitoring  
- Communication system anomaly detection  
- On-orbit fault detection  
- Predictive maintenance  
- Ground station diagnostics  

---


