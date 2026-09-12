# NDRF Guardian — AIoT-Enabled Edge Intelligent Wearable System

  **Real-time monitoring and safety system for disaster response personnel**

  ![Project Status](https://img.shields.io/badge/status-completed-success)
  ![Platform](https://img.shields.io/badge/platform-ESP32--S3-blue)
  ![TinyML](https://img.shields.io/badge/model-5.8KB-orange)

  ---

  ## Overview

  NDRF Guardian is an edge-intelligent wearable system designed to monitor the health and environmental conditions of
  disaster response personnel in real time. The system combines physiological sensors, environmental monitoring, and
  on-device TinyML inference to predict risk levels and prevent emergencies before they occur.

  **Developed during:** AICTE IDEA Lab Summer Internship 2026, GGSIPU
  **Team:** Hitesh Tomar (ML Pipeline), Lakshya (IoT/Firmware Lead), Paras, Shivam, Kundan, Gaurav
  **Mentor:** Ms. Ankita Sarkar

  ---

  ## Problem Statement

  Disaster response personnel operate in extreme conditions where real-time health monitoring can mean the difference
  between life and death. This system runs entirely on-device with no network dependency — because in a collapsed
  building, there is no network.

  ---

  ## My Contribution: The Machine Learning Pipeline

  I owned the complete ML pipeline from problem framing to a deployable model file that the hardware team could flash
  onto the ESP32. This repo documents that work.

  ### The Data Problem

  No public dataset exists for disaster-responder vitals under hazard conditions. You cannot ethically induce hypoxia or
  heat stress in test subjects.

  **Solution:** Generated 15,000 synthetic samples anchored to medical literature:
  - **AHA heart rate zones** (resting, moderate, vigorous, max)
  - **WHO SpO₂ thresholds** (hypoxia at <90%, severe <85%)
  - **NIOSH work-rest cycles** (heat stress limits)
  - **Bosch IAQ scale** (air quality index 0–500)

  Validated feature ranges against real sensor readings from hardware testing.

  ### Model Architecture

  **Dual-output MLP (Multi-Layer Perceptron):**
  Input (11 features)
      ↓
  Dense(32, ReLU)
      ↓
  Dense(16, ReLU)
      ↓         ↓
  risk_level   wsi_score
  softmax(3)   sigmoid×100

  **11 Input Features:**
  1. `heart_rate` — beats per minute
  2. `spo2` — blood oxygen saturation (%)
  3. `amb_temp` — ambient temperature (°C)
  4. `iaq` — indoor air quality index
  5. `humidity` — relative humidity (%)
  6. `activity` — still / walking / running
  7. `work_duration` — minutes since shift start
  8. `fall_flag` — binary fall detection
  9. `heat_index` — computed from temp + humidity
  10. `fatigue_score` — heuristic from HR elevation + duration
  11. `spo2_drop` — deviation from baseline SpO₂

  ### Training Results

  | Metric | Value |
  |--------|-------|
  | **Accuracy** | 95% |
  | **Precision (weighted avg)** | 95% |
  | **Recall (weighted avg)** | 95% |
  | **Critical class precision** | 0.94 |
  | **Critical class recall** | 0.96 |
  | **WSI MAE** | 5.03 points |

  **The 100% accuracy trap:** The first model achieved 100% accuracy on synthetic data with no class overlap. I treated
  this as a bug, not a success — introduced 7% controlled label noise to force realistic decision boundaries and brought
  it to a trustworthy 95%.

  ### Deployment

  - **Framework:** TensorFlow/Keras → TFLite
  - **Quantization:** INT8 post-training quantization
  - **Model size:** 5,960 bytes (5.8 KB)
  - **Exported as:** C array (`nirvana_model_data.h`)
  - **Inference:** TensorFlow Lite Micro (runs on ESP32-S3)

  The model runs entirely on-device. No cloud, no network dependency, ~7-second inference cycle alongside all other
  firmware tasks.

  ---

  ## System Context (Hardware Team's Work)

  The wearable integrates:
  - **MAX30102** pulse oximeter (HR, SpO₂)
  - **DHT22** temp/humidity sensor
  - **MQ-2** smoke/gas detection
  - **MPU6050** IMU for fall detection
  - **LoRa SX1278** long-range telemetry
  - **BLE** zone detection
  - **ESP32-S3** dual-core MCU running FreeRTOS

  The firmware, hardware integration, and IoT pipeline were handled by Lakshya and the rest of the team.

  ---

  ## Repository Structure

  .
  ├── ml-pipeline/
  │   ├── ML_Pipeline_AIOT_Project.ipynb  # Full training notebook
  │   ├── generate_data.py                # Synthetic data generation
  │   ├── train_model.py                  # Model training
  │   ├── export_tflite.py                # TFLite conversion + C array
  │   ├── scaler_params.json              # Feature normalization params
  │   └── models/
  │       ├── nirvana_dual_model.tflite   # Quantized TFLite model
  │       └── nirvana_model_data.h        # C array for firmware
  ├── docs/
  │   └── ml_handoff.md                   # Model specs + integration notes
  └── README.md

  ---

  ## Getting Started (ML Pipeline)

  ### Prerequisites
  
  pip install tensorflow numpy pandas scikit-learn matplotlib seaborn

  Training the Model

  1. Generate synthetic data:
  python generate_data.py
  # Outputs: data/synthetic_data.csv (15,000 samples)

  2. Train the model:
  python train_model.py
  # Outputs: models/nirvana_dual_model.h5

  3. Export to TFLite + C array:
  python export_tflite.py
  # Outputs:
  #   models/nirvana_dual_model.tflite
  #   models/nirvana_model_data.h (for ESP32)
  #   models/scaler_params.json

  4. Hand off to hardware team:
     - nirvana_model_data.h → include in firmware
     - scaler_params.json → hardcode mean/std arrays in firmware

  ---

  Key Technical Decisions

  1. Why Synthetic Data?

  Real disaster-response vital data doesn't exist in public datasets, and ethically you can't induce the conditions
  (hypoxia, heat stress, toxic gas exposure) needed to collect it. Synthetic data let us anchor to medical thresholds
  while the hardware team validated sensor ranges.

  2. Why Dual-Output?

  The system needs both a discrete alert level (Safe/Warning/Critical for rule-based actions) and a continuous score
  (0–100 WSI for trending and dashboards). Training two separate models would double inference cost. A shared backbone
  learns the representation once.

  3. Why Only 95% Accuracy?

  The first model hit 100% because the synthetic classes had no overlap. That's not realistic — a person at HR=85 and
  SpO₂=92% could be Safe or Warning depending on context the features don't capture. Label noise forced the model to
  learn probabilistic boundaries instead of memorizing clean separations.

  4. Feature Scaling

  The model expects 11 features normalized with hardcoded mean/std (from scaler_params.json). The firmware must apply
  the same normalization before inference. Critical bug caught: the iaq feature trained on 0–380 scale, but the MQ-2
  sensor reports raw ADC ~2000–3200. The hardware team added a mapping function — without it, the model saw every sample
  as "maximum smoke."

  ---

  Results & Validation

  Confusion Matrix (Test Set)

  ┌─────────────────┬────────────────┬───────────────────┬────────────────────┐
  │                 │ Predicted Safe │ Predicted Warning │ Predicted Critical │
  ├─────────────────┼────────────────┼───────────────────┼────────────────────┤
  │ Actual Safe     │ 938            │ 55                │ 0                  │
  ├─────────────────┼────────────────┼───────────────────┼────────────────────┤
  │ Actual Warning  │ 53             │ 933               │ 28                 │
  ├─────────────────┼────────────────┼───────────────────┼────────────────────┤
  │ Actual Critical │ 0              │ 37                │ 956                │
  └─────────────────┴────────────────┴───────────────────┴────────────────────┘

  Critical recall: 96% — the model catches 956 out of 993 true Critical cases. The 37 misses went to Warning (still
  flagged), not Safe.

  WSI Regression

  - MAE: 5.03 points on 0–100 scale
  - Model slightly overestimates safety in the 40–60 range (compressed toward the mean)
  - Falls and SpO₂ <85% correctly produce WSI <20 (hard-capped in training data)

  ---

  Limitations

  1. Synthetic data — not validated against real NDRF field data
  2. Feature weights in WSI formula — based on domain knowledge, not data-driven optimization
  3. No motion artifact handling — SpO₂ assumes still finger; model doesn't know if the reading is noisy
  4. Static thresholds — no personalization (baseline HR varies by individual)

  ---

  Future Work

  - Train on real NDRF operational data (pending ethical approval and data access)
  - Tune WSI feature weights via survival analysis on incident outcomes
  - Add confidence scores (model uncertainty quantification)
  - Explore temporal models (LSTM) to use HR/SpO₂ trends, not just snapshots

  ---

  Acknowledgments

  - AICTE IDEA Lab, GGSIPU for resources and mentorship
  - Ms. Ankita Sarkar for guidance on the ML problem framing
  - Lakshya for handling the entire IoT pipeline, firmware integration, and sensor debugging — made it possible to focus
    on the model
  - Team NIRVANA (Paras, Shivam, Kundan, Gaurav) for hardware assembly and testing

  ---

  License

  MIT License — see LICENSE file for details.

  ---

  Citation

  If you use this work in research, please cite:

  bibtex
  @misc{ndrf_guardian_ml_2026,
    title={TinyML Pipeline for NDRF Guardian Wearable System},
    author={Tomar, Hitesh},
    year={2026},
    publisher={AICTE IDEA Lab, GGSIPU},
    howpublished={\url{https://github.com/HiteshTomar2004/AIOT_Wearable_device}}
  }

  ---
