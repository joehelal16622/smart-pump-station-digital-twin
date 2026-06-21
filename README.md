# Smart Pump Station Digital Twin

A mechanical engineering portfolio project built with Python and Streamlit.

This app simulates a pump station feeding a pipeline. It calculates the operating point using a simplified pump curve and system head curve, generates synthetic sensor readings, injects different fault conditions, uses machine learning to classify the system condition, and recommends optimized operating settings.

## Version 0.4 Features

- Pump and pipeline simulation
- Pump curve vs system curve
- Flow rate, pressure head, power, efficiency, and energy cost
- Synthetic time-series sensor data
- Fault injection:
  - Healthy
  - Leak
  - Blockage
  - Bearing wear
  - Impeller fouling
  - Sensor drift
- Rule-based fault diagnosis
- Machine learning fault classifier
- Random Forest model trained on synthetic pump-station cases
- Model accuracy and confusion matrix
- ML fault probability chart
- Feature importance chart
- Energy optimization tab
- Recommended pump RPM and valve opening
- Optimization search chart
- Top candidate operating points
- Engineering recommendation system

## Planned Upgrades

- GitHub repository polish
- Streamlit Cloud deployment
- Professional engineering case-study writeup
- LinkedIn post and project description

## How to Run

```bash
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Engineering Concepts Used

- Pump curves
- System head curves
- Darcy-Weisbach friction loss
- Reynolds number
- Swamee-Jain friction factor approximation
- Pump efficiency approximation
- Energy cost estimation
- Synthetic sensor data generation
- Fault signature detection
- Supervised machine learning classification
- Random Forest feature importance
- Grid-search optimization
- Engineering decision support

## Note on Model Accuracy

The machine learning classifier is trained and tested on synthetic digital-twin data. High accuracy is expected because the simulated fault signatures are controlled and separable. Real-world deployment would require validation using actual pump-station sensor data.
