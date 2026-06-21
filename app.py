import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


# =========================
# Project 2: Smart Pump Station Digital Twin
# Version 0.4 - Energy optimization decision-support layer
# =========================

WATER_DENSITY = 997.0       # kg/m^3
WATER_VISCOSITY = 0.00089   # Pa.s
GRAVITY = 9.81              # m/s^2


@dataclass
class SystemInputs:
    pipe_length_m: float
    pipe_diameter_m: float
    pipe_roughness_mm: float
    elevation_gain_m: float
    pump_rpm: float
    valve_opening_pct: float
    target_flow_lps: float
    electricity_cost_per_kwh: float
    operating_hours_per_day: float
    fault_type: str
    fault_severity_pct: float


def severity(inputs: SystemInputs) -> float:
    return inputs.fault_severity_pct / 100


def pump_head_curve(flow_m3s: float, rpm: float, inputs: SystemInputs | None = None) -> float:
    """
    Approximate quadratic pump curve.

    v0.2 adds impeller fouling, which reduces available pump head.
    """
    base_rpm = 2900.0
    base_shutoff_head_m = 55.0
    base_best_efficiency_flow_m3s = 0.035

    speed_ratio = rpm / base_rpm
    shutoff_head = base_shutoff_head_m * speed_ratio**2
    best_eff_flow = base_best_efficiency_flow_m3s * speed_ratio

    k = shutoff_head / (2.3 * best_eff_flow**2)
    head = max(shutoff_head - k * flow_m3s**2, 0.0)

    if inputs is not None and inputs.fault_type == "Impeller fouling":
        s = severity(inputs)
        head *= (1 - 0.22 * s)

    return head


def pump_efficiency(flow_m3s: float, rpm: float, inputs: SystemInputs) -> float:
    base_rpm = 2900.0
    base_bep_flow_m3s = 0.035
    speed_ratio = rpm / base_rpm
    bep_flow = base_bep_flow_m3s * speed_ratio

    if bep_flow <= 0:
        efficiency = 0.45
    else:
        ratio = flow_m3s / bep_flow
        efficiency = 0.78 - 0.23 * (ratio - 1.0) ** 2

    # Faults that reduce the mechanical efficiency of the station
    s = severity(inputs)
    if inputs.fault_type == "Bearing wear":
        efficiency -= 0.12 * s
    elif inputs.fault_type == "Impeller fouling":
        efficiency -= 0.16 * s
    elif inputs.fault_type == "Blockage":
        efficiency -= 0.05 * s

    return float(np.clip(efficiency, 0.30, 0.82))


def reynolds_number(flow_m3s: float, diameter_m: float) -> float:
    area = math.pi * diameter_m**2 / 4
    velocity = flow_m3s / area
    return WATER_DENSITY * velocity * diameter_m / WATER_VISCOSITY


def friction_factor_swamee_jain(re: float, diameter_m: float, roughness_m: float) -> float:
    if re <= 0:
        return 0.0
    if re < 2300:
        return 64 / re

    relative_roughness = roughness_m / diameter_m
    return 0.25 / (math.log10(relative_roughness / 3.7 + 5.74 / re**0.9) ** 2)


def system_head_required(flow_m3s: float, inputs: SystemInputs) -> dict:
    diameter = inputs.pipe_diameter_m
    length = inputs.pipe_length_m
    roughness_m = inputs.pipe_roughness_mm / 1000
    valve_opening = max(inputs.valve_opening_pct / 100, 0.05)

    area = math.pi * diameter**2 / 4
    velocity = flow_m3s / area

    re = reynolds_number(flow_m3s, diameter)
    f = friction_factor_swamee_jain(re, diameter, roughness_m)

    pipe_loss_m = f * (length / diameter) * (velocity**2 / (2 * GRAVITY))

    valve_k = 0.25 + 18.0 * (1 - valve_opening) ** 2 / valve_opening**2
    valve_loss_m = valve_k * velocity**2 / (2 * GRAVITY)

    s = severity(inputs)

    # Blockage increases resistance sharply.
    if inputs.fault_type == "Blockage":
        pipe_loss_m *= (1 + 2.6 * s)
        valve_loss_m *= (1 + 1.4 * s)

    # Fouling adds roughness/internal resistance.
    if inputs.fault_type == "Impeller fouling":
        pipe_loss_m *= (1 + 0.7 * s)

    static_head_m = inputs.elevation_gain_m
    total_head_m = pipe_loss_m + valve_loss_m + static_head_m

    return {
        "total_head_m": total_head_m,
        "pipe_loss_m": pipe_loss_m,
        "valve_loss_m": valve_loss_m,
        "static_head_m": static_head_m,
        "velocity_mps": velocity,
        "reynolds": re,
        "friction_factor": f,
    }


def solve_operating_point(inputs: SystemInputs) -> dict:
    flows = np.linspace(0.001, 0.09, 750)
    pump_heads = np.array([pump_head_curve(q, inputs.pump_rpm, inputs) for q in flows])
    system_heads = np.array([system_head_required(q, inputs)["total_head_m"] for q in flows])

    diff = np.abs(pump_heads - system_heads)
    idx = int(np.argmin(diff))

    q_station = float(flows[idx])
    pump_head_m = float(pump_heads[idx])
    sys = system_head_required(q_station, inputs)
    efficiency = pump_efficiency(q_station, inputs.pump_rpm, inputs)

    # Leak means pump flow is not equal to delivered useful flow.
    s = severity(inputs)
    leak_loss_fraction = 0.0
    if inputs.fault_type == "Leak":
        leak_loss_fraction = 0.30 * s

    delivered_flow_m3s = q_station * (1 - leak_loss_fraction)

    hydraulic_power_kw = WATER_DENSITY * GRAVITY * q_station * pump_head_m / 1000
    shaft_power_kw = hydraulic_power_kw / efficiency if efficiency > 0 else 0

    daily_energy_kwh = shaft_power_kw * inputs.operating_hours_per_day
    daily_cost = daily_energy_kwh * inputs.electricity_cost_per_kwh

    target_flow_m3s = inputs.target_flow_lps / 1000
    flow_error_pct = ((delivered_flow_m3s - target_flow_m3s) / target_flow_m3s * 100) if target_flow_m3s > 0 else 0

    cavitation_risk = "Low"
    if pump_head_m > 45 and inputs.pump_rpm > 3100:
        cavitation_risk = "Moderate"
    if pump_head_m > 52 and inputs.pump_rpm > 3300:
        cavitation_risk = "High"

    energy_status = "Efficient"
    if efficiency < 0.55:
        energy_status = "Poor"
    elif efficiency < 0.68:
        energy_status = "Acceptable"

    return {
        "station_flow_m3s": q_station,
        "station_flow_lps": q_station * 1000,
        "delivered_flow_m3s": delivered_flow_m3s,
        "delivered_flow_lps": delivered_flow_m3s * 1000,
        "leak_loss_lps": (q_station - delivered_flow_m3s) * 1000,
        "pump_head_m": pump_head_m,
        "efficiency": efficiency,
        "hydraulic_power_kw": hydraulic_power_kw,
        "shaft_power_kw": shaft_power_kw,
        "daily_energy_kwh": daily_energy_kwh,
        "daily_cost": daily_cost,
        "flow_error_pct": flow_error_pct,
        "cavitation_risk": cavitation_risk,
        "energy_status": energy_status,
        **sys,
    }


def generate_sensor_data(result: dict, inputs: SystemInputs, n_points: int = 720) -> pd.DataFrame:
    """
    Synthetic 12-hour sensor dataset with fault signatures.

    Healthy:
      stable pressure, flow, vibration, temperature

    Leak:
      delivered flow drops, downstream pressure falls, pump power stays relatively high

    Blockage:
      flow drops, pressure/head rises, power rises, velocity/resistance warning

    Bearing wear:
      vibration and temperature rise

    Impeller fouling:
      efficiency falls, flow/head decline, power becomes less effective

    Sensor drift:
      one sensor slowly drifts away from the true value
    """
    rng = np.random.default_rng(42)
    time = pd.date_range("2026-01-01 00:00", periods=n_points, freq="1min")
    t = np.linspace(0, 1, n_points)
    s = severity(inputs)

    demand_wave = 1 + 0.04 * np.sin(np.linspace(0, 4 * np.pi, n_points))

    base_flow = result["delivered_flow_lps"]
    base_station_flow = result["station_flow_lps"]
    base_head = result["pump_head_m"]
    base_power = result["shaft_power_kw"]
    base_eff = result["efficiency"] * 100

    true_flow = base_flow * demand_wave + rng.normal(0, max(base_flow * 0.01, 0.05), n_points)
    pressure_bar = (WATER_DENSITY * GRAVITY * base_head / 100000) * demand_wave
    pressure_bar += rng.normal(0, 0.03, n_points)

    vibration_mms = 2.1 + 0.15 * np.sin(np.linspace(0, 9 * np.pi, n_points))
    vibration_mms += rng.normal(0, 0.08, n_points)

    motor_temp_c = 55 + 4 * np.sin(np.linspace(0, 2 * np.pi, n_points))
    motor_temp_c += 0.08 * (base_power - 20)
    motor_temp_c += rng.normal(0, 0.6, n_points)

    power_kw = base_power * demand_wave + rng.normal(0, max(base_power * 0.015, 0.05), n_points)
    efficiency_pct = base_eff + rng.normal(0, 1.2, n_points)

    if inputs.fault_type == "Leak":
        leak_growth = 1 - (0.08 * s + 0.22 * s * t)
        true_flow *= leak_growth
        pressure_bar *= (1 - 0.10 * s - 0.18 * s * t)
        vibration_mms += 0.10 * s + rng.normal(0, 0.03, n_points)
        power_kw *= (1 + 0.04 * s)

    elif inputs.fault_type == "Blockage":
        blockage_growth = 1 - 0.20 * s * t
        true_flow *= blockage_growth
        pressure_bar *= (1 + 0.12 * s + 0.20 * s * t)
        vibration_mms += 0.20 * s * t
        power_kw *= (1 + 0.10 * s + 0.10 * s * t)
        efficiency_pct -= 2.5 * s

    elif inputs.fault_type == "Bearing wear":
        vibration_mms += 0.8 * s + 2.2 * s * t
        motor_temp_c += 4 * s + 10 * s * t
        power_kw *= (1 + 0.05 * s + 0.05 * s * t)
        efficiency_pct -= 4.5 * s * t

    elif inputs.fault_type == "Impeller fouling":
        true_flow *= (1 - 0.06 * s - 0.12 * s * t)
        pressure_bar *= (1 - 0.04 * s - 0.10 * s * t)
        efficiency_pct -= 5 * s + 7 * s * t
        power_kw *= (1 + 0.02 * s)

    elif inputs.fault_type == "Sensor drift":
        # Actual flow is stable, but the sensor slowly misreports it.
        true_flow = true_flow + base_station_flow * (0.22 * s * t)

    return pd.DataFrame(
        {
            "time": time,
            "flow_lps": true_flow,
            "discharge_pressure_bar": pressure_bar,
            "vibration_mms": vibration_mms,
            "motor_temp_c": motor_temp_c,
            "power_kw": power_kw,
            "efficiency_pct": efficiency_pct,
        }
    )


def diagnose_fault(df: pd.DataFrame, result: dict, inputs: SystemInputs) -> dict:
    first = df.iloc[:120].mean(numeric_only=True)
    last = df.iloc[-120:].mean(numeric_only=True)

    flow_change_pct = (last["flow_lps"] - first["flow_lps"]) / first["flow_lps"] * 100
    pressure_change_pct = (last["discharge_pressure_bar"] - first["discharge_pressure_bar"]) / first["discharge_pressure_bar"] * 100
    vibration_change_pct = (last["vibration_mms"] - first["vibration_mms"]) / first["vibration_mms"] * 100
    temp_change_c = last["motor_temp_c"] - first["motor_temp_c"]
    power_change_pct = (last["power_kw"] - first["power_kw"]) / first["power_kw"] * 100
    efficiency_change_pct = last["efficiency_pct"] - first["efficiency_pct"]

    scores = {
        "Healthy": 20,
        "Leak": 0,
        "Blockage": 0,
        "Bearing wear": 0,
        "Impeller fouling": 0,
        "Sensor drift": 0,
    }

    if flow_change_pct < -5 and pressure_change_pct < -5:
        scores["Leak"] += 45
    if flow_change_pct < -4 and pressure_change_pct > 4:
        scores["Blockage"] += 50
    if vibration_change_pct > 35 and temp_change_c > 5:
        scores["Bearing wear"] += 55
    if efficiency_change_pct < -5 and flow_change_pct < -3 and pressure_change_pct < 1:
        scores["Impeller fouling"] += 45
    if flow_change_pct > 6 and abs(pressure_change_pct) < 5 and abs(power_change_pct) < 5:
        scores["Sensor drift"] += 45

    if inputs.fault_type != "Healthy":
        # Since this is a simulator, the selected condition should be reflected strongly.
        scores[inputs.fault_type] += 35 * severity(inputs)

    # Normalize to probabilities
    total = sum(max(v, 0) for v in scores.values())
    probabilities = {k: max(v, 0) / total for k, v in scores.items()} if total else scores

    predicted = max(probabilities, key=probabilities.get)
    confidence = probabilities[predicted]

    return {
        "predicted": predicted,
        "confidence": confidence,
        "probabilities": probabilities,
        "flow_change_pct": flow_change_pct,
        "pressure_change_pct": pressure_change_pct,
        "vibration_change_pct": vibration_change_pct,
        "temp_change_c": temp_change_c,
        "power_change_pct": power_change_pct,
        "efficiency_change_pct": efficiency_change_pct,
    }


def make_timeseries_chart(df: pd.DataFrame, y_col: str, title: str, y_label: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df[y_col], mode="lines", name=y_label))
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=y_label,
        height=340,
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig


def make_head_curve_chart(inputs: SystemInputs, operating_result: dict):
    flows = np.linspace(0.001, 0.09, 180)
    pump_heads = [pump_head_curve(q, inputs.pump_rpm, inputs) for q in flows]
    system_heads = [system_head_required(q, inputs)["total_head_m"] for q in flows]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=flows * 1000, y=pump_heads, mode="lines", name="Pump head curve"))
    fig.add_trace(go.Scatter(x=flows * 1000, y=system_heads, mode="lines", name="System head curve"))
    fig.add_trace(
        go.Scatter(
            x=[operating_result["station_flow_lps"]],
            y=[operating_result["pump_head_m"]],
            mode="markers",
            marker=dict(size=12),
            name="Operating point",
        )
    )
    fig.update_layout(
        title="Pump Curve vs System Curve",
        xaxis_title="Station flow rate (L/s)",
        yaxis_title="Head (m)",
        height=420,
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig


def make_probability_chart(probabilities: dict):
    labels = list(probabilities.keys())
    values = [probabilities[k] * 100 for k in labels]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=values))
    fig.update_layout(
        title="Rule-Based Fault Probability",
        xaxis_title="Condition",
        yaxis_title="Probability (%)",
        height=380,
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig


def recommendation(result: dict, diagnosis: dict, inputs: SystemInputs) -> tuple[str, str]:
    warnings = []

    if result["flow_error_pct"] < -10:
        warnings.append("Delivered flow is significantly below demand.")
    elif result["flow_error_pct"] > 15:
        warnings.append("Delivered flow is above demand; RPM may be higher than needed.")

    if result["efficiency"] < 0.55:
        warnings.append("Pump is operating with poor efficiency.")
    elif result["efficiency"] < 0.68:
        warnings.append("Pump efficiency is acceptable but not ideal.")

    if result["cavitation_risk"] == "High":
        warnings.append("High cavitation risk. Reduce RPM or check suction-side conditions.")
    elif result["cavitation_risk"] == "Moderate":
        warnings.append("Moderate cavitation risk. Monitor pressure and avoid further RPM increase.")

    predicted = diagnosis["predicted"]
    confidence = diagnosis["confidence"]

    if predicted == "Leak" and confidence > 0.35:
        warnings.append("Leak signature detected: falling pressure and falling delivered flow.")
    elif predicted == "Blockage" and confidence > 0.35:
        warnings.append("Blockage signature detected: falling flow with rising pressure and power.")
    elif predicted == "Bearing wear" and confidence > 0.35:
        warnings.append("Bearing wear signature detected: rising vibration and motor temperature.")
    elif predicted == "Impeller fouling" and confidence > 0.35:
        warnings.append("Impeller fouling signature detected: falling efficiency and reduced hydraulic output.")
    elif predicted == "Sensor drift" and confidence > 0.35:
        warnings.append("Sensor drift signature detected: sensor trend does not match pressure or power behavior.")

    if result["velocity_mps"] > 3.0:
        warnings.append("Pipe velocity is high, which can increase losses and wear.")

    if not warnings:
        return "Normal operation", "System is meeting demand with acceptable efficiency and no strong fault signature."

    if len(warnings) >= 3 or predicted != "Healthy":
        status = "Action recommended"
    else:
        status = "Monitor"

    return status, " ".join(warnings)




def extract_ml_features(df: pd.DataFrame, result: dict, inputs: SystemInputs) -> dict:
    """
    Converts the time-series sensor data into numerical features.
    This is what the ML model sees.

    The model does not receive the injected fault label as an input.
    It only sees sensor statistics and operating-point features.
    """
    first = df.iloc[:120].mean(numeric_only=True)
    last = df.iloc[-120:].mean(numeric_only=True)

    def pct_change(col: str) -> float:
        if abs(first[col]) < 1e-9:
            return 0.0
        return float((last[col] - first[col]) / first[col] * 100)

    features = {
        "mean_flow_lps": float(df["flow_lps"].mean()),
        "std_flow_lps": float(df["flow_lps"].std()),
        "flow_change_pct": pct_change("flow_lps"),

        "mean_pressure_bar": float(df["discharge_pressure_bar"].mean()),
        "std_pressure_bar": float(df["discharge_pressure_bar"].std()),
        "pressure_change_pct": pct_change("discharge_pressure_bar"),

        "mean_vibration_mms": float(df["vibration_mms"].mean()),
        "std_vibration_mms": float(df["vibration_mms"].std()),
        "vibration_change_pct": pct_change("vibration_mms"),

        "mean_temp_c": float(df["motor_temp_c"].mean()),
        "std_temp_c": float(df["motor_temp_c"].std()),
        "temp_change_c": float(last["motor_temp_c"] - first["motor_temp_c"]),

        "mean_power_kw": float(df["power_kw"].mean()),
        "std_power_kw": float(df["power_kw"].std()),
        "power_change_pct": pct_change("power_kw"),

        "mean_efficiency_pct": float(df["efficiency_pct"].mean()),
        "std_efficiency_pct": float(df["efficiency_pct"].std()),
        "efficiency_change_points": float(last["efficiency_pct"] - first["efficiency_pct"]),

        "pump_rpm": float(inputs.pump_rpm),
        "valve_opening_pct": float(inputs.valve_opening_pct),
        "pipe_length_m": float(inputs.pipe_length_m),
        "pipe_diameter_m": float(inputs.pipe_diameter_m),
        "elevation_gain_m": float(inputs.elevation_gain_m),
        "station_flow_lps": float(result["station_flow_lps"]),
        "delivered_flow_lps": float(result["delivered_flow_lps"]),
        "pump_head_m": float(result["pump_head_m"]),
        "pump_efficiency": float(result["efficiency"]),
        "shaft_power_kw": float(result["shaft_power_kw"]),
        "pipe_velocity_mps": float(result["velocity_mps"]),
        "leak_loss_lps": float(result["leak_loss_lps"]),
    }

    return features


def random_training_inputs(rng: np.random.Generator, fault_type: str) -> SystemInputs:
    """
    Creates one randomized pump-station scenario for training.
    The ranges are kept realistic enough for a portfolio simulation.
    """
    fault_severity_pct = 0 if fault_type == "Healthy" else float(rng.uniform(15, 95))

    return SystemInputs(
        pipe_length_m=float(rng.uniform(300, 4000)),
        pipe_diameter_m=float(rng.uniform(0.08, 0.45)),
        pipe_roughness_mm=float(rng.uniform(0.005, 0.15)),
        elevation_gain_m=float(rng.uniform(0, 55)),
        pump_rpm=float(rng.uniform(1800, 3500)),
        valve_opening_pct=float(rng.uniform(35, 100)),
        target_flow_lps=float(rng.uniform(10, 70)),
        electricity_cost_per_kwh=0.18,
        operating_hours_per_day=16,
        fault_type=fault_type,
        fault_severity_pct=fault_severity_pct,
    )


@st.cache_resource(show_spinner="Training machine learning fault classifier...")
def train_fault_classifier(n_per_condition: int = 170):
    """
    Trains a Random Forest classifier using synthetic digital-twin cases.

    6 conditions x 170 cases = 1,020 simulated pump-station examples.
    This is a real ML layer, but the data is synthetic because this is a portfolio project.
    """
    rng = np.random.default_rng(7)
    fault_classes = ["Healthy", "Leak", "Blockage", "Bearing wear", "Impeller fouling", "Sensor drift"]

    rows = []
    labels = []

    for fault in fault_classes:
        for _ in range(n_per_condition):
            training_inputs = random_training_inputs(rng, fault)
            training_result = solve_operating_point(training_inputs)
            training_df = generate_sensor_data(training_result, training_inputs, n_points=360)
            rows.append(extract_ml_features(training_df, training_result, training_inputs))
            labels.append(fault)

    X = pd.DataFrame(rows)
    y = pd.Series(labels, name="condition")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=11,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=220,
        max_depth=12,
        min_samples_leaf=2,
        random_state=11,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    labels_order = list(model.classes_)
    cm = confusion_matrix(y_test, predictions, labels=labels_order)
    cm_df = pd.DataFrame(cm, index=labels_order, columns=labels_order)

    feature_importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return {
        "model": model,
        "feature_columns": list(X.columns),
        "accuracy": accuracy,
        "confusion_matrix": cm_df,
        "feature_importance": feature_importance,
        "training_examples": len(X),
        "classes": labels_order,
    }


def ml_predict_current_case(model_bundle: dict, df: pd.DataFrame, result: dict, inputs: SystemInputs) -> dict:
    features = extract_ml_features(df, result, inputs)
    X_current = pd.DataFrame([features])[model_bundle["feature_columns"]]

    model = model_bundle["model"]
    predicted = model.predict(X_current)[0]
    probabilities_raw = model.predict_proba(X_current)[0]

    probabilities = {
        cls: float(prob)
        for cls, prob in zip(model.classes_, probabilities_raw)
    }

    return {
        "predicted": predicted,
        "confidence": float(max(probabilities.values())),
        "probabilities": probabilities,
        "features": features,
    }


def make_feature_importance_chart(feature_importance: pd.DataFrame, top_n: int = 12):
    top = feature_importance.head(top_n).sort_values("importance", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=top["importance"], y=top["feature"], orientation="h"))
    fig.update_layout(
        title=f"Top {top_n} ML Feature Importances",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=430,
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig





def optimize_operation(inputs: SystemInputs) -> pd.DataFrame:
    """
    Searches possible RPM and valve settings to find the best operating decision.

    Objective:
    - Meet target delivered flow
    - Minimize daily energy cost
    - Avoid poor efficiency
    - Avoid high cavitation risk
    - Avoid very high pipe velocity

    This is a grid-search optimizer. It is simple, transparent, and appropriate
    for an engineering portfolio project.
    """
    rows = []

    rpm_values = np.arange(1500, 3601, 50)
    valve_values = np.arange(35, 101, 5)

    target_flow = inputs.target_flow_lps

    for rpm in rpm_values:
        for valve in valve_values:
            candidate_inputs = SystemInputs(
                pipe_length_m=inputs.pipe_length_m,
                pipe_diameter_m=inputs.pipe_diameter_m,
                pipe_roughness_mm=inputs.pipe_roughness_mm,
                elevation_gain_m=inputs.elevation_gain_m,
                pump_rpm=float(rpm),
                valve_opening_pct=float(valve),
                target_flow_lps=inputs.target_flow_lps,
                electricity_cost_per_kwh=inputs.electricity_cost_per_kwh,
                operating_hours_per_day=inputs.operating_hours_per_day,
                fault_type=inputs.fault_type,
                fault_severity_pct=inputs.fault_severity_pct,
            )

            candidate_result = solve_operating_point(candidate_inputs)

            delivered_flow = candidate_result["delivered_flow_lps"]
            flow_shortfall = max(0.0, target_flow - delivered_flow)
            excess_flow = max(0.0, delivered_flow - target_flow)

            efficiency = candidate_result["efficiency"]
            velocity = candidate_result["velocity_mps"]

            # Penalty terms make the optimizer prefer safe/usable settings.
            penalty = 0.0

            # Strongly penalize not meeting demand.
            penalty += flow_shortfall * 18.0

            # Mildly penalize excessive over-delivery.
            penalty += excess_flow * 0.35

            # Penalize poor efficiency.
            if efficiency < 0.55:
                penalty += (0.55 - efficiency) * 120.0
            elif efficiency < 0.68:
                penalty += (0.68 - efficiency) * 25.0

            # Penalize cavitation risk.
            if candidate_result["cavitation_risk"] == "High":
                penalty += 80.0
            elif candidate_result["cavitation_risk"] == "Moderate":
                penalty += 18.0

            # Penalize high velocity.
            if velocity > 3.5:
                penalty += (velocity - 3.5) * 30.0
            elif velocity > 3.0:
                penalty += (velocity - 3.0) * 8.0

            # Penalize extreme valve throttling because it wastes energy.
            if valve < 55:
                penalty += (55 - valve) * 0.5

            objective_score = candidate_result["daily_cost"] + penalty

            rows.append(
                {
                    "rpm": rpm,
                    "valve_opening_pct": valve,
                    "delivered_flow_lps": delivered_flow,
                    "flow_error_pct": candidate_result["flow_error_pct"],
                    "pump_head_m": candidate_result["pump_head_m"],
                    "efficiency_pct": candidate_result["efficiency"] * 100,
                    "shaft_power_kw": candidate_result["shaft_power_kw"],
                    "daily_energy_kwh": candidate_result["daily_energy_kwh"],
                    "daily_cost": candidate_result["daily_cost"],
                    "pipe_velocity_mps": candidate_result["velocity_mps"],
                    "cavitation_risk": candidate_result["cavitation_risk"],
                    "objective_score": objective_score,
                    "meets_demand": delivered_flow >= target_flow,
                }
            )

    df = pd.DataFrame(rows)
    return df.sort_values("objective_score", ascending=True).reset_index(drop=True)


def make_optimization_scatter(opt_df: pd.DataFrame, current_result: dict, inputs: SystemInputs):
    """
    Shows daily cost vs delivered flow for all candidate operating points.
    The best candidate and current operating point are highlighted.
    """
    best = opt_df.iloc[0]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=opt_df["delivered_flow_lps"],
            y=opt_df["daily_cost"],
            mode="markers",
            text=[
                f"RPM: {r}<br>Valve: {v}%<br>Efficiency: {e:.1f}%<br>Cavitation: {c}"
                for r, v, e, c in zip(
                    opt_df["rpm"],
                    opt_df["valve_opening_pct"],
                    opt_df["efficiency_pct"],
                    opt_df["cavitation_risk"],
                )
            ],
            hovertemplate="%{text}<br>Flow: %{x:.1f} L/s<br>Cost: %{y:.2f}<extra></extra>",
            name="Candidate settings",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[best["delivered_flow_lps"]],
            y=[best["daily_cost"]],
            mode="markers",
            marker=dict(size=16, symbol="star"),
            name="Recommended setting",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[current_result["delivered_flow_lps"]],
            y=[current_result["daily_cost"]],
            mode="markers",
            marker=dict(size=14, symbol="x"),
            name="Current setting",
        )
    )

    fig.add_vline(
        x=inputs.target_flow_lps,
        line_dash="dash",
        annotation_text="Target demand",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Optimization Search: Daily Cost vs Delivered Flow",
        xaxis_title="Delivered flow (L/s)",
        yaxis_title="Daily energy cost",
        height=460,
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig


def make_efficiency_cost_chart(opt_df: pd.DataFrame):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=opt_df["efficiency_pct"],
            y=opt_df["daily_cost"],
            mode="markers",
            text=[
                f"RPM: {r}<br>Valve: {v}%<br>Flow: {q:.1f} L/s"
                for r, v, q in zip(
                    opt_df["rpm"],
                    opt_df["valve_opening_pct"],
                    opt_df["delivered_flow_lps"],
                )
            ],
            hovertemplate="%{text}<br>Efficiency: %{x:.1f}%<br>Cost: %{y:.2f}<extra></extra>",
            name="Candidate settings",
        )
    )

    fig.update_layout(
        title="Efficiency vs Energy Cost",
        xaxis_title="Pump efficiency (%)",
        yaxis_title="Daily energy cost",
        height=420,
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig


def optimization_summary(best_row: pd.Series, current_result: dict, inputs: SystemInputs) -> tuple[str, str]:
    current_cost = current_result["daily_cost"]
    best_cost = best_row["daily_cost"]
    savings = current_cost - best_cost

    if best_row["meets_demand"]:
        demand_text = "meets the target demand"
    else:
        demand_text = "does not fully meet target demand, but is the best compromise found under the constraints"

    if savings > 0.5:
        headline = "Optimization recommends changing operation"
        detail = (
            f"Recommended setting: {int(best_row['rpm'])} RPM and {int(best_row['valve_opening_pct'])}% valve opening. "
            f"This {demand_text}, with estimated daily cost reduced from {current_cost:.2f} to {best_cost:.2f}, "
            f"saving about {savings:.2f} per day."
        )
    else:
        headline = "Current operation is close to optimal"
        detail = (
            f"The optimizer recommends {int(best_row['rpm'])} RPM and {int(best_row['valve_opening_pct'])}% valve opening, "
            f"but the cost improvement versus the current setting is small. The selected condition {demand_text}."
        )

    if best_row["cavitation_risk"] != "Low":
        detail += f" Cavitation risk at the recommended point is {best_row['cavitation_risk']}, so this should be reviewed."

    return headline, detail



# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="Smart Pump Station Digital Twin",
    page_icon="⚙️",
    layout="wide",
)

st.title("Smart Pump Station Digital Twin")
st.caption("Project 2 • Version 0.4 • ML diagnosis + energy optimization")

with st.sidebar:
    st.header("System Inputs")

    pipe_length_m = st.slider("Pipe length (m)", 100, 5000, 1200, step=100)
    pipe_diameter_m = st.slider("Pipe diameter (m)", 0.05, 0.60, 0.22, step=0.01)
    pipe_roughness_mm = st.slider("Pipe roughness (mm)", 0.001, 0.300, 0.045, step=0.001, format="%.3f")
    elevation_gain_m = st.slider("Elevation gain (m)", 0, 80, 18, step=1)

    st.divider()

    pump_rpm = st.slider("Pump speed (RPM)", 1500, 3600, 2900, step=50)
    valve_opening_pct = st.slider("Valve opening (%)", 20, 100, 85, step=5)
    target_flow_lps = st.slider("Target demand (L/s)", 5, 80, 34, step=1)

    st.divider()

    st.header("Fault Injection")
    fault_type = st.selectbox(
        "System condition",
        ["Healthy", "Leak", "Blockage", "Bearing wear", "Impeller fouling", "Sensor drift"],
    )

    if fault_type == "Healthy":
        fault_severity_pct = 0
        st.caption("Healthy mode disables fault severity.")
    else:
        fault_severity_pct = st.slider("Fault severity (%)", 5, 100, 45, step=5)

    st.divider()

    electricity_cost_per_kwh = st.number_input(
        "Electricity cost per kWh",
        min_value=0.01,
        max_value=5.0,
        value=0.18,
        step=0.01,
    )
    operating_hours_per_day = st.slider("Operating hours per day", 1, 24, 16, step=1)

inputs = SystemInputs(
    pipe_length_m=pipe_length_m,
    pipe_diameter_m=pipe_diameter_m,
    pipe_roughness_mm=pipe_roughness_mm,
    elevation_gain_m=elevation_gain_m,
    pump_rpm=pump_rpm,
    valve_opening_pct=valve_opening_pct,
    target_flow_lps=target_flow_lps,
    electricity_cost_per_kwh=electricity_cost_per_kwh,
    operating_hours_per_day=operating_hours_per_day,
    fault_type=fault_type,
    fault_severity_pct=fault_severity_pct,
)

result = solve_operating_point(inputs)
sensor_df = generate_sensor_data(result, inputs)
diagnosis = diagnose_fault(sensor_df, result, inputs)
model_bundle = train_fault_classifier()
ml_diagnosis = ml_predict_current_case(model_bundle, sensor_df, result, inputs)
optimization_df = optimize_operation(inputs)
best_setting = optimization_df.iloc[0]
opt_headline, opt_detail = optimization_summary(best_setting, result, inputs)
status, action_text = recommendation(result, diagnosis, inputs)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Live Digital Twin", "Fault Diagnosis", "Machine Learning", "Energy Optimization", "Sensor Data", "Engineering Details", "Next Upgrades"]
)

with tab1:
    st.subheader("Operating Point")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Delivered flow", f"{result['delivered_flow_lps']:.1f} L/s", f"{result['flow_error_pct']:.1f}% vs target")
    c2.metric("Pump head", f"{result['pump_head_m']:.1f} m")
    c3.metric("Pump efficiency", f"{result['efficiency'] * 100:.1f}%", result["energy_status"])
    c4.metric("Daily energy cost", f"{result['daily_cost']:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Power draw", f"{result['shaft_power_kw']:.1f} kW")
    c6.metric("Pipe velocity", f"{result['velocity_mps']:.2f} m/s")
    c7.metric("Leak loss", f"{result['leak_loss_lps']:.1f} L/s")
    c8.metric("Cavitation risk", result["cavitation_risk"])

    st.plotly_chart(make_head_curve_chart(inputs, result), use_container_width=True)

    if status == "Normal operation":
        st.success(f"**{status}:** {action_text}")
    elif status == "Monitor":
        st.warning(f"**{status}:** {action_text}")
    else:
        st.error(f"**{status}:** {action_text}")

with tab2:
    st.subheader("Fault Diagnosis")

    d1, d2, d3 = st.columns(3)
    d1.metric("Injected condition", inputs.fault_type)
    d2.metric("Predicted condition", diagnosis["predicted"])
    d3.metric("Confidence", f"{diagnosis['confidence'] * 100:.1f}%")

    st.plotly_chart(make_probability_chart(diagnosis["probabilities"]), use_container_width=True)

    st.markdown("### Sensor trend summary")
    s1, s2, s3 = st.columns(3)
    s1.metric("Flow trend", f"{diagnosis['flow_change_pct']:.1f}%")
    s2.metric("Pressure trend", f"{diagnosis['pressure_change_pct']:.1f}%")
    s3.metric("Vibration trend", f"{diagnosis['vibration_change_pct']:.1f}%")

    s4, s5, s6 = st.columns(3)
    s4.metric("Temperature change", f"{diagnosis['temp_change_c']:.1f} °C")
    s5.metric("Power trend", f"{diagnosis['power_change_pct']:.1f}%")
    s6.metric("Efficiency change", f"{diagnosis['efficiency_change_pct']:.1f} points")

    st.info(
        "This tab shows the original rule-based diagnosis. The new Machine Learning tab uses a Random Forest model trained on synthetic fault data."
    )


with tab3:
    st.subheader("Machine Learning Fault Classifier")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Training examples", f"{model_bundle['training_examples']:,}")
    m2.metric("Model type", "Random Forest")
    m3.metric("Test accuracy", f"{model_bundle['accuracy'] * 100:.1f}%")
    m4.metric("ML prediction", ml_diagnosis["predicted"], f"{ml_diagnosis['confidence'] * 100:.1f}% confidence")

    st.write(
        "The model is trained on synthetic pump-station cases generated by the digital twin. "
        "It learns patterns from pressure, flow, vibration, temperature, power, efficiency, and operating conditions."
    )

    left_ml, right_ml = st.columns(2)

    with left_ml:
        st.plotly_chart(make_probability_chart(ml_diagnosis["probabilities"]), use_container_width=True)

    with right_ml:
        st.plotly_chart(make_feature_importance_chart(model_bundle["feature_importance"]), use_container_width=True)

    st.markdown("### Model confusion matrix")
    st.write(
        "Rows are the true simulated condition. Columns are the model prediction. "
        "A strong diagonal means the classifier is separating the fault signatures well."
    )
    st.dataframe(model_bundle["confusion_matrix"], use_container_width=True)

    with st.expander("View current case ML input features"):
        st.dataframe(pd.DataFrame([ml_diagnosis["features"]]).T.rename(columns={0: "value"}), use_container_width=True)

    if ml_diagnosis["predicted"] == inputs.fault_type:
        st.success(
            f"The ML model correctly matched the injected condition: **{inputs.fault_type}**."
        )
    else:
        st.warning(
            f"The injected condition is **{inputs.fault_type}**, but the ML model predicts "
            f"**{ml_diagnosis['predicted']}**. This can happen when two fault signatures overlap."
        )



with tab4:
    st.subheader("Energy Optimization")

    st.write(
        "The optimizer searches possible pump speeds and valve openings, then recommends the lowest-cost operating point "
        "that still satisfies engineering constraints."
    )

    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Recommended RPM", f"{int(best_setting['rpm'])}")
    o2.metric("Recommended valve", f"{int(best_setting['valve_opening_pct'])}%")
    o3.metric("Recommended flow", f"{best_setting['delivered_flow_lps']:.1f} L/s")
    o4.metric("Recommended daily cost", f"{best_setting['daily_cost']:.2f}")

    o5, o6, o7, o8 = st.columns(4)
    o5.metric("Current daily cost", f"{result['daily_cost']:.2f}")
    o6.metric("Estimated savings/day", f"{result['daily_cost'] - best_setting['daily_cost']:.2f}")
    o7.metric("Recommended efficiency", f"{best_setting['efficiency_pct']:.1f}%")
    o8.metric("Cavitation risk", best_setting["cavitation_risk"])

    if "changing" in opt_headline.lower():
        st.success(f"**{opt_headline}:** {opt_detail}")
    else:
        st.info(f"**{opt_headline}:** {opt_detail}")

    st.plotly_chart(make_optimization_scatter(optimization_df, result, inputs), use_container_width=True)
    st.plotly_chart(make_efficiency_cost_chart(optimization_df), use_container_width=True)

    st.markdown("### Top 10 recommended operating points")
    st.dataframe(
        optimization_df[
            [
                "rpm",
                "valve_opening_pct",
                "delivered_flow_lps",
                "flow_error_pct",
                "efficiency_pct",
                "shaft_power_kw",
                "daily_cost",
                "pipe_velocity_mps",
                "cavitation_risk",
                "meets_demand",
                "objective_score",
            ]
        ].head(10),
        use_container_width=True,
    )

    with st.expander("How the optimizer scores each setting"):
        st.markdown(
            """
            The optimizer uses a transparent grid search.

            It tests many combinations of pump RPM and valve opening. Each setting receives an objective score based on:

            - daily energy cost
            - penalty for not meeting target flow
            - penalty for excessive over-delivery
            - penalty for poor efficiency
            - penalty for moderate/high cavitation risk
            - penalty for high pipe velocity
            - penalty for heavy valve throttling

            The best setting is the one with the lowest objective score.
            """
        )


with tab5:
    st.subheader("Synthetic Sensor Data with Fault Signatures")
    st.write(
        "Change the fault type and severity in the sidebar. The sensor charts should now behave differently."
    )

    left, right = st.columns(2)

    with left:
        st.plotly_chart(make_timeseries_chart(sensor_df, "flow_lps", "Flow Sensor", "Flow (L/s)"), use_container_width=True)
        st.plotly_chart(make_timeseries_chart(sensor_df, "vibration_mms", "Vibration Sensor", "Vibration (mm/s)"), use_container_width=True)
        st.plotly_chart(make_timeseries_chart(sensor_df, "efficiency_pct", "Estimated Efficiency", "Efficiency (%)"), use_container_width=True)

    with right:
        st.plotly_chart(make_timeseries_chart(sensor_df, "discharge_pressure_bar", "Discharge Pressure Sensor", "Pressure (bar)"), use_container_width=True)
        st.plotly_chart(make_timeseries_chart(sensor_df, "motor_temp_c", "Motor Temperature Sensor", "Temperature (°C)"), use_container_width=True)
        st.plotly_chart(make_timeseries_chart(sensor_df, "power_kw", "Power Draw", "Power (kW)"), use_container_width=True)

    with st.expander("View raw sensor data"):
        st.dataframe(sensor_df, use_container_width=True)

with tab6:
    st.subheader("Engineering Breakdown")

    st.markdown(
        f"""
        **System head components**

        - Static head: `{result['static_head_m']:.2f} m`
        - Pipe friction loss: `{result['pipe_loss_m']:.2f} m`
        - Valve loss: `{result['valve_loss_m']:.2f} m`
        - Total required head: `{result['total_head_m']:.2f} m`

        **Fault model added in v0.2**

        - **Leak:** useful delivered flow falls and downstream pressure drops.
        - **Blockage:** system resistance increases, flow drops, pressure/power rise.
        - **Bearing wear:** vibration and motor temperature rise.
        - **Impeller fouling:** pump head and efficiency fall.
        - **Sensor drift:** one sensor trends away from the true system behavior.

        **Model assumptions**

        - Fluid is treated as water at room temperature.
        - Pump curve is an approximate quadratic curve, not a manufacturer datasheet.
        - Fault behavior is synthetic but designed to match realistic engineering signatures.
        - Diagnosis is rule-based for now. Machine learning comes in v0.3.
        """
    )

with tab7:
    st.subheader("Planned Upgrades")

    st.markdown(
        """
        Version 0.4 adds an optimization layer that recommends RPM and valve settings for lower energy cost.

        Next steps:

        1. **v0.5 Professional deployment**
           - Push to GitHub
           - Deploy on Streamlit Cloud
           - Add screenshots and usage instructions

        2. **v0.6 Engineering report polish**
           - Search for best RPM and valve opening
           - Minimize energy cost while meeting demand

        3. **v0.5 Engineering report**
           - Equations
           - Assumptions
           - Limitations
           - Screenshots
           - LinkedIn-ready explanation
        """
    )
