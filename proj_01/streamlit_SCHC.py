import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import datetime  # for current month default

st.set_page_config(page_title="Healthcare + Supply Chain (SCHC) Advisor", layout="wide")

# -------------------------------
# Paths
# -------------------------------
ROOT   = Path(__file__).parent
DATA   = ROOT / "data"
MODELS = ROOT / "models"
DATA.mkdir(exist_ok=True, parents=True)
MODELS.mkdir(exist_ok=True, parents=True)

# -------------------------------
# Loaders (cached)
# -------------------------------
@st.cache_data
def load_csvs():
    """Load required CSVs from /data."""
    sc_shipments = pd.read_csv(DATA / "supply_chain_shipments.csv")
    hc_df        = pd.read_csv(DATA / "healthcare_demand.csv")
    # lane stats optional; if missing we'll compute on the fly
    lane_stats = None
    p = DATA / "carrier_lane_stats.csv"
    if p.exists():
        lane_stats = pd.read_csv(p)
    return sc_shipments, hc_df, lane_stats

@st.cache_resource
def load_models():
    """Load models from /models, return dict with possibly None values."""
    def _try(p):
        try:
            return joblib.load(p)
        except Exception:
            return None

    return {
        "sc_on_time": _try(MODELS / "supply_chain_on_time_model.pkl"),
        "sc_cost":    _try(MODELS / "supply_chain_cost_model.pkl"),
        "hc_demand":  _try(MODELS / "healthcare_demand_model.pkl"),
        "hc_risk":    _try(MODELS / "healthcare_shortage_model.pkl"),
    }

# -------------------------------
# Helpers (stats-only fallbacks)
# -------------------------------
def compute_lane_stats(sc_shipments: pd.DataFrame):
    lane_stats = (
        sc_shipments
        .groupby(["Origin","Destination","Carrier","ServiceLevel"], as_index=False)
        .agg(on_time_prob=("OnTimeDelivery","mean"),
             est_cost=("ShipmentCost","mean"),
             avg_dist=("Distance","mean"),
             n=("OnTimeDelivery","size"))
    )
    return lane_stats

def baseline_demand(hc_df: pd.DataFrame):
    return (
        hc_df.groupby(["Hospital","Medicine","Month"], as_index=False)
             .agg(pred_monthly_demand=("Demand","mean"))
    )

# -------------------------------
# Load data/models
# -------------------------------
try:
    sc_shipments, hc_df, lane_stats = load_csvs()
except FileNotFoundError as e:
    st.error(f"CSV missing: {e}. Ensure required files exist in {DATA}")
    st.stop()

models = load_models()
if lane_stats is None:
    lane_stats = compute_lane_stats(sc_shipments)
demand_baseline = baseline_demand(hc_df)

# -------------------------------
# Collapsible readiness section
# -------------------------------
with st.sidebar.expander("Load Data & Model", expanded=False):
    csv_ok = {
        "supply_chain_shipments.csv": (DATA / "supply_chain_shipments.csv").exists(),
        "healthcare_demand.csv": (DATA / "healthcare_demand.csv").exists()
    }
    for name, ok in csv_ok.items():
        (st.success if ok else st.error)(f"CSV: {name}")

    model_status = {
        "supply_chain_on_time_model.pkl": models["sc_on_time"] is not None,
        "supply_chain_cost_model.pkl":    models["sc_cost"]   is not None,
        "healthcare_demand_model.pkl":    models["hc_demand"] is not None,
        "healthcare_shortage_model.pkl":  models["hc_risk"]   is not None,
    }
    for name, ok in model_status.items():
        (st.success if ok else st.warning)(f"Model: {name}")

st.title("🩺🚚 SCHC — Unified Inventory & Delivery Risk Advisor")

# -------------------------------
# UI Inputs (FORM so nothing updates until submit)
# -------------------------------
hospitals      = sorted(hc_df["Hospital"].unique().tolist())
medicines      = sorted(hc_df["Medicine"].unique().tolist())
months         = list(range(1,13))
current_month  = datetime.datetime.now().month  # 1–12

origins        = sorted(sc_shipments["Origin"].unique().tolist())
destinations   = sorted(sc_shipments["Destination"].unique().tolist())
carriers       = sorted(sc_shipments["Carrier"].unique().tolist())
service_levels = sorted(sc_shipments["ServiceLevel"].unique().tolist())

with st.sidebar.form("inputs_form"):
    st.header("Inputs")
    hospital = st.selectbox("Hospital", hospitals)
    medicine = st.selectbox("Medicine", medicines)
    # Default to current month
    month    = st.selectbox("Month", months, index=current_month - 1)

    origin   = st.selectbox("Origin", origins)
    destination = st.selectbox("Destination", destinations)
    carrier  = st.selectbox("Carrier", carriers)
    service  = st.selectbox("Service Level", service_levels)
    current_inventory = st.number_input("Current Inventory", min_value=0, value=150, step=10)
    lead_time_days    = st.number_input("Lead Time (days)", min_value=1, value=7, step=1)
    stops    = st.slider("Stops", 1, 3, 1)
    weight   = st.number_input("Weight (lbs)", min_value=50, value=35000)
    distance_input = st.number_input("Distance (mi) (0 = auto)", min_value=0, value=0)

    run_clicked = st.form_submit_button("Run Recommendation")

# -------------------------------
# Core computations (hybrid: models if available; else stats)
# -------------------------------
def predict_demand_and_risk(hospital, medicine, month, current_inventory, lead_time_days):
    region = hc_df.loc[hc_df["Hospital"] == hospital, "Region"].iloc[0] if hospital in hc_df["Hospital"].unique() else "MW"
    # Demand
    if models["hc_demand"] is not None:
        x_reg = pd.DataFrame([{
            "Hospital": hospital, "Region": region, "Medicine": medicine,
            "Month": month, "Inventory": current_inventory, "LeadTimeDays": lead_time_days
        }])
        demand_pred = float(models["hc_demand"].predict(x_reg)[0])
    else:
        row = demand_baseline.query("Hospital == @hospital and Medicine == @medicine and Month == @month")
        demand_pred = float(row["pred_monthly_demand"].iloc[0]) if not row.empty else float(hc_df.loc[hc_df["Medicine"]==medicine,"Demand"].mean())
    # Shortage probability
    if models["hc_risk"] is not None:
        x_cls = pd.DataFrame([{
            "Hospital": hospital, "Region": region, "Medicine": medicine,
            "Month": month, "Inventory": current_inventory, "LeadTimeDays": lead_time_days
        }])
        shortage_prob = float(models["hc_risk"].predict_proba(x_cls)[0,1])
    else:
        daily = demand_pred / 30.0
        buffer_need = daily * lead_time_days
        shortage_prob = 0.8 if current_inventory < buffer_need else 0.2

    daily_need = demand_pred / 30.0
    buffer_need = daily_need * lead_time_days
    inventory_gap = current_inventory - buffer_need

    return demand_pred, shortage_prob, buffer_need, inventory_gap

def estimate_lane(origin, destination, carrier, service, distance_override=None, weight=35000, stops=1):
    if distance_override and distance_override > 0:
        distance = float(distance_override)
    else:
        lane = sc_shipments[(sc_shipments["Origin"]==origin) & (sc_shipments["Destination"]==destination)]
        distance = float(lane["Distance"].mean()) if len(lane) >= 5 else 800.0

    if (models["sc_on_time"] is not None) and (models["sc_cost"] is not None):
        x_sc = pd.DataFrame([{
            "Origin": origin, "Destination": destination, "Carrier": carrier, "ServiceLevel": service,
            "Distance": distance, "Weight": weight, "Stops": stops
        }])
        on_time_prob = float(models["sc_on_time"].predict_proba(x_sc)[0,1])
        cost_est     = float(models["sc_cost"].predict(x_sc)[0])
    else:
        row = lane_stats.query("Origin == @origin and Destination == @destination and Carrier == @carrier and ServiceLevel == @service")
        if row.empty:
            lane_only = sc_shipments.query("Origin == @origin and Destination == @destination")
            on_time_prob = float(lane_only["OnTimeDelivery"].mean()) if not lane_only.empty else 0.65
            cost_est     = float(lane_only["ShipmentCost"].mean()) if not lane_only.empty else 300.0
        else:
            on_time_prob = float(row["on_time_prob"].iloc[0])
            cost_est     = float(row["est_cost"].iloc[0])

    return distance, on_time_prob, cost_est

# -------------------------------
# Run only when the form is submitted
# -------------------------------
if run_clicked:
    demand_pred, shortage_prob, buffer_need, inventory_gap = predict_demand_and_risk(
        hospital, medicine, month, current_inventory, lead_time_days
    )
    distance, on_time_prob, cost_est = estimate_lane(
        origin, destination, carrier, service, distance_override=distance_input, weight=weight, stops=stops
    )

    order_now = (shortage_prob >= 0.5) or (inventory_gap < 0)
    risk_flag = on_time_prob < 0.7

    recommendation = "✅ ORDER NOW" if order_now else "🕒 OK TO WAIT"
    logistics_note = "⚠️ Lane/Carrier risk is HIGH (consider expedited or different carrier)" if risk_flag else "✅ Lane/Carrier risk acceptable"

    # -------- Display (rounded integers for unit counts) --------
    display_demand   = f"{int(round(demand_pred)):,}"
    display_shortage = f"{shortage_prob * 100:.2f}%"
    display_ontime   = f"{on_time_prob * 100:.2f}%"
    display_cost     = f"{cost_est:,.2f}"
    display_buffer   = f"{int(round(buffer_need)):,}"
    display_gap      = f"{int(round(inventory_gap)):,}"

    st.subheader("Recommendation")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Predicted Monthly Demand", display_demand)
        st.metric("Shortage Probability", display_shortage)
        st.metric("Inventory Buffer Need", display_buffer)
    with c2:
        st.metric("On-Time Probability", display_ontime)
        st.metric("Estimated Cost ($)", display_cost)
        st.metric("Inventory Gap (Units)", display_gap)

    st.info(f"**Decision:** {recommendation}\n\n**Logistics:** {logistics_note}\n\n**Distance Used:** {round(distance,1)} mi")

    # Compare carriers
    st.subheader("Compare Carriers (Same Lane/Inputs)")
    rows = []
    for c in carriers:
        _, otp, ce = estimate_lane(
            origin, destination, c, service,
            distance_override=distance_input, weight=weight, stops=stops
        )
        rows.append({
            "Carrier": c,
            "on_time_probability": otp,   # keep numeric for sorting, format after
            "estimated_cost": ce,
            "recommendation": recommendation if otp >= 0.7 else "Consider alternate/expedite",
        })
    comp = pd.DataFrame(rows).sort_values("on_time_probability", ascending=False)

    # format columns for display
    comp["on_time_probability"] = (comp["on_time_probability"] * 100).map(lambda x: f"{x:.2f}%")
    comp["estimated_cost"] = comp["estimated_cost"].map(lambda x: f"{x:,.2f}")

    st.dataframe(comp, use_container_width=True)
else:
    st.caption("Adjust inputs in the sidebar form, then click **Run Recommendation**. The view will not update until you submit.")
