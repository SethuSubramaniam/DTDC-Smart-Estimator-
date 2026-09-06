"""
DTDC Smart Estimator v4 — Complete Streamlit Application
=========================================================
Changes from v3:
  • 100 cities (24 new cities added, well distributed across states)
  • Dark blue/purple gradient background with glass-style cards
  • Removed: Total Pieces input
  • Added: Order Value (₹), Risk Surcharge dropdown
  • Moved: Nature of Consignment → Left column
  • Moved: VAS → Right column row 4 (below Shipping Mode)
  • VAS options: None, COD, Insurance (Express removed)
  • Fully rule-based cost calculation (no ML for cost)
  • Route optimization auto-displays after prediction (no separate button)
  • ML delivery model preserved
"""

import math
import warnings
import os
from math import ceil

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# SECTION 1 — CITY DATA  (100 cities, all states)
# ══════════════════════════════════════════════════════════════

CITY_COORDS: dict = {
    # ── Andhra Pradesh (5) ────────────────────────────────────
    "Guntur":             (16.3067, 80.4365),
    "Kurnool":            (15.8281, 78.0373),
    "Nellore":            (14.4426, 79.9865),
    "Vijayawada":         (16.5062, 80.6480),
    "Tirupati":            (13.6288, 79.4192),
    "Vizag":              (17.6868, 83.2185),
    # ── Assam (3) ─────────────────────────────────────────────
    "Dibrugarh":          (27.4728, 94.9120),
    "Guwahati":           (26.1445, 91.7362),
    "Silchar":            (24.8333, 92.7789),
    # ── Bihar (3) ─────────────────────────────────────────────
    "Bhagalpur":          (25.2425, 86.9842),
    "Gaya":               (24.7955, 84.9994),
    "Patna":              (25.5941, 85.1376),
    # ── Chhattisgarh (3) ──────────────────────────────────────
    "Bhilai":             (21.1938, 81.3509),
    "Bilaspur":           (22.0796, 82.1391),
    "Raipur":             (21.2514, 81.6296),
    # ── Delhi (1) ─────────────────────────────────────────────
    "Delhi":              (28.6139, 77.2090),
    # ── Goa (2) ───────────────────────────────────────────────
    "Margao":             (15.2832, 73.9862),
    "Panaji":             (15.4909, 73.8278),
    # ── Gujarat (5) ───────────────────────────────────────────
    "Ahmedabad":          (23.0225, 72.5714),
    "Anand":              (22.5645, 72.9289),
    "Gandhinagar":        (23.2156, 72.6369),
    "Rajkot":             (22.3039, 70.8022),
    "Surat":              (21.1702, 72.8311),
    "Vadodara":           (22.3072, 73.1812),
    # ── Haryana (3) ───────────────────────────────────────────
    "Faridabad":          (28.4089, 77.3178),
    "Gurugram":           (28.4595, 77.0266),
    "Hisar":              (29.1492, 75.7217),
    # ── Himachal Pradesh (3) ──────────────────────────────────
    "Dharamshala":        (32.2190, 76.3234),
    "Manali":             (32.2396, 77.1887),
    "Shimla":             (31.1048, 77.1734),
    # ── Jammu & Kashmir (2) ───────────────────────────────────
    "Jammu":              (32.7266, 74.8570),
    "Srinagar":           (34.0837, 74.7973),
    # ── Jharkhand (3) ─────────────────────────────────────────
    "Dhanbad":            (23.7957, 86.4304),
    "Jamshedpur":         (22.8046, 86.2029),
    "Ranchi":             (23.3441, 85.3096),
    # ── Karnataka (5) ─────────────────────────────────────────
    "Bangalore":          (12.9716, 77.5946),
    "Belgaum":            (15.8497, 74.4977),
    "Hubli":              (15.3647, 75.1240),
    "Mangalore":          (12.9141, 74.8560),
    "Mysore":             (12.2958, 76.6394),
    # ── Kerala (4) ────────────────────────────────────────────
    "Kannur":             (11.8745, 75.3704),
    "Kochi":              ( 9.9312, 76.2673),
    "Kozhikode":          (11.2588, 75.7804),
    "Thiruvananthapuram": ( 8.5241, 76.9366),
    "Thrissur":           (10.5276, 76.2144),
    # ── Madhya Pradesh (4) ────────────────────────────────────
    "Bhopal":             (23.2599, 77.4126),
    "Gwalior":            (26.2183, 78.1828),
    "Indore":             (22.7196, 75.8577),
    "Jabalpur":           (23.1815, 79.9864),
    "Ujjain":             (23.1765, 75.7885),
    # ── Maharashtra (6) ───────────────────────────────────────
    "Amravati":           (20.9320, 77.7523),
    "Aurangabad":         (19.8762, 75.3433),
    "Kolhapur":           (16.7050, 74.2433),
    "Mumbai":             (19.0760, 72.8777),
    "Nagpur":             (21.1458, 79.0882),
    "Nashik":             (19.9975, 73.7898),
    "Pune":               (18.5204, 73.8567),
    "Solapur":            (17.6599, 75.9064),
    # ── Manipur (1) ───────────────────────────────────────────
    "Imphal":             (24.8170, 93.9368),
    # ── Meghalaya (1) ─────────────────────────────────────────
    "Shillong":           (25.5788, 91.8933),
    # ── Odisha (3) ────────────────────────────────────────────
    "Bhubaneswar":        (20.2961, 85.8245),
    "Cuttack":            (20.4625, 85.8830),
    "Rourkela":           (22.2604, 84.8536),
    # ── Punjab (3) ────────────────────────────────────────────
    "Amritsar":           (31.6340, 74.8723),
    "Jalandhar":          (31.3260, 75.5762),
    "Ludhiana":           (30.9010, 75.8573),
    # ── Rajasthan (4) ─────────────────────────────────────────
    "Ajmer":              (26.4499, 74.6399),
    "Jaipur":             (26.9124, 75.7873),
    "Jodhpur":            (26.2389, 73.0243),
    "Kota":               (25.2138, 75.8648),
    "Udaipur":            (24.5854, 73.7125),
    # ── Tamil Nadu (7) ────────────────────────────────────────
    "Chennai":            (13.0827, 80.2707),
    "Coimbatore":         (11.0168, 76.9558),
    "Dindigul":           (10.3673, 77.9803),
    "Madurai":            ( 9.9252, 78.1198),
    "Salem":              (11.6643, 78.1460),
    "Thanjavur":          (10.7870, 79.1378),
    "Tiruchirappalli":    (10.7905, 78.7047),
    "Tirunelveli":        ( 8.7139, 77.7567),
    # ── Telangana (3) ─────────────────────────────────────────
    "Hyderabad":          (17.3850, 78.4867),
    "Karimnagar":         (18.4386, 79.1288),
    "Warangal":           (17.9689, 79.5941),
    # ── Uttar Pradesh (6) ─────────────────────────────────────
    "Agra":               (27.1767, 78.0081),
    "Aligarh":            (27.8974, 78.0880),
    "Allahabad":          (25.4358, 81.8463),
    "Kanpur":             (26.4499, 80.3319),
    "Lucknow":            (26.8467, 80.9462),
    "Meerut":             (28.9845, 77.7064),
    "Varanasi":           (25.3176, 82.9739),
    # ── Uttarakhand (3) ───────────────────────────────────────
    "Dehradun":           (30.3165, 78.0322),
    "Haridwar":           (29.9457, 78.1642),
    "Roorkee":            (29.8543, 77.8880),
    # ── West Bengal (4) ───────────────────────────────────────
    "Asansol":            (23.6889, 86.9661),
    "Durgapur":           (23.5204, 87.3119),
    "Kolkata":            (22.5726, 88.3639),
    "Siliguri":           (26.7271, 88.3953),
    # ── Chandigarh (UT) (1) ───────────────────────────────────
    "Chandigarh":         (30.7333, 76.7794),
    # ── Tripura (1) ───────────────────────────────────────────
    "Agartala":           (23.8315, 91.2868),
    # ── Nagaland (1) ──────────────────────────────────────────
    "Kohima":             (25.6751, 94.1086),
    # ── Puducherry (UT) (1) ───────────────────────────────────
    "Puducherry":         (11.9416, 79.8083),
}

CITY_STATE_MAP: dict = {
    # Andhra Pradesh
    "Guntur": "Andhra Pradesh",         "Kurnool": "Andhra Pradesh",
    "Nellore": "Andhra Pradesh",        "Vijayawada": "Andhra Pradesh",
    "Tirupati": "Andhra Pradesh",       "Vizag": "Andhra Pradesh",
    # Assam
    "Dibrugarh": "Assam",               "Guwahati": "Assam",
    "Silchar": "Assam",
    # Bihar
    "Bhagalpur": "Bihar",               "Gaya": "Bihar",
    "Patna": "Bihar",
    # Chhattisgarh
    "Bhilai": "Chhattisgarh",           "Bilaspur": "Chhattisgarh",
    "Raipur": "Chhattisgarh",
    # Delhi
    "Delhi": "Delhi",
    # Goa
    "Margao": "Goa",                    "Panaji": "Goa",
    # Gujarat
    "Ahmedabad": "Gujarat",             "Anand": "Gujarat",
    "Gandhinagar": "Gujarat",           "Rajkot": "Gujarat",
    "Surat": "Gujarat",                 "Vadodara": "Gujarat",
    # Haryana
    "Faridabad": "Haryana",             "Gurugram": "Haryana",
    "Hisar": "Haryana",
    # Himachal Pradesh
    "Dharamshala": "Himachal Pradesh",  "Manali": "Himachal Pradesh",
    "Shimla": "Himachal Pradesh",
    # Jammu & Kashmir
    "Jammu": "Jammu & Kashmir",         "Srinagar": "Jammu & Kashmir",
    # Jharkhand
    "Dhanbad": "Jharkhand",             "Jamshedpur": "Jharkhand",
    "Ranchi": "Jharkhand",
    # Karnataka
    "Bangalore": "Karnataka",           "Belgaum": "Karnataka",
    "Hubli": "Karnataka",               "Mangalore": "Karnataka",
    "Mysore": "Karnataka",
    # Kerala
    "Kannur": "Kerala",                 "Kochi": "Kerala",
    "Kozhikode": "Kerala",              "Thiruvananthapuram": "Kerala",
    "Thrissur": "Kerala",
    # Madhya Pradesh
    "Bhopal": "Madhya Pradesh",         "Gwalior": "Madhya Pradesh",
    "Indore": "Madhya Pradesh",         "Jabalpur": "Madhya Pradesh",
    "Ujjain": "Madhya Pradesh",
    # Maharashtra
    "Amravati": "Maharashtra",          "Aurangabad": "Maharashtra",
    "Kolhapur": "Maharashtra",          "Mumbai": "Maharashtra",
    "Nagpur": "Maharashtra",            "Nashik": "Maharashtra",
    "Pune": "Maharashtra",              "Solapur": "Maharashtra",
    # Manipur
    "Imphal": "Manipur",
    # Meghalaya
    "Shillong": "Meghalaya",
    # Nagaland
    "Kohima": "Nagaland",
    # Odisha
    "Bhubaneswar": "Odisha",            "Cuttack": "Odisha",
    "Rourkela": "Odisha",
    # Puducherry
    "Puducherry": "Puducherry",
    # Punjab
    "Amritsar": "Punjab",               "Jalandhar": "Punjab",
    "Ludhiana": "Punjab",
    # Rajasthan
    "Ajmer": "Rajasthan",               "Jaipur": "Rajasthan",
    "Jodhpur": "Rajasthan",             "Kota": "Rajasthan",
    "Udaipur": "Rajasthan",
    # Tamil Nadu
    "Chennai": "Tamil Nadu",            "Coimbatore": "Tamil Nadu",
    "Dindigul": "Tamil Nadu",           "Madurai": "Tamil Nadu",
    "Salem": "Tamil Nadu",              "Thanjavur": "Tamil Nadu",
    "Tiruchirappalli": "Tamil Nadu",    "Tirunelveli": "Tamil Nadu",
    # Telangana
    "Hyderabad": "Telangana",           "Karimnagar": "Telangana",
    "Warangal": "Telangana",
    # Tripura
    "Agartala": "Tripura",
    # Uttar Pradesh
    "Agra": "Uttar Pradesh",            "Aligarh": "Uttar Pradesh",
    "Allahabad": "Uttar Pradesh",       "Kanpur": "Uttar Pradesh",
    "Lucknow": "Uttar Pradesh",         "Meerut": "Uttar Pradesh",
    "Varanasi": "Uttar Pradesh",
    # Uttarakhand
    "Dehradun": "Uttarakhand",          "Haridwar": "Uttarakhand",
    "Roorkee": "Uttarakhand",
    # West Bengal
    "Asansol": "West Bengal",           "Durgapur": "West Bengal",
    "Kolkata": "West Bengal",           "Siliguri": "West Bengal",
    # Chandigarh
    "Chandigarh": "Chandigarh",
}

ALL_CITIES  = sorted(CITY_COORDS.keys())
ALL_STATES  = sorted(set(CITY_STATE_MAP.values()))
NONE_OPTION = "— None (Show All) —"


# ══════════════════════════════════════════════════════════════
# SECTION 2 — GEO / HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km (Haversine formula)."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def calculate_distance(city1: str, city2: str) -> float:
    """Haversine km between two known cities."""
    if city1 == city2:
        return 0.0
    lat1, lon1 = CITY_COORDS[city1]
    lat2, lon2 = CITY_COORDS[city2]
    return round(haversine(lat1, lon1, lat2, lon2), 2)


def transport_mode_rule(shipping_mode: str, distance_km: float) -> str:
    """Rule-based vehicle recommendation."""
    if shipping_mode in ("Air Cargo", "Express"):
        return "✈️ Aeroplane \n🚛/🚐 Truck or van (To warehouse)"
    return "🚐 Delivery Van" if distance_km <= 100 else "🚛 Truck"


def business_delivery_estimate(shipping_mode: str, distance_km: float) -> int:
    """DTDC SLA-based delivery days."""
    if shipping_mode == "Express":
        if distance_km <= 500:   return 1
        if distance_km <= 1500:  return 2
        return 3
    elif shipping_mode == "Air Cargo":
        if distance_km <= 1000:  return 2
        if distance_km <= 2500:  return 3
        return 4
    else:  # Surface
        if distance_km <= 200:   return 1
        if distance_km <= 600:   return 2
        if distance_km <= 1200:  return 3
        if distance_km <= 2000:  return 4
        return 5


def cities_for_state(state: str) -> list:
    """Sorted cities in a state. Returns ALL_CITIES for NONE_OPTION."""
    if state == NONE_OPTION:
        return ALL_CITIES
    return sorted(c for c, s in CITY_STATE_MAP.items() if s == state)


def auto_state_for_city(city: str) -> str:
    """Return state label for a city."""
    return CITY_STATE_MAP.get(city, NONE_OPTION)


# ══════════════════════════════════════════════════════════════
# SECTION 3 — ROUTE OPTIMIZATION  (NetworkX Dijkstra, cached)
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def build_geo_graph():
    G = nx.Graph()
    cities = list(CITY_COORDS.keys())

    for i, c1 in enumerate(cities):
        for c2 in cities[i + 1:]:
            dist = calculate_distance(c1, c2)

            # 🔥 Only connect nearby cities
            if dist <= 600:
                G.add_edge(c1, c2, weight=dist)

    return G


def find_shortest_path(G: nx.Graph, origin: str, destination: str) -> tuple:
    """Return (path_list, total_km) via Dijkstra."""
    try:
        path   = nx.dijkstra_path(G, origin, destination, weight="weight")
        length = nx.dijkstra_path_length(G, origin, destination, weight="weight")
        return path, round(length, 2)
    except nx.NetworkXNoPath:
        return [origin, destination], calculate_distance(origin, destination)


# ══════════════════════════════════════════════════════════════
# SECTION 4 — RULE-BASED COST CALCULATION
# ══════════════════════════════════════════════════════════════

# Base rate per kg by shipping mode
MODE_RATE_PER_KG = {
    "Surface":   80.0,
    "Air Cargo": 120.0,
    "Express":   150.0,
}

DISTANCE_RATE_PER_KM = 0.15   # ₹ per km (distance cost component)


def calc_dox_cost(weight_kg: float) -> float:
    """
    Nature = Dox slab pricing:
      0 – 0.5 kg  → ₹100
      0.5 – 1 kg  → ₹175
      every additional 500 g → +₹75
    """
    weight_g = weight_kg * 1000
    if weight_g <= 500:
        return 100.0
    elif weight_g <= 1000:
        return 175.0
    else:
        extra_slabs = ceil((weight_g - 1000) / 500)
        return 175.0 + extra_slabs * 75.0


def calc_vas_cost(base_cost: float, vas: str,
                  order_value: float, risk_surcharge: str) -> float:
    """
    VAS cost rules:
      COD       → 1.5% of order_value
      Insurance → 0.2% of order_value if Carrier, 2% if Owner
      None      → ₹0
    """
    if vas == "COD":
        return order_value * 0.015
    elif vas == "Insurance":
        rate = 0.002 if risk_surcharge == "Carrier" else 0.02
        return order_value * rate
    return 0.0


def calculate_total_cost(weight: float, mode: str, distance_km: float,
                         nature: str, vas: str,
                         order_value: float, risk_surcharge: str) -> dict:
    """
    Full rule-based cost breakdown.
    Returns dict with individual components and total.
    """
    # 1. Weight-based base cost
    base_cost     = weight * MODE_RATE_PER_KG.get(mode, 80.0)

    # 2. Distance cost (existing logic retained)
    distance_cost = distance_km * DISTANCE_RATE_PER_KM

    # 3. Nature of consignment cost
    dox_cost      = calc_dox_cost(weight) if nature == "Dox" else 0.0

    # 4. VAS cost
    vas_cost      = calc_vas_cost(base_cost, vas, order_value, risk_surcharge)

    total = base_cost + distance_cost + dox_cost + vas_cost

    return {
        "base_cost":     round(base_cost, 2),
        "distance_cost": round(distance_cost, 2),
        "dox_cost":      round(dox_cost, 2),
        "vas_cost":      round(vas_cost, 2),
        "total":         round(total, 2),
    }


# ══════════════════════════════════════════════════════════════
# SECTION 5 — DATA LOADING & ML DELIVERY MODEL
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="📂 Loading & cleaning dataset…")
def load_data(path: str) -> pd.DataFrame:
    """Load DTDC CSV, clean, engineer features for delivery model."""
    df = pd.read_csv(path, encoding="latin1")

    df["Sender Date"]   = pd.to_datetime(df["Sender Date"],  errors="coerce")
    df["Receive Date"]  = pd.to_datetime(df["Receive Date"], errors="coerce")
    df["delivery_days"] = (df["Receive Date"] - df["Sender Date"]).dt.days
    df = df[df["delivery_days"].between(0, 60)]

    df["Value Added Services"] = df["Value Added Services"].fillna("None")
    df["Total Amount"]         = df["Total Amount"].fillna(df["Total Amount"].median())
    df["Chargeable Wt"]        = df["Chargeable Wt"].fillna(df["Chargeable Wt"].median())
    df["Total Pieces"]         = df["Total Pieces"].fillna(1).astype(int)

    df.dropna(
        subset=["Origin", "Destination", "Mode", "Nature of Consignment",
                "Total Amount", "delivery_days"],
        inplace=True,
    )

    df["distance_km"] = df.apply(
        lambda r: calculate_distance(r["Origin"], r["Destination"])
        if r["Origin"] in CITY_COORDS and r["Destination"] in CITY_COORDS else 0.0,
        axis=1,
    )
    df["cost_per_kg"] = df["Total Amount"] / df["Chargeable Wt"].replace(0, np.nan)
    df["cost_per_kg"] = df["cost_per_kg"].fillna(df["cost_per_kg"].median())

    for col in ["Origin", "Destination", "Mode", "Nature of Consignment"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    return df


DELIVERY_FEATURES = [
    "distance_km", "Mode_enc", "Origin_enc",
    "Destination_enc", "Total Pieces", "Chargeable Wt",
]


@st.cache_resource(show_spinner="🤖 Training delivery model…")
def train_delivery_model(df: pd.DataFrame):
    """RandomForest for delivery_days (SLA rules are primary, ML as reference)."""
    X = df[DELIVERY_FEATURES].copy()
    y = df["delivery_days"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(
        n_estimators=150, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    return model, round(r2_score(y_te, model.predict(X_te)), 4)


def build_label_maps(df: pd.DataFrame) -> dict:
    maps = {}
    for col in ["Origin", "Destination", "Mode", "Nature of Consignment"]:
        maps[col] = dict(zip(df[col].astype(str), df[col + "_enc"]))
    return maps


def predict_delivery(model, label_maps: dict, origin: str, destination: str,
                     mode: str, distance_km: float, weight: float) -> int:
    """SLA rule-based delivery days (dataset uniform 1-5, rule gives meaning)."""
    return business_delivery_estimate(mode, distance_km)


# ══════════════════════════════════════════════════════════════
# SECTION 6 — PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="DTDC Smart Estimator",
    page_icon="🚚",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ════════════════════════════════════════════════════════════
   DTDC SMART ESTIMATOR v4 — AMBER / GOLDEN YELLOW THEME
   ════════════════════════════════════════════════════════════ */

/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ════════════════════════════════════════════
   BACKGROUND — dark amber / mustard gradient
   ════════════════════════════════════════════ */
.stApp {
    background: linear-gradient(135deg,
        #3d2000 0%,
        #5c3100 20%,
        #7a4100 40%,
        #6b3800 60%,
        #4a2600 80%,
        #3d2000 100%);
    background-attachment: fixed;
}

/* Make Streamlit's internal containers transparent */
section[data-testid="stAppViewContainer"] > .main,
.block-container {
    background: transparent !important;
}

/* ── Amber sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #3d2000 0%, #5c3100 100%) !important;
    border-right: 1px solid rgba(251,192,45,0.25) !important;
}
section[data-testid="stSidebar"] * { color: #ffe082 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #ffd54f !important; }
section[data-testid="stSidebar"] .stMetric label { color: #ffca28 !important; }
section[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] { color: #fff8e1 !important; }

/* ── Hero banner ── */
.hero {
    text-align: center;
    padding: 2.5rem 2rem 2rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #e65100 0%, #f57f17 40%, #f9a825 75%, #ffb300 100%);
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 12px 40px rgba(230,81,0,0.55);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute; top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: rgba(255,255,255,0.09);
    border-radius: 50%;
}
.hero::after {
    content: "";
    position: absolute; bottom: -30px; left: -30px;
    width: 150px; height: 150px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
}

/* ── Glass-morphism section card — warm amber tint ── */
.section-card {
    background: rgba(255,236,153,0.10);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(251,192,45,0.28);
    border-radius: 18px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.45);
}

/* ── Widget labels & text on amber dark bg ── */
.stSelectbox label,
.stNumberInput label,
.stTextInput label,
label[data-baseweb="label"],
p, .stMarkdown p {
    color: #ffe082 !important;
}
h1, h2, h3, h4 { color: #fff8e1 !important; }
.stSubheader { color: #ffe57f !important; }

/* ════════════════════════════════════════════
   INPUT FIELDS — uniform warm cream styling
   Applies to: selectbox, number_input, text_input, dropdowns
   Fixes white-field & black-on-focus problems
   ════════════════════════════════════════════ */

/* Native HTML inputs & textareas */
input, textarea, select {
    background-color: #fff8e1 !important;
    color: #3e2000 !important;
    border-radius: 8px !important;
    border: 1px solid #fbc02d !important;
}

/* Focus state — stays light, NO black bg */
input:focus, textarea:focus, select:focus {
    background-color: #fff8e1 !important;
    color: #1a0a00 !important;
    outline: none !important;
    box-shadow: 0 0 0 2px #f9a825 !important;
    border-color: #f9a825 !important;
}

/* Streamlit BaseWeb number / text input wrapper */
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea,
.stNumberInput input,
.stTextInput input {
    background-color: #fff8e1 !important;
    color: #3e2000 !important;
    border: 1px solid #fbc02d !important;
    border-radius: 8px !important;
}

div[data-baseweb="input"] input:focus,
div[data-baseweb="textarea"] textarea:focus,
.stNumberInput input:focus,
.stTextInput input:focus {
    background-color: #fff8e1 !important;
    color: #1a0a00 !important;
    border-color: #f9a825 !important;
    box-shadow: 0 0 0 2px #f9a825 !important;
    outline: none !important;
}

/* Number input stepper buttons */
div[data-baseweb="input"] button,
.stNumberInput button {
    background-color: #ffe082 !important;
    color: #5c3100 !important;
    border: 1px solid #fbc02d !important;
}
div[data-baseweb="input"] button:hover,
.stNumberInput button:hover {
    background-color: #ffd54f !important;
}

/* Selectbox / dropdown container */
div[data-baseweb="select"] > div,
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #fff8e1 !important;
    color: #3e2000 !important;
    border: 1px solid #fbc02d !important;
    border-radius: 8px !important;
}

/* Selectbox focused / open state */
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="select"] > div[aria-expanded="true"] {
    background-color: #fff8e1 !important;
    border-color: #f9a825 !important;
    box-shadow: 0 0 0 2px #f9a825 !important;
}

/* Dropdown option text */
div[data-baseweb="select"] span,
div[data-baseweb="select"] [data-testid="stSelectboxVirtualDropdown"] {
    color: #3e2000 !important;
}

/* Dropdown chevron/arrow icon */
div[data-baseweb="select"] svg { fill: #8d5000 !important; }

/* Dropdown menu popup list */
ul[data-baseweb="menu"],
div[data-baseweb="popover"] ul {
    background-color: #fff8e1 !important;
    border: 1px solid #fbc02d !important;
    border-radius: 8px !important;
}

/* Each dropdown list item */
ul[data-baseweb="menu"] li,
div[data-baseweb="popover"] li {
    background-color: #fff8e1 !important;
    color: #3e2000 !important;
}
ul[data-baseweb="menu"] li:hover,
div[data-baseweb="popover"] li:hover {
    background-color: #ffe082 !important;
    color: #3e2000 !important;
}

/* Selected item highlight */
ul[data-baseweb="menu"] li[aria-selected="true"],
div[data-baseweb="popover"] li[aria-selected="true"] {
    background-color: #ffd54f !important;
    color: #3e2000 !important;
}

/* ── Result metric box — warm cream for contrast ── */
.metric-box {
    background: rgba(255,248,225,0.97);
    border-left: 5px solid #e65100;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 4px 20px rgba(230,81,0,0.20);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.metric-box:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 28px rgba(230,81,0,0.32);
}
.metric-box .label {
    font-size: 0.78rem; color: #8d5000; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
}
.metric-box .value {
    font-size: 1.85rem; font-weight: 800; color: #3e2000; margin-top: 0.15rem;
}

/* ── Mode badge ── */
.mode-tag {
    display: inline-block; border-radius: 20px;
    padding: 0.2rem 0.8rem; font-size: 0.73rem; font-weight: 700; margin-top: 0.35rem;
}
.tag-surface { background:#e8f5e9; color:#2e7d32; border:1px solid #a5d6a7; }
.tag-air     { background:#e3f2fd; color:#1565c0; border:1px solid #90caf9; }
.tag-express { background:#fff3e0; color:#e65100; border:1px solid #ffcc80; }

/* ── Route card — warm cream ── */
.route-card {
    background: rgba(255,248,225,0.97);
    border: 2px solid rgba(245,127,23,0.35);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-top: 1.2rem;
    box-shadow: 0 4px 20px rgba(230,81,0,0.15);
}
.route-step {
    display: inline-block;
    background: linear-gradient(135deg, #e65100, #f9a825);
    color: white;
    padding: 0.28rem 0.85rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    margin: 0.2rem 0.1rem;
    box-shadow: 0 2px 8px rgba(230,81,0,0.35);
}
.route-arrow {
    color: #e65100; font-size: 1.0rem; font-weight: 700;
    margin: 0 0.05rem; vertical-align: middle;
}
.route-total {
    font-size: 0.87rem; color: #5c3100; margin-top: 0.7rem; line-height: 1.6;
}

/* ── Cost breakdown card ── */
.cost-breakdown {
    background: rgba(255,248,225,0.96);
    border: 1px solid rgba(245,127,23,0.20);
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
    margin-top: 0.8rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.10);
}

/* ── Divider ── */
.divider { border:none; border-top:1px solid rgba(251,192,45,0.25); margin:1.2rem 0; }

/* ── State info pill ── */
.state-pill {
    display: inline-block;
    background: rgba(249,168,37,0.20);
    border: 1px solid rgba(251,192,45,0.50);
    border-radius: 20px;
    padding: 0.15rem 0.65rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: #ffe082;
    margin-top: 0.2rem;
}

/* ── Expander header on amber dark bg ── */
details summary,
details summary p,
.streamlit-expanderHeader,
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary span {
    color: #ffe082 !important;
    background: rgba(251,192,45,0.08) !important;
    border-radius: 8px !important;
}

/* Expander body text inside output sections */
div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] p,
div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] span {
    color: #3e2000 !important;
}

/* ── Dataframe / table styling ── */
div[data-testid="stDataFrame"] {
    background: rgba(255,248,225,0.95) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(251,192,45,0.30) !important;
}

/* ── Success / error / info alert boxes ── */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
}
div[data-testid="stAlert"][data-baseweb="notification"] {
    background-color: rgba(255,248,225,0.95) !important;
    border-color: #fbc02d !important;
    color: #3e2000 !important;
}

/* ── Metric widget in main area ── */
div[data-testid="stMetric"] label { color: #ffe082 !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #fff8e1 !important; }
div[data-testid="stMetric"] [data-testid="stMetricDelta"] { color: #ffd54f !important; }

/* ── Caption text ── */
small, .stCaption, div[data-testid="stCaptionContainer"] {
    color: #ffca28 !important;
    opacity: 0.85;
}

/* ═══════════════════════════════════════════
   CALCULATE BUTTON — amber/orange gradient
   ═══════════════════════════════════════════ */
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #e65100 0%, #f57f17 50%, #f9a825 100%) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-size: 1.05rem !important;
    font-weight: 700 !important; padding: 0.7rem 1.6rem !important;
    box-shadow: 0 6px 20px rgba(230,81,0,0.50) !important;
    transition: all 0.25s ease !important;
    letter-spacing: 0.02em !important;
    width: 100% !important;
}
div[data-testid="stButton"] button:hover {
    opacity: 0.90 !important;
    box-shadow: 0 10px 30px rgba(230,81,0,0.65) !important;
    transform: translateY(-2px) !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SECTION 7 — HEADER
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <h1 style="font-size:2.9rem; margin:0; font-weight:800; letter-spacing:-0.01em;">
        🚚 DTDC Smart Estimator
    </h1>
    <p style="font-size:1.2rem; margin:12px 0 0; opacity:0.92; font-weight:500;">
        Instant, accurate shipping price &amp; route estimates across India
    </p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SECTION 8 — LOAD DATA, TRAIN MODELS, BUILD GRAPH
# ══════════════════════════════════════════════════════════════

_dataset_path = "Dataset_Generator_for_DTDC.csv"
if not os.path.exists(_dataset_path):
    _dataset_path = "/mnt/user-data/uploads/Dataset_Generator_for_DTDC.csv"

df             = load_data(_dataset_path)
delivery_model, delivery_r2 = train_delivery_model(df)
label_maps     = build_label_maps(df)
G              = build_geo_graph()

with st.expander("📊 Model & Pricing Info", expanded=False):
    mc1, mc2 = st.columns(2)
    with mc1:
        st.metric("⏱ Delivery Model R²", f"{delivery_r2:.4f}", delta="RandomForest")
        st.metric("🏙️ Cities Available", f"{len(ALL_CITIES)}")
    with mc2:
        st.metric("📊 Training Records", f"{len(df):,}")
        st.metric("🗺️ States Covered", f"{len(ALL_STATES)}")
    st.caption(
        "**Pricing:** Surface ₹80/kg · Air Cargo ₹120/kg · Express ₹150/kg · "
        "Distance ₹0.15/km · Dox slab: ₹100/175/+₹75 per 500g"
    )


# ══════════════════════════════════════════════════════════════
# SECTION 9 — INPUT FORM
# Layout:
#   LEFT  col: State filter, City, Weight, Order Value, Nature of Consignment
#   RIGHT col: State filter, City, Shipping Mode, VAS, Risk Surcharge
# ══════════════════════════════════════════════════════════════

st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("📋 Shipment Details")
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

col_input1, col_input2 = st.columns(2)

# ── LEFT COLUMN ─────────────────────────────────────────────
with col_input1:
    st.markdown("### 🗺️ State of Origin")
    origin_state_opts = [NONE_OPTION] + ALL_STATES
    selected_origin_state = st.selectbox(
        "Origin state",
        options=origin_state_opts,
        index=origin_state_opts.index("Delhi"),
        key="origin_state",
        label_visibility="collapsed",
        help="'None' shows all cities; pick a state to filter.",
    )
    origin_cities  = cities_for_state(selected_origin_state)
    default_origin = origin_cities.index("Delhi") if "Delhi" in origin_cities else 0
    origin = st.selectbox(
        "📍 Origin City",
        options=origin_cities,
        index=default_origin,
        key="origin_city",
    )
    auto_origin_state = auto_state_for_city(origin)
    st.markdown(f'<span class="state-pill">📍 {auto_origin_state}</span>',
                unsafe_allow_html=True)

    weight = st.number_input(
        "⚖️ Weight (kg)",
        min_value=0.1, max_value=200.0, value=2.0, step=0.5,
        help="Chargeable weight of the shipment.",
    )

    # NEW: Order Value
    order_value = st.number_input(
        "💰 Order Value (₹)",
        min_value=0.0, max_value=10_000_000.0,
        value=0.0, step=100.0, format="%.2f",
        help="Declared value of shipment. Used for COD & Insurance calculation.",
    )

    # MOVED from right: Nature of Consignment
    nature = st.selectbox(
        "📄 Nature of Consignment",
        options=["Dox", "Non-Dox"],
        help="Dox = Documents (slab pricing) · Non-Dox = Products/parcels.",
    )

# ── RIGHT COLUMN ─────────────────────────────────────────────
with col_input2:
    st.markdown("### 🗺️ State of Destination")
    dest_state_opts  = [NONE_OPTION] + ALL_STATES
    selected_dest_state = st.selectbox(
        "Destination state",
        options=dest_state_opts,
        index=dest_state_opts.index("Maharashtra"),
        key="dest_state",
        label_visibility="collapsed",
        help="'None' shows all cities; pick a state to filter.",
    )
    dest_cities  = cities_for_state(selected_dest_state)
    default_dest = dest_cities.index("Mumbai") if "Mumbai" in dest_cities else 0
    destination = st.selectbox(
        "📍 Destination City",
        options=dest_cities,
        index=default_dest,
        key="dest_city",
    )
    auto_dest_state = auto_state_for_city(destination)
    st.markdown(f'<span class="state-pill">📍 {auto_dest_state}</span>',
                unsafe_allow_html=True)

    # Shipping Mode (row 3 right)
    shipping_mode = st.selectbox(
        "🚚 Shipping Mode",
        options=["Surface", "Air Cargo", "Express"],
        help="Surface ₹80/kg | Air Cargo ₹120/kg | Express ₹150/kg",
    )

    # MOVED here (row 4 right) — VAS with Express option REMOVED
    vas = st.selectbox(
        "💼 Value Added Services",
        options=["None", "COD", "Insurance"],
        help="COD: 1.5% of order value · Insurance: depends on Risk Surcharge",
    )

    # NEW: Risk Surcharge (row 5 right)
    risk_surcharge = st.selectbox(
        "⚠️ Risk Surcharge",
        options=["None", "Carrier", "Owner"],
        help="Used for Insurance VAS. Carrier = 0.2% of order value, Owner = 2%.",
    )

st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SECTION 10 — CALCULATE BUTTON & RESULTS
# ══════════════════════════════════════════════════════════════

if st.button("🚀 Calculate Shipping Price & Route",
             use_container_width=True, type="primary"):

    if origin == destination:
        st.error("⚠️ Origin and destination must be different cities.")
        st.stop()

    with st.spinner("🔄 Computing cost, delivery time, and optimal route…"):

        # Distance
        distance_km = calculate_distance(origin, destination)

        # Delivery time (SLA rules)
        delivery_days = predict_delivery(
            delivery_model, label_maps, origin, destination,
            shipping_mode, distance_km, weight
        )

        # Rule-based cost
        cost_breakdown = calculate_total_cost(
            weight, shipping_mode, distance_km,
            nature, vas, order_value, risk_surcharge
        )
        estimated_cost = cost_breakdown["total"]

        # Transport vehicle
        transport = transport_mode_rule(shipping_mode, distance_km)

        # Route optimization (auto, no button)
        path, path_dist = find_shortest_path(G, origin, destination)

    # ── Four result metric boxes ───────────────────────────────
    st.markdown("---")
    st.subheader("📦 Shipment Estimate Results")

    mode_cls = {"Surface": "tag-surface", "Air Cargo": "tag-air",
                "Express": "tag-express"}.get(shipping_mode, "tag-surface")

    r1, r2_col = st.columns(2)
    with r1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">💰 Estimated Cost</div>
            <div class="value">&#8377;{estimated_cost:,.2f}</div>
            <span class="mode-tag {mode_cls}">{shipping_mode}</span>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-box">
            <div class="label">📏 Distance</div>
            <div class="value">{distance_km:,.1f} km</div>
        </div>""", unsafe_allow_html=True)

    with r2_col:
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">⏱ Estimated Delivery</div>
            <div class="value">{delivery_days} day{"s" if delivery_days != 1 else ""}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-box">
            <div class="label">🚛 Recommended Transport</div>
            <div class="value" style="font-size:1.25rem">{transport}</div>
        </div>""", unsafe_allow_html=True)

    # Summary banner
    o_state = auto_state_for_city(origin)
    d_state = auto_state_for_city(destination)
    st.success(
        f"✅ **{origin}** ({o_state}) → **{destination}** ({d_state}) | "
        f"Mode: **{shipping_mode}** | VAS: **{vas}** | "
        f"Weight: **{weight} kg** | Order Value: ₹{order_value:,.2f}"
    )

    # ── ROUTE OPTIMIZATION (auto-displayed) ───────────────────
    steps_html = ""
    for i, city in enumerate(path):
        cst = CITY_STATE_MAP.get(city, "")
        steps_html += f'<span class="route-step" title="{cst}">{city}</span>'
        if i < len(path) - 1:
            steps_html += '<span class="route-arrow"> ➜ </span>'

    segments = []
    for i in range(len(path) - 1):
        d = calculate_distance(path[i], path[i + 1])
        segments.append((path[i], path[i + 1], d))

    n_stops    = len(path) - 2
    stops_text = (f"{n_stops} intermediate stop{'s' if n_stops != 1 else ''}"
                  if n_stops > 0 else "Direct route")

    st.markdown(f"""
    <div class="route-card">
        <h4 style="margin:0 0 0.9rem; color:#1a1a4e; font-weight:800; font-size:1.05rem;">
            🧭 Optimal Route &nbsp;—&nbsp; Dijkstra's Algorithm
        </h4>
        <div style="line-height:2.4">{steps_html}</div>
        <div class="route-total">
            &nbsp;|&nbsp; 🔵 {stops_text}
            &nbsp;|&nbsp; 🏙️ {len(path)} cities on route
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Cost breakdown expander ────────────────────────────────
    with st.expander("📑 Cost Breakdown", expanded=True):
        cb = cost_breakdown
        breakdown_df = pd.DataFrame({
            "Component": [
                "Base Cost (Weight × Rate/kg)",
                "Distance Cost (km × ₹0.15/km)",
                "Dox Slab Cost",
                "VAS / Insurance Cost",
                "─────────────────────",
                "TOTAL",
            ],
            "Amount (₹)": [
                f"₹{cb['base_cost']:,.2f}",
                f"₹{cb['distance_cost']:,.2f}",
                f"₹{cb['dox_cost']:,.2f}",
                f"₹{cb['vas_cost']:,.2f}",
                "",
                f"₹{cb['total']:,.2f}",
            ],
            "Note": [
                f"{weight} kg × ₹{MODE_RATE_PER_KG[shipping_mode]}/kg",
                f"{distance_km:.1f} km × ₹0.15",
                f"{'Dox slab pricing' if nature == 'Dox' else 'Non-Dox: no charge'}",
                f"{vas} — {risk_surcharge}" if vas != "None" else "No VAS",
                "",
                "",
            ],
        })
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    # Full estimate table
    with st.expander("📋 Full Estimate Details", expanded=False):
        full_df = pd.DataFrame({
            "Parameter": [
                "Origin City", "Origin State",
                "Destination City", "Destination State",
                "Distance", "Weight", "Order Value",
                "Shipping Mode", "Nature of Consignment",
                "Value Added Services", "Risk Surcharge",
                "Final Cost", "Delivery Time", "Transport Mode",
            ],
            "Value": [
                origin, o_state, destination, d_state,
                f"{distance_km:.1f} km", f"{weight} kg",
                f"₹{order_value:,.2f}",
                shipping_mode, nature, vas, risk_surcharge,
                f"₹{estimated_cost:,.2f}",
                f"{delivery_days} day(s)", transport,
            ],
        })
        st.dataframe(full_df, use_container_width=True, hide_index=True)

    # Segment distances
    if segments:
        with st.expander("📍 Route Segment Distances", expanded=False):
            seg_df = pd.DataFrame(
                {"From": [s[0] for s in segments],
                 "To":   [s[1] for s in segments],
                 "Distance (km)": [f"{s[2]:,.1f}" for s in segments]}
            )
            st.dataframe(seg_df, use_container_width=True, hide_index=True)
            st.caption(
                "On a fully-connected 100-city graph, Dijkstra confirms the "
                "direct Haversine edge as the shortest path."
            )


# ══════════════════════════════════════════════════════════════
# SECTION 11 — SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("📊 App Summary")
    st.metric("Total Records",    f"{len(df):,}")
    st.metric("Cities",           f"{len(ALL_CITIES)}")
    st.metric("States / UTs",     f"{len(ALL_STATES)}")
    st.metric("Avg. Delivery",    f"{df['delivery_days'].mean():.1f} days")
    st.metric("Avg. Distance",    f"{df['distance_km'].mean():,.0f} km")

    st.markdown("---")
    st.caption("🤖 ML Model")
    st.write(f"⏱ Delivery R²: **{delivery_r2}**")

    st.markdown("---")
    st.caption("💸 Rate Card")
    st.write("🟢 Surface   → **₹80/kg**")
    st.write("🔵 Air Cargo → **₹120/kg**")
    st.write("🟠 Express   → **₹150/kg**")
    st.write("📏 Distance  → **₹0.15/km**")

    st.markdown("---")
    st.caption("📄 Dox Slab Pricing")
    st.write("≤ 500g → ₹100")
    st.write("≤ 1 kg → ₹175")
    st.write("> 1 kg → +₹75 per 500g")

    st.markdown("---")
    st.caption("💼 VAS Rules")
    st.write("• COD: 1.5% of order value")
    st.write("• Insurance (Carrier): 0.2%")
    st.write("• Insurance (Owner): 2.0%")

    st.markdown("---")
    st.caption("© 2025 DTDC Smart Estimator")