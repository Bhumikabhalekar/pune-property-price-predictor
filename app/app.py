import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="Pune Property Price Predictor", page_icon="🏠", layout="wide")


# ---------- load files (cached so they load only once) ----------
@st.cache_resource
def load_model():
    model = joblib.load(BASE / "models" / "rf_price_model.joblib")
    cols = joblib.load(BASE / "models" / "feature_columns.joblib")
    loc_info = joblib.load(BASE / "models" / "locality_psf.joblib")
    return model, cols, loc_info


@st.cache_data
def load_data():
    return pd.read_csv(BASE / "data" / "processed" / "pune_property_clean.csv")


model, feature_cols, loc_info = load_model()
data = load_data()
locality_stats = pd.read_csv(BASE / "data" / "processed" / "locality_stats.csv")


def fmt(price):
    """Show a price as lakh or crore."""
    return f"₹{price / 1e7:.2f} Cr" if price >= 1e7 else f"₹{price / 1e5:.1f} L"


def predict_with_range(row):
    """Return (estimate, low, high) using every tree in the forest."""
    prep = model.named_steps["prep"]
    forest = model.named_steps["rf"]
    X = prep.transform(row)
    if hasattr(X, "toarray"):
        X = X.toarray()
    tree_preds = np.exp([t.predict(X)[0] for t in forest.estimators_])
    estimate = np.exp(forest.predict(X)[0])
    return estimate, np.percentile(tree_preds, 10), np.percentile(tree_preds, 90)


# ---------- page ----------
st.title("🏠 Pune Property Price Predictor")
st.write("Enter the property details to get an estimated price.")

amenity_cols = [c for c in feature_cols if c.startswith("am_")]
nice = {c: c.replace("am_", "").replace("_", " ").title() for c in amenity_cols}

left, right = st.columns(2)

with left:
    locality = st.selectbox("Locality", sorted(data["locality"].unique()))
    property_type = st.selectbox("Property type", sorted(data["property_type"].unique()))
    area = st.number_input("Area (sq ft)", 250, 10000, 1000, step=50)
    bedrooms = st.number_input("Bedrooms (BHK)", 1, 8, 2)
    bathroom = st.number_input("Bathrooms", 1, 8, 2)
    balconies = st.number_input("Balconies", 0, 8, 1)

with right:
    floor = st.number_input("Floor", 0, 40, 3)
    totalfloor = st.number_input("Total floors in building", 1, 40, 10)
    age = st.number_input("Property age (years)", 0, 50, 3)
    neworold = st.selectbox("New or resale", sorted(data["neworold"].unique()))
    facing = st.selectbox("Facing", sorted(data["facing"].unique()))
    chosen = st.multiselect("Amenities", amenity_cols, format_func=lambda c: nice[c])
    asking_lakh = st.number_input("Seller's asking price in ₹ lakh (optional)", 0.0, 5000.0, 0.0, step=5.0)

if st.button("Predict price", type="primary"):
    # start with every column empty / default, then fill what the user gave us
    row = {c: np.nan for c in feature_cols}
    row.update({"ownership": "unknown", "status": "unknown", "addi_rooms": 0})
    row.update({c: 0 for c in amenity_cols})
    row.update({c: 1 for c in chosen})
    row.update({
        "locality": locality, "property_type": property_type, "area": area,
        "bedrooms": bedrooms, "bathroom": bathroom, "balconies": balconies,
        "floor": floor, "totalfloor": totalfloor, "age": age,
        "neworold": neworold, "facing": facing, "amenity_count": len(chosen),
    })
    row["locality_psf"] = loc_info["map"].get(locality, loc_info["default"])
    row = pd.DataFrame([row])[feature_cols]

    estimate, low, high = predict_with_range(row)

    st.success(f"Estimated price: **{fmt(estimate)}**")
    st.info(f"Likely range: **{fmt(low)} to {fmt(high)}**")
    st.caption(f"That is about ₹{estimate / area:,.0f} per sq ft.")

    
    # ---------- 1. Fair price check ----------
    if asking_lakh > 0:
        asking = asking_lakh * 1e5
        st.subheader("Is the asking price fair?")
        if asking > high:
            st.error(f"Overpriced: {fmt(asking)} is above our likely range.")
        elif asking < low:
            st.info(f"Below our range: {fmt(asking)} may be a good deal, or something may be wrong. Check carefully.")
        else:
            st.success(f"Fair: {fmt(asking)} is within our likely range.")

    # ---------- 2. Locality comparison ----------
    st.subheader(f"How does {locality.title()} compare?")
    pune_psf = (data["price"] / data["area"]).median()
    loc_row = locality_stats[locality_stats["locality"] == locality]
    c1, c2, c3 = st.columns(3)
    c1.metric("Your estimate", f"₹{estimate / area:,.0f} / sq ft")
    if not loc_row.empty:
        c2.metric(f"{locality.title()} median", f"₹{loc_row['median_psf'].iloc[0]:,.0f} / sq ft")
        n = int(loc_row["listings"].iloc[0])
        if n < 20:
            st.warning(f"Only {n} listings in this locality, so treat this estimate with extra care.")
        else:
            st.caption(f"Based on {n} listings in this locality.")
    c3.metric("Pune median", f"₹{pune_psf:,.0f} / sq ft")

    # ---------- 3. Similar listings ----------
    st.subheader("Similar listings from our data")
    same_type = (data["bedrooms"] == bedrooms) & (data["property_type"] == property_type)

    # first try: same locality, area within 20%
    sim = data[same_type & (data["locality"] == locality)
               & data["area"].between(area * 0.8, area * 1.2)].copy()
    where = f"in {locality.title()}"
    fallback = False

    # second try: all of Pune, area within 10%
    if sim.empty:
        sim = data[same_type & data["area"].between(area * 0.9, area * 1.1)].copy()
        where = "in other Pune localities (none found in this one), closest to your estimated price"
        fallback = True

    if sim.empty:
        st.write("No close matches found.")
    else:
        st.caption(f"Closest matches {where}")
        if fallback:
            sim["gap"] = (sim["price"] - estimate).abs()   # closest price to our estimate
        else:
            sim["gap"] = (sim["area"] - area).abs()        # closest size
        sim = sim.sort_values("gap").head(5)
        show = sim[["locality", "area", "bedrooms", "bathroom", "age", "price"]].copy()
        show["locality"] = show["locality"].str.title()
        show["price"] = show["price"].apply(fmt)
        show.columns = ["Locality", "Area (sq ft)", "Bedrooms", "Bathrooms", "Age (yrs)", "Price"]
        st.dataframe(show, hide_index=True)