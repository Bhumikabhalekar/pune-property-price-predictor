import pandas as pd
import streamlit as st
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
FIG = BASE / "reports" / "figures"

st.set_page_config(page_title="Pune Market Insights", page_icon="📊", layout="wide")
st.title("📊 Pune Property Market Insights")
st.write("What we learned from analysing about 28,000 Pune listings.")


@st.cache_data
def load_stats():
    return pd.read_csv(BASE / "data" / "processed" / "locality_stats.csv")


# ---------- locality price lookup ----------
st.subheader("Check any locality")
stats = load_stats()
stats = stats[stats["listings"] >= 20].sort_values("locality")
pick = st.selectbox("Locality", stats["locality"], format_func=str.title)
row = stats[stats["locality"] == pick].iloc[0]
c1, c2 = st.columns(2)
c1.metric("Median price per sq ft", f"₹{row['median_psf']:,.0f}")
c2.metric("Listings in our data", int(row["listings"]))

# ---------- key charts ----------
st.subheader("Key findings")

st.image(str(FIG / "top_localities_psf.png"))
st.write("**Location matters most.** Sangamvadi and Boat Club Road cost about double "
         "per sq ft compared with areas like Wanowrie.")

col_a, col_b = st.columns(2)
with col_a:
    st.image(str(FIG / "area_vs_price.png"))
    st.write("**Bigger means pricier**, but at the same size, prices can differ 4 to 5 times.")
with col_b:
    st.image(str(FIG / "bedrooms_type_vs_price.png"))
    st.write("**Each extra bedroom raises the price sharply.** Villas and penthouses cost the most.")

col_c, col_d = st.columns(2)
with col_c:
    st.image(str(FIG / "amenities_vs_psf.png"))
    st.write("**More amenities do not mean a higher price per sq ft.** Big amenity-rich "
             "projects tend to be in cheaper outer areas.")
with col_d:
    st.image(str(FIG / "age_vs_psf.png"))
    st.write("**Older homes look pricier per sq ft** because they sit in established central areas.")