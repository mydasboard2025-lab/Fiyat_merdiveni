import re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fiyat Merdiveni", layout="wide")
st.title("Fiyat Merdiveni")

# ---------- Helpers ----------
def parse_money(x):
    """
    Accepts values like:
      4.927.695 ₺
      4,927,695
      4927695
      4.927.695
    Returns float or None.
    """
    if x is None:
        return None
    if isinstance(x, (int, float)) and pd.notna(x):
        return float(x)

    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None

    # keep digits only
    digits = re.sub(r"[^\d]", "", s)
    if digits == "":
        return None
    return float(digits)

def parse_percent(x):
    """
    Accepts:
      6%
      0.06
      6
    Returns fraction (0.06) or 0.0 if empty.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return 0.0
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return 0.0
    s = s.replace(",", ".")
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except:
            return 0.0
    try:
        v = float(s)
        # if user typed 6, interpret as 6%
        if v > 1.0:
            return v / 100.0
        return v
    except:
        return 0.0

def format_try(v):
    if v is None or pd.isna(v):
        return ""
    v = float(v)
    return f"{v:,.0f} ₺".replace(",", ".")

# ---------- Input section ----------
st.subheader("1) Veri Girişi")

tab1, tab2 = st.tabs(["Manuel giriş", "CSV yükle"])

with tab1:
    st.caption("Her satır bir araç. Fiyat: Liste fiyatı. İndirim opsiyonel (%). GP opsiyonel. X_pos: grafikte yatay konum (1,2,3...).")
    default_rows = 8
    df_manual = pd.DataFrame({
        "model": ["BMW X1 xDrive25e – M Sport"] + [""]*(default_rows-1),
        "price_list": ["4.927.695 ₺"] + [""]*(default_rows-1),
        "discount_pct": ["6%"] + [""]*(default_rows-1),
        "gross_profit": [""] + [""]*(default_rows-1),
        "note": [""] + [""]*(default_rows-1),
        "x_pos": ["1"] + [""]*(default_rows-1),  # ✅ yeni kolon
    })

    edited = st.data_editor(
        df_manual,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "model": st.column_config.TextColumn("Model", required=True),
            "price_list": st.column_config.TextColumn("Liste Fiyatı (₺)", required=True),
            "discount_pct": st.column_config.TextColumn("İndirim (%)", help="6% veya 6 veya 0.06"),
            "gross_profit": st.column_config.TextColumn("GP (opsiyonel)", help="Örn: 4.033€ veya 4033"),
            "note": st.column_config.TextColumn("Not (opsiyonel)"),
            "x_pos": st.column_config.TextColumn(
                "X (Yatay Konum)",
                help="Ondalıklı girebilirsin: Virgül kabul, boş bırakılırsa otomatik.",
            ),
        },
    )
    df_in = edited.copy()


with tab2:
    st.caption("CSV kolonları: model, price_list, discount_pct, gross_profit, note")
    up = st.file_uploader("CSV yükle", type=["csv"])
    if up is not None:
        df_csv = pd.read_csv(up)
        df_in = df_csv.copy()

# ---------- Process ----------
st.subheader("2) Grafik Ayarları")

if df_in is None:
    st.info("Manuel giriş sekmesinden veri gir veya CSV yükle.")
    st.stop()

# Clean + compute
df = df_in.copy()
if "model" not in df.columns or "price_list" not in df.columns:
    st.error("Veride en az 'model' ve 'price_list' kolonları olmalı.")
    st.stop()

df["model"] = df["model"].astype(str).str.strip()
df = df[df["model"].ne("")].copy()

df["price_list_num"] = df["price_list"].apply(parse_money)
df = df[df["price_list_num"].notna()].copy()

if "discount_pct" not in df.columns:
    df["discount_pct"] = 0
df["discount_frac"] = df["discount_pct"].apply(parse_percent)

# Net price (after discount)
df["price_net_num"] = df["price_list_num"] * (1 - df["discount_frac"])

# Optional GP
if "gross_profit" not in df.columns:
    df["gross_profit"] = ""
df["gross_profit_str"] = df["gross_profit"].astype(str).fillna("").str.strip()

# Optional note
if "note" not in df.columns:
    df["note"] = ""
df["note"] = df["note"].astype(str).fillna("").str.strip()

# Sort by net price
df = df.sort_values("price_net_num").reset_index(drop=True)

# Controls
currency_mode = st.radio(
    "Sıralama hangi fiyatla olsun?",
    ["Net fiyat (indirim sonrası)", "Liste fiyatı"],
    horizontal=True,
)
y_col = "price_net_num" if currency_mode.startswith("Net") else "price_list_num"

title_left = st.text_input("Sol başlık", value="Fiyat Merdiveni")
subtitle = st.text_input("Alt başlık (opsiyonel)", value="")

show_labels = st.checkbox("Etiketleri göster", value=True)
label_mode = st.selectbox("Etiket içeriği", ["Model + Fiyat", "Model + Fiyat + İndirim", "Model + Fiyat + İndirim + GP"])

# ---------- Plot ----------
st.subheader("3) Fiyat Merdiveni")

if len(df) < 2:
    st.warning("En az 2 araç girilmeli.")
    st.stop()

min_p = float(df[y_col].min())
max_p = float(df[y_col].max())
rng = max(max_p - min_p, 1.0)

# x positions = ladder steps
# X positions: user-defined (x_pos) if provided, otherwise auto 1..N
if "x_pos" in df.columns:
    df["x_pos_num"] = pd.to_numeric(
        df["x_pos"].astype(str).str.replace(",", ".", regex=False).str.strip(),
        errors="coerce"
    )
else:
    df["x_pos_num"] = None

    
# auto-fill missing x positions with 1..N
missing = df["x_pos_num"].isna()
if missing.any():
    auto_vals = list(range(1, missing.sum() + 1))
    df.loc[missing, "x_pos_num"] = auto_vals

df["x"] = df["x_pos_num"].astype(float)



fig_w = max(9, min(16, 0.9 * len(df) + 6))
fig_h = 6

fig, ax = plt.subplots(figsize=(fig_w, fig_h))

# points
ax.scatter(df["x"], df[y_col])

# axis formatting
ax.set_title(title_left + (f"\n{subtitle}" if subtitle else ""))
ax.set_xlabel("")
ax.set_ylabel("Fiyat (₺)")

# Format y tick labels with Turkish thousands dots
yticks = ax.get_yticks()
ax.set_yticklabels([format_try(v) for v in yticks])

# Remove x ticks (we label points)
ax.set_xticks([])

# Labels (2-line: model in bold blue, details smaller underneath)
if show_labels:
    for i in range(len(df)):
        x = df.loc[i, "x"]
        y = df.loc[i, y_col]

        model = df.loc[i, "model"]

        # What to show as "price" (net or list depending on selection)
        price_show = df.loc[i, y_col]

        discount = df.loc[i, "discount_frac"]
        gp = df.loc[i, "gross_profit_str"]
        note = df.loc[i, "note"]

        # --- Line 1: model (blue + bold)
        ax.annotate(
            model,
            (x, y),
            textcoords="offset points",
            xytext=(0, 13),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#1f77b4",  # nice blue (BMW-like)
        )

        # --- Line 2: details (smaller)
        details_parts = [format_try(price_show)]

        # discount in parentheses
        if discount and discount > 0:
            details_parts.append(f"({discount*100:.0f}% indirim)")

        # GP
        if gp and gp.lower() not in {"nan", "none", ""}:
            details_parts.append(f"GP: {gp}")

        # optional note (you can comment this out if you don't want it)
        if note:
            details_parts.append(note)

        details = " ".join(details_parts)

        ax.annotate(
            details,
            (x, y),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )


ax.grid(True, axis="y", linestyle="--", alpha=0.3)
st.pyplot(fig, use_container_width=True)

# Table view
with st.expander("Veri tablosu"):
    out = df[["model", "price_list_num", "discount_frac", "price_net_num", "gross_profit_str", "note"]].copy()
    out["price_list_num"] = out["price_list_num"].apply(format_try)
    out["price_net_num"] = out["price_net_num"].apply(format_try)
    out["discount_frac"] = out["discount_frac"].apply(lambda x: f"{x*100:.0f}%" if x else "")
    out = out.rename(columns={
        "price_list_num": "Liste Fiyatı",
        "price_net_num": "Net Fiyat",
        "discount_frac": "İndirim",
        "gross_profit_str": "GP",
        "note": "Not",
    })
    st.dataframe(out, use_container_width=True)

