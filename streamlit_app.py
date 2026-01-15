import re
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


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


def list_image_codes(brand_folder: str) -> list[str]:
    """
    Reads available png files under: assets/images/<brand_folder>/*.png
    Returns list like ['320i','z4','x1'] (filename stems).
    """
    folder = Path("assets") / "images" / brand_folder
    if not folder.exists():
        return []
    return sorted([p.stem for p in folder.glob("*.png")])


# ---------- Sidebar: brand + available codes ----------
st.sidebar.header("Görsel Ayarları")
brand_folder = st.sidebar.selectbox("Marka klasörü", ["bmw", "mercedes"], index=0)

available_codes = list_image_codes(brand_folder)
if not available_codes:
    st.sidebar.warning(f"assets/images/{brand_folder} içinde PNG bulunamadı. (Örn: 320i.png)")


# ---------- Input section ----------
st.subheader("1) Veri Girişi")

tab1, tab2 = st.tabs(["Manuel giriş", "CSV yükle"])

df_in = None  # net state

with tab1:
    st.caption(
        "Her satır bir araç. Fiyat: Liste fiyatı. İndirim opsiyonel (%). GP opsiyonel. "
        "X_pos: grafikte yatay konum (0–1). Resim Kodu: listeden seç."
    )
    default_rows = 8

    df_manual = pd.DataFrame({
        "model": ["BMW X1 xDrive25e – M Sport"] + [""] * (default_rows - 1),
        "price_list": ["4.927.695 ₺"] + [""] * (default_rows - 1),
        "discount_pct": ["6%"] + [""] * (default_rows - 1),
        "gross_profit": [""] + [""] * (default_rows - 1),
        "note": [""] + [""] * (default_rows - 1),
        "x_pos": ["0.50"] + [""] * (default_rows - 1),      # 0..1 arası
        "img_code": [""] + [""] * (default_rows - 1),       # kullanıcı dropdown’dan seçecek
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
                "X (0–1)",
                help="0=sol, 0.5=orta, 1=sağ. Ondalıklı girebilirsin (virgül kabul). Boş bırakılırsa otomatik."
            ),

            # ✅ burada: kullanıcı kod yazmıyor, listeden seçiyor
            "img_code": st.column_config.SelectboxColumn(
                "Resim Kodu (png)",
                help=f"assets/images/{brand_folder}/ içindeki PNG adından seç (ör: 320i). Seçilmezse nokta gösterilir.",
                options=[""] + available_codes,
            ),
        },
    )
    df_in = edited.copy()

with tab2:
    st.caption("CSV kolonları: model, price_list, discount_pct, gross_profit, note, x_pos, img_code")
    up = st.file_uploader("CSV yükle", type=["csv"])
    if up is not None:
        df_csv = pd.read_csv(up)
        df_in = df_csv.copy()


# ---------- Process ----------
st.subheader("2) Grafik Ayarları")

if df_in is None:
    st.info("Manuel giriş sekmesinden veri gir veya CSV yükle.")
    st.stop()

df = df_in.copy()

# Required columns
required_cols = {"model", "price_list"}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    st.error(f"Veride en az şu kolonlar olmalı: {', '.join(sorted(required_cols))}")
    st.stop()

# Clean
df["model"] = df["model"].astype(str).str.strip()
df = df[df["model"].ne("")].copy()

df["price_list_num"] = df["price_list"].apply(parse_money)
df = df[df["price_list_num"].notna()].copy()

if "discount_pct" not in df.columns:
    df["discount_pct"] = 0
df["discount_frac"] = df["discount_pct"].apply(parse_percent)

df["price_net_num"] = df["price_list_num"] * (1 - df["discount_frac"])

if "gross_profit" not in df.columns:
    df["gross_profit"] = ""
df["gross_profit_str"] = df["gross_profit"].astype(str).fillna("").str.strip()

if "note" not in df.columns:
    df["note"] = ""
df["note"] = df["note"].astype(str).fillna("").str.strip()

if "img_code" not in df.columns:
    df["img_code"] = ""
df["img_code"] = df["img_code"].astype(str).fillna("").str.strip().str.lower()

# Sort by net price (still used for y-order; x is independent)
df = df.sort_values("price_net_num").reset_index(drop=True)

# Controls
currency_mode = st.radio(
    "Sıralama hangi fiyatla olsun?",
    ["Net fiyat (indirim sonrası)", "Liste fiyatı"],
    horizontal=True,
)
y_col = "price_net_num" if currency_mode.startswith("Net") else "price_list_num"

title_left = st.text_input("Grafik Başlığı", value="Fiyat Merdiveni")
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

# X positions: user-provided (0..1). Accept 0,5 or 0.5
if "x_pos" in df.columns:
    df["x_pos_num"] = pd.to_numeric(
        df["x_pos"].astype(str).str.replace(",", ".", regex=False).str.strip(),
        errors="coerce"
    )
else:
    df["x_pos_num"] = None

# Auto-fill missing x positions evenly (optional fallback)
missing = df["x_pos_num"].isna()
if missing.any():
    # spread missing points across 0..1
    n = missing.sum()
    auto_vals = [0.1 + (i / max(n - 1, 1)) * 0.8 for i in range(n)]  # 0.1..0.9
    df.loc[missing, "x_pos_num"] = auto_vals

df["x"] = df["x_pos_num"].astype(float).clip(0.0, 1.0)

fig_w = 16
fig_h = 7
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

# Fixed X axis 0..1
ax.set_xlim(0, 1)
ax.margins(x=0)
ax.set_xticks([])

# Title higher + bolder
ax.set_title(
    title_left,
    fontsize=18,
    fontweight="bold",
    pad=32
)
if subtitle:
    ax.text(
        0.5, 1.02,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        color="#555555"
    )

ax.set_xlabel("")
ax.set_ylabel("Fiyat (₺)")

# Give labels breathing room on Y
ax.set_ylim(min_p - rng * 0.05, max_p + rng * 0.18)

# Y tick formatting (safe way)
import matplotlib.ticker as mticker
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: format_try(v)))

# --- Draw images if selected; otherwise draw a small dot ---
for i in range(len(df)):
    x = float(df.loc[i, "x"])
    y = float(df.loc[i, y_col])
    code = str(df.loc[i, "img_code"]).strip().lower()

    img_path = Path("assets") / "images" / brand_folder / f"{code}.png"

    if code and img_path.exists():
        im = Image.open(img_path).convert("RGBA")
        imagebox = OffsetImage(im, zoom=0.22)  # adjust 0.18–0.30 as needed
        ab = AnnotationBbox(
            imagebox,
            (x, y),
            xybox=(0, 24),   # move image upward from the point
            xycoords="data",
            boxcoords="offset points",
            frameon=False
        )
        ax.add_artist(ab)
    else:
        ax.scatter([x], [y], s=30)

# Labels (2-line: model in bold blue, details smaller underneath)
if show_labels:
    for i in range(len(df)):
        x = float(df.loc[i, "x"])
        y = float(df.loc[i, y_col])

        model = df.loc[i, "model"]
        price_show = df.loc[i, y_col]

        discount = df.loc[i, "discount_frac"]
        gp = df.loc[i, "gross_profit_str"]
        note = df.loc[i, "note"]

        # Line 1: model (blue + bold) -> under image
        ax.annotate(
            model,
            (x, y),
            textcoords="offset points",
            xytext=(0, -16),
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="#1f77b4",
        )

        # Line 2: details (smaller)
        details_parts = [format_try(price_show)]

        if discount and discount > 0:
            details_parts.append(f"({discount*100:.0f}% indirim)")

        if label_mode.endswith("+ GP") and gp and gp.lower() not in {"nan", "none", ""}:
            details_parts.append(f"GP: {gp}")

        if note:
            details_parts.append(note)

        details = " ".join(details_parts)

        ax.annotate(
            details,
            (x, y),
            textcoords="offset points",
            xytext=(0, -30),
            ha="center",
            va="top",
            fontsize=8,
            color="#333333",
        )

ax.grid(True, axis="y", linestyle="--", alpha=0.25)

fig.tight_layout()
st.pyplot(fig, use_container_width=True)

# Table view
with st.expander("Veri tablosu"):
    out_cols = ["model", "price_list_num", "discount_frac", "price_net_num", "gross_profit_str", "note", "x", "img_code"]
    out = df[out_cols].copy()

    out["price_list_num"] = out["price_list_num"].apply(format_try)
    out["price_net_num"] = out["price_net_num"].apply(format_try)
    out["discount_frac"] = out["discount_frac"].apply(lambda x: f"{x*100:.0f}%" if x else "")
    out["x"] = out["x"].apply(lambda v: f"{v:.2f}")

    out = out.rename(columns={
        "price_list_num": "Liste Fiyatı",
        "price_net_num": "Net Fiyat",
        "discount_frac": "İndirim",
        "gross_profit_str": "GP",
        "note": "Not",
        "x": "X (0–1)",
        "img_code": "Resim Kodu",
    })

    st.dataframe(out, use_container_width=True)
