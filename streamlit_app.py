import re
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.ticker as mticker


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


def list_all_image_options(brands: list[str]) -> list[str]:
    """
    Returns ["", "bmw/320i", "bmw/z4", "mercedes/c200", ...]
    Reads from assets/images/<brand>/*.png
    """
    out = [""]
    for b in brands:
        folder = Path("assets") / "images" / b
        if folder.exists():
            codes = sorted([p.stem for p in folder.glob("*.png")])
            out += [f"{b}/{c}" for c in codes]
    return out


def resize_to_width(im: Image.Image, target_w: int) -> Image.Image:
    """
    Resize PIL image to target width while keeping aspect ratio.
    Uses LANCZOS for good quality.
    """
    w, h = im.size
    if w == 0 or target_w <= 0:
        return im
    scale = target_w / float(w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return im.resize((new_w, new_h), Image.LANCZOS)


BRANDS = ["bmw", "mercedes"]
all_img_options = list_all_image_options(BRANDS)


# ---------- Input section ----------
st.subheader("1) Veri Girişi")

tab1, tab2 = st.tabs(["Manuel giriş", "CSV yükle"])
df_in = None

with tab1:
    st.caption(
        "Her satır bir araç. Fiyat: Liste fiyatı. İndirim opsiyonel (%). GP opsiyonel. "
        "X: 0–1 (0=sol, 0.5=orta, 1=sağ). Resim: tek dropdown’dan bmw/xxx veya mercedes/xxx seç."
    )

    default_rows = 8
    df_manual = pd.DataFrame({
        "model": ["BMW X1 xDrive25e – M Sport"] + [""] * (default_rows - 1),
        "price_list": ["4.927.695 ₺"] + [""] * (default_rows - 1),
        "discount_pct": ["6%"] + [""] * (default_rows - 1),
        "gross_profit": [""] + [""] * (default_rows - 1),
        "note": [""] + [""] * (default_rows - 1),
        "x_pos": ["0.50"] + [""] * (default_rows - 1),
        "img_code": [""] + [""] * (default_rows - 1),  # now stores "bmw/320i" etc.
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

            "img_code": st.column_config.SelectboxColumn(
                "Resim (Marka/Model)",
                help="Örn: bmw/320i veya mercedes/c200. Seçilmezse nokta gösterilir.",
                options=all_img_options,
            ),
        },
    )
    df_in = edited.copy()

with tab2:
    st.caption("CSV kolonları: model, price_list, discount_pct, gross_profit, note, x_pos, img_code")
    st.caption("Not: img_code artık 'bmw/320i' gibi olmalı.")
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

required_cols = {"model", "price_list"}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    st.error(f"Veride en az şu kolonlar olmalı: {', '.join(sorted(required_cols))}")
    st.stop()

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

# Sort by net price (y-order). x is independent.
df = df.sort_values("price_net_num").reset_index(drop=True)

currency_mode = st.radio(
    "Sıralama hangi fiyatla olsun?",
    ["Net fiyat (indirim sonrası)", "Liste fiyatı"],
    horizontal=True,
)
y_col = "price_net_num" if currency_mode.startswith("Net") else "price_list_num"

title_left = st.text_input("Grafik Başlığı", value="Fiyat Merdiveni")
subtitle = st.text_input("Alt başlık (opsiyonel)", value="")

show_labels = st.checkbox("Etiketleri göster", value=True)
label_mode = st.selectbox(
    "Etiket içeriği",
    ["Model + Fiyat", "Model + Fiyat + İndirim", "Model + Fiyat + İndirim + GP"]
)

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

# Auto-fill missing x positions across 0..1 (fallback)
missing = df["x_pos_num"].isna()
if missing.any():
    n = missing.sum()
    auto_vals = [0.1 + (i / max(n - 1, 1)) * 0.8 for i in range(n)]  # 0.1..0.9
    df.loc[missing, "x_pos_num"] = auto_vals

df["x"] = df["x_pos_num"].astype(float).clip(0.0, 1.0)

fig, ax = plt.subplots(figsize=(16, 7))

# Fixed X axis 0..1
ax.set_xlim(0, 1)
ax.margins(x=0)
ax.set_xticks([])

# Title higher + bolder
ax.set_title(title_left, fontsize=18, fontweight="bold", pad=32)
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

# Give top room for images/labels
ax.set_ylim(min_p - rng * 0.14, max_p + rng * 0.28)

# Y tick formatter
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: format_try(v)))

# TARGET width for all images (px) — same for all images
TARGET_W = 260

# Draw images if selected; otherwise draw a dot
for i in range(len(df)):
    x = float(df.loc[i, "x"])
    y = float(df.loc[i, y_col])

    sel = str(df.loc[i, "img_code"]).strip().lower()  # expects "bmw/320i"
    img_path = None
    if sel and "/" in sel:
        b, c = sel.split("/", 1)
        img_path = Path("assets") / "images" / b / f"{c}.png"

    if img_path is not None and img_path.exists():
        im = Image.open(img_path).convert("RGBA")

        # normalize width
        im = resize_to_width(im, TARGET_W)

        imagebox = OffsetImage(im, zoom=1.0)  # draw the resized image as-is
        ab = AnnotationBbox(
            imagebox,
            (x, y),
            xybox=(0, 34),
            xycoords="data",
            boxcoords="offset points",
            frameon=False
        )
        ax.add_artist(ab)
    else:
        ax.scatter([x], [y], s=30)

# Labels under the image
if show_labels:
    for i in range(len(df)):
        x = float(df.loc[i, "x"])
        y = float(df.loc[i, y_col])

        model = df.loc[i, "model"]
        price_show = df.loc[i, y_col]
        discount = df.loc[i, "discount_frac"]
        gp = df.loc[i, "gross_profit_str"]
        note = df.loc[i, "note"]

        # Model (blue + bold)
        ax.annotate(
            model,
            (x, y),
            textcoords="offset points",
            xytext=(0, -23),
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="#1f77b4",
        )

        # Details
        details_parts = [format_try(price_show)]
        if ("İndirim" in label_mode) and (discount and discount > 0):
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
            xytext=(0, -37),
            ha="center",
            va="top",
            fontsize=8,
            color="#333333",
        )

ax.grid(True, axis="y", linestyle="--", alpha=0.25)

fig.tight_layout()
st.pyplot(fig, use_container_width=True)

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
        "img_code": "Resim",
    })

    st.dataframe(out, use_container_width=True)

