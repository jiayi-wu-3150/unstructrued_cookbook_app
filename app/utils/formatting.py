import io
import json
from PIL import Image, ImageDraw
import streamlit as st


AUDIO_TYPE_COLORS = {
    "speech":       "blue",
    "music":        "violet",
    "sound_effect": "orange",
    "ambient":      "green",
    "mixed":        "red",
    "unknown":      "gray",
}


def confidence_color(score: float) -> str:
    if score is None:
        return "gray"
    if score >= 0.9:
        return "green"
    if score >= 0.8:
        return "orange"
    return "red"


def recipe_header(icon: str, title: str, subtitle: str, badge_label: str, badge_value: str):
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title(f"{icon} {title}")
        st.caption(subtitle)
    with col2:
        st.metric(badge_label, badge_value)


def code_tab_content(snippet: str, language: str = "sql"):
    st.code(snippet, language=language)


def requirements_tab_content(permissions: list[str], resources: list[str], dependencies: list[str]):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Permissions**")
        for p in permissions:
            st.markdown(f"- {p}")
    with c2:
        st.markdown("**Databricks Resources**")
        for r in resources:
            st.markdown(f"- {r}")
    with c3:
        st.markdown("**Dependencies**")
        for d in dependencies:
            st.markdown(f"- {d}")


def show_endpoint_health(endpoint_names: list[str]):
    """Display live health status for a list of serving endpoints."""
    from utils.db import get_endpoint_statuses
    with st.spinner("Checking endpoint health..."):
        statuses = get_endpoint_statuses(tuple(endpoint_names))
    for name, (ready, state) in statuses.items():
        icon = "🟢" if ready else "🔴"
        st.markdown(f"{icon} `{name}` — **{state}**")


# Colors for bounding boxes by element type
BBOX_COLORS = {
    "title": (255, 87, 34),       # deep orange
    "text": (33, 150, 243),       # blue
    "table": (76, 175, 80),       # green
    "figure": (156, 39, 176),     # purple
    "page_header": (255, 193, 7), # amber
    "page_footer": (158, 158, 158),  # gray
    "list": (0, 188, 212),        # cyan
    "caption": (233, 30, 99),     # pink
}
BBOX_DEFAULT_COLOR = (100, 100, 100)


def draw_bboxes_on_image(img_bytes: bytes, bboxes: list[dict], alpha: int = 60) -> bytes:
    """Draw bounding boxes on an image.

    Args:
        img_bytes: Raw image bytes (JPEG/PNG).
        bboxes: List of dicts with keys: coord [x1,y1,x2,y2], element_type (optional), label (optional).
        alpha: Fill transparency (0-255).

    Returns:
        Annotated image as PNG bytes.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for box in bboxes:
        coord = box.get("coord")
        if not coord or len(coord) != 4:
            continue
        x1, y1, x2, y2 = coord
        etype = box.get("element_type", "")
        color = BBOX_COLORS.get(etype, BBOX_DEFAULT_COLOR)
        # Semi-transparent fill
        draw.rectangle([x1, y1, x2, y2], fill=(*color, alpha), outline=(*color, 200), width=2)
        # Label
        label = box.get("label") or etype
        if label:
            draw.text((x1 + 4, y1 + 2), label, fill=(*color, 255))

    result = Image.alpha_composite(img, overlay).convert("RGB")
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


def render_pdf_page(pdf_bytes: bytes, page_num: int = 0, dpi: int = 200) -> bytes:
    """Render a PDF page to PNG bytes using PyMuPDF."""
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes(output="png")
