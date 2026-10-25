"""Build the CEP presentation deck from the saved results and figures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FIGURES, MODELS, REPORTS

OUT_DIR = REPORTS / "presentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAVY = RGBColor(0x16, 0x2B, 0x3C)
INK = RGBColor(0x1F, 0x2A, 0x33)
MUTED = RGBColor(0x5B, 0x65, 0x73)
BLUE = RGBColor(0x2E, 0x86, 0xAB)
RED = RGBColor(0xD1, 0x49, 0x5B)
GREEN = RGBColor(0x6A, 0x99, 0x4E)
AMBER = RGBColor(0xC7, 0x6D, 0x2E)
PALE = RGBColor(0xF4, 0xF7, 0xFA)
LINE = RGBColor(0xD9, 0xE2, 0xEA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

W, H = 13.333, 7.5


def new_slide(prs: Presentation, bg=PALE):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = bg
    return slide


def textbox(slide, left, top, width, height, text, *, size=16, color=INK, bold=False,
            align=PP_ALIGN.LEFT, font="Arial", anchor=MSO_ANCHOR.TOP, spacing=1.0):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = spacing
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font
    return box


def bullets(slide, left, top, width, height, items, *, size=15, color=INK, gap=8, spacing=1.0):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    first = True
    for item in items:
        if isinstance(item, tuple):
            title, body = item
        else:
            title, body = None, item
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap)
        p.line_spacing = spacing
        if title:
            r = p.add_run()
            r.text = "▪  "
            r.font.size = Pt(size)
            r.font.color.rgb = BLUE
            r.font.name = "Arial"
            r2 = p.add_run()
            r2.text = title
            r2.font.size = Pt(size)
            r2.font.bold = True
            r2.font.color.rgb = color
            r2.font.name = "Arial"
            r3 = p.add_run()
            r3.text = "  " + body if body else ""
            r3.font.size = Pt(size)
            r3.font.color.rgb = color
            r3.font.name = "Arial"
        else:
            r = p.add_run()
            r.text = "▪  " + body
            r.font.size = Pt(size)
            r.font.color.rgb = color
            r.font.name = "Arial"
    return box


def header(slide, kicker, title, subtitle=None):
    textbox(slide, 0.62, 0.36, 11.8, 0.28, kicker.upper(), size=9.5, color=BLUE, bold=True)
    textbox(slide, 0.62, 0.68, 12.0, 0.62, title, size=30, color=NAVY, bold=True, spacing=0.95)
    accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.64), Inches(1.39), Inches(1.05), Inches(0.055))
    accent.fill.solid(); accent.fill.fore_color.rgb = RED; accent.line.fill.background()
    if subtitle:
        textbox(slide, 0.62, 1.52, 12.0, 0.4, subtitle, size=13, color=MUTED)


def footer(slide, number):
    textbox(slide, 0.62, 7.12, 9.6, 0.22,
            "Piecuch & Żabiński, Scientific Data 12, 650 (2025) · CC BY 4.0 · ME-446 CEP",
            size=8, color=MUTED)
    textbox(slide, 11.9, 7.12, 0.8, 0.22, f"{number:02d}", size=9, color=MUTED, align=PP_ALIGN.RIGHT)


def picture_fit(slide, path, left, top, width, height, *, border=False):
    with Image.open(path) as img:
        iw, ih = img.size
    scale = min(width / iw, height / ih)
    w, h = iw * scale, ih * scale
    x = left + (width - w) / 2
    y = top + (height - h) / 2
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    if border:
        pic.line.color.rgb = LINE
        pic.line.width = Pt(0.75)
    return pic


def card(slide, left, top, width, height, *, fill=WHITE, line=LINE, radius=0.06):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.adjustments[0] = radius
    shape.fill.solid(); shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line; shape.line.width = Pt(0.75)
    shape.shadow.inherit = False
    return shape


def kpi(slide, left, top, width, height, value, label, *, color=NAVY, note=None):
    card(slide, left, top, width, height)
    textbox(slide, left + 0.18, top + 0.16, width - 0.3, 0.46, value, size=24, color=color, bold=True)
    textbox(slide, left + 0.18, top + 0.62, width - 0.3, 0.44, label, size=9.5, color=MUTED)
    if note:
        textbox(slide, left + 0.18, top + height - 0.36, width - 0.3, 0.24, note, size=8.5, color=color)


def table(slide, left, top, width, height, data, *, col_widths=None, font_size=11,
          header_fill=NAVY, highlight_row=None):
    rows, cols = len(data), len(data[0])
    shape = slide.shapes.add_table(rows, cols, Inches(left), Inches(top), Inches(width), Inches(height))
    tbl = shape.table
    if col_widths:
        total = sum(col_widths)
        for i, cw in enumerate(col_widths):
            tbl.columns[i].width = Emu(int(Inches(width) * cw / total))
    for r in range(rows):
        for c in range(cols):
            cell = tbl.cell(r, c)
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.035)
            cell.margin_bottom = Inches(0.035)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = header_fill
            elif highlight_row is not None and r == highlight_row:
                cell.fill.fore_color.rgb = RGBColor(0xE8, 0xF3, 0xE4)
            else:
                cell.fill.fore_color.rgb = WHITE if r % 2 else RGBColor(0xF7, 0xF9, 0xFB)
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            run = p.add_run()
            run.text = str(data[r][c])
            run.font.size = Pt(font_size)
            run.font.name = "Arial"
            run.font.bold = r == 0 or (highlight_row is not None and r == highlight_row)
            run.font.color.rgb = WHITE if r == 0 else INK
    return shape


def main() -> None:
    metrics = json.loads((MODELS / "model_metrics.json").read_text(encoding="utf-8"))
    sel = metrics["selected_model_metrics"]
    loo = metrics["selected_model_tool_loo_metrics"]
    comparison = pd.read_csv(MODELS / "model_comparison_pooled.csv")
    primary = comparison[comparison["scheme"] == "tool_5fold"].set_index("model")
    unseen = pd.read_csv(MODELS / "unseen_experiment_summary.csv")

    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)

    # 1 — title -----------------------------------------------------------------
    s = new_slide(prs, NAVY)
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.18), Inches(H))
    band.fill.solid(); band.fill.fore_color.rgb = RED; band.line.fill.background()
    textbox(s, 0.85, 0.75, 11.9, 0.3, "ME-446  ·  AI & INTERNET OF THINGS  ·  COMPLEX ENGINEERING PROBLEM", size=11, color=RGBColor(0x9F, 0xC6, 0xDE), bold=True)
    textbox(s, 0.85, 1.5, 11.2, 1.8, "Detecting Dangerous\nMilling Conditions", size=44, color=WHITE, bold=True, spacing=0.92)
    textbox(s, 0.85, 3.35, 10.8, 0.8, "from vibration and spindle/drive-current statistics", size=21, color=RGBColor(0xD6, 0xE4, 0xEE))
    line = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.87), Inches(4.35), Inches(0.9), Inches(0.06))
    line.fill.solid(); line.fill.fore_color.rgb = BLUE; line.line.fill.background()
    textbox(s, 0.85, 4.62, 10.0, 0.4, "Random forest  ·  96.2 % recall  ·  MCC 0.762  ·  grouped validation", size=15, color=WHITE, bold=True)
    textbox(s, 0.85, 5.95, 6.5, 0.7, "Group ______  ·  NED University of Engineering & Technology", size=11, color=RGBColor(0xB7, 0xCB, 0xD9))
    textbox(s, 7.4, 5.95, 5.1, 0.7, "Data: Piecuch & Żabiński, Scientific Data 12, 650 (2025) · CC BY 4.0", size=8.5, color=RGBColor(0x9F, 0xB5, 0xC4), align=PP_ALIGN.RIGHT)

    # 2 — problem ---------------------------------------------------------------
    s = new_slide(prs)
    header(s, "01 · Engineering problem", "A safety monitor for an aggressive cut",
           "The machine does not know which radial depth of cut it is running. The sensor signature does.")
    cols = [
        ("The condition", "RDOC 8 mm is 80 % of the cutter diameter. Chip load, cutting force and spindle torque rise; chatter and overload become likely.", BLUE),
        ("The risk", "A missed dangerous cut means faster wear, scrap or a broken tool. A false alarm costs an inspection, not a spindle.", RED),
        ("The task", "Binary classification: optimal (4.5 mm) vs non-optimal (8 mm). 120 vibration/current statistics per cycle; no process settings as inputs.", GREEN),
    ]
    for i, (title, body, color) in enumerate(cols):
        left = 0.62 + i * 4.12
        card(s, left, 1.95, 3.85, 2.5)
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(1.95), Inches(0.07), Inches(2.5))
        bar.fill.solid(); bar.fill.fore_color.rgb = color; bar.line.fill.background()
        textbox(s, left + 0.28, 2.18, 3.3, 0.35, title, size=17, color=NAVY, bold=True)
        textbox(s, left + 0.28, 2.72, 3.3, 1.55, body, size=12.5, color=INK, spacing=1.12)
    callout = card(s, 0.62, 4.78, 12.1, 1.35, fill=WHITE)
    textbox(s, 0.92, 4.98, 11.55, 0.95,
            "Why it is a genuine CEP problem: RDOC is constant within every tool and experiment. "
            "A random row split lets the model memorise tool identity, so the validation design — not just the algorithm — decides whether the result is believable.",
            size=13, color=INK, spacing=1.12)
    footer(s, 2)

    # 3 — dataset ---------------------------------------------------------------
    s = new_slide(prs)
    header(s, "02 · Dataset", "968 cycles from a public milling campaign",
           "Piecuch & Żabiński (2025), Scientific Data — processed 120-statistic feature matrix, CC BY 4.0.")
    stats = [
        ("968", "milling cycles", NAVY), ("24", "experiments", NAVY), ("14", "cutting tools", NAVY),
        ("120", "sensor statistics", BLUE), ("212", "dangerous cycles · 21.9 %", RED), ("2", "RDOC levels: 4.5 / 8 mm", GREEN),
    ]
    for i, (value, label, color) in enumerate(stats):
        left = 0.62 + (i % 3) * 4.12
        top = 1.95 + (i // 3) * 1.28
        kpi(s, left, top, 3.85, 1.08, value, label, color=color)
    bullets(s, 0.72, 4.85, 7.0, 1.7, [
        ("Two independent sources", "feature CSV and metadata workbook cross-checked row for row."),
        ("Zero missing values", "no duplicate rows, no duplicate sensor vectors, no zero-variance features."),
        ("Difficult records kept", "14 censored cycles and one single-cycle failure retained, not deleted."),
    ], size=10.5, gap=5)
    card(s, 8.05, 4.78, 4.65, 1.72, fill=WHITE)
    textbox(s, 8.3, 4.96, 4.15, 0.3, "Features the model sees", size=10, color=BLUE, bold=True)
    textbox(s, 8.3, 5.3, 4.15, 1.05, "8 accelerometer channels + 12 current channels, each with min, max, mean, std, skew and kurtosis. Metadata, hardness, wear counters and IDs are excluded.",
            size=9.5, color=INK, spacing=1.1)
    footer(s, 3)

    # 4 — confounding -----------------------------------------------------------
    s = new_slide(prs)
    header(s, "03 · The trap", "RDOC is perfectly confounded with tool identity",
           "0 experiments and 0 tools contain both classes — so the split strategy is the result.")
    bullets(s, 0.68, 1.95, 5.55, 4.3, [
        ("One condition per tool", "Every tool runs at exactly one RDOC/ADOC combination."),
        ("One condition per experiment", "So do all 24 experiments; the class is a property of the setup."),
        ("Tool type 2 never appears at 8 mm", "Material and tool type are entangled with the target."),
        ("Random split leaks", "The same tool appears in train and test; scores become memorisation."),
    ], size=12.2, gap=12)
    picture_fit(s, FIGURES / "fig02_tool_condition_confounding.png", 6.5, 1.95, 6.25, 4.45, border=True)
    textbox(s, 0.68, 6.25, 5.6, 0.55, "Tool 11 has one cycle; tool type 2 is only ever optimal.", size=9, color=MUTED)
    footer(s, 4)

    # 5 — method ----------------------------------------------------------------
    s = new_slide(prs)
    header(s, "04 · Method", "Cleaning → grouped validation → four models",
           "Every model sees the same 120 features and the same folds. Nothing is tuned on the held-out tool.")
    steps = [
        ("1", "Audit & clean", "Parse export quirks; cross-check sources; keep all 968 cycles."),
        ("2", "Features", "signed-log1p on 120 sensor statistics; balanced class weights."),
        ("3", "Grouped CV", "5 tool folds; tuning nested inside each outer fold."),
        ("4", "Compare", "Dummy · logistic · random forest · histogram gradient boosting."),
        ("5", "Select", "Tune threshold on out-of-fold probabilities; stress-test by tool."),
    ]
    for i, (num, title, body) in enumerate(steps):
        left = 0.6 + i * 2.52
        card(s, left, 2.0, 2.32, 2.35, fill=WHITE)
        circle = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(left + 0.18), Inches(2.2), Inches(0.42), Inches(0.42))
        circle.fill.solid(); circle.fill.fore_color.rgb = BLUE; circle.line.fill.background()
        textbox(s, left + 0.18, 2.24, 0.42, 0.2, num, size=12, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
        textbox(s, left + 0.18, 2.72, 1.95, 0.35, title, size=12, color=NAVY, bold=True)
        textbox(s, left + 0.18, 3.12, 1.98, 1.0, body, size=9.2, color=INK, spacing=1.08)
    card(s, 0.62, 4.72, 5.9, 1.6, fill=WHITE)
    textbox(s, 0.88, 4.9, 5.4, 0.3, "Primary estimate", size=10, color=GREEN, bold=True)
    textbox(s, 0.88, 5.22, 5.4, 0.9, "5-fold tool-group CV, pooled out-of-fold predictions. Leave-one-tool-out and leave-one-experiment-out are sensitivity tests.",
            size=10.5, color=INK, spacing=1.1)
    card(s, 6.75, 4.72, 5.95, 1.6, fill=RGBColor(0xFD, 0xEE, 0xEF), line=RGBColor(0xF3, 0xCF, 0xD3))
    textbox(s, 7.0, 4.9, 5.45, 0.3, "Leakage control: never the headline", size=10, color=RED, bold=True)
    textbox(s, 7.0, 5.22, 5.45, 0.9, "Random 5-fold: RF ROC-AUC 0.998. Tool-grouped: 0.942. The gap is the leakage. All reported numbers are grouped.",
            size=10.5, color=INK, spacing=1.1)
    footer(s, 5)

    # 6 — model comparison ------------------------------------------------------
    s = new_slide(prs)
    header(s, "05 · Model comparison", "Four models, one honest comparison",
           "5-fold tool-group CV, threshold 0.5. ROC-AUC and PR-AUC are threshold-independent; the rest are operating-point metrics.")
    rows = [["Model", "ROC-AUC", "PR-AUC", "Recall", "Precision", "F1", "MCC", "Brier"]]
    labels = {"Dummy": "Dummy (prior)", "LogisticRegression": "Logistic regression",
              "RandomForest": "Random forest", "HistGradientBoosting": "Hist gradient boosting"}
    for name in ["Dummy", "LogisticRegression", "RandomForest", "HistGradientBoosting"]:
        r = primary.loc[name]
        rows.append([labels[name], f"{r.roc_auc:.3f}", f"{r.pr_auc:.3f}", f"{r.recall:.3f}",
                     f"{r.precision:.3f}", f"{r.f1:.3f}", f"{r.mcc:.3f}", f"{r.brier:.3f}"])
    table(s, 0.62, 1.95, 7.6, 2.75, rows, col_widths=[2.1, 1, 1, 1, 1.1, 0.8, 0.8, 0.8], font_size=9.5, highlight_row=3)
    textbox(s, 0.65, 4.83, 7.5, 1.4,
            "Random forest wins the operating point (F1 0.799, MCC 0.742, Brier 0.082). "
            "Gradient boosting wins threshold-independent PR-AUC (0.782) but is weaker at 0.5 — reported honestly, not hidden.",
            size=11, color=INK, spacing=1.12)
    picture_fit(s, FIGURES / "fig07_model_comparison.png", 8.35, 1.95, 4.4, 4.3, border=True)
    footer(s, 6)

    # 7 — selected model --------------------------------------------------------
    s = new_slide(prs)
    header(s, "06 · Selected model", "At the tuned operating point, missing a cut is rare",
           "Threshold 0.444 maximises F1 on pooled out-of-fold probabilities; it sits below 0.5 on purpose.")
    items = [
        ("96.2 %", "recall (dangerous)", RED), ("69.9 %", "precision", BLUE),
        ("0.762", "MCC", GREEN), ("0.923", "balanced accuracy", BLUE),
        ("0.810", "F1", GREEN), ("0.082", "Brier score", NAVY),
    ]
    for i, (value, label, color) in enumerate(items):
        left = 0.62 + (i % 3) * 2.1
        top = 1.95 + (i // 3) * 1.18
        kpi(s, left, top, 1.95, 1.0, value, label, color=color)
    textbox(s, 0.65, 4.5, 6.1, 1.7,
            "8 missed dangerous cycles out of 212 (3.8 %), 88 false alarms out of 756 (11.6 %). "
            "Leave-one-tool-out agrees: recall 0.962, MCC 0.778. The model is not relying on one grouping of tools.",
            size=10.8, color=INK, spacing=1.12)
    card(s, 0.62, 5.82, 5.95, 0.78, fill=WHITE)
    textbox(s, 0.85, 5.98, 5.5, 0.5, "Threshold is a policy: 0.444 favours recall; the dashboard exposes the trade-off.",
            size=9.5, color=MUTED)
    picture_fit(s, FIGURES / "fig09_confusion_calibration.png", 6.95, 1.95, 5.75, 4.65, border=True)
    footer(s, 7)

    # 8 — error anatomy ---------------------------------------------------------
    s = new_slide(prs)
    header(s, "07 · Error analysis", "One optimal tool drives most false alarms",
           "Tool 7 accounts for 75 of the 88 false alarms; the 8 misses are spread across tools 4–6.")
    rows = [
        ["Tool", "Actual", "Cycles", "Mean P", "Flagged"],
        ["4", "Dangerous", "41", "0.712", "92.7 %"],
        ["5", "Dangerous", "51", "0.623", "94.1 %"],
        ["6", "Dangerous", "28", "0.676", "92.9 %"],
        ["7", "Optimal", "124", "0.563", "60.5 %"],
        ["8", "Optimal", "116", "0.149", "5.2 %"],
        ["105", "Optimal", "125", "0.104", "5.6 %"],
    ]
    table(s, 0.65, 1.98, 4.7, 2.85, rows, col_widths=[0.7, 1.2, 0.9, 0.9, 1.0], font_size=10, highlight_row=4)
    textbox(s, 0.68, 5.05, 4.6, 1.3,
            "Tool 7 false alarms cluster late in tool life (mean normalised life 0.33). Wear pushes an optimal tool into a dangerous-looking signature — the central confound.",
            size=10.2, color=INK, spacing=1.1)
    picture_fit(s, FIGURES / "fig11_per_tool_probabilities.png", 5.65, 1.95, 7.05, 4.75, border=True)
    footer(s, 8)

    # 9 — feature importance ----------------------------------------------------
    s = new_slide(prs)
    header(s, "08 · Interpretation", "It reads force, not the answer key",
           "Permutation importance on the fitted forest. Correlated channels share importance; the ranking is the message.")
    picture_fit(s, FIGURES / "fig10_feature_importance.png", 0.55, 1.95, 8.0, 4.8, border=True)
    bullets(s, 8.85, 2.0, 3.85, 4.6, [
        ("Drive-axis mean acceleration", "the top signal — a static force offset grows with radial engagement."),
        ("Current variability", "drive and spindle current std follow the heavier, more interrupted cut."),
        ("No wear counter or ID", "metadata is excluded, so the model cannot read the target."),
        ("Physically plausible", "the ranking matches cutting mechanics, not a dataset artefact."),
    ], size=10.3, gap=10)
    footer(s, 9)

    # 10 — unseen ---------------------------------------------------------------
    s = new_slide(prs)
    header(s, "09 · Unseen cases", "Held-out experiments behave as they should",
           "A fresh forest is trained without P002 and P108, then applied to all 127 of their cycles.")
    p002 = unseen[unseen.ExperimentID == "P002"].iloc[0]
    p108 = unseen[unseen.ExperimentID == "P108"].iloc[0]
    rows = [
        ["Experiment", "Actual", "Cycles", "Mean P", "Flagged"],
        ["P002", "Dangerous", str(int(p002.n_cycles)), f"{p002.prob_mean:.3f}", f"{p002.frac_predicted_dangerous:.1%}"],
        ["P108", "Optimal", str(int(p108.n_cycles)), f"{p108.prob_mean:.3f}", f"{p108.frac_predicted_dangerous:.1%}"],
    ]
    table(s, 0.65, 2.0, 6.0, 1.75, rows, col_widths=[1.4, 1.4, 0.9, 0.9, 1.0], font_size=11, highlight_row=1)
    card(s, 0.62, 4.05, 5.95, 1.9, fill=WHITE)
    textbox(s, 0.88, 4.22, 5.5, 0.3, "The one miss is a real engineering finding", size=10, color=RED, bold=True)
    textbox(s, 0.88, 4.55, 5.45, 1.25, "P002's first cycle scores 0.439 — just under the 0.444 threshold. A fresh tool has not yet developed the wear-driven rise in vibration and current. Use a rolling window, not a verdict on one cycle.",
            size=9.8, color=INK, spacing=1.1)
    card(s, 6.95, 2.0, 5.75, 3.95, fill=WHITE)
    textbox(s, 7.2, 2.18, 5.25, 0.3, "Single-cycle scoring with the final model", size=10, color=BLUE, bold=True)
    rows2 = [
        ["Cycle", "Actual", "P(dangerous)", "Decision"],
        ["P002_F12_C4", "Dangerous", "0.967", "Flag"],
        ["P108_F12_C7", "Optimal", "0.000", "Clear"],
    ]
    table(s, 7.2, 2.6, 5.25, 1.35, rows2, col_widths=[1.7, 1.3, 1.1, 0.9], font_size=9.5, highlight_row=2)
    textbox(s, 7.2, 4.22, 5.25, 1.45, "Same 120-feature interface a deployment would use: one row of sensor statistics, no metadata. P108 is cleared on every cycle; P002 is flagged on 45 of 46.",
            size=9.8, color=INK, spacing=1.1)
    footer(s, 10)

    # 11 — interpretation -------------------------------------------------------
    s = new_slide(prs)
    header(s, "10 · Engineering interpretation", "A data-driven cutting-force monitor",
           "The model's signal and its errors both make physical sense.")
    cols = [
        ("Signal", "Larger RDOC → higher chip load → higher drive-axis mean acceleration and more variable spindle/drive current.", BLUE),
        ("Model", "The forest combines correlated channels and shifts its decision boundary at 0.444 to favour recall.", GREEN),
        ("Error", "Wear, hardness and tool geometry move the signature in the same direction as RDOC, so tool 7 overlaps.", RED),
    ]
    for i, (title, body, color) in enumerate(cols):
        left = 0.62 + i * 4.12
        card(s, left, 1.98, 3.85, 2.15)
        textbox(s, left + 0.22, 2.16, 3.4, 0.3, title.upper(), size=10, color=color, bold=True)
        textbox(s, left + 0.22, 2.55, 3.4, 1.4, body, size=11.8, color=INK, spacing=1.1)
    card(s, 0.62, 4.38, 12.1, 1.7, fill=NAVY)
    textbox(s, 0.92, 4.58, 11.5, 0.3, "CONSEQUENCE OF AN ERROR", size=9.5, color=RGBColor(0x9F, 0xC6, 0xDE), bold=True)
    textbox(s, 0.92, 4.92, 11.45, 1.0, "A false negative continues a dangerous cut — faster wear, chatter, scrap or a broken tool. A false positive costs an inspection. The threshold and the human-in-the-loop workflow are chosen around that asymmetry.",
            size=12.5, color=WHITE, spacing=1.12)
    footer(s, 11)

    # 12 — limitations and recommendations --------------------------------------
    s = new_slide(prs)
    header(s, "11 · Caveats & next steps", "Strong screening alarm, not a causal instrument",
           "The confound is a property of the experiment, not something a better model can remove.")
    card(s, 0.62, 1.95, 5.9, 4.55, fill=WHITE)
    textbox(s, 0.88, 2.12, 5.4, 0.3, "LIMITATIONS", size=10, color=RED, bold=True)
    bullets(s, 0.9, 2.5, 5.35, 3.85, [
        ("Confounded design", "RDOC constant within each tool and experiment."),
        ("Wear confound", "Dangerous tools fail sooner; signal drifts with life."),
        ("Small tool count", "14 tools, 5 dangerous; tool 7 dominates errors."),
        ("Pre-computed features", "No raw waveforms; no re-optimised windows."),
        ("Uncalibrated units", "ADC counts, single machine, one campaign."),
    ], size=9.6, gap=7)
    card(s, 6.8, 1.95, 5.9, 4.55, fill=WHITE)
    textbox(s, 7.05, 2.12, 5.4, 0.3, "RECOMMENDATIONS", size=10, color=GREEN, bold=True)
    bullets(s, 7.07, 2.5, 5.35, 3.85, [
        ("Two-cycle confirmation", "Do not retract on one marginal cycle."),
        ("Tool-specific baseline", "Compare with the tool's own early-life signature."),
        ("Balanced experiment", "Cut the same tool at both RDOC levels."),
        ("Calibrate channels", "Convert ADC to g and A before transfer."),
        ("Tune to cost ratio", "Inspection cost vs missed-cut risk."),
    ], size=9.6, gap=7)
    footer(s, 12)

    # 13 — close ----------------------------------------------------------------
    s = new_slide(prs, NAVY)
    textbox(s, 0.85, 0.95, 11.7, 0.3, "ME-446 · COMPLEX ENGINEERING PROBLEM", size=10.5, color=RGBColor(0x9F, 0xC6, 0xDE), bold=True)
    textbox(s, 0.85, 1.5, 11.6, 0.9, "Delivered and reproducible", size=38, color=WHITE, bold=True)
    artifacts = [
        ("11 figures", "EDA, model comparison, calibration, per-tool evidence"),
        ("Executed notebook", "cleaning → models → threshold → unseen cases"),
        ("Streamlit dashboard", "6 tabs, threshold slider, live scoring, CSV upload"),
        ("Tableau extracts", "8 tables + calculated-field build guide"),
        ("Technical report", "17 pages, 5 400 words, 11 figures"),
    ]
    for i, (title, body) in enumerate(artifacts):
        top = 2.75 + i * 0.72
        textbox(s, 0.88, top, 2.5, 0.3, title, size=12, color=RGBColor(0x9F, 0xC6, 0xDE), bold=True)
        textbox(s, 3.45, top, 8.8, 0.3, body, size=11, color=WHITE)
    textbox(s, 0.85, 6.28, 11.7, 0.35, "Live dashboard:  milling-cep-iguwxgwerngzavkpz2aguj.streamlit.app", size=10.5, color=RGBColor(0x9F, 0xC6, 0xDE), bold=True)
    textbox(s, 0.85, 6.65, 11.7, 0.35, "github.com/Shahbaz2104/milling-cep  ·  Questions", size=10.5, color=RGBColor(0xB7, 0xCB, 0xD9))

    out = OUT_DIR / "MILLING_CEP_SLIDES.pptx"
    prs.save(out)
    print(f"Wrote {out} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
