"""Generate a short 2-page PDF summarising M2 v2 and v3 updates."""

from pathlib import Path
from fpdf import FPDF

ROOT    = Path(__file__).resolve().parent.parent
V3_DIR  = ROOT / "reports" / "assets" / "m2_v3_results"
OUT     = ROOT / "reports" / "results" / "m2_update_report.pdf"

DARK_BG    = (15, 28, 44)
PANEL_BG   = (25, 42, 62)
CARD_BG    = (32, 52, 74)
TEAL       = (0, 196, 168)
GOLD       = (255, 196, 50)
WHITE      = (255, 255, 255)
LIGHT_GREY = (200, 210, 220)
MID_GREY   = (140, 155, 170)
GREEN      = (60, 200, 120)
RED        = (220, 80, 70)
BLUE       = (80, 160, 220)


class Report(FPDF):
    def footer(self):
        self.set_y(-11)
        self.set_font("Helvetica", "I", 7.5)
        self.set_text_color(*MID_GREY)
        self.cell(0, 6, "SSGA Meta-Labeling  |  Brandeis MSF 2026  |  CONFIDENTIAL", align="C")

    def dark_page(self):
        self.add_page()
        self.set_fill_color(*DARK_BG)
        self.rect(0, 0, 210, 297, "F")
        self.set_fill_color(*TEAL)
        self.rect(0, 0, 210, 3, "F")
        self.set_left_margin(15)
        self.set_right_margin(15)

    def rule(self, color=None):
        self.set_fill_color(*(color or TEAL))
        self.rect(15, self.get_y(), 180, 0.4, "F")
        self.ln(3)

    def h1(self, t):
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(*WHITE)
        self.cell(0, 9, t, align="C", ln=True)

    def h2(self, t, color=None):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*(color or TEAL))
        self.multi_cell(0, 6, t)

    def body(self, t, size=9.5):
        self.set_font("Helvetica", "", size)
        self.set_text_color(*LIGHT_GREY)
        self.multi_cell(0, 6, t)

    def stat(self, label, val, color=None):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*LIGHT_GREY)
        self.cell(100, 6, label)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*(color or TEAL))
        self.cell(0, 6, val, ln=True)

    def callout(self, title, text, color=None):
        c = color or GOLD
        y0 = self.get_y()
        lines = text.split("\n")
        h = 7 + len(lines) * 5.5 + 3
        self.set_fill_color(*CARD_BG)
        self.set_draw_color(*c)
        self.rect(15, y0, 180, h, "FD")
        self.set_xy(19, y0 + 3)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*c)
        self.cell(0, 5, title, ln=True)
        self.set_x(19)
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*LIGHT_GREY)
        self.multi_cell(170, 5.5, text)
        self.ln(3)


pdf = Report()
pdf.set_auto_page_break(auto=True, margin=13)

# ===========================================================================
# PAGE 1
# ===========================================================================
pdf.dark_page()

# Header strip
pdf.set_fill_color(*TEAL)
pdf.rect(0, 0, 210, 22, "F")
pdf.set_y(4)
pdf.set_font("Helvetica", "B", 14)
pdf.set_text_color(*DARK_BG)
pdf.cell(0, 7, "M2 Model Update", align="C", ln=True)
pdf.set_font("Helvetica", "", 9)
pdf.cell(0, 6, "Feature Selection + Position Sizing  |  March 2026", align="C", ln=True)

pdf.set_y(28)

# --- What changed ---
pdf.h2("What We Changed and Why")
pdf.rule()
pdf.ln(2)

pdf.body(
    "The original M2 model used all 22 features including VIX and OAS regime "
    "indicators, and made a binary yes/no decision on each M1 signal using a "
    "fixed 0.5 probability threshold. Two problems were identified:")
pdf.ln(3)

problems = [
    ("Problem 1: Wrong features",
     RED,
     "EDA showed VIX and OAS have no independent signal (chi-square p=0.89, p=0.77). "
     "They are redundant with M1's own macro indicators. Keeping them added noise."),
    ("Problem 2: Binary gate costs returns",
     RED,
     "When M2 rejected a trade, the portfolio earned 0. But M1 wins 72% of the time, "
     "so most rejected months had positive returns that were simply missed. "
     "This created a negative Information Ratio even when selected trades looked good."),
]
for title, col, body in problems:
    pdf.set_fill_color(*CARD_BG)
    pdf.set_draw_color(*col)
    y0 = pdf.get_y()
    pdf.rect(15, y0, 180, 20, "FD")
    pdf.set_xy(19, y0 + 3)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*col)
    pdf.cell(0, 5, title, ln=True)
    pdf.set_x(19)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*LIGHT_GREY)
    pdf.multi_cell(170, 4.8, body)
    pdf.ln(2)

pdf.ln(2)

# --- Fix 1 ---
pdf.h2("Fix 1 - Feature Selection (M2 v2)")
pdf.rule()
pdf.ln(2)
pdf.body(
    "Removed the 10 VIX and OAS features, keeping only the 12 core M1 context "
    "features: M1 direction, composite score, the 6 macro z-scores, and M1's "
    "trailing track record. This reduced noise without losing meaningful signal.")
pdf.ln(2)
pdf.body(
    "We also swept thresholds from 0.10 to 0.90 in 0.01 steps to find the "
    "optimal cutoff. The binary gate at threshold 0.51 approved 28 of 78 trades "
    "and achieved Sharpe 2.34 on those selected trades -- but the IR remained "
    "negative (-0.607) because 50 months sat in cash earning nothing.")
pdf.ln(3)

# --- Fix 2 ---
pdf.h2("Fix 2 - Position Sizing Instead of Binary Gate (M2 v3)")
pdf.rule()
pdf.ln(2)
pdf.body(
    "Rather than yes/no, M2's probability now directly scales how much of each "
    "trade is taken. The sizes are normalised so the average exposure equals M1:")
pdf.ln(2)

pdf.set_fill_color(*PANEL_BG)
pdf.rect(15, pdf.get_y(), 180, 22, "F")
y0 = pdf.get_y() + 3
pdf.set_xy(19, y0)
pdf.set_font("Courier", "", 8.5)
pdf.set_text_color(*TEAL)
rows = [
    "position_size = m2_prob / mean(m2_prob)",
    "",
    "m2_prob = 0.73  ->  size = 1.16x  (above average confidence, trade more)",
    "m2_prob = 0.55  ->  size = 0.88x  (below average confidence, trade less)",
    "m2_prob = 0.35  ->  size = 0.56x  (low confidence, trade much less)",
]
for row in rows:
    pdf.set_x(19)
    pdf.cell(0, 4.2, row, ln=True)
pdf.set_y(y0 + 22)
pdf.ln(3)

pdf.body(
    "Every trade still participates. High confidence months are amplified (up to "
    "1.72x). Low confidence months are reduced (down to 0.35x). No month earns "
    "zero. The opportunity cost of the binary gate is eliminated entirely.")


# ===========================================================================
# PAGE 2
# ===========================================================================
pdf.dark_page()

pdf.set_y(7)
pdf.h1("Results")
pdf.rule()
pdf.ln(3)

# Results table
pdf.set_font("Helvetica", "B", 8.5)
pdf.set_text_color(*MID_GREY)
pdf.cell(70, 6, "Setup")
pdf.cell(18, 6, "Trades", align="C")
pdf.cell(32, 6, "Ann Return", align="C")
pdf.cell(28, 6, "Sharpe", align="C")
pdf.cell(0, 6, "Info Ratio", align="C", ln=True)
pdf.set_fill_color(*PANEL_BG)
pdf.rect(15, pdf.get_y(), 180, 0.4, "F")
pdf.ln(1)

# Header row
pdf.set_font("Helvetica", "B", 8)
pdf.set_text_color(*MID_GREY)
pdf.cell(72, 5.5, "Setup")
pdf.cell(14, 5.5, "N", align="C")
pdf.cell(28, 5.5, "Ann Return", align="C")
pdf.cell(24, 5.5, "Sharpe", align="C")
pdf.cell(0, 5.5, "Info Ratio", align="C", ln=True)
pdf.set_fill_color(*PANEL_BG)
pdf.rect(15, pdf.get_y(), 180, 0.3, "F")
pdf.ln(1)

def table_row(pdf, name, n, ret, sh, ir, col, bold=False):
    pdf.set_font("Helvetica", "B" if bold else "", 8.5)
    pdf.set_text_color(*col)
    pdf.cell(72, 5.8, name)
    pdf.cell(14, 5.8, n, align="C")
    pdf.cell(28, 5.8, ret, align="C")
    pdf.cell(24, 5.8, sh, align="C")
    pdf.cell(0, 5.8, ir, align="C", ln=True)

table_row(pdf, "M1 baseline (no filter)",      "78", "9.53%",  "1.310", "+0.000", WHITE, bold=True)
pdf.ln(1)

# v1 section
pdf.set_font("Helvetica", "B", 7.5)
pdf.set_text_color(*BLUE)
pdf.cell(0, 5, "M2 v1 - Binary Gate (22 features incl. VIX/OAS)", ln=True)
rows_v1 = [
    ("  t=0.30",  "69", "9.10%",  "1.283", "-0.236"),
    ("  t=0.40",  "56", "6.98%",  "1.056", "-0.737"),
    ("  t=0.50",  "46", "5.42%",  "0.891", "-0.937"),
    ("  t=0.60",  "38", "6.19%",  "1.146", "-0.634"),
    ("  t=0.70",  "25", "5.22%",  "1.161", "-0.705"),
    ("  t=0.80",  "10", "1.89%",  "0.924", "-1.062"),
]
for name, n, ret, sh, ir in rows_v1:
    table_row(pdf, name, n, ret, sh, ir, LIGHT_GREY)
pdf.ln(1)

# v2 section
pdf.set_font("Helvetica", "B", 7.5)
pdf.set_text_color(*GOLD)
pdf.cell(0, 5, "M2 v2 - Binary Gate (12 core features, no VIX/OAS)", ln=True)
rows_v2 = [
    ("  t=0.30",  "73", "9.34%",  "1.371", "-0.072"),
    ("  t=0.40",  "58", "6.13%",  "1.003", "-0.781"),
    ("  t=0.50",  "31", "5.14%",  "0.990", "-0.801"),
    ("  t=0.51",  "28", "6.01%",  "1.233", "-0.607"),
    ("  t=0.60",  "16", "4.84%",  "1.196", "-0.729"),
    ("  t=0.70",  " 5", "1.89%",  "0.660", "-1.109"),
    ("  t=0.80",  " 2", "1.05%",  "0.453", "-1.210"),
]
for name, n, ret, sh, ir in rows_v2:
    table_row(pdf, name, n, ret, sh, ir, LIGHT_GREY)
pdf.ln(1)

# v3
pdf.set_font("Helvetica", "B", 7.5)
pdf.set_text_color(*GREEN)
pdf.cell(0, 5, "M2 v3 - Position Sizing (12 core features)", ln=True)
table_row(pdf, "  Normalised sizing (avg=1.0x)", "78", "10.84%", "1.345", "+0.457", GREEN, bold=True)

pdf.ln(3)

# Charts
pdf.img_path = str(V3_DIR / "3_summary_bars.png")
if Path(pdf.img_path).exists():
    pdf.image(pdf.img_path, x=15, y=pdf.get_y(), w=180, h=62)
    pdf.set_y(pdf.get_y() + 63)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(*MID_GREY)
    pdf.cell(0, 5, "Figure 1: Ann Return, Sharpe, and IR across all four setups. Green = M2 v3 (normalised sizing).", ln=True)

pdf.ln(2)

cum_path = str(V3_DIR / "1_cumulative_return_sizing.png")
if Path(cum_path).exists():
    pdf.image(cum_path, x=15, y=pdf.get_y(), w=180, h=60)
    pdf.set_y(pdf.get_y() + 61)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(*MID_GREY)
    pdf.cell(0, 5, "Figure 2: Cumulative return 2009-2025. M2 normalised sizing (green) consistently leads M1 baseline (blue).", ln=True)

pdf.ln(3)

# Callout
pdf.set_fill_color(*CARD_BG)
pdf.set_draw_color(*GREEN)
y0 = pdf.get_y()
pdf.rect(15, y0, 180, 28, "FD")
pdf.set_xy(19, y0 + 3)
pdf.set_font("Helvetica", "B", 9.5)
pdf.set_text_color(*GREEN)
pdf.cell(0, 6, "Key Takeaway", ln=True)
pdf.set_x(19)
pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(*LIGHT_GREY)
pdf.multi_cell(170, 5.5,
    "Position sizing is the first M2 configuration to beat M1 on all three metrics "
    "simultaneously: higher annual return (+1.3%), higher Sharpe (1.345 vs 1.310), "
    "and a positive Information Ratio (+0.457 vs 0.000). The next step is adding "
    "features that are orthogonal to M1 to give M2 genuinely new information.")

OUT.parent.mkdir(parents=True, exist_ok=True)
pdf.output(str(OUT))
print(f"PDF saved: {OUT}")
