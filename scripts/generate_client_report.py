"""Generate a comprehensive client-facing PDF for the SSGA Meta-Labeling project."""

from pathlib import Path
from fpdf import FPDF

ROOT     = Path(__file__).resolve().parent.parent
EDA_DIR  = ROOT / "reports" / "assets" / "secondary_eda"
M2_DIR   = ROOT / "reports" / "assets" / "m2_results"
ASSET_DIR = ROOT / "reports" / "assets"
OUT_PATH = ROOT / "reports" / "results" / "ssga_metalabeling_client_report.pdf"

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
DARK_BG     = (15, 28, 44)
PANEL_BG    = (25, 42, 62)
CARD_BG     = (32, 52, 74)
TEAL        = (0, 196, 168)
GOLD        = (255, 196, 50)
RED_SOFT    = (220, 80, 70)
WHITE       = (255, 255, 255)
LIGHT_GREY  = (200, 210, 220)
MID_GREY    = (140, 155, 170)
SOFT_BLUE   = (80, 160, 220)
GREEN_SOFT  = (60, 200, 120)
TEAL_DARK   = (0, 140, 120)


# ---------------------------------------------------------------------------
# PDF class
# ---------------------------------------------------------------------------
class ClientReport(FPDF):

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7.5)
        self.set_text_color(*MID_GREY)
        self.cell(0, 8,
            "SSGA Meta-Labeling Project  |  Brandeis MSF 2026  |  "
            "Raviv - Samweg - Cristian  |  CONFIDENTIAL",
            align="C")

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------
    def dark_page(self):
        self.add_page()
        self.set_fill_color(*DARK_BG)
        self.rect(0, 0, 210, 297, "F")
        self.set_fill_color(*TEAL)
        self.rect(0, 0, 210, 3, "F")

    def panel(self, x, y, w, h, color=None, border_color=None):
        c = color or PANEL_BG
        self.set_fill_color(*c)
        if border_color:
            self.set_draw_color(*border_color)
            self.rect(x, y, w, h, "FD")
        else:
            self.rect(x, y, w, h, "F")

    def h1(self, text, color=None):
        self.set_font("Helvetica", "B", 17)
        self.set_text_color(*(color or WHITE))
        self.cell(0, 10, text, align="C", ln=True)

    def h2(self, text, color=None):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*(color or TEAL))
        self.multi_cell(0, 7, text)

    def h3(self, text, color=None):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*(color or GOLD))
        self.multi_cell(0, 6, text)

    def body(self, text, size=9.5, color=None):
        self.set_font("Helvetica", "", size)
        self.set_text_color(*(color or LIGHT_GREY))
        self.multi_cell(0, 6, text)

    def small(self, text, color=None):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*(color or MID_GREY))
        self.multi_cell(0, 5, text)

    def rule(self, color=None):
        self.set_fill_color(*(color or TEAL))
        self.rect(15, self.get_y(), 180, 0.4, "F")
        self.ln(3)

    def spacer(self, h=4):
        self.ln(h)

    def callout(self, title, text, x=15, w=180, title_color=None, bg_color=None):
        tc = title_color or GOLD
        bg = bg_color or CARD_BG
        y0 = self.get_y()
        lines = text.split("\n")
        h = 6 + 5 + len(lines) * 5.5 + 4
        self.set_fill_color(*bg)
        self.set_draw_color(*tc)
        self.rect(x, y0, w, h, "FD")
        self.set_xy(x + 4, y0 + 3)
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(*tc)
        self.cell(w - 8, 5, title, ln=True)
        self.set_x(x + 4)
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*LIGHT_GREY)
        self.multi_cell(w - 8, 5.5, text)
        self.ln(3)

    def code_box(self, title, code_lines, x=15, w=180):
        y0 = self.get_y()
        line_h = 4.8
        h = 8 + len(code_lines) * line_h + 5
        self.set_fill_color(10, 18, 28)
        self.set_draw_color(*SOFT_BLUE)
        self.rect(x, y0, w, h, "FD")
        self.set_xy(x + 4, y0 + 3)
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*SOFT_BLUE)
        self.cell(w - 8, 5, title, ln=True)
        self.set_x(x + 4)
        self.set_font("Courier", "", 7.5)
        self.set_text_color(180, 220, 180)
        for line in code_lines:
            self.set_x(x + 4)
            self.cell(w - 8, line_h, line, ln=True)
        self.ln(3)

    def stat_row(self, label, value, color=None):
        vc = color or TEAL
        y0 = self.get_y()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*LIGHT_GREY)
        self.set_xy(15, y0)
        self.cell(110, 6, label)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*vc)
        self.cell(0, 6, value, ln=True)

    def img(self, path, w=180, h=70, caption=""):
        if Path(path).exists():
            y0 = self.get_y()
            self.image(str(path), x=15, y=y0, w=w, h=h)
            self.set_y(y0 + h + 1)
            if caption:
                self.small(caption)
                self.spacer(2)

    # ------------------------------------------------------------------
    # Section divider page
    # ------------------------------------------------------------------
    def section_page(self, number, title, subtitle, desc):
        self.dark_page()
        self.set_fill_color(*TEAL)
        self.rect(0, 100, 210, 70, "F")
        self.set_y(106)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*DARK_BG)
        self.cell(0, 7, f"SECTION {number}", align="C", ln=True)
        self.set_font("Helvetica", "B", 26)
        self.set_text_color(*DARK_BG)
        self.cell(0, 13, title, align="C", ln=True)
        self.set_font("Helvetica", "", 13)
        self.cell(0, 8, subtitle, align="C", ln=True)
        self.set_y(185)
        self.set_left_margin(30)
        self.set_right_margin(30)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*LIGHT_GREY)
        self.multi_cell(0, 6, desc)
        self.set_left_margin(10)
        self.set_right_margin(10)

    # ------------------------------------------------------------------
    # Content page template
    # ------------------------------------------------------------------
    def content_page(self, title, subtitle=""):
        self.dark_page()
        self.set_y(7)
        self.set_left_margin(15)
        self.set_right_margin(15)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*WHITE)
        self.cell(0, 8, title, align="C", ln=True)
        if subtitle:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(*MID_GREY)
            self.cell(0, 5, subtitle, align="C", ln=True)
        self.rule()
        self.spacer(2)


# ===========================================================================
# BUILD PDF
# ===========================================================================
pdf = ClientReport()
pdf.set_auto_page_break(auto=True, margin=14)


# ---------------------------------------------------------------------------
# COVER
# ---------------------------------------------------------------------------
pdf.dark_page()

# Large teal block top
pdf.set_fill_color(*TEAL)
pdf.rect(0, 0, 210, 55, "F")

pdf.set_y(10)
pdf.set_font("Helvetica", "B", 10)
pdf.set_text_color(*DARK_BG)
pdf.cell(0, 7, "BRANDEIS MASTERS IN FINANCE  |  2026", align="C", ln=True)
pdf.set_font("Helvetica", "B", 9)
pdf.cell(0, 6, "SSGA QUANTITATIVE RESEARCH PROJECT", align="C", ln=True)

pdf.set_y(64)
pdf.set_font("Helvetica", "B", 32)
pdf.set_text_color(*WHITE)
pdf.cell(0, 16, "Meta-Labeling:", align="C", ln=True)
pdf.set_font("Helvetica", "B", 22)
pdf.set_text_color(*TEAL)
pdf.cell(0, 12, "Teaching a Model When to Trust Itself", align="C", ln=True)

pdf.spacer(8)
pdf.set_left_margin(25)
pdf.set_right_margin(25)
pdf.set_font("Helvetica", "", 10.5)
pdf.set_text_color(*LIGHT_GREY)
pdf.multi_cell(0, 7,
    "This report presents a quantitative investment research project exploring "
    "whether a machine learning filter can improve upon a rules-based macro "
    "trading model. We explain every concept from first principles, with "
    "technical depth available for quantitative readers.")
pdf.set_left_margin(10)
pdf.set_right_margin(10)

pdf.spacer(10)
pdf.set_fill_color(*TEAL)
pdf.rect(25, pdf.get_y(), 160, 0.5, "F")
pdf.spacer(8)

# Three summary cards
def cover_card(pdf, x, y, icon_label, title, body):
    pdf.panel(x, y, 52, 52, CARD_BG, TEAL)
    pdf.set_xy(x + 3, y + 4)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*TEAL)
    pdf.cell(46, 5, icon_label, ln=True)
    pdf.set_x(x + 3)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*WHITE)
    pdf.multi_cell(46, 5, title)
    pdf.set_x(x + 3)
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(*LIGHT_GREY)
    pdf.multi_cell(46, 4.5, body)

cy = pdf.get_y()
cover_card(pdf, 15,  cy, "THE PRIMARY MODEL", "M1: When to BUY or SELL",
    "A rules-based macro signal using 6 market indicators with dynamic weights.")
cover_card(pdf, 79,  cy, "THE SECONDARY MODEL", "M2: Should We Trust M1?",
    "A machine learning filter that predicts when M1's signal is likely to succeed.")
cover_card(pdf, 143, cy, "THE RESULT", "An Honest Finding",
    "M2 did not add value in this iteration. We explain why and what comes next.")

pdf.set_y(cy + 56)
pdf.set_font("Helvetica", "", 8.5)
pdf.set_text_color(*MID_GREY)
pdf.cell(0, 6, "Raviv  -  Samweg  -  Cristian        Week of March 16, 2026        CONFIDENTIAL - INTERNAL USE ONLY", align="C")


# ---------------------------------------------------------------------------
# TABLE OF CONTENTS
# ---------------------------------------------------------------------------
pdf.dark_page()
pdf.set_y(12)
pdf.set_left_margin(15)
pdf.set_right_margin(15)
pdf.h1("Table of Contents")
pdf.rule()
pdf.spacer(4)

toc_items = [
    ("SECTION 1", "The Problem We Are Solving",         "What gap in investing does this project address?"),
    ("SECTION 2", "The Data",                           "28 years of monthly market data across 4 asset classes"),
    ("SECTION 3", "M1 - The Primary Trading Model",     "How we generate BUY and SELL signals"),
    ("           3a", "The 6 Market Indicators",        "What each one measures and why"),
    ("           3b", "Z-Scores: A Universal Language", "Making different indicators comparable"),
    ("           3c", "Dynamic Weights: Spearman IC",   "How the model decides which indicators to trust"),
    ("           3d", "From Score to Signal",           "BUY / SELL decision logic"),
    ("           3e", "M1 Performance Results",         "28-year backtest vs benchmarks"),
    ("SECTION 4", "M2 - The Meta-Labeling Filter",      "Teaching the system when to trust M1"),
    ("           4a", "What is Meta-Labeling?",         "The concept explained simply"),
    ("           4b", "New Features: VIX and OAS",      "Market fear and credit stress as context"),
    ("           4c", "Feature Engineering",            "Turning raw data into model inputs"),
    ("           4d", "Walk-Forward Training",          "Training without any future data"),
    ("           4e", "Logistic Regression",            "How M2 makes its decision"),
    ("           4f", "M2 Results",                     "What the data showed"),
    ("SECTION 5", "EDA: Testing Our Assumptions",       "Statistical tests before building the model"),
    ("SECTION 6", "Conclusions and Next Steps",         "What we learned and where we go from here"),
]

for section, title, desc in toc_items:
    is_main = not section.startswith("  ")
    pdf.set_font("Helvetica", "B" if is_main else "", 9 if is_main else 8.5)
    pdf.set_text_color(*(TEAL if is_main else LIGHT_GREY))
    pdf.set_x(15)
    pdf.cell(28, 6, section)
    pdf.set_font("Helvetica", "B" if is_main else "", 9 if is_main else 8.5)
    pdf.set_text_color(*(WHITE if is_main else LIGHT_GREY))
    pdf.cell(90, 6, title)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(*MID_GREY)
    pdf.multi_cell(0, 6, desc)
    if is_main:
        pdf.set_fill_color(*PANEL_BG)
        pdf.rect(15, pdf.get_y(), 180, 0.3, "F")
        pdf.spacer(1)


# ===========================================================================
# SECTION 1 - THE PROBLEM
# ===========================================================================
pdf.section_page("1", "THE PROBLEM", "What gap in investing are we addressing?",
    "Most investment models tell you WHAT to buy or sell. They rarely tell you "
    "HOW CONFIDENT to be in each individual signal. This section explains why "
    "that gap matters and how meta-labeling is designed to fill it.")

# --- Page 1.1 ---
pdf.content_page("The Core Problem: Not All Signals Are Equal",
    "Why even a good model sometimes gets things wrong")
pdf.h2("Imagine a weather forecast")
pdf.spacer(2)
pdf.body(
    "A weather app tells you: 'It will rain tomorrow.' You trust that. But what "
    "if the app also told you its confidence level? '70% chance of rain in summer, "
    "but only 45% confidence in spring because spring weather is unpredictable.' "
    "That extra layer of information is exactly what this project builds.")
pdf.spacer(4)
pdf.h2("The investing equivalent")
pdf.spacer(2)
pdf.body(
    "Our primary model (M1) looks at financial market data and says BUY or SELL "
    "each month. It is right about 70% of the time over 28 years - a strong result. "
    "But that means it is wrong 30% of the time. The question is: can we predict "
    "WHICH 30% is likely to be wrong - before it is wrong?")
pdf.spacer(4)
pdf.h2("What meta-labeling tries to do")
pdf.spacer(2)
pdf.body(
    "Meta-labeling adds a second layer. After M1 says BUY or SELL, a second model "
    "(M2) asks: 'Given everything I know about current market conditions, should I "
    "trust this signal?' If M2 says yes, we trade. If M2 says no, we sit in cash.")
pdf.spacer(4)

pdf.panel(15, pdf.get_y(), 180, 40, CARD_BG, TEAL)
y0 = pdf.get_y() + 4
pdf.set_xy(19, y0)
pdf.set_font("Helvetica", "B", 10)
pdf.set_text_color(*TEAL)
pdf.cell(0, 6, "The Two-Model Architecture", ln=True)
pdf.set_x(19)
pdf.set_font("Courier", "", 9)
pdf.set_text_color(*LIGHT_GREY)
pdf.cell(0, 5.5, "MARKET DATA  -->  M1 (Primary)  -->  BUY or SELL signal", ln=True)
pdf.set_x(19)
pdf.cell(0, 5.5, "                                         |", ln=True)
pdf.set_x(19)
pdf.cell(0, 5.5, "                                   M2 (Secondary)", ln=True)
pdf.set_x(19)
pdf.cell(0, 5.5, "                               YES: execute trade", ln=True)
pdf.set_x(19)
pdf.cell(0, 5.5, "                               NO:  sit in cash", ln=True)
pdf.set_y(y0 + 40)
pdf.spacer(2)

pdf.callout("Why this matters for investors",
    "A model that trades less but only when it has high conviction can achieve better "
    "risk-adjusted returns than one that acts on every signal. Fewer but higher-quality "
    "trades means lower transaction costs, lower drawdowns, and more consistent performance.",
    title_color=GREEN_SOFT)


# ===========================================================================
# SECTION 2 - THE DATA
# ===========================================================================
pdf.section_page("2", "THE DATA", "28 years of monthly market history",
    "Good models require good data. This section explains what market data we use, "
    "why these specific assets were chosen, and how 28 years of history gives us "
    "a picture of how markets behave across full economic cycles including recessions, "
    "crashes, and recoveries.")

pdf.content_page("What Data Do We Use?",
    "Four asset classes, monthly frequency, 1996 to 2025")

pdf.h2("The four market building blocks")
pdf.spacer(2)
pdf.body(
    "Every month from 1996 to 2025 we observe four key financial instruments. "
    "These were chosen because together they capture most of what drives "
    "investment returns across economic cycles:")
pdf.spacer(5)

assets = [
    ("S&P 500 (SPX)", "SOFT_BLUE",
     "The US stock market. Tracks the 500 largest US companies. When the "
     "economy is growing, stocks rise. When investors are fearful, stocks fall. "
     "This is our primary gauge of equity market health.",
     "Technical: price index used for trend calculation. Monthly total returns "
     "used for performance attribution."),
    ("Bloomberg Commodity Index (BCOM)", "GOLD",
     "A basket of raw materials: oil, gold, wheat, copper, etc. Commodities "
     "tend to rise when the economy is strong and inflation is picking up. "
     "They fall when demand slows.",
     "Technical: price index for trend; return series for breadth calculation. "
     "Provides non-equity economic momentum signal."),
    ("US Corporate Bonds", "GREEN_SOFT",
     "Bonds issued by companies. They pay more than government bonds because "
     "they carry some risk. When companies are doing well, corporate bonds "
     "are in demand. When credit stress rises, they underperform government bonds.",
     "Technical: monthly total returns. Used in credit vs rates spread "
     "calculation to measure risk appetite in fixed income."),
    ("US 10-Year Treasury Bond", "RED_SOFT",
     "The safest investment - the US government's 10-year borrowing. "
     "When investors are scared, they flee to Treasuries (prices rise, "
     "yields fall). Used as the risk-free benchmark in our model.",
     "Technical: price index for yield momentum; returns as risk-free "
     "benchmark. Duration adjustment applied for total return calculation."),
]

for name, col_key, simple, technical in assets:
    col = eval(col_key)
    pdf.set_fill_color(*CARD_BG)
    pdf.set_draw_color(*col)
    y0 = pdf.get_y()
    pdf.rect(15, y0, 180, 28, "FD")
    pdf.set_xy(19, y0 + 3)
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_text_color(*col)
    pdf.cell(0, 5.5, name, ln=True)
    pdf.set_x(19)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*LIGHT_GREY)
    pdf.multi_cell(170, 4.8, simple)
    pdf.set_x(19)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(*MID_GREY)
    pdf.multi_cell(170, 4.5, "For technical readers: " + technical)
    pdf.spacer(3)

pdf.spacer(2)
pdf.callout("Why monthly frequency?",
    "M1 is designed as a strategic allocation model, not a day-trading system. Monthly "
    "signals reduce noise from short-term price fluctuations and align with how "
    "institutional investors rebalance portfolios. 28 years gives us 336 monthly "
    "observations covering 4 full market cycles: dot-com boom/bust, 2008 crisis, "
    "COVID crash, and the post-2022 rate hiking cycle.", title_color=TEAL)


# ===========================================================================
# SECTION 3 - M1
# ===========================================================================
pdf.section_page("3", "M1: THE PRIMARY MODEL", "How we generate BUY and SELL signals",
    "M1 is the foundation of the system. It watches 6 financial market indicators, "
    "converts them into a single combined score, and decides each month whether "
    "to go risk-on (BUY) or risk-off (SELL). Every step uses only data that would "
    "have been available at that point in history - no future information is ever used.")

# --- 3a: The 6 Indicators ---
pdf.content_page("3a  -  The 6 Market Indicators",
    "What each one measures and why it belongs in the model")

pdf.h2("What is an indicator?")
pdf.spacer(2)
pdf.body(
    "An indicator is a number calculated from market data that tells us something "
    "useful about the current state of the economy or financial markets. "
    "Think of them as six different gauges on a car dashboard - each measuring "
    "something different, and together giving a full picture of the engine.")
pdf.spacer(5)

indicators = [
    ("1. SPX Trend", TEAL,
     "Is the stock market in an uptrend or downtrend?",
     "Compares the current S&P 500 price to its average over the last 6 months. "
     "A positive number means the market is above its recent average - momentum is up. "
     "A negative number means prices have been falling.",
     "spx_price / rolling_mean_6m  - 1.0"),
    ("2. BCOM Trend", GOLD,
     "Is the commodity market trending up or down?",
     "Same calculation as SPX Trend but applied to commodities. Rising commodity "
     "prices generally signal strong economic demand and inflationary pressure.",
     "bcom_price / rolling_mean_6m  - 1.0"),
    ("3. Credit vs Rates", SOFT_BLUE,
     "Are corporate bonds beating government bonds?",
     "Takes the average return of corporate bonds over the last 3 months and "
     "subtracts the average return of government bonds. A positive value means "
     "investors are comfortable taking credit risk - a risk-on signal.",
     "mean(corp_ret, 3m) - mean(treasury_ret, 3m)"),
    ("4. Risk Breadth", GREEN_SOFT,
     "How broad is the risk-on sentiment across all assets?",
     "Averages the returns of stocks, commodities, and corporate bonds, then "
     "subtracts treasury returns. When multiple asset classes are outperforming "
     "safe bonds simultaneously, market confidence is broad and genuine.",
     "mean(spx + bcom + corp, 3m) - mean(treasury, 3m)"),
    ("5. BCOM Acceleration", RED_SOFT,
     "Is commodity momentum speeding up or slowing down?",
     "Compares the short-term commodity trend (3-month average return) to the "
     "longer-term trend (12-month average return). Positive means momentum is "
     "accelerating - early signal of a commodity cycle turning.",
     "bcom_ret.rolling(3m).mean() - bcom_ret.rolling(12m).mean()"),
    ("6. Yield Momentum", (200, 140, 255),
     "Which way are interest rates moving?",
     "Tracks the direction of 10-year treasury prices (which move opposite to "
     "yields). A falling treasury price means rising rates - often a sign of "
     "economic strength or inflationary pressure.",
     "-(treasury_price.diff(3))"),
]

for name, col, simple_q, body_text, formula in indicators:
    y0 = pdf.get_y()
    pdf.set_fill_color(*CARD_BG)
    pdf.set_draw_color(*col)
    pdf.rect(15, y0, 180, 24, "FD")
    pdf.set_xy(19, y0 + 2.5)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*col)
    pdf.cell(0, 5, name + "   -   " + simple_q, ln=True)
    pdf.set_x(19)
    pdf.set_font("Helvetica", "", 8.2)
    pdf.set_text_color(*LIGHT_GREY)
    pdf.multi_cell(120, 4.5, body_text)
    pdf.set_xy(143, y0 + 6)
    pdf.set_font("Courier", "", 7)
    pdf.set_text_color(*col)
    pdf.multi_cell(47, 4.5, formula)
    pdf.spacer(2)


# --- 3b: Z-Scores ---
pdf.content_page("3b  -  Z-Scores: A Universal Language",
    "How we make 6 different indicators comparable to each other")

pdf.h2("The problem with raw numbers")
pdf.spacer(2)
pdf.body(
    "Each of our 6 indicators uses completely different units and scales. "
    "SPX Trend might be 0.04 (4% above average), while Credit vs Rates might be "
    "0.002 (0.2% per month difference). How do you add these together meaningfully? "
    "You cannot - unless you convert them to a common language first.")
pdf.spacer(4)

pdf.h2("The intuition: how unusual is this reading?")
pdf.spacer(2)
pdf.body(
    "Imagine you are told today's temperature is 25 degrees. Is that unusual? "
    "It depends on WHERE you are and WHAT TIME OF YEAR it is. In Siberia in "
    "January, that is extreme. In Miami in July, it is normal. Context is everything.")
pdf.spacer(3)
pdf.body(
    "A Z-score answers: 'How unusual is today's indicator reading compared to "
    "its own history?' It measures the distance from the historical average in "
    "units of standard deviation. This makes all 6 indicators directly comparable.")
pdf.spacer(4)

pdf.panel(15, pdf.get_y(), 180, 28, CARD_BG, TEAL)
y0 = pdf.get_y() + 3
pdf.set_xy(19, y0)
pdf.set_font("Helvetica", "B", 9)
pdf.set_text_color(*TEAL)
pdf.cell(0, 5, "The Z-Score Formula - Explained Simply", ln=True)
pdf.set_x(19)
pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(*LIGHT_GREY)
pdf.cell(0, 5.5, "Z-score = (Today's value  -  Historical average)  /  Historical standard deviation", ln=True)
pdf.set_x(19)
pdf.set_font("Helvetica", "I", 8.5)
pdf.set_text_color(*MID_GREY)
pdf.cell(0, 5, "Z = 0.0  means perfectly normal.   Z = +2.0  means unusually strong.   Z = -2.0  means unusually weak.", ln=True)
pdf.set_y(y0 + 28)
pdf.spacer(3)

pdf.h3("A concrete example")
pdf.spacer(2)
pdf.body(
    "Say the SPX Trend over 28 years has averaged +0.5% with a standard deviation "
    "of 2.0%. Today it reads +4.5%. The Z-score is (4.5 - 0.5) / 2.0 = +2.0. "
    "This tells us today's equity trend is 2 standard deviations above normal - "
    "a strong bullish signal. A Z-score of -1.5 on Credit vs Rates would mean "
    "credit is underperforming bonds by 1.5 standard deviations - a warning sign.")
pdf.spacer(4)

pdf.h2("The critical detail: expanding history")
pdf.spacer(2)
pdf.body(
    "A naive Z-score would use the full 28-year average to evaluate every single "
    "month - including future months the model hadn't seen yet. That would be "
    "cheating. Our implementation uses an EXPANDING window: the Z-score in March "
    "2005 is computed using only data from 1996 to February 2005. The model only "
    "knows what was knowable at that moment in time.")
pdf.spacer(4)

pdf.code_box("CODE: Expanding Z-Score (src/metalabel/primary/signals.py)", [
    "def expanding_zscore(series, min_periods=12):",
    "    x        = series                          # today's indicator value",
    "    history  = x.shift(1)                      # only use PAST data (shift by 1)",
    "    mean     = history.expanding(min_periods).mean()   # growing historical avg",
    "    std      = history.expanding(min_periods).std()    # growing historical std",
    "    z        = (x - mean) / std                # how unusual is today?",
    "    return z",
    "",
    "# The shift(1) is critical: it ensures today's value is never used",
    "# to compute the mean/std against which today is evaluated.",
])


# --- 3c: Dynamic Weights ---
pdf.content_page("3c  -  Dynamic Weights: How M1 Decides Which Indicators to Trust",
    "Spearman Information Coefficient and inverse-correlation weighting explained")

pdf.h2("The problem with fixed weights")
pdf.spacer(2)
pdf.body(
    "The simplest approach would be to weight all 6 indicators equally - "
    "just average their Z-scores. But this ignores two important problems: "
    "first, some indicators may be more predictive than others in the current "
    "market environment. Second, some indicators may be telling you the same "
    "thing as others, so you are effectively double-counting that signal.")
pdf.spacer(4)

pdf.h2("Problem 1: Redundancy - Are two indicators saying the same thing?")
pdf.spacer(2)
pdf.body(
    "If SPX Trend and Risk Breadth both rise and fall together, counting both "
    "equally gives too much weight to that one underlying factor. Our model "
    "penalises indicators that are highly correlated with each other.")
pdf.spacer(3)
pdf.body(
    "The solution is called inverse-correlation weighting. Think of it like this: "
    "imagine 5 friends voting on where to go for dinner. If 4 of them always vote "
    "the same way and 1 always votes differently, the 1 dissenter has more "
    "independent information. We give more weight to the 'independent dissenter' "
    "indicators and less to the 'groupthink' indicators.")
pdf.spacer(4)

pdf.h2("Problem 2: Relevance - Has this indicator been useful recently?")
pdf.spacer(2)
pdf.body(
    "This is where the Spearman Information Coefficient (IC) comes in. Once we "
    "have 36 months of history, we look back at the last 3 years and ask for "
    "each indicator: did high readings in that indicator tend to coincide with "
    "good returns? If yes, keep trusting it. If no - zero it out.")
pdf.spacer(4)

pdf.panel(15, pdf.get_y(), 180, 30, CARD_BG, GOLD)
y0 = pdf.get_y() + 3
pdf.set_xy(19, y0)
pdf.set_font("Helvetica", "B", 9.5)
pdf.set_text_color(*GOLD)
pdf.cell(0, 6, "What is Spearman Correlation?", ln=True)
pdf.set_x(19)
pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(*LIGHT_GREY)
pdf.multi_cell(170, 5.5,
    "Spearman correlation ranks things from lowest to highest, then asks: "
    "when the indicator was ranked HIGH, was the return also ranked HIGH? "
    "It is robust to outliers and does not assume a straight-line relationship - "
    "it just asks 'do they move in the same direction?'")
pdf.set_y(y0 + 30)
pdf.spacer(3)

pdf.h3("A simple example of Spearman IC in action")
pdf.spacer(2)
pdf.body(
    "Suppose over the last 3 years, every time Credit vs Rates was in its top "
    "half of readings, the next month's portfolio return was also in its top half. "
    "Spearman IC would be close to +1.0: strong positive relationship. "
    "Now suppose SPX Trend had no consistent relationship - sometimes high readings "
    "led to good returns, sometimes not. Spearman IC near 0. The model would "
    "downweight SPX Trend and upweight Credit vs Rates.")
pdf.spacer(4)

pdf.code_box("CODE: Dynamic Weighting with Spearman IC (src/metalabel/primary/signals.py)", [
    "# Step 1: compute inverse-correlation weights (reduce redundancy)",
    "corr_matrix   = zscore_window.corr()          # how correlated are indicators?",
    "sum_corr      = corr_matrix.sum(axis=0)       # total correlation each has",
    "inv_corr      = 1.0 / sum_corr                # invert: less correlated = more weight",
    "base_weights  = inv_corr / inv_corr.sum()     # normalise to sum to 1",
    "",
    "# Step 2: Spearman IC filter (has this indicator predicted returns recently?)",
    "if months_of_history >= 36:",
    "    ic_window = last_36_months_of_data",
    "    corrs     = ic_window.corr(method='spearman')['target_return']",
    "    ic_mask   = (corrs > 0.0).astype(float)   # 1 if useful, 0 if misleading",
    "",
    "# Step 3: combine - zero out misleading indicators",
    "final_weights = base_weights * ic_mask",
    "final_weights = final_weights / final_weights.sum()  # re-normalise",
])

pdf.callout("Why this makes M1 'dynamic'",
    "These weights are recalculated fresh every single month. In a period where "
    "equity trends are not predicting returns well (e.g. a liquidity-driven rally), "
    "SPX Trend gets zeroed out and credit indicators get upweighted. The model "
    "continuously rewires itself based on what has been working in the recent past.",
    title_color=TEAL)


# --- 3d: Score to Signal ---
pdf.content_page("3d  -  From Score to Signal: The BUY / SELL Decision",
    "How 6 numbers become one trading decision")

pdf.h2("The composite score")
pdf.spacer(2)
pdf.body(
    "Once we have Z-scores for all 6 indicators and their dynamic weights, we "
    "compute a single weighted average called the composite score. This is "
    "one number that summarises the entire macro environment as the model sees it.")
pdf.spacer(3)
pdf.body(
    "A high positive composite score means most indicators are signalling "
    "risk-on conditions - equities trending up, credit outperforming, broad "
    "momentum positive. A large negative score means the opposite - defensive "
    "positioning is warranted.")
pdf.spacer(5)

pdf.panel(15, pdf.get_y(), 180, 42, CARD_BG, TEAL)
y0 = pdf.get_y() + 4
pdf.set_xy(19, y0)
pdf.set_font("Helvetica", "B", 10)
pdf.set_text_color(*TEAL)
pdf.cell(0, 6, "The Signal Decision Rule", ln=True)
pdf.spacer(2)

rows = [
    ("Composite Score > +0.31", "BUY", "Go risk-on: overweight equities and commodities", GREEN_SOFT),
    ("Composite Score < -0.31", "SELL", "Go risk-off: overweight treasuries defensively", RED_SOFT),
    ("-0.31 to +0.31",          "HOLD", "No clear signal: maintain previous allocation", GOLD),
]
for cond, sig, desc, col in rows:
    pdf.set_xy(19, pdf.get_y())
    pdf.set_font("Courier", "", 9)
    pdf.set_text_color(*LIGHT_GREY)
    pdf.cell(68, 6, cond)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*col)
    pdf.cell(18, 6, sig)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*MID_GREY)
    pdf.cell(0, 6, desc, ln=True)
pdf.set_y(y0 + 42)
pdf.spacer(4)

pdf.h2("What happens when a signal fires?")
pdf.spacer(2)
pdf.body(
    "When M1 says BUY, the portfolio rotates into equities and commodities in "
    "equal proportion. When it says SELL, it rotates defensively into government "
    "bonds. The model holds its position until the composite score crosses the "
    "threshold in the opposite direction. HOLD months maintain the previous "
    "allocation unchanged.")
pdf.spacer(4)
pdf.body(
    "Over 28 years, M1 generated 138 actionable signals: 66 BUY signals and "
    "72 SELL signals. These are the events on which M2 is trained and evaluated.")
pdf.spacer(4)

pdf.callout("Why +/- 0.31?",
    "The threshold was calibrated on the training data to produce a reasonable signal "
    "frequency. Too tight a threshold generates too many signals (high turnover, "
    "high costs). Too wide a threshold misses meaningful moves. The 0.31 value "
    "balanced these competing objectives across the full historical sample.",
    title_color=GOLD)


# --- 3e: M1 Performance ---
pdf.content_page("3e  -  M1 Performance Results",
    "28-year backtest vs four benchmarks, net of transaction costs")

pdf.h2("How does M1 compare to simply holding the market?")
pdf.spacer(2)
pdf.body(
    "The ultimate test of any active model is whether it outperforms passive "
    "alternatives. We compared M1 against four benchmarks over the full 28-year "
    "period from 1996 to 2025:")
pdf.spacer(4)

benchmarks = [
    ("Buy and Hold S&P 500", "Simply hold US equities forever.", "The classic passive equity benchmark."),
    ("60/40 Portfolio", "60% equities, 40% bonds, rebalanced monthly.", "The standard institutional benchmark."),
    ("Equal Weight Portfolio", "Equal allocation across all 4 assets.", "Naive diversification baseline."),
    ("Simple Trend", "A naive momentum strategy without dynamic weights.", "Tests whether complexity adds value over simple trend-following."),
]
for name, desc, context in benchmarks:
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*WHITE)
    pdf.cell(65, 5.5, name)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*LIGHT_GREY)
    pdf.cell(75, 5.5, desc)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*MID_GREY)
    pdf.cell(0, 5.5, context, ln=True)

pdf.spacer(4)
pdf.img(ASSET_DIR / "equity_curves.png", w=180, h=80,
    caption="Figure 1: Cumulative equity curves 1996-2025, net of transaction costs. "
            "PrimaryV1 (purple) is our M1 model.")
pdf.spacer(2)

pdf.h3("Key performance metrics (M1 - PrimaryV1)")
pdf.spacer(2)

m1_stats = [
    ("Dataset period", "1996 - 2025 (28 years, monthly)"),
    ("Total M1 signals", "138 (66 BUY, 72 SELL)"),
    ("Out-of-sample win rate (actionable signals)", "71.8%"),
    ("Average monthly return per signal", "+0.76%"),
    ("Annualised Sharpe Ratio (OOS)", "1.31"),
    ("M1 vs Buy-and-Hold S&P 500", "Competitive returns with lower drawdowns"),
    ("M1 vs 60/40", "Higher Sharpe through dynamic allocation"),
]
for label, val in m1_stats:
    pdf.stat_row(label, val)
    pdf.spacer(1)

pdf.spacer(3)
pdf.callout("What the equity curve tells us",
    "M1 (purple line) tracks closely with the best-performing benchmarks throughout "
    "history, but critically preserves capital during major drawdowns (2008, 2020) "
    "where its SELL signals moved the portfolio to safety. This defensiveness "
    "is visible in the chart as shallower dips during crisis periods.",
    title_color=GREEN_SOFT)


# ===========================================================================
# SECTION 4 - M2
# ===========================================================================
pdf.section_page("4", "M2: THE META-LABELING FILTER", "Teaching the model when to trust itself",
    "M1 is a strong model but it is not perfect. M2 is a machine learning layer "
    "that sits on top of M1 and asks: given today's market conditions, how confident "
    "should we be in M1's signal? This section explains the concept, the construction, "
    "and - honestly - the results.")

# --- 4a: Meta-labeling concept ---
pdf.content_page("4a  -  What is Meta-Labeling?",
    "A concept from quantitative finance by Marcos Lopez de Prado")

pdf.h2("The sports analogy")
pdf.spacer(2)
pdf.body(
    "Imagine a football coach (M1) who calls plays based on studying the opponent. "
    "The coach is good - right about 70% of the time. Now imagine an assistant "
    "coach (M2) who watches film and says: 'Coach, I notice you tend to get this "
    "wrong in wet weather against fast defenses - maybe sit this one out.' "
    "M2 does not override the coach. It just says when to trust the coach and "
    "when to hold back.")
pdf.spacer(4)

pdf.h2("The technical concept")
pdf.spacer(2)
pdf.body(
    "Meta-labeling was introduced by Dr. Marcos Lopez de Prado in his book "
    "'Advances in Financial Machine Learning' (2018). The key insight is:")
pdf.spacer(3)

points = [
    "M1 decides the DIRECTION of the trade (BUY or SELL)",
    "M2 decides the SIZE or CONFIDENCE of that trade (trade vs do not trade)",
    "M2 is only trained on the events where M1 already fired a signal",
    "M2's target is binary: did M1's signal turn out to be profitable? (1=yes, 0=no)",
]
for p in points:
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*LIGHT_GREY)
    pdf.set_x(22)
    pdf.cell(5, 6, "-")
    pdf.multi_cell(0, 6, p)

pdf.spacer(4)
pdf.h2("Why not just improve M1 directly?")
pdf.spacer(2)
pdf.body(
    "Because M1 uses one type of information (macro momentum trends) and M2 "
    "can use completely different information (market regime, fear levels, "
    "M1's own recent track record). These are complementary, not competing. "
    "Meta-labeling preserves M1's directional logic while adding a regime-aware "
    "confidence filter on top.")
pdf.spacer(4)

pdf.panel(15, pdf.get_y(), 180, 32, CARD_BG, SOFT_BLUE)
y0 = pdf.get_y() + 4
pdf.set_xy(19, y0)
pdf.set_font("Helvetica", "B", 9.5)
pdf.set_text_color(*SOFT_BLUE)
pdf.cell(0, 6, "The Meta-Label Target Variable", ln=True)
pdf.set_x(19)
pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(*LIGHT_GREY)
pdf.multi_cell(170, 5.5,
    "For each of the 138 M1 signals, we look at what actually happened next month. "
    "If M1 said BUY and the portfolio made money: meta_label = 1. "
    "If M1 said BUY and the portfolio lost money: meta_label = 0. "
    "M2's job is to predict this 1 or 0 before the trade is made.")
pdf.set_y(y0 + 32)


# --- 4b: VIX and OAS ---
pdf.content_page("4b  -  New Features: VIX and OAS",
    "Market fear and credit stress as regime context for M2")

pdf.h2("Why add VIX and OAS?")
pdf.spacer(2)
pdf.body(
    "M1 uses macro trend indicators to decide direction. But it has no awareness "
    "of the broader market regime - specifically, how fearful or stressed the "
    "market is right now. Our hypothesis was: maybe M1's signals are more reliable "
    "in calm markets than in panicking ones.")
pdf.spacer(4)

pdf.h2("VIX - The Fear Index")
pdf.spacer(2)
pdf.body(
    "VIX is the CBOE Volatility Index. It measures how much volatility options "
    "traders expect in the S&P 500 over the next 30 days. Think of it as "
    "a thermometer for market anxiety. VIX of 12 = calm. VIX of 40 = panic "
    "(as seen in March 2020, October 2008).")
pdf.spacer(3)
pdf.body(
    "Our hypothesis: when VIX is very high, markets are chaotic and trend signals "
    "break down. M2 should learn to be more cautious with M1's signals during "
    "high-fear regimes.")
pdf.spacer(5)

pdf.h2("OAS - The Credit Stress Gauge")
pdf.spacer(2)
pdf.body(
    "OAS (Option-Adjusted Spread) is the extra yield investors demand to hold "
    "corporate bonds instead of risk-free government bonds. Wide spreads mean "
    "investors fear corporate defaults - a sign of financial stress and illiquidity. "
    "Tight spreads mean credit markets are functioning normally.")
pdf.spacer(3)
pdf.body(
    "Our hypothesis: wide and widening credit spreads signal that the financial "
    "plumbing is under stress. In such conditions, M1's macro signals may be "
    "less reliable because dislocations overwhelm fundamentals.")
pdf.spacer(4)

pdf.callout("The theoretical link to M1",
    "M1 already captures some of this information indirectly: credit vs rates "
    "spread and risk breadth both respond to market stress. This turned out to be "
    "an important clue about why M2 struggled - more on this in Section 5.",
    title_color=GOLD)

pdf.spacer(3)
pdf.h2("What we expected M2 to learn")
pdf.spacer(2)

expected = [
    ("High VIX + VIX Rising",  "Markets panicking. M1 trend signals unreliable. Skip trade."),
    ("Low VIX + VIX Falling",  "Markets calm. M1 operating in favourable conditions. Take trade."),
    ("Wide OAS + OAS Widening","Credit stress mounting. M1 SELL signals more likely correct."),
    ("Tight OAS + OAS Falling","Credit normalising. BUY signals more trustworthy."),
]
for condition, expected_action in expected:
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_text_color(*TEAL)
    pdf.cell(80, 6, condition)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*LIGHT_GREY)
    pdf.cell(0, 6, expected_action, ln=True)


# --- 4c: Feature Engineering ---
pdf.content_page("4c  -  Feature Engineering",
    "Turning raw VIX and OAS numbers into 10 model-ready inputs")

pdf.h2("Why not just use the raw VIX number?")
pdf.spacer(2)
pdf.body(
    "A raw reading of VIX = 28 is not interpretable on its own. Is 28 high? "
    "It depends on history. In 1993, it would be extraordinary. In 2020, it was "
    "moderate. The model needs context. We apply the same Z-score logic as M1, "
    "plus additional features that capture different dimensions of the regime.")
pdf.spacer(5)

vix_features = [
    ("vix_level_z",     "How high is VIX relative to its own full history? (expanding Z-score)"),
    ("vix_change_z",    "How much did VIX move this month, vs. typical monthly moves?"),
    ("vix_trend",       "Is VIX above or below its own 6-month rolling average?"),
    ("vix_high_regime", "BINARY: is VIX above its all-time expanding median? (1=yes, 0=no)"),
    ("vix_rising",      "BINARY: did VIX increase this month? (1=yes, 0=no)"),
]
oas_features = [
    ("oas_level_z",     "How wide are spreads relative to their full history?"),
    ("oas_change_z",    "How much did OAS move this month vs. typical moves?"),
    ("oas_trend",       "Is OAS above or below its own 6-month rolling average?"),
    ("oas_wide_regime", "BINARY: are spreads in a historically wide/stressed regime?"),
    ("oas_widening",    "BINARY: did spreads widen this month? (credit getting worse?)"),
]

pdf.set_font("Helvetica", "B", 9.5)
pdf.set_text_color(*SOFT_BLUE)
pdf.cell(0, 6, "VIX Features (5 inputs)", ln=True)
pdf.spacer(1)
for name, desc in vix_features:
    pdf.set_font("Courier", "", 8.5)
    pdf.set_text_color(*TEAL)
    pdf.cell(42, 5.5, name)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*LIGHT_GREY)
    pdf.cell(0, 5.5, desc, ln=True)

pdf.spacer(4)
pdf.set_font("Helvetica", "B", 9.5)
pdf.set_text_color(*SOFT_BLUE)
pdf.cell(0, 6, "OAS Features (5 inputs)", ln=True)
pdf.spacer(1)
for name, desc in oas_features:
    pdf.set_font("Courier", "", 8.5)
    pdf.set_text_color(*GOLD)
    pdf.cell(42, 5.5, name)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*LIGHT_GREY)
    pdf.cell(0, 5.5, desc, ln=True)

pdf.spacer(4)
pdf.code_box("CODE: Building VIX Features (src/metalabel/secondary/features.py)", [
    "def build_vix_features(vix, trend_window=6, zscore_min_periods=12):",
    "    # Level: how high is VIX vs its own history?",
    "    level_z      = expanding_zscore(vix)              # same no-lookahead zscore",
    "",
    "    # Change: how much did VIX move this month?",
    "    change       = vix.diff()                         # month-over-month change",
    "    change_z     = expanding_zscore(change)           # z-score of that change",
    "",
    "    # Trend: is VIX above its recent average?",
    "    rolling_mean = vix.rolling(trend_window).mean()",
    "    trend        = (vix / rolling_mean) - 1.0         # % deviation from avg",
    "",
    "    # Regime flags: binary 0/1 signals",
    "    high_regime  = (vix > expanding_median(vix)).astype(float)",
    "    rising       = (change > 0).astype(float)         # did VIX go up?",
    "",
    "    return [level_z, change_z, trend, high_regime, rising]",
])


# --- 4d: Walk-Forward ---
pdf.content_page("4d  -  Walk-Forward Training: No Peeking at the Future",
    "How we ensure M2 is evaluated fairly on data it has never seen")

pdf.h2("The overfitting problem in finance")
pdf.spacer(2)
pdf.body(
    "Any model can be made to look perfect if it is trained and tested on the "
    "same data. Like studying with the exact exam questions - you would score "
    "100% but learn nothing. In finance this is called overfitting, and it is "
    "the most common way models fail when deployed.")
pdf.spacer(4)

pdf.h2("The walk-forward solution")
pdf.spacer(2)
pdf.body(
    "We use an expanding walk-forward approach. Imagine you are back in January "
    "2005. You train M2 on every signal from 1996 to December 2004 (60 months). "
    "Then you predict February 2005. Then add February 2005 to your training set "
    "and predict March 2005. And so on - never looking at future data.")
pdf.spacer(4)

pdf.panel(15, pdf.get_y(), 180, 50, CARD_BG, TEAL)
y0 = pdf.get_y() + 4
pdf.set_xy(19, y0)
pdf.set_font("Helvetica", "B", 9)
pdf.set_text_color(*TEAL)
pdf.cell(0, 5.5, "Walk-Forward Illustration (simplified)", ln=True)
pdf.set_font("Courier", "", 8)
pdf.set_text_color(*LIGHT_GREY)
walk_lines = [
    "Step 1:  TRAIN on events 1-60    -->  PREDICT event 61",
    "Step 2:  TRAIN on events 1-61    -->  PREDICT event 62",
    "Step 3:  TRAIN on events 1-62    -->  PREDICT event 63",
    "  ...    (window expands by 1 each step)",
    "Step 78: TRAIN on events 1-137   -->  PREDICT event 138",
    "",
    "Result: 78 out-of-sample predictions. M2 never saw any of these",
    "events during training. This is a genuine simulation of live use.",
]
for line in walk_lines:
    pdf.set_x(19)
    pdf.cell(0, 4.8, line, ln=True)
pdf.set_y(y0 + 50)
pdf.spacer(3)

pdf.h2("Why 60 months minimum training?")
pdf.spacer(2)
pdf.body(
    "Logistic regression needs enough examples to learn meaningful patterns. "
    "With only 12 training events, the model would be fitting noise. 60 months "
    "(5 years) gives a statistically reasonable sample while still leaving 78 "
    "months of genuine out-of-sample evaluation - enough to draw conclusions.")
pdf.spacer(4)

pdf.callout("The result: 78 genuinely out-of-sample predictions",
    "The 78 OOS events span 2009 to 2025 - a period that includes the European debt "
    "crisis, China slowdown, COVID crash, rate hiking cycle, and the AI boom. "
    "M2 was evaluated across a wide range of market conditions it had never trained on.",
    title_color=SOFT_BLUE)


# --- 4e: Logistic Regression ---
pdf.content_page("4e  -  Logistic Regression: How M2 Makes Its Decision",
    "The machine learning model and why we chose it for this problem")

pdf.h2("What M2 is doing")
pdf.spacer(2)
pdf.body(
    "M2 takes 22 input features (the 6 M1 indicators + 10 VIX/OAS features + "
    "4 M1 track record features + M1 direction) and outputs a single number: "
    "the probability that M1's next signal will be profitable.")
pdf.spacer(4)

pdf.h2("Why Logistic Regression?")
pdf.spacer(2)
pdf.body(
    "Logistic regression is the standard starting point for binary classification "
    "in quantitative finance. It takes all the input features, multiplies each "
    "by a learned weight, sums them up, and squashes the result into a probability "
    "between 0 and 1 using the sigmoid function. Simple, interpretable, and it "
    "produces well-calibrated probabilities.")
pdf.spacer(3)

reasons = [
    ("Small dataset (138 events)",
     "Complex models like neural networks or gradient boosting need thousands of "
     "examples to avoid overfitting. With 138 events, logistic regression's built-in "
     "regularisation (the C parameter) is essential."),
    ("Calibrated probabilities",
     "The threshold gate needs real probabilities, not just rankings. Logistic "
     "regression produces well-calibrated probabilities out of the box. Tree-based "
     "models require additional calibration steps."),
    ("Interpretability",
     "At this research stage, we want to understand what the model is learning. "
     "Logistic regression coefficients directly show which features drive decisions."),
    ("Research discipline",
     "Starting simple is standard practice. You establish a baseline with logistic "
     "regression before justifying the added complexity of ensemble methods."),
]
for title, body in reasons:
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*TEAL)
    pdf.cell(0, 5.5, title, ln=True)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*LIGHT_GREY)
    pdf.multi_cell(0, 5, body)
    pdf.spacer(2)

pdf.spacer(2)
pdf.code_box("CODE: M2 Model Configuration (src/metalabel/secondary/model.py)", [
    "def _make_model():",
    "    return LogisticRegression(",
    "        C             = 0.5,          # regularisation: prevents overfitting",
    "        max_iter      = 500,          # iterations for solver convergence",
    "        solver        = 'lbfgs',      # efficient for small datasets",
    "        class_weight  = 'balanced',   # adjusts for 70/30 win/loss imbalance",
    "        random_state  = 42,           # reproducibility",
    "    )",
    "",
    "# At each walk-forward step:",
    "scaler      = StandardScaler()        # normalise features (mean=0, std=1)",
    "X_train_s   = scaler.fit_transform(X_train)   # fit scaler on train only",
    "X_test_s    = scaler.transform(X_test)         # apply same scale to test",
    "model.fit(X_train_s, y_train)         # learn weights",
    "probs       = model.predict_proba(X_test_s)[:,1]  # P(M1 wins)",
    "m2_approve  = (probs >= 0.5).astype(int)       # threshold gate",
])


# --- 4f: M2 Results ---
pdf.content_page("4f  -  M2 Results: What the Data Showed",
    "An honest assessment of the first iteration")

pdf.h2("The headline numbers")
pdf.spacer(2)

result_stats = [
    ("Out-of-sample events",              "78 (spanning 2009-2025)"),
    ("Minimum training window",           "60 months (5 years)"),
    ("Threshold used",                    "0.5 (default)"),
    ("M2 classification accuracy",        "53.9%   (random guessing = 50%)"),
    ("M2 ROC-AUC",                        "0.523   (random coin flip = 0.500)"),
]
for label, val in result_stats:
    pdf.stat_row(label, val)
    pdf.spacer(1)

pdf.spacer(5)
pdf.h2("Economic performance comparison")
pdf.spacer(2)

pdf.set_font("Helvetica", "B", 8.5)
pdf.set_text_color(*MID_GREY)
pdf.cell(55, 6, "Group")
pdf.cell(18, 6, "N", align="C")
pdf.cell(28, 6, "Win Rate", align="C")
pdf.cell(35, 6, "Avg Return", align="C")
pdf.cell(0, 6, "Ann. Sharpe", align="C", ln=True)
pdf.set_fill_color(*PANEL_BG)
pdf.rect(15, pdf.get_y(), 180, 0.3, "F")
pdf.spacer(1)

perf_rows = [
    ("M1 baseline (all 78 OOS trades)",  "78", "71.8%", "+0.76%", "1.31", WHITE),
    ("M2 Approved (trade these)",        "46", "71.7%", "+0.75%", "1.18", RED_SOFT),
    ("M2 Rejected (sit in cash)",        "32", "71.9%", "+0.78%", "n/a",  GOLD),
]
for name, n, wr, ret, sharpe, col in perf_rows:
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*col)
    pdf.cell(55, 6, name)
    pdf.cell(18, 6, n, align="C")
    pdf.cell(28, 6, wr, align="C")
    pdf.cell(35, 6, ret, align="C")
    pdf.cell(0, 6, sharpe, align="C", ln=True)

pdf.spacer(4)
pdf.img(M2_DIR / "2_approve_reject_performance.png", w=180, h=65,
    caption="Figure 2: M2 Approve vs Reject economic performance comparison. "
            "All three groups show near-identical win rates - M2 has no discriminating power.")

pdf.spacer(2)
pdf.callout("The honest conclusion",
    "M2 did not add value in this iteration. The trades it approved performed slightly "
    "WORSE than the ones it rejected. The Sharpe ratio fell from 1.31 to 1.18 when "
    "filtering through M2. The model is not distinguishing winners from losers.",
    title_color=RED_SOFT, bg_color=(40, 20, 20))

pdf.content_page("4f continued  -  Understanding Why M2 Struggled",
    "Three hypotheses for the null result")

pdf.img(M2_DIR / "3_cumulative_return.png", w=180, h=70,
    caption="Figure 3: Cumulative return - M1 baseline (blue) vs M2-approved only (green). "
            "M2 filtering consistently underperforms full M1 across the OOS period.")
pdf.spacer(3)

hypotheses = [
    ("Hypothesis 1: M1 already absorbs the stress signal",
     TEAL,
     "M1's own indicators - credit vs rates, risk breadth, SPX trend - all respond to "
     "market stress. When VIX is high, these indicators have already adjusted. By adding "
     "VIX and OAS, we may be giving M2 redundant information that M1 already processed. "
     "This is called multicollinearity: two features measuring the same underlying thing.",
     "This is the most likely explanation. VIX and OAS are consequences of the same "
     "market stress that M1's indicators already capture - they add no independent signal."),
    ("Hypothesis 2: M1 is robust across all regimes by design",
     GOLD,
     "M1 has both BUY and SELL signals. In a market crash (high VIX), M1 is likely "
     "already positioned SELL - and that SELL is correct. M1's win rate of ~72% holds "
     "across calm and stressed markets alike, meaning there is no regime where M1 "
     "predictably fails - so no regime filter can help.",
     "The data supports this: high VIX + wide OAS actually showed the BEST M1 "
     "performance (72.7% win rate, +1.02% avg return). The 'worst' regime was the best."),
    ("Hypothesis 3: 138 events is statistically too small",
     SOFT_BLUE,
     "Even if a real pattern exists, detecting it requires enough data. With 138 monthly "
     "observations across 28 years, we have very few genuine crisis periods (2008, 2011, "
     "2020). Statistical tests simply cannot confirm a pattern that appears in only a "
     "handful of observations.",
     "The chi-square p-values of 0.89 (VIX) and 0.77 (OAS) reflect this uncertainty. "
     "We need either more events or stronger features to get below the 0.05 threshold."),
]

for title, col, simple_body, technical in hypotheses:
    pdf.set_fill_color(*CARD_BG)
    pdf.set_draw_color(*col)
    y0 = pdf.get_y()
    pdf.rect(15, y0, 180, 38, "FD")
    pdf.set_xy(19, y0 + 3)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*col)
    pdf.cell(0, 5, title, ln=True)
    pdf.set_x(19)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*LIGHT_GREY)
    pdf.multi_cell(170, 5, simple_body)
    pdf.set_x(19)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(*MID_GREY)
    pdf.multi_cell(170, 4.5, "For technical readers: " + technical)
    pdf.spacer(3)


# ===========================================================================
# SECTION 5 - EDA
# ===========================================================================
pdf.section_page("5", "EDA: TESTING OUR ASSUMPTIONS", "Statistical tests before building the model",
    "Before training M2, we conducted Exploratory Data Analysis (EDA) to test "
    "whether VIX and OAS even have a measurable relationship with M1's performance. "
    "This section presents 4 of the 6 key charts and their statistical conclusions.")

pdf.content_page("5  -  What the EDA Charts Showed",
    "Six charts, one consistent finding")

pdf.h2("Chart 1 - Does VIX predict M1 wins?")
pdf.spacer(2)
pdf.body(
    "We split all 138 M1 events into Low VIX and High VIX regimes and compared "
    "win rates. A chi-square test measured whether any observed difference was "
    "statistically real or just random variation with our small sample.")
pdf.spacer(2)
pdf.img(EDA_DIR / "1_win_rate_by_vix_regime.png", w=180, h=62,
    caption="Low VIX: 67.9% win rate  |  High VIX: 70.6% win rate  |  Chi-square p = 0.8882 (not significant)")
pdf.spacer(1)
pdf.callout("Interpretation",
    "p = 0.8882 means there is an 89% chance this 2.7% difference is pure random "
    "noise. We need p < 0.05 to call something real. VIX regime has zero "
    "statistically significant relationship with M1 winning.", title_color=RED_SOFT)

pdf.content_page("5 continued  -  More EDA Findings")

pdf.h2("Chart 2 - Does OAS predict M1 wins?")
pdf.spacer(2)
pdf.img(EDA_DIR / "2_win_rate_by_oas_regime.png", w=180, h=62,
    caption="Tight OAS: 66.7%  |  Wide OAS: 70.8%  |  Chi-square p = 0.7730 (not significant)")
pdf.spacer(1)
pdf.callout("Interpretation",
    "OAS tells the same story as VIX. Wide spreads (stressed credit) are not "
    "associated with worse M1 performance. p = 0.77 is far from significance.",
    title_color=RED_SOFT)
pdf.spacer(4)

pdf.h2("Chart 3 - The most striking result: combined VIX + OAS regimes")
pdf.spacer(2)
pdf.img(EDA_DIR / "3_joint_regime_heatmap.png", w=180, h=65,
    caption="High VIX + Wide OAS (worst expected) delivered 72.7% win rate and +1.02% avg return - the best of all four regimes.")
pdf.spacer(1)
pdf.callout("The counter-intuitive finding",
    "The regime most investors would label 'dangerous' produced the best M1 "
    "performance. This strongly supports the thesis that M1's SELL logic already "
    "capitalises on stress periods by positioning defensively.", title_color=GOLD)

pdf.content_page("5 continued  -  Feature Correlations and Regime Simulation")

pdf.h2("Chart 4 - Feature correlations with M1 win/loss")
pdf.spacer(2)
pdf.img(EDA_DIR / "4_feature_correlations.png", w=180, h=85,
    caption="No feature exceeds 0.11 correlation with meta_label. None are statistically significant (no * markers).")
pdf.spacer(2)
pdf.callout("What this means",
    "The highest correlation with M1 winning is only 0.108 (trailing average return). "
    "All 10 VIX and OAS features are below 0.07. In a dataset of 138 observations, "
    "you would need at least 0.17 correlation to be statistically significant at "
    "p < 0.05. None of our features come close.", title_color=RED_SOFT)

pdf.spacer(3)
pdf.h2("Chart 6 - What if we used regime as a simple gate?")
pdf.spacer(2)
pdf.img(EDA_DIR / "6_regime_gate_simulation.png", w=180, h=65,
    caption="Filtering by 'favourable' VIX+OAS reduces Sharpe from 1.19 to 0.68. The 'bad' regime (High VIX+Wide OAS) has Sharpe 1.49.")


# ===========================================================================
# SECTION 6 - CONCLUSIONS
# ===========================================================================
pdf.section_page("6", "CONCLUSIONS & NEXT STEPS", "What we learned and where we go from here",
    "This iteration of M2 did not add value. But the framework, the data pipeline, "
    "and the walk-forward infrastructure are all now in place. We have a clear "
    "diagnostic of why this approach failed, which directly informs the next steps.")

pdf.content_page("6  -  Summary and Next Steps",
    "Honest findings and a roadmap for improvement")

pdf.h2("What we built")
pdf.spacer(2)
pdf.body(
    "In this research sprint we built an end-to-end meta-labeling system: "
    "a 28-year dataset of 138 labeled M1 signals, 22 features including 10 new "
    "VIX and OAS regime features, a leakage-free walk-forward training framework, "
    "and a logistic regression M2 classifier. The entire codebase is modular, "
    "tested, and version-controlled.")
pdf.spacer(4)

pdf.h2("What we found")
pdf.spacer(2)
findings = [
    ("M1 is strong on its own",
     "71.8% win rate and 1.31 annualised Sharpe over 28 years. Outperforms 60/40 "
     "and simple buy-and-hold on a risk-adjusted basis."),
    ("M2 did not improve M1",
     "ROC-AUC of 0.523 (barely above random). M2-approved trades had slightly "
     "lower Sharpe (1.18) than simply trading all M1 signals (1.31)."),
    ("VIX and OAS carry no independent signal",
     "Chi-square tests (p=0.89, p=0.77) confirm no statistically significant "
     "relationship between these regime features and M1 win/loss outcomes."),
    ("M1 is regime-robust - a feature, not a bug",
     "M1 performs equally well or better in stressed markets, likely because its "
     "SELL logic correctly positions defensively during downturns."),
    ("Sample size is a binding constraint",
     "138 monthly events over 28 years is too small to detect subtle regime effects "
     "even if they exist. This is the most honest structural limitation."),
]
for title, body in findings:
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*TEAL)
    pdf.cell(5, 6, "-")
    pdf.cell(0, 6, title, ln=True)
    pdf.set_x(20)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*LIGHT_GREY)
    pdf.multi_cell(0, 5, body)
    pdf.spacer(2)

pdf.spacer(3)
pdf.h2("What comes next")
pdf.spacer(2)

next_steps = [
    ("01", "Find features orthogonal to M1",
     "The key lesson is that VIX and OAS are redundant with M1's existing inputs. "
     "The next iteration should look for information M1 structurally cannot see: "
     "sentiment data, options market signals, cross-asset dispersion."),
    ("02", "Investigate ROC curve threshold optimisation",
     "The 0.5 threshold was never optimised. A walk-forward threshold selection "
     "within the training folds may unlock value that fixed thresholds miss."),
    ("03", "Cross-validation across multiple window sizes",
     "Test minimum training windows of 3, 6, 12, 24, and 36 months to understand "
     "how much history M2 actually needs and whether results are stable."),
    ("04", "Separate BUY and SELL meta-labelers",
     "BUY and SELL signals may have different risk profiles. Training one M2 "
     "for each direction could improve precision at the cost of sample size."),
    ("05", "Production integration once validated",
     "Once a specification shows genuine out-of-sample improvement, integrate "
     "it into the live monthly allocation process with defined guardrails."),
]

for num, title, body in next_steps:
    y0 = pdf.get_y()
    pdf.set_fill_color(*CARD_BG)
    pdf.set_draw_color(*TEAL)
    pdf.rect(15, y0, 180, 22, "FD")
    pdf.set_xy(19, y0 + 3)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*TEAL)
    pdf.cell(12, 5.5, num)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 5.5, title, ln=True)
    pdf.set_x(31)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*LIGHT_GREY)
    pdf.multi_cell(160, 4.8, body)
    pdf.spacer(2)

# Final closing note
pdf.spacer(4)
pdf.set_fill_color(*TEAL)
pdf.rect(15, pdf.get_y(), 180, 0.4, "F")
pdf.spacer(4)
pdf.set_left_margin(20)
pdf.set_right_margin(20)
pdf.set_font("Helvetica", "I", 9)
pdf.set_text_color(*LIGHT_GREY)
pdf.multi_cell(0, 6,
    "A null result is a real result. We now know that VIX and OAS do not help M2, "
    "that M1 is robust across market regimes, and that the binding constraint is "
    "sample size and feature orthogonality. This diagnostic is what makes the "
    "next iteration faster and more targeted. The infrastructure is built. "
    "The next question is sharper.")
pdf.set_left_margin(10)
pdf.set_right_margin(10)

# ---------------------------------------------------------------------------
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
pdf.output(str(OUT_PATH))
print(f"PDF saved: {OUT_PATH}")
