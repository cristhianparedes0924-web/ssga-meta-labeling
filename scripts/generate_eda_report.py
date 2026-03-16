"""Generate a simple PDF report explaining EDA and Feature Engineering results."""

from pathlib import Path
from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
EDA_DIR = ROOT / "reports" / "assets" / "secondary_eda"
OUT_PATH = ROOT / "reports" / "results" / "eda_feature_engineering_report.pdf"

DARK_BG   = (18, 32, 47)
TEAL      = (0, 200, 170)
WHITE     = (255, 255, 255)
LIGHT_GREY = (220, 225, 230)
YELLOW    = (255, 200, 50)
SOFT_BLUE = (52, 152, 219)


class Report(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*LIGHT_GREY)
        self.cell(0, 8, f"Brandeis MSF 2026  |  SSGA Meta-Labeling Project  |  Page {self.page_no()}", align="C")

    def cover_page(self):
        self.add_page()
        # Dark background
        self.set_fill_color(*DARK_BG)
        self.rect(0, 0, 210, 297, "F")

        # Teal accent bar top
        self.set_fill_color(*TEAL)
        self.rect(0, 0, 210, 4, "F")

        # Title
        self.set_y(80)
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(*WHITE)
        self.cell(0, 14, "EDA & Feature Engineering", align="C", ln=True)

        self.set_font("Helvetica", "", 16)
        self.set_text_color(*TEAL)
        self.cell(0, 10, "What We Did and What We Found", align="C", ln=True)

        self.ln(10)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(*LIGHT_GREY)
        self.cell(0, 8, "M2 Secondary Meta-Labeling  |  Branch: sam_M2", align="C", ln=True)
        self.cell(0, 8, "Raviv - Samweg - Cristian  |  Week of Mar 16, 2026", align="C", ln=True)

        # Divider
        self.ln(20)
        self.set_fill_color(*TEAL)
        self.rect(30, self.get_y(), 150, 1, "F")
        self.ln(10)

        # Two boxes side by side
        self._summary_box(15, self.get_y(), 85, 60, "FEATURE ENGINEERING",
            "Turning raw VIX and OAS data\ninto 10 meaningful inputs\nthe M2 model can learn from.")
        self._summary_box(110, self.get_y() - 60 + 60, 85, 60, "EDA",
            "Testing whether those features\nactually relate to M1 winning\nbefore building any model.")

        self.ln(80)
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(*LIGHT_GREY)
        self.cell(0, 8, "Key finding: neither VIX nor OAS significantly predicts M1 performance", align="C")

    def _summary_box(self, x, y, w, h, title, body):
        self.set_fill_color(30, 50, 70)
        self.set_draw_color(*TEAL)
        self.rect(x, y, w, h, "FD")
        self.set_xy(x + 3, y + 4)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*TEAL)
        self.cell(w - 6, 7, title, ln=True)
        self.set_x(x + 3)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*LIGHT_GREY)
        self.multi_cell(w - 6, 6, body)

    def section_divider(self, title, subtitle=""):
        self.add_page()
        self.set_fill_color(*DARK_BG)
        self.rect(0, 0, 210, 297, "F")
        self.set_fill_color(*TEAL)
        self.rect(0, 0, 210, 4, "F")

        self.set_y(110)
        self.set_font("Helvetica", "B", 24)
        self.set_text_color(*WHITE)
        self.cell(0, 12, title, align="C", ln=True)
        if subtitle:
            self.ln(4)
            self.set_font("Helvetica", "", 13)
            self.set_text_color(*TEAL)
            self.cell(0, 8, subtitle, align="C", ln=True)

    def content_page(self, title, body_paragraphs, image_path=None, image_caption="", result_box=None):
        self.add_page()
        self.set_fill_color(*DARK_BG)
        self.rect(0, 0, 210, 297, "F")
        self.set_fill_color(*TEAL)
        self.rect(0, 0, 210, 3, "F")

        # Page title
        self.set_y(8)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(*WHITE)
        self.cell(0, 9, title, align="C", ln=True)

        self.set_fill_color(*TEAL)
        self.rect(20, self.get_y(), 170, 0.5, "F")
        self.ln(4)

        # Body text
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*LIGHT_GREY)
        self.set_left_margin(15)
        self.set_right_margin(15)

        for para in body_paragraphs:
            if para.startswith("**") and para.endswith("**"):
                self.set_font("Helvetica", "B", 10)
                self.set_text_color(*TEAL)
                self.multi_cell(0, 6, para.strip("**"))
                self.set_font("Helvetica", "", 10)
                self.set_text_color(*LIGHT_GREY)
            else:
                self.multi_cell(0, 6, para)
            self.ln(2)

        # Result box
        if result_box:
            self.ln(2)
            self.set_fill_color(20, 40, 60)
            self.set_draw_color(*YELLOW)
            box_y = self.get_y()
            self.rect(15, box_y, 180, 18, "FD")
            self.set_xy(18, box_y + 3)
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(*YELLOW)
            self.cell(30, 6, "RESULT:", ln=False)
            self.set_font("Helvetica", "", 9)
            self.set_text_color(*WHITE)
            self.multi_cell(150, 6, result_box)
            self.ln(4)

        # Image
        if image_path and Path(image_path).exists():
            img_y = self.get_y() + 2
            available = 280 - img_y
            img_h = min(available, 75)
            self.image(str(image_path), x=10, y=img_y, w=190, h=img_h)
            if image_caption:
                self.set_y(img_y + img_h + 1)
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(*LIGHT_GREY)
                self.cell(0, 5, image_caption, align="C")

        self.set_left_margin(10)
        self.set_right_margin(10)

    def summary_page(self):
        self.add_page()
        self.set_fill_color(*DARK_BG)
        self.rect(0, 0, 210, 297, "F")
        self.set_fill_color(*TEAL)
        self.rect(0, 0, 210, 3, "F")

        self.set_y(10)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*WHITE)
        self.cell(0, 10, "Summary & Conclusions", align="C", ln=True)
        self.set_fill_color(*TEAL)
        self.rect(20, self.get_y(), 170, 0.5, "F")
        self.ln(6)

        sections = [
            ("FEATURE ENGINEERING", TEAL, [
                "We took raw monthly VIX and OAS data and engineered 10 features:",
                "  VIX: level z-score, change z-score, 6-month trend, high-regime flag, rising flag",
                "  OAS: level z-score, change z-score, 6-month trend, wide-regime flag, widening flag",
                "All z-scores used only past data - no future information leaked into any feature.",
            ]),
            ("EDA FINDINGS", SOFT_BLUE, [
                "Chi-square test (VIX vs M1 win rate):  p = 0.8882  - no significant relationship",
                "Chi-square test (OAS vs M1 win rate):  p = 0.7730  - no significant relationship",
                "T-test on returns by VIX regime:       p = 0.1597  - not significant",
                "T-test on returns by OAS regime:       p = 0.0787  - not significant",
                "All feature correlations with meta_label were below 0.11 - essentially zero.",
            ]),
            ("M2 MODEL RESULTS", YELLOW, [
                "Logistic regression trained walk-forward on 60-month expanding windows.",
                "OOS accuracy: 53.9%  |  ROC-AUC: 0.523  (random = 0.50)",
                "M1 baseline Sharpe: 1.31  ->  M2-filtered Sharpe: 1.18  (got worse)",
                "M2-rejected trades had higher win rate than M2-approved trades.",
            ]),
            ("CONCLUSION", (200, 80, 80), [
                "VIX and OAS features are theoretically sound but carry no detectable signal",
                "for predicting M1 wins over 28 years of monthly data.",
                "Most likely explanation: M1's own indicators (SPX trend, credit vs rates,",
                "risk breadth) already absorb the stress information that VIX captures.",
                "Next step: find features that are orthogonal to M1's existing inputs.",
            ]),
        ]

        for title, color, lines in sections:
            box_y = self.get_y()
            self.set_fill_color(25, 45, 65)
            self.set_draw_color(*color)
            self.rect(12, box_y, 186, 6 + len(lines) * 6, "FD")
            self.set_xy(15, box_y + 2)
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(*color)
            self.cell(0, 5, title, ln=True)
            self.set_font("Helvetica", "", 8.5)
            self.set_text_color(*LIGHT_GREY)
            for line in lines:
                self.set_x(15)
                self.cell(0, 6, line, ln=True)
            self.ln(4)


# ---------------------------------------------------------------------------
# Build the PDF
# ---------------------------------------------------------------------------
pdf = Report()
pdf.set_auto_page_break(auto=True, margin=15)

# Cover
pdf.cover_page()

# ---- SECTION 1: FEATURE ENGINEERING ----
pdf.section_divider("PART 1: FEATURE ENGINEERING", "How we turned raw data into model inputs")

pdf.content_page(
    title="What is Feature Engineering?",
    body_paragraphs=[
        "**The problem with raw data**",
        "A model cannot learn from raw numbers like 'VIX = 28.3' alone. It has no "
        "context - is 28.3 high or low? Is it getting worse or better? Has it been "
        "like this for months, or did it just spike?",
        "**Feature engineering solves this**",
        "We transform raw VIX and OAS data into 5 meaningful features each - ones that "
        "carry context the model can actually use.",
        "**The no-lookahead rule**",
        "Every single feature is computed using only past data. The z-score at March 2020 "
        "uses only data from before March 2020. This ensures the model never cheats "
        "by seeing future information during training.",
    ],
    result_box="10 features engineered from 2 raw data sources (VIX + OAS). Zero lookahead bias."
)

pdf.content_page(
    title="VIX Features - Measuring Market Fear",
    body_paragraphs=[
        "**What is VIX?**",
        "VIX is the 'fear index' - it measures how much volatility investors expect in "
        "the stock market. High VIX = panic and uncertainty. Low VIX = calm markets.",
        "**The 5 features we built from VIX:**",
        "  vix_level_z     How high is VIX vs its own history? (+1.8 = 1.8 std devs above normal)",
        "  vix_change_z    How much did VIX move this month, vs typical monthly moves?",
        "  vix_trend       Is VIX above or below its 6-month rolling average?",
        "  vix_high_regime Is VIX above its all-time expanding median? (1 = yes, 0 = no)",
        "  vix_rising      Did VIX go up this month? (1 = yes, 0 = no)",
        "**Why these specific features?**",
        "Together they answer: where is fear (level), how fast is it changing (change), "
        "which direction is it trending (trend), and are we in a sustained fear regime "
        "vs a one-off spike (regime flag).",
    ],
    result_box="5 VIX features capture: level, momentum, trend, regime, and direction of market fear."
)

pdf.content_page(
    title="OAS Features - Measuring Credit Stress",
    body_paragraphs=[
        "**What is OAS?**",
        "OAS (Option-Adjusted Spread) measures the extra yield investors demand to hold "
        "corporate bonds instead of safe government bonds. Wide OAS = credit stress and "
        "illiquidity. Tight OAS = easy credit conditions.",
        "**The 5 features we built from OAS:**",
        "  oas_level_z     How wide are spreads vs their own history?",
        "  oas_change_z    How much did spreads move this month?",
        "  oas_trend       Are spreads above or below their 6-month average?",
        "  oas_wide_regime Are spreads in a historically wide (stressed) regime?",
        "  oas_widening    Did spreads widen this month? (credit getting worse?)",
        "**The hypothesis for using OAS**",
        "When credit spreads are wide and widening, it signals financial stress that "
        "might cause M1's signals to be less reliable - similar logic to VIX.",
    ],
    result_box="5 OAS features capture: level, momentum, trend, regime, and direction of credit stress."
)

# ---- SECTION 2: EDA ----
pdf.section_divider("PART 2: EDA", "Testing whether the features actually matter")

pdf.content_page(
    title="What is EDA and Why Did We Do It?",
    body_paragraphs=[
        "**EDA = Exploratory Data Analysis**",
        "Before building any model, we asked a simple question: does VIX or OAS actually "
        "relate to whether M1's signals win or lose? If they don't, adding them to M2 "
        "will only add noise.",
        "**The hypothesis we were testing**",
        "M1 follows market trends. In calm markets (low VIX, tight OAS), trends are "
        "stable and M1 should win more often. In fearful, stressed markets (high VIX, "
        "wide OAS), trends break down and M1 should win less often.",
        "**What we actually measured**",
        "We split 138 M1 events into two groups - high and low VIX - and compared "
        "M1's win rate in each group. We then ran statistical tests to check if any "
        "difference was real or just random chance.",
        "**The standard for 'real'**",
        "A p-value below 0.05 means there is less than 5% chance the difference is "
        "random. A p-value above 0.05 means we cannot rule out that it is just noise.",
    ],
    result_box="6 EDA charts were produced. All pointed to the same conclusion."
)

pdf.content_page(
    title="Chart 1 - Does VIX Predict M1 Wins?",
    body_paragraphs=[
        "**What this chart shows**",
        "M1's win rate split by Low VIX (calm markets) vs High VIX (fearful markets), "
        "across all 138 events. Also broken out by BUY and SELL signals separately.",
        "**The numbers**",
        "Low VIX:  67.9% win rate (n=53 events)",
        "High VIX: 70.6% win rate (n=85 events)",
        "The difference is only 2.7 percentage points - and it goes the wrong direction. "
        "M1 actually wins slightly MORE when VIX is high.",
        "**The statistical test**",
        "Chi-square p-value = 0.8882. This means there is an 89% chance this small "
        "difference is pure random noise. Far above the 0.05 threshold needed to call "
        "it meaningful.",
    ],
    image_path=EDA_DIR / "1_win_rate_by_vix_regime.png",
    image_caption="Chart 1: M1 Win Rate by VIX Regime. Chi-square p=0.8882 - no significant relationship.",
    result_box="VIX regime has no statistically significant effect on M1's win rate (p=0.8882)."
)

pdf.content_page(
    title="Chart 2 - Does OAS Predict M1 Wins?",
    body_paragraphs=[
        "**What this chart shows**",
        "Same test but for OAS. M1's win rate in Tight OAS (easy credit) vs Wide OAS "
        "(credit stress) markets.",
        "**The numbers**",
        "Tight OAS: 66.7% win rate (n=42 events)",
        "Wide OAS:  70.8% win rate (n=96 events)",
        "Again, the direction is backwards - M1 wins more when credit is stressed. "
        "The hypothesis said stressed markets should hurt M1, but the data disagrees.",
        "**The statistical test**",
        "Chi-square p-value = 0.7730. 77% chance this is random noise. "
        "Not significant.",
    ],
    image_path=EDA_DIR / "2_win_rate_by_oas_regime.png",
    image_caption="Chart 2: M1 Win Rate by OAS Regime. Chi-square p=0.7730 - no significant relationship.",
    result_box="OAS regime has no statistically significant effect on M1's win rate (p=0.7730)."
)

pdf.content_page(
    title="Chart 3 - VIX and OAS Combined",
    body_paragraphs=[
        "**What this chart shows**",
        "A heatmap of M1's win rate and average return across all four combinations "
        "of VIX and OAS regimes simultaneously.",
        "**The most striking finding**",
        "The supposedly 'worst' regime - High VIX + Wide OAS (fearful market, stressed "
        "credit) - actually has the BEST performance: 72.7% win rate and +1.02% avg "
        "monthly return.",
        "The supposedly 'best' regime - Low VIX + Tight OAS (calm, easy credit) - "
        "had only 70.6% win rate and just +0.31% avg monthly return.",
        "**What this means**",
        "M1 is not just robust to market stress - it may actually perform better during "
        "stress, possibly because its SELL signals correctly call downturns. This makes "
        "VIX and OAS useless as filters.",
    ],
    image_path=EDA_DIR / "3_joint_regime_heatmap.png",
    image_caption="Chart 3: Joint VIX x OAS heatmap. Stressed regimes show equal or better M1 performance.",
    result_box="High VIX + Wide OAS (worst expected regime) delivered the best actual M1 performance."
)

pdf.content_page(
    title="Chart 4 - Feature Correlations with M1 Wins",
    body_paragraphs=[
        "**What this chart shows**",
        "The correlation of every single feature with the meta_label (did M1 win?). "
        "A perfect predictor would show +1.0 or -1.0. A useless feature shows 0.",
        "**The key finding**",
        "The highest correlation is only +0.108 (trailing average return) - essentially "
        "zero predictive power. All VIX and OAS features sit between -0.03 and +0.07.",
        "Importantly, none of the correlations have a * or ** significance marker, "
        "meaning none are statistically significant at p<0.05.",
        "**What this confirms**",
        "No individual feature - including VIX and OAS - has a meaningful linear "
        "relationship with M1 winning. The signal simply is not there in the data.",
    ],
    image_path=EDA_DIR / "4_feature_correlations.png",
    image_caption="Chart 4: All feature correlations with meta_label are below 0.11 - none statistically significant.",
    result_box="Highest correlation with M1 wins: 0.108. All VIX/OAS features below 0.07."
)

pdf.content_page(
    title="Charts 5 & 6 - Return Distributions and Regime Gate",
    body_paragraphs=[
        "**Chart 5: Return distributions by regime (t-test)**",
        "Compared the full distribution of M1 monthly returns in High vs Low VIX. "
        "High VIX averaged +0.93%, Low VIX averaged +0.41% - but the t-test p-value "
        "was 0.1597. Not significant. The distributions overlap too much.",
        "**Chart 6: What if we used VIX/OAS as a simple gate?**",
        "Simulated trading only in 'favorable' regimes (Low VIX + Tight OAS) vs "
        "trading only in 'stressful' regimes (High VIX + Wide OAS):",
        "  All events (no filter):      Sharpe 1.19, Win Rate 69.6%",
        "  Low VIX only:                Sharpe 0.84, Win Rate 67.9%  (worse)",
        "  Low VIX + Tight OAS:         Sharpe 0.68, Win Rate 70.6%  (much worse)",
        "  High VIX + Wide OAS:         Sharpe 1.49, Win Rate 72.7%  (best!)",
        "Filtering out stressed markets actually destroyed value. The 'bad' regime "
        "was the best-performing one.",
    ],
    image_path=EDA_DIR / "6_regime_gate_simulation.png",
    image_caption="Chart 6: Regime gate simulation. Filtering by 'favorable' VIX/OAS conditions hurt performance.",
    result_box="Simple regime gate using VIX+OAS reduces Sharpe from 1.19 to 0.68. No improvement found."
)

# ---- Summary ----
pdf.summary_page()

# Save
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
pdf.output(str(OUT_PATH))
print(f"PDF saved to: {OUT_PATH}")
