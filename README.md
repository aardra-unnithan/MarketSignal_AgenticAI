# MarketSignal

**An agentic AI platform for market expansion and pricing decisions**

*Kautz-Uible Economics Institute AI Competition — Spring 2026 — Team Ctrl+Agents*

---

## The Problem

Every day, business owners and executives face a deceptively simple question: *should we expand into this market, and what should we charge?* Answering it properly means synthesizing Federal Reserve monetary policy, national macroeconomic conditions, neighborhood-level demographics, and local competitive structure — all at once. Normally that takes a team of analysts hours of work, and even then the results are often too slow to act on and too generic to be useful for one specific location.

MarketSignal does this autonomously, in under 30 seconds, grounded in live data and published economic research. Enter a business type, price tier, firm type, and a target ZIP code — the agent reads the current macroeconomic environment, reads what the Federal Reserve just said, pulls real neighborhood data from the U.S. Census Bureau, fetches live competitor data from Google Places, and produces a structured recommendation with four research-grounded scores, a SWOT analysis, and a board-level advisory memo. Every output traces back to a specific data source or academic paper.

## How It Works: Brain and Muscle

MarketSignal separates number-crunching from language-based reasoning into two layers.

- **The Muscle** is pure Python. It fetches data from external sources and computes all four scores using fixed, research-based rules. It never guesses — the same inputs always produce the same scores, and the language model never touches the numbers.
- **The Brain** is Claude. It receives everything the Muscle collected and computed, then does what requires judgment: classifying the tone of the Fed's latest statement, writing the SWOT analysis, and producing the board advisory memo — always citing the actual numbers, never inventing them.

This separation means every score is fully auditable, and the agent can't quietly inflate a recommendation just because a business type sounds appealing.

## The Four Scores

- **Expansion Score (0–100)** — should this firm enter this market at all? Weighted primarily on competitor density (Bresnahan & Reiss, 1991), plus local income, unemployment, population, and prime rate.
- **Local Market Score (0–100)** — how attractive is this specific neighborhood? Built on Huff's (1964) gravity model, weighting population and income equally.
- **Pricing Power Score (0–100)** — how much can this firm charge? Driven mainly by brand equity by firm type (Ailawadi & Keller, 2004), plus income premium and competitive structure.
- **Market Entry Window (Open / Cautious / Closed)** — is now a good time? Combines national macro timing (Nakamura & Steinsson, 2013), local market readiness, and the competitive landscape.

Full weighting rationale and the academic paper behind every factor is in [`docs/methodology.md`](docs/methodology.md).

## Firm-Type Differentiation

The same market produces different correct answers for different firms. An Independent coffee shop facing 28 competitors per 10,000 residents is in a severely saturated market (its Bresnahan-Reiss threshold is 4 per 10,000); a National Brand facing the same density is fine (its threshold is 12), because brand equity changes what "saturated" means. MarketSignal applies different entry thresholds, brand premiums, and income-sensitivity parameters depending on whether the firm is Independent, a Regional Chain, or a National Brand.

## Data Sources

| Source | What it provides |
|---|---|
| **FRED API** | 8 live macroeconomic series: Fed Funds Rate, Retail Sales, Consumer Sentiment, CPI, GDP, PCE growth, Prime Rate, Disposable Income |
| **federalreserve.gov** | Scrapes the latest FOMC statement for tone analysis, with a live source link on the dashboard |
| **U.S. Census Bureau (2023 ACS 5-Year)** | Median household income, population, and unemployment rate for the target ZIP |
| **Google Places API** | Live competitor count, ratings, and reviews within a 2-mile radius, paginated up to 60 results |
| **Zillow rent data** | Local rent comparables (`zillow_rent.csv`) |

## Proxy HHI

Traditional market-concentration analysis (HHI) needs real revenue data, which isn't publicly available for small businesses. MarketSignal follows Anderson & Magruder (2012) and uses each competitor's Google review count as a proxy for market share: `HHI = Σ(share²) × 10,000`. This catches cases plain competitor counts miss — e.g. a ZIP code with 56 coffee shops can still be *unconcentrated* (HHI ≈ 1,089) if no single competitor dominates attention, meaning there's more room to differentiate than the raw count suggests.

## Scenario Simulator

An 11-slider simulator (8 national indicators, 3 local) lets you ask "what if" questions — e.g. *how does the recommendation change if the Fed raises rates to 6%, or local unemployment drops to 3%* — and recalculates all four scores deterministically without another API call.

## Tech Stack

- **Dashboard:** Plotly Dash
- **Reasoning:** Claude API (Anthropic)
- **Data:** FRED, Census Bureau, Google Places, Zillow
- **Deployment:** Render

## Getting Started

**1. Clone and enter the project**
```bash
git clone <this-repo-url>
cd MarketSignal_AgenticAI
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up your API keys**

Copy `.env.example` to `.env` and fill in your own keys:
```
ANTHROPIC_API_KEY=your_key_here
FRED_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```
`.env` is gitignored — never commit real keys.

**4. Run it**
```bash
python app.py
```
Open `http://127.0.0.1:8050` in your browser.

## Limitations

MarketSignal is a decision-support tool, not a definitive answer:

- **Proxy HHI is an approximation** — review counts track attention, not actual revenue or foot traffic.
- **Scoring weights are theoretically derived, not empirically calibrated** — they reflect each source paper's own hierarchy of factors, not a regression against real outcomes.
- **The model is static** — it scores current conditions, not trajectory (a gentrifying low-income neighborhood scores the same as a stable one).
- **Firm-type classification is simplified** — three categories capture brand equity but not supply chain, loyalty programs, or other firm-specific advantages.
- **Google Places caps at 60 results** — in extremely dense markets, true competitor count may be undercounted.

## Future Work

- **Empirical calibration** of scoring weights against a real dataset of market-entry outcomes (the main blocker: no clean public dataset of this kind currently exists).
- **Establishment-level market data** (e.g. SafeGraph, Placer.ai, BLS QCEW) in place of the review-count HHI proxy.
- **Temporal dynamics** — multi-year Census trends and building-permit data to distinguish a stable neighborhood from a rapidly changing one.
- **Backtesting** historical recommendations against actual outcomes, pending access to longitudinal expansion data.

## Team

Built by **Team Ctrl+Agents** for the Kautz-Uible Economics Institute AI Competition, Spring 2026 — 1st runner-up.
