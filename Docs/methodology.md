# Methodology

Every score in MarketSignal is grounded in a published academic paper. The formulas are operationalizations of established economic models, not invented scoring systems.

## Economic Research Framework

**Bresnahan and Reiss (1991)** established that markets support a finite number of competitors before entry becomes unprofitable. MarketSignal uses their entry threshold logic to set competitor density limits by firm type: 4 per 10,000 residents for an Independent, 8 for a Regional Chain, 12 for a National Brand — reflecting brand equity's ability to support entry into denser markets.

**Berry and Waldfogel (1999)** demonstrated that market size, measured by population, determines how many differentiated competitors a market can sustain. Used directly in the Expansion Score's population thresholds.

**Bernanke and Gertler (1995)** showed monetary policy transmits to firm-level investment through the credit channel — specifically the prime lending rate. Included in the Expansion Score because borrowing cost directly affects a small firm's ability to finance expansion.

**Huff (1964)** developed the gravity model of retail trade, predicting consumer patronage as a function of store attractiveness and market size. Adapted as the Local Market Score, weighting population and income equally as Huff's model specifies.

**Dube, Hitsch, and Rossi (2010)** demonstrated that price sensitivity varies systematically with local income. Used to calculate the income premium factor in the Pricing Power Score, via the ratio of local median income to the national median ($74,580).

**Ailawadi and Keller (2004)** showed brand equity allows firms to charge a price premium above what local income alone would support. Firm type is used as a proxy for brand equity: National Brands receive 28 of 40 possible brand points, Regional Chains 16, Independents 0 — with local income providing additional support points in each case.

**Anderson and Magruder (2012)** demonstrated that online review counts can serve as proxies for market share and consumer attention — the basis for the Proxy HHI (see below).

**Nakamura and Steinsson (2013)** showed monetary policy has heterogeneous real effects across regions and conditions. Used in the Market Entry Window, weighting the Federal Funds Rate as the primary transmission variable alongside GDP growth, consumer sentiment, and inflation.

## The Four Scores

Weights within each score reflect the relative explanatory power each factor holds in its source paper — the variable most central to a paper's core finding gets the highest weight.

### Expansion Score (0–100)
*Should this firm enter this market at all?*
- Competitor density — 40 pts (Bresnahan & Reiss, 1991 — the paper's central finding)
- Local median income — 20 pts
- Local unemployment — 15 pts
- Population — 15 pts (Berry & Waldfogel, 1999)
- Prime rate — 10 pts (Bernanke-Gertler credit channel)

A score above 70 supports expansion; below 45 means do not expand.

### Local Market Score (0–100)
*How attractive is this specific neighborhood?*
- Population — 35 pts
- Local income — 35 pts *(equal weight to population, per Huff's gravity formula treating size and purchasing power as co-equal)*
- Local unemployment — 20 pts
- Disposable income growth — 10 pts

### Pricing Power Score (0–100)
*How much can this firm charge?*
- Brand equity by firm type — 40 pts (Ailawadi & Keller, 2004 — the strongest predictor of sustained price premiums)
- Income premium vs. national median — 25 pts (Dube, Hitsch & Rossi, 2010)
- Competition level (Google Places) — 20 pts
- PCE growth — 10 pts
- Retail sales growth — 5 pts

### Market Entry Window (Open / Cautious / Closed)
*Is right now a good time to enter?*
- National macro timing — 40 pts (Nakamura & Steinsson, 2013: Fed rate, GDP growth, sentiment, inflation)
- Local market readiness — 35 pts (Huff, 1964: local income, unemployment, population)
- Competitive landscape — 25 pts (Bresnahan-Reiss density by firm type + Proxy HHI)

Open requires above 70 points; Cautious is 45–70; Closed is below 45.

## Proxy HHI Methodology

The traditional Herfindahl-Hirschman Index needs real revenue or sales data per competitor, which isn't publicly available for most small businesses. MarketSignal follows Anderson and Magruder (2012), who showed consumer attention (measured via online engagement) correlates strongly with market share, and treats each competitor's Google Places review count as a proxy for accumulated consumer attention and relative market presence.

**Calculation:**
1. For each competitor within a 2-mile radius, retrieve total review count.
2. Each competitor's market-share proxy = their review count ÷ total review count across all competitors.
3. Proxy HHI = Σ(share²) × 10,000.

Markets below 1,500 are unconcentrated; 1,500–2,500 are moderately concentrated; above 2,500 are highly concentrated (standard DOJ thresholds).

**Example:** For ZIP 45219, a paginated Google Places search returned 56 coffee shop competitors within two miles — a density of 28.98 per 10,000 residents — but a Proxy HHI of 1,089, classified as *unconcentrated*. 56 competitors sounds severely saturated, but the HHI tells a different story: no single player dominates, attention is fragmented, and there may be room to differentiate despite the high raw count. The agent uses both signals together — density for the entry threshold test, HHI for the concentration test — rather than relying on either alone.

## Agentic Behavior

MarketSignal qualifies as agentic in three specific ways:

1. **It autonomously selects its analytical framework based on inputs.** Specifying National Brand vs. Independent silently changes which Bresnahan-Reiss thresholds, Ailawadi-Keller brand premiums, and Dube-Hitsch-Rossi pricing rules apply — the user never specifies which formulas to use.
2. **It uses tools autonomously in sequence.** It calls FRED, the Federal Reserve website, the Census Bureau, and Google Places without user instruction, aggregates the results, and synthesizes a unified recommendation.
3. **Its reasoning is visible.** The Agent Reasoning Trace panel shows the step-by-step logic — how it read the Fed signal, what the macro indicators mean for this firm, how it assessed the local market, how it applied the pricing model, and how it reached the final recommendation — grounded in the specific numbers from that run, not generic statements.

## Technical Implementation

All scoring logic lives in a single deterministic function, `calculate_scores_deterministic()`, which can be tested independently of the dashboard — the same inputs always produce the same scores, and the language model never touches this calculation. The Google Places fetch paginates through up to three pages of results (60 total) to avoid the default 20-result cap that would undercount competitors in dense markets.
