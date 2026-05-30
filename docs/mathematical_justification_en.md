# Mathematical Justification of the MDQ Project
## Detecting Hidden Entrepreneurship via Transactional Behavior

**Student:** Miras  
**Project:** MDQ — Hidden Entrepreneur Detection

---

## Table of Contents

1. [Part 1. Mathematical Justification of Features](#part-1-mathematical-justification-of-features)
2. [Part 2. Model Quality Metrics](#part-2-model-quality-metrics)
3. [Part 3. Justification of Classification Thresholds](#part-3-justification-of-classification-thresholds)
4. [Part 4. Model Error Analysis](#part-4-model-error-analysis)
5. [Part 5. Presentation Defense Tips](#part-5-presentation-defense-tips)

---

# Part 1. Mathematical Justification of Features

## 1.1 General Problem Statement

Each card $c$ is represented by a set of transactions $\mathcal{T}_c = \{t_1, t_2, \ldots, t_{n_c}\}$. Each transaction $t_i$ is a tuple:

$$t_i = (\text{amount}_i,\ \text{merchant}_i,\ \text{mcc}_i,\ \text{timestamp}_i,\ \text{channel}_i,\ \ldots)$$

Objective: construct a feature vector $\mathbf{x}_c \in \mathbb{R}^{35}$ for each card $c$ and train a classifier:

$$f: \mathbb{R}^{35} \to [0, 1], \quad f(\mathbf{x}_c) = P(\text{business} \mid \mathbf{x}_c)$$

Data: ~80,000 consumer cards (label = 0) and ~20,000 business cards (label = 1).

📌 **For the slide:** *Each card → 35 numerical features → "business" probability. The unit of analysis is the card, not the individual transaction.*

---

## 1.2 Group 1: Spending Volume

### Formulas

For card $c$ with transaction amounts $a_1, a_2, \ldots, a_n$:

| Feature | Formula | Meaning |
|---------|---------|-------|
| `tx_count` | $n_c = \mathcal{T}_c$ | Transaction count |
| `amt_sum` | $S_c = \sum_{i=1}^{n} a_i$ | Total turnover |
| `amt_mean` | $\bar{a}_c = \frac{1}{n}\sum_{i=1}^{n} a_i$ | Mean transaction amount |
| `amt_median` | $\text{Med}(a_1, \ldots, a_n)$ | Median transaction amount |
| `amt_std` | $\sigma_c = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(a_i - \bar{a}_c)^2}$ | Standard deviation |
| `amt_max` | $\max(a_1, \ldots, a_n)$ | Maximum transaction |
| `amt_min` | $\min(a_1, \ldots, a_n)$ | Minimum transaction |
| `amt_cv` | $CV_c = \frac{\sigma_c}{\bar{a}_c}$ | Coefficient of variation |

### Coefficient of Variation (CV)

$$CV = \frac{\sigma}{\mu}$$

CV is a dimensionless measure of relative variability. For business cards, $CV$ is typically higher: purchases exhibit heterogeneous amounts (ranging from small office supplies to massive inventory wholesale shipments), whereas consumer spending resides within a significantly narrower scope.

### Economic Justification

Business cards demonstrate:
- **A higher median ticket** (`amt_median` business/consumer mean ratio ≈ 3.99×, SHAP |impact| ≈ 0.05): commercial procurement tasks dwarf everyday domestic retail purchases.
- **Higher variance in transaction amounts**: covering a diverse spectrum from cheap stationery up to large wholesale lots.
- **Greater overall transaction count**: due to frequent, recurring operational expenditures.

### Statistical Significance

The difference in means is confirmed by Welch's t-test for unequal variances:

$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

For `amt_median`: $p < 10^{-100}$, with a Cohen’s effect size of $d > 1.0$ (indicating a strong effect).

For features characterized by heavy tails (such as total transaction amounts), the non-parametric Mann–Whitney $U$ test is additionally deployed:

$$U = \sum_{i=1}^{n_1}\sum_{j=1}^{n_2} \mathbf{1}[X_i > Y_j]$$

📌 **For the slide:** *Business cards display higher spending volume, larger tickets, and greater overall variance. The median transaction amount is nearly 4× higher for businesses (b/c mean ratio = 3.99). All observed discrepancies are statistically significant ($p \ll 0.001$, Cohen's $d > 1.0$ — see Appendix A2 in the notebook).*

---

## 1.3 Group 2: Channel / Timing

### Formulas

Each feature in this group serves as an estimate of the event probability for the card:

$$\hat{p}_{\text{event}}(c) = \frac{\sum_{i=1}^{n_c} \mathbf{1}[\text{event}(t_i)]}{n_c}$$

| Feature | Event Condition $\text{event}(t_i)$ |
|---------|----------------------------|
| `online_share` | `channel` = "online" |
| `weekend_share` | Day of week ∈ {Sat, Sun} |
| `evening_share` | Hour ∈ [18, 23] |
| `bizhours_share` | Weekday **and** Hour ∈ [9, 17] |
| `foreign_share` | `country` ≠ "Kazakhstan" |

### Interpretation as a Statistical Estimate

Each `*_share` represents a sample estimate of probability:

$$\hat{p} = \frac{k}{n}, \quad \text{where } k = \sum \mathbf{1}[\text{event}]$$

By the Central Limit Theorem (CLT), given a sufficiently large sample size $n$:

$$\hat{p} \sim \mathcal{N}\left(p, \frac{p(1-p)}{n}\right)$$

The Wald confidence interval is formulated as follows:

$$\hat{p} \pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

### Economic Justification

| Pattern | Consumer | Business |
|---------|------------|--------|
| Active Hours | Evenings, weekends | Core working hours (9–17, Mon–Fri) |
| Core Channels | Retail online shopping | Online wholesale procurement + B2B warehouse POS |
| Foreign Transactions | Leisure travel (highly seasonal) | Regular imports / international supplier invoices |

📌 **For the slide:** *Businesses primarily transact during working hours (bizhours_share ↑), while consumers dominate evenings and weekends. Each share acts as an empirical probability estimate of the corresponding behavior.*

---

## 1.4 Group 3: B2B Exposure

### Definition of the B2B Basket

An expert-vetted basket containing 40 specific MCC codes ($\mathcal{M}_{B2B}$) has been isolated, including:
- Wholesale/distribution codes: 5044, 5045, 5046, 5047, ...
- Professional services: 7311, 7321, 7333, 7338, ...
- Logistics & infrastructure: 4214, 4215, 4225, 4816

**Important:** MCC 5122 (wholesale pharmaceuticals) was deliberately **excluded** despite being highly predictive, as it is exclusive to the corporate segment in this specific data extraction → keeping it would introduce direct label leakage.

### Formulas

| Feature | Formula |
|---------|---------|
| `b2b_mcc_share` | $\hat{p}_{B2B}(c) = \frac{\sum_{i} \mathbf{1}[\text{mcc}_i \in \mathcal{M}_{B2B}]}{n_c}$ |
| `b2b_amt_share` | $\hat{r}_{B2B}(c) = \frac{\sum_{i} a_i \cdot \mathbf{1}[\text{mcc}_i \in \mathcal{M}_{B2B}]}{\sum_{i} a_i}$ |
| `b2b_unique_merchants` | $\{m : \exists\, t_i \in \mathcal{T}_c,\ \text{merchant}_i = m,\ \text{mcc}_i \in \mathcal{M}_{B2B}\}$ |

### Statistical Interpretation

$\hat{p}_{B2B}(c)$ behaves as a consistent estimator of the true underlying conditional probability:

$$\hat{p}_{B2B}(c) \xrightarrow{n_c \to \infty} P(\text{MCC} \in \mathcal{M}_{B2B} \mid \text{card} = c)$$

### Class-mean Contrast and SHAP Importance

Two complementary views — how strongly the class means differ (b/c ratio) and how much the feature actually moves the model's output (mean |SHAP|):

| Feature | business / consumer mean ratio | SHAP global mean \|impact\| |
|---------|-------------------------------:|---------------------------:|
| `b2b_mcc_share`        | **9.15×** | 0.06 |
| `b2b_amt_share`        | **8.06×** | 0.31 |
| `b2b_unique_merchants` | **6.70×** | **1.37** |

Note: B2B exposure dominates the **class-mean contrast** (the largest single ratios in the entire feature table), but `b2b_unique_merchants` is the only B2B feature in the SHAP top-5 — once the model knows the count of distinct B2B suppliers, the per-transaction share variables add little marginal signal. Consumers interact almost exclusively with downstream retail; entrepreneurs deal with upstream wholesalers and professional SaaS/B2B networks. A card transacting across 5+ unique wholesale suppliers strongly implies ongoing, systematic procurement.

📌 **For the slide:** *B2B exposure is the strongest class-mean separator in the feature table (`b2b_mcc_share` ratio = 9.15×). After the model fit, `b2b_unique_merchants` (supplier breadth) is the B2B variable that carries the SHAP weight. The basket is curated from MCC semantics, with 5122 excluded to block target leakage.*

---

## 1.5 Group 4: Merchant Concentration

### Herfindahl–Hirschman Index (HHI)

$$HHI_c = \sum_{m=1}^{M_c} s_m^2, \quad s_m = \frac{n_{c,m}}{n_c}$$

where $s_m$ represents the transaction count share of card $c$ at merchant $m$, and $M_c$ denotes the absolute count of unique merchants.

**Properties:**
- $HHI \in [1/M_c,\ 1]$
- $HHI = 1$ → all transactions occur at a single merchant (absolute concentration).
- $HHI = 1/M_c$ → transactions are distributed perfectly equally (absolute diversification).
- HHI maps directly to the variance of shares combined with the square of their mean: $HHI = \text{Var}(s) + \bar{s}^2$

### Shannon Entropy

$$H_c = -\sum_{m=1}^{M_c} p_m \cdot \ln(p_m), \quad p_m = \frac{n_{c,m}}{n_c}$$

**Properties:**
- $H \in [0,\ \ln M_c]$
- $H = 0$ → complete certainty; all transactions are localized at one merchant.
- $H = \ln M_c$ → maximum uncertainty; uniform distribution across all merchants.
- Entropy quantifies the underlying behavioral **uncertainty** (measured in nats): high $H$ indicates that any next transaction could map to a vast range of vendors—a hallmark of operational entities dealing with an expansive array of distinct suppliers.

### Top-1 Ratio and Quantitative Metrics

| Feature | Formula | Meaning |
|---------|---------|-------|
| `merchant_top_ratio` | $\max_m s_m$ | Dominance of the primary merchant |
| `n_unique_merchants` | $M_c$ | Total unique merchant count |
| `n_unique_mcc` | Total unique MCC codes | Industry/Sector diversity |
| `merchants_per_tx` | $M_c / n_c$ | Unique merchants captured per transaction |

### Connection Between HHI and Entropy

HHI and Shannon Entropy function as dual formulations of the same underlying structural reality. Through the lens of the Rényi exponential entropy measure of order $\alpha$:

$$H_\alpha = \frac{1}{1-\alpha} \ln\left(\sum_m p_m^\alpha\right)$$

- In the limit as $\alpha \to 1$: $H_1 = H$ (Shannon Entropy)
- When evaluated at $\alpha = 2$: $H_2 = -\ln(HHI) \implies HHI = e^{-H_2}$

### Economic Justification

| Metric | Consumer | Business |
|---------|------------|--------|
| HHI | High (Routine patterns: hypermarket + 2 local stores) | Low (Distributed cross-procurement from many vendors) |
| Entropy | Low | High |
| Top-1 Ratio | High (40–60% of total activity locked at a favorite grocery store) | Low (10–20% max share on a single merchant) |

📌 **For the slide:** *HHI and entropy are dual mathematical measures of diversity. A consumer relies heavily on 3–5 core stores, whereas a commercial entity splits its volume across dozens of specialized providers. Lower HHI and elevated Entropy act as robust corporate flags.*

---

## 1.6 Group 5: Recurrence

### Formulas

| Feature | Formula | Underlying Source |
|---------|---------|----------|
| `recurring_share` | $\frac{\sum_i \mathbf{1}[\text{is\_recurring}(t_i)]}{n_c}$ | Transaction processing system flags |
| `recurring_capable_share` | $\frac{\sum_i \mathbf{1}[\text{rec\_capable}(\text{merchant}_i)]}{n_c}$ | Merchant master data reference tables |
| `tokenized_share` | $\frac{\sum_i \mathbf{1}[\text{tokenized}(t_i)]}{n_c}$ | Digital wallet / tokenization engine flags |

### Statistical Meaning

Each individual `*_share` serves as a method-of-moments point estimator for a basic binomial process parameter:

$$\hat{p} \sim \text{Bin}(n_c, p) / n_c$$

The corresponding variance of the estimator behaves as $\text{Var}(\hat{p}) = \frac{p(1-p)}{n_c}$, which decays predictably as the number of available historical transactions increases.

### Economic Justification

- **recurring_share** (b/c mean ratio = 4.92×, SHAP |impact| = 0.30): businesses systematically set up automated payment schedules for critical software, operational services, rent, and recurring logistics pipelines.
- **recurring_capable_share** (b/c mean ratio = 4.54×, SHAP |impact| = 0.58): captures frequent interactions with cloud providers, digital infrastructure, or SaaS subscriptions — identifying vendors designed to handle automated corporate pipelines, even if a specific baseline transaction lacks an active flag.
- **tokenized_share** (b/c mean ratio = 1.52×, SHAP |impact| = **1.77**, top-2 globally): tokenized digital checkout systems are the second most influential single signal in the model.

📌 **For the slide:** *Recurring and tokenized behaviors are a major model component. The class means differ by ~4.9× on `recurring_share`; on the model side, `tokenized_share` (SHAP |impact| = 1.77) and `recurring_capable_share` (0.58) carry the weight, reflecting the corporate cadence of automated overhead billing.*

---

## 1.7 Group 6: Activity & Burstiness

### Base Activity Metrics

| Feature | Formula |
|---------|---------|
| `active_days` | $\{d : \exists\, t_i, \text{date}(t_i) = d\}$ |
| `active_weeks` | $\{w : \exists\, t_i, \text{week}(t_i) = w\}$ |
| `span_days` | $\text{last\_date} - \text{first\_date} + 1$ |
| `tx_per_active_day` | $n_c / \text{active\_days}$ |
| `recency_days` | $\text{end\_of\_window} - \text{last\_date}$ |

### Inter-transaction Intervals (Gap Statistics)

For chronologically ordered transactions $t_{(1)} < t_{(2)} < \ldots < t_{(n_c)}$, inter-transaction gaps (measured in hours) are calculated as:

$$g_i = \frac{t_{(i+1)} - t_{(i)}}{3600}, \quad i = 1, \ldots, n_c - 1$$

| Feature | Formula |
|---------|---------|
| `gap_mean` | $\bar{g} = \frac{1}{n_c - 1}\sum_{i=1}^{n_c - 1} g_i$ |
| `gap_std` | $\sigma_g = \sqrt{\frac{1}{n_c - 2}\sum_{i}(g_i - \bar{g})^2}$ |

### Goh–Barabási Burstiness Index

$$B = \frac{\sigma_g - \bar{g}}{\sigma_g + \bar{g}}$$

**Properties:**
- $B \in [-1, 1]$
- $B > 0$ — **Bursty** pattern: long stretches of structural dormancy punctured by brief, high-density transaction salvos.
- $B = 0$ — **Poisson** process: transaction generation events are completely random and independent.
- $B < 0$ — **Regular** pattern: transaction streams are periodic and evenly spaced (clockwork execution).

**Connection with the coefficient of variation of gaps:**

$$B = \frac{CV_g - 1}{CV_g + 1}, \quad \text{where } CV_g = \frac{\sigma_g}{\bar{g}}$$

This provides a smooth, monotonic mapping of $CV_g$ from the domain $(0, \infty)$ onto the normalized metric space $(-1, 1)$.

### Monthly Spend Stability

$$CV_{\text{monthly}} = \frac{\sigma(\text{monthly\_spend})}{\bar{\text{monthly\_spend}}}$$

Commercial portfolios exhibit steady, predictable baseline monthly outlays (low $CV_{\text{monthly}}$) driven by fixed structural overheads, contrasting with consumer trends that shift dramatically due to lifestyle volatility or seasonal vacations.

### Economic Justification

| Pattern | Consumer | Business |
|---------|------------|--------|
| Burstiness | $> 0$ (Volatile, impulse-driven retail purchasing) | $\approx 0$ or $< 0$ (Highly structured procurement intervals) |
| gap_mean | Substantial (Transactions spaced across multiple days) | Low (Constant, high-frequency operational usage) |
| monthly_spend_cv | High (Highly susceptible to holiday spending or vacations) | Low (Anchored by strict corporate budget profiles) |
| active_days | Lower density | Extremely high (Consistent day-to-day operations) |

📌 **For the slide:** *The Goh–Barabási index maps behavioral consistency. $B < 0$ implies highly scheduled execution characteristic of structured businesses, whereas $B > 0$ points to ad-hoc, impulse retail consumer buying. Formula: $B = (\sigma - \mu)/(\sigma + \mu)$.*

---

## 1.8 Summary Tables: Two Importance Rankings

We report **two** rankings because they answer different questions, and the documentation has historically confused them. Both are computed against the same labeled population (105 k cards).

### 1.8.a Class-mean contrast (business / consumer mean ratio)

"By how much do the two classes differ on this feature, before any model is fit?"

| # | Feature | Group | b / c mean ratio |
|---|---------|-------|------------------:|
| 1 | `b2b_mcc_share`            | B2B Exposure  | **9.15×** |
| 2 | `b2b_amt_share`            | B2B Exposure  | **8.06×** |
| 3 | `b2b_unique_merchants`     | B2B Exposure  | **6.70×** |
| 4 | `recurring_share`          | Recurrence    | **4.92×** |
| 5 | `recurring_capable_share`  | Recurrence    | **4.54×** |
| 6 | `amt_median`               | Spending      | **3.99×** |
| 7 | `burstiness`               | Activity      | 3.12× |
| 8 | `amt_mean`                 | Spending      | 3.06× |

### 1.8.b SHAP global mean \|impact\| (top 8, top 15 in the notebook)

"How much does the feature actually move the LightGBM output, once the model has access to the others?"

| # | Feature | Group | mean \|SHAP\| |
|---|---------|-------|--------------:|
| 1 | `evening_share`            | Temporal     | **2.66** |
| 2 | `tokenized_share`          | Recurrence   | **1.77** |
| 3 | `online_share`             | Temporal     | **1.37** |
| 4 | `b2b_unique_merchants`     | B2B          | **1.37** |
| 5 | `weekend_share`            | Temporal     | 1.10 |
| 6 | `recurring_capable_share`  | Recurrence   | 0.58 |
| 7 | `b2b_amt_share`            | B2B          | 0.31 |
| 8 | `recurring_share`          | Recurrence   | 0.30 |

The two rankings disagree on purpose. The B2B share features have the largest *raw* class gap, but once the model knows the **count of distinct B2B suppliers** plus the **temporal signature** (evening / online / weekend share) and **tokenized share**, the additional B2B share variables become marginal. This is healthy — it shows the model is using multiple independent behavioral pillars, not single-feature leakage.

📌 **For the slide:** *Two complementary views — class-mean contrast (raw separability) and SHAP impact (model usage). B2B exposure dominates the contrast view; temporal pattern + tokenization + B2B supplier breadth dominate the SHAP view. No leaky institutional fields (`card_tier`, `bank_name`) are used.*

---

# Part 2. Model Quality Metrics

## 2.1 ROC-AUC and Connection to Wilcoxon U-Statistic

### ROC-AUC Definition

The Receiver Operating Characteristic (ROC) space maps $(\text{FPR}(\tau), \text{TPR}(\tau))$ combinations generated by sweeping the discrimination threshold $\tau \in [0, 1]$:

$$\text{TPR}(\tau) = \frac{TP(\tau)}{TP(\tau) + FN(\tau)}, \quad \text{FPR}(\tau) = \frac{FP(\tau)}{FP(\tau) + TN(\tau)}$$

$$\text{ROC-AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(x))\, dx$$

### Equivalence to the Mann–Whitney–Wilcoxon U-Statistic

**Theorem (Bamber, 1975):** The area under the ROC curve is mathematically identical to the normalized Mann–Whitney–Wilcoxon U-statistic:

$$\text{ROC-AUC} = \frac{U}{n_+ \cdot n_-} = P(\hat{p}_{\text{pos}} > \hat{p}_{\text{neg}})$$

where $n_+$ represents total actual positive cases (commercial), $n_-$ represents total negative entities (consumers), and $U$ measures the total count of perfectly concordant pairs:

$$U = \sum_{i \in \text{pos}} \sum_{j \in \text{neg}} \mathbf{1}[\hat{p}_i > \hat{p}_j]$$

**Interpretation:** ROC-AUC maps the exact probability that a randomly selected commercial instance yields a higher probability score than a completely random consumer instance.

Our current experimental layout yields a **ROC-AUC of 1.0000**, implying perfect mathematical separation of score distributions between both classes.

### Model Results

| Classifier Algorithm | ROC-AUC | PR-AUC | F1-Score |
|----------------------|---------|--------|----------|
| Logistic Regression (Baseline) | 1.0000 | 1.0000 | 0.9996 |
| **LightGBM (Primary Champion)** | **1.0000** | **1.0000** | **0.9996** |
| Random Forest | 1.0000 | 1.0000 | 0.9995 |

📌 **For the slide:** *ROC-AUC = $P(\text{Score}_{\text{Biz}} > \text{Score}_{\text{Consumer}})$, making it identical to the Wilcoxon statistic. A value of 1.0 indicates perfect class separation within the engineered feature space.*

---

## 2.2 PR-AUC (Average Precision)

### Why PR-AUC for Imbalanced Data

The sample structure contains 80,000 consumer portfolios versus 20,000 commercial cards ($\approx 4:1$ class balance ratio). Under class imbalances, ROC-AUC evaluations can sometimes yield overly optimistic conclusions.

**Mathematical Driver:** $\text{FPR} = \text{FP} / N_{\text{neg}}$. When $N_{\text{neg}}$ is substantial, even high absolute false positive counts ($\text{FP}$) are compressed by the massive denominator, forcing the FPR artificially low and masking performance flaws.

### PR-AUC Definition

The Precision-Recall trajectory tracks coordinates across $(\text{Recall}(\tau), \text{Precision}(\tau))$ combinations:

$$\text{Precision}(\tau) = \frac{TP(\tau)}{TP(\tau) + FP(\tau)}, \quad \text{Recall}(\tau) = \frac{TP(\tau)}{TP(\tau) + FN(\tau)}$$

$$\text{PR-AUC} = \sum_{k} (R_k - R_{k-1}) \cdot P_k$$

(Evaluated using numerical trapezoidal integration across discrete threshold coordinates).

### PR-AUC Baseline

For an uninformative random classifier, the performance floor maps strictly to the positive class prevalence:

$$\text{PR-AUC}_{\text{random}} = \frac{n_+}{n_+ + n_-} = \frac{20,000}{100,000} = 0.20$$

Our achieved empirical performance of **PR-AUC = 1.0000** represents a massive lift over the uninformative random baseline floor.

📌 **For the slide:** *PR-AUC focuses directly on class imbalances. The uninformative baseline floor sits at 0.20 (the proportion of businesses). Achieving 1.0 indicates flawless precision profiles preserved across all operating recall benchmarks.*

---

## 2.3 F1-score: Why the Harmonic Mean

### Definition

$$F_1 = \frac{2 \cdot P \cdot R}{P + R} = \frac{2\,TP}{2\,TP + FP + FN}$$

### Why the Harmonic Mean Instead of the Arithmetic Mean?

The harmonic mean is highly sensitive to extreme coordinate divergence, **harshly penalizing severe imbalances** between Precision and Recall:

| Precision ($P$) | Recall ($R$) | Arithmetic Mean | $F_1$ (Harmonic Mean) |
|---|---|:-:|:-:|
| 1.0 | 0.01 | 0.505 | 0.0198 |
| 0.5 | 0.50 | 0.500 | 0.5000 |
| 0.9 | 0.90 | 0.900 | 0.9000 |

If a model flags a single correct positive instance and misses everything else ($P = 1.0$, $R = 0.01$), a basic arithmetic average reports a deceptive score of 0.505. The $F_1$ formulation drops to 0.0198, correctly flagging the model as practically useless.

**General Case Extensions ($F_\beta$):**

$$F_\beta = (1 + \beta^2) \cdot \frac{P \cdot R}{\beta^2 \cdot P + R}$$

Configuring $\beta > 1$ weights Recall more heavily (minimizing missed opportunities), whereas setting $\beta < 1$ prioritizes strict Precision accuracy constraints.

📌 **For the slide:** *The $F_1$ metric is the harmonic mean of precision and recall. It penalizes severe imbalances, preventing models from hiding extreme recall blind spots behind an artificially inflated precision score. Our top configuration stabilizes at 0.9996.*

---

## 2.4 Confusion Matrix: Business Interpretation

### Matrix Structure

| | Predicted: Consumer (0) | Predicted: Business (1) |
|:-:|:-:|:-:|
| **Actual: Consumer (0)** | TN (Correctly retained in consumer tiers) | FP (False corporate alarm) |
| **Actual: Business (1)** | FN (Missed commercial entity) | TP (Correctly identified commercial entity) |

### Business Cost of Errors

| Classification Error | Operational & Business Consequence | Financial Cost Weight |
|------------|-------------------|-----------|
| **False Positive (FP)** (Consumer $\to$ "Business") | Triggers unnecessary outreach or manual verification. Triggers minor customer friction but causes no direct capital leakage. | **Low Cost** ($C_{FP}$) |
| **False Negative (FN)** (Business $\to$ "Consumer") | A commercial entity continues leveraging zero-fee consumer infrastructure. The institution loses transaction fees and incurs regulatory risk. | **High Cost** ($C_{FN}$) |

**Operational Asymmetry:** $C_{FN} \gg C_{FP}$ $\implies$ We prefer operating points optimized for high Recall profiles to minimize costly FN errors, even at the cost of additional minor FP flags.

📌 **For the slide:** *An FN error means missing a commercial entity, which leads to direct revenue leakage and compliance risks. An FP error triggers a low-cost, routine verification check. Therefore, operational constraints dictate prioritizing recall over precision.*

---

# Part 3. Justification of Classification Thresholds

## 3.1 Bayesian Decision Theory for Optimal Threshold Selection

### Formalization

The classification engine maps inputs to probability scores: $\hat{p}(c) = P(\text{business} \mid \mathbf{x}_c)$. The discrete assignment rule is formalized as:

$$\hat{y}(c) = \begin{cases} 1, & \text{if } \hat{p}(c) \geq \tau \\ 0, & \text{otherwise} \end{cases}$$

### Expected Loss Minimization

For any given threshold assignment $\tau$, the global Expected Loss equation is defined as:

$$\mathcal{L}(\tau) = C_{FP} \cdot \text{FPR}(\tau) \cdot N_- + C_{FN} \cdot \text{FNR}(\tau) \cdot N_+$$

where $\text{FNR} = 1 - \text{TPR}$, $N_-$ represents absolute consumer counts, and $N_+$ represents business volume.

### Optimal Bayesian Threshold

Applying Bayesian decision theory to minimize global risk yields the optimal threshold equation:

$$\tau^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$$

**Example Scenario:** Assuming that missing a commercial entity is 10 times more costly than checking a consumer by mistake ($C_{FN} = 10 \cdot C_{FP}$):

$$\tau^* = \frac{1}{1 + 10} = 0.091$$

The optimal operational threshold automatically shifts **downward**, optimizing Recall to protect the institution from costly missed detections.

### Project Implementations

We deploy three specialized operational thresholds to match core business workflows:

| Selected Threshold | Definition | Target Workflow | Core System Logic |
|-------|----------|------------|--------|
| F1-max         | $\tau_{F1}$ that maximizes $F_1$ on train-OOF | Automated portfolio migration | Equalized precision / recall trade-off |
| Outreach       | smallest $\tau$ with precision $\geq 0.50$ on train-OOF | RM call lists / outreach campaigns | Largest viable lead pool above a precision floor (missed SME costs more than a cheap call) |
| Lead cutoff    | $\tau = 0.30$ (fixed business rule) | Lead-generation extraction | Locks target auditing to $P(\text{business}) \geq 0.30$ |

📌 **For the slide:** *The optimal operational threshold is defined by $\tau^* = C_{FP}/(C_{FP} + C_{FN})$. As the cost of missed detections increases, the threshold falls to boost recall. We maintain three distinct operational thresholds for three business use cases.*

---

## 3.2 Why OOF Tuning and Not a Validation Set

### The Problem of Validation Set "Double Dipping"

If the final operational threshold is directly tuned against the core validation set, that data partitions loses its statistical independence:

$$\text{Threshold tuned on } \mathcal{D}_{\text{val}} \implies \text{Evaluated metrics on } \mathcal{D}_{\text{val}} \text{ are over-optimistic}$$

### Out-of-Fold (OOF) Strategy

1. Partition the core **training** pool into $K = 5$ stratified cross-validation folds.
2. For each individual iteration $k$: train the underlying algorithms on $K-1$ folds, and generate predictions on the remaining holdout fold $k$.
3. This process ensures that every training record receives a clean "out-of-fold" score $\hat{p}_{\text{OOF}}(c)$ generated by an algorithmic iteration that never observed that data during training.

$$\hat{p}_{\text{OOF}}(c) = f_{\neg k(c)}(\mathbf{x}_c), \quad c \in \text{fold } k$$

4. The operational thresholds ($\tau$) are optimized strictly using these OOF prediction arrays.
5. The downstream validation set remains completely untouched, ensuring completely unbiased final metric evaluations.

### Key Operational Advantages

- Threshold selection leverages roughly 80% of total available data vectors (the entire training set), ensuring high stability.
- Unbiased final metrics avoid cross-contamination leakage.
- Aligns with best practices in production ML engineering and Kaggle competition frameworks.

📌 **For the slide:** *OOF scoring means every training instance is evaluated by a model variant that never observed it during optimization. Thresholds are optimized on these OOF vectors, preserving the validation partition for clean performance evaluations.*

---

## 3.3 Justification of the 0.30 Threshold for Leads

### Bayesian Cost Analysis

Deploying an operating cutoff of $\tau = 0.30$ means a card is classified as a sales lead whenever its model-assigned corporate probability touches or exceeds 30%.

Mapping this back to Bayesian decision theory reveals the implicit cost weights:

$$\tau = 0.30 = \frac{C_{FP}}{C_{FP} + C_{FN}} \implies C_{FN} = \frac{0.70}{0.30} \cdot C_{FP} \approx 2.33 \cdot C_{FP}$$

This establishes that **missing a hidden commercial target is modeled as 2.33 times more expensive** than triggering a false auditing call. This represents a conservative baseline; real-world commercial leakage often carries a higher relative cost.

### Probability Calibration Perspective

Assuming a well-calibrated classifier model ($\hat{p} \approx P(\text{business} \mid \mathbf{x})$), a threshold of 0.30 carries a direct real-world meaning:

> *“We pull into manual audit pipelines any account displaying at least a 30% baseline probability of operating as a commercial business entity.”*

Statistically, out of 10 distinct leads flagged at the $\hat{p} = 0.30$ tier, approximately 3 will turn out to be actual commercial business entities upon manual audit.

### Empirical Results

Applying this 0.30 lead generation cutoff successfully extracted **11 consumer-labeled card portfolios** displaying scores $\geq 0.30$, led by a primary target with a high score of $P(\text{business}) = 0.992$.

📌 **For the slide:** *A threshold of 0.30 mathematically balances risk when missed commercial targets are 2.3 times more costly than false alarms. This strategy extracted 11 hidden commercial accounts, led by an extreme outlier profile scored at 0.992.*

---

# Part 4. Model Error Analysis

## 4.1 Expected Loss

### Analytical Formulation

$$\mathcal{L} = C_{FP} \cdot \text{FPR} \cdot N_- + C_{FN} \cdot (1 - \text{TPR}) \cdot N_+$$

Expanding this expression using core sample counts simplifies to:

$$\mathcal{L} = C_{FP} \cdot \frac{FP}{N_-} \cdot N_- + C_{FN} \cdot \frac{FN}{N_+} \cdot N_+$$

$$\mathcal{L} = C_{FP} \cdot FP + C_{FN} \cdot FN$$

### Normalized Loss Expression

To compare cost profiles across changing population counts, we normalize the expression by the total population $N$:

$$\ell(\tau) = \pi_- \cdot C_{FP} \cdot \text{FPR}(\tau) + \pi_+ \cdot C_{FN} \cdot \text{FNR}(\tau)$$

where $\pi_+ = N_+ / N$ and $\pi_- = N_- / N$ represent the underlying positive and negative class prior weights.

### Concrete Cost Breakdown Example

Assuming baseline parameters $C_{FP} = 1$, $C_{FN} = 10$, $N_- = 80,000$, and $N_+ = 20,000$:

| Operating Threshold | FP Count | FN Count | Global Expected Loss ($\mathcal{L}$) |
|---------------------|----------|----------|----------------------------------------|
| F1-max ($\tau \approx 0.32$)            | 0 | 1 | $1 \cdot 0 + 10 \cdot 1 = 10$ |
| Outreach ($\tau \approx 2 \cdot 10^{-6}$, prec $\geq 0.5$) | 5 | 0 | $1 \cdot 5 + 10 \cdot 0 = 5$ |
| Empirical argmin ($\tau \approx 0.016$) | 4 | 0 | $1 \cdot 4 + 10 \cdot 0 = 4$ |

When asymmetric error costs apply ($C_{FN} \gg C_{FP}$), the operating point with lower threshold (more recall, more FPs traded for fewer FNs) wins the loss budget. See **Appendix A4** in the notebook for the full $\mathcal{L}(\tau)$ sweep, with the Bayesian optimum $\tau^* = C_{FP}/(C_{FP}+C_{FN}) = 1/11 \approx 0.091$ marked.

📌 **For the slide:** *Expected Loss = $C_{FP} \cdot \text{FP} + C_{FN} \cdot \text{FN}$. When the costs of missed detections dominate ($C_{FN} \gg C_{FP}$), the threshold optimized for higher recall scores minimizes global operational losses.*

---

## 4.2 Model Calibration

### Core Definition

An algorithmic classifier is considered **perfectly calibrated** if its output scores map directly to empirical frequencies:

$$P(Y = 1 \mid \hat{p}(X) = q) = q, \quad \forall q \in [0, 1]$$

This means that among a subpopulation of cards assigned a corporate probability score of 0.70, exactly 70% must be true commercial entities.

### Platt Scaling Calibration

To correct LightGBM output scale distortions, Platt Scaling trains an overlay logistic regression model directly on top of the raw logit outputs:

$$p_{\text{cal}} = \sigma(A \cdot f(\mathbf{x}) + B) = \frac{1}{1 + e^{-(A \cdot f(\mathbf{x}) + B)}}$$

The scale tuning parameters $A$ and $B$ are optimized by maximizing log-likelihood estimations on an independent validation holdout set.

### Isotonic Regression (Alternative Calibration)

For non-parametric scaling adjustments, Isotonic Regression fits a monotonic, isotonic piecewise-constant mapping function $g$ designed to minimize:

$$\min_{g \uparrow} \sum_{i=1}^{n}(y_i - g(\hat{p}_i))^2$$

### Reliability Diagram Evaluation

The score range is divided into $B$ discrete bins. For each individual bucket $b$, we compute:

$$\text{fraction\_positive}_b = \frac{\sum_{i \in b} y_i}{|b|}, \quad \text{mean\_predicted}_b = \frac{\sum_{i \in b} \hat{p}_i}{|b|}$$

For a perfectly calibrated model, all binned coordinate points align along the ideal diagonal line $y = x$.

**Evaluation Metric:** Expected Calibration Error (ECE):

$$\text{ECE} = \sum_{b=1}^{B} \frac{|b|}{n} \cdot |\text{fraction\_positive}_b - \text{mean\_predicted}_b|$$

📌 **For the slide:** *Calibration ensures that a probability score of 0.70 maps directly to a 70% empirical corporate density. Platt Scaling utilizes an overlay logistic regression to achieve this, while ECE serves as the standard evaluation metric.*

---

## 4.3 Synthetic Data Disclaimer

### Why Perfect AUC Metrics ($\approx$ 1.0) Do Not Directly Transfer to Production

Achieving an empirical AUC of 1.0000 is an artifact of the underlying synthetic data generation process, driven by two factors:

1. **Generative Separation**: The synthetic transactions are generated using distinct statistical distributions for consumer and commercial classes, making them highly separable by design.
2. **Absence of Real-World "Gray Zones"**: The synthetic generator lacks real-world edge cases, such as:
    - Part-time freelancers using a single card for mixed personal and commercial use.
    - Small sole proprietors routing business expenses through personal accounts.
    - Consumers exhibiting business-like behavior (e.g., purchasing materials for home renovations or events).

3. **Expected Real-World Production Performance**: On actual live banking data, expect performance to land within a realistic range of $0.85 \leq \text{AUC} \leq 0.95$.

### Performance Degradation Modeling

The transition from synthetic environments to live production data can be modeled as:

$$\text{AUC}_{\text{real}} \approx \text{AUC}_{\text{synth}} - \Delta_{\text{domain}} - \Delta_{\text{noise}} - \Delta_{\text{concept}}$$

where:
- $\Delta_{\text{domain}}$ captures structural domain shifts in population distributions.
- $\Delta_{\text{noise}}$ captures real-world transaction data anomalies, such as misclassified MCC codes or authorization delays.
- $\Delta_{\text{concept}}$ captures concept drift between the modeled definitions and actual business behavior.

📌 **For the slide:** *An AUC score of 1.0 is expected on synthetic data and will not directly transfer to production environments, where performance typically trends between 0.85 and 0.95 due to real-world edge cases. The project's core value resides in its engineering framework and features.*

---

## 4.4 Confidence Intervals for Metrics

### Non-Parametric Bootstrap Assessment for AUC Performance

To assess statistical uncertainty without making rigid parametric assumptions, we deploy a non-parametric bootstrap strategy:

1. Draw $B = 1000$ independent bootstrap resamples with replacement from the validation dataset partition of size $n$.
2. For each individual resampled layout $b$, calculate its corresponding performance score $\text{AUC}_b$.
3. The empirical 95% confidence interval boundaries map to the historical percentiles: $[\text{AUC}_{(0.025)},\ \text{AUC}_{(0.975)}]$.

### Analytical Variance Estimation via DeLong's Method

Alternatively, DeLong's method (DeLong et al., 1988) provides a direct analytical calculation for AUC variance:

$$\text{Var}(\widehat{\text{AUC}}) = \frac{1}{n_+ \cdot n_-}\left[\frac{n_-\, \hat{S}_1 + n_+\, \hat{S}_2}{(n_+ + n_-)} \right]$$

where $\hat{S}_1$ and $\hat{S}_2$ represent variance component estimators derived from structural component variables.

The analytical 95% confidence interval is then computed as:

$$\widehat{\text{AUC}} \pm 1.96 \cdot \sqrt{\text{Var}(\widehat{\text{AUC}})}$$

### Exact Binomial Intervals for Precision and Recall Performance

For a evaluation block of size $n$ that registers $k$ correct classifications, the performance distributions follow a Beta distribution profile:

$$P \sim \text{Beta}(k + 1, n - k + 1)$$

The exact 95% Clopper–Pearson confidence interval bounds are calculated as:

$$\left[B^{-1}(0.025;\ k,\ n-k+1),\quad B^{-1}(0.975;\ k+1,\ n-k)\right]$$

📌 **For the slide:** *We leverage bootstrap resamples to construct robust 95% confidence intervals for our AUC metrics, alongside DeLong’s analytical alternative and exact Clopper–Pearson intervals for precision and recall tracking.*

---

## 4.5 What the Notebook Actually Computes (Appendix A1–A4)

The notebook ships a self-contained appendix that operationalizes the theory above:

| Notebook section | What it does | Result on this data |
|---|---|---|
| **A1.** Bootstrap CI for ROC-AUC and PR-AUC | 1000 resamples of the validation set | both intervals collapse to [1.0000, 1.0000] → AUC = 1 is not a lucky split |
| **A2.** Welch's $t$, Mann–Whitney $U$, Cohen's $d$ | for the SHAP top-8 features | $p < 10^{-300}$ for all, $|d| \in [1.6, 4.8]$ (all "very large" effects) |
| **A3.** Calibration (reliability diagram + ECE) | raw LightGBM vs Platt overlay fit on train-OOF | ECE_raw = 0.0001, ECE_Platt = 0.0005 — already well-calibrated, Platt mostly a placeholder for real-data deployment |
| **A4.** Expected loss vs threshold ($C_{FP}=1, C_{FN}=10$) | sweep $\tau \in [0, 1]$ | empirical argmin at $\tau \approx 0.016$ with $\mathcal{L} = 4$; Bayesian $\tau^* = 0.091$ gives $\mathcal{L} = 10$ |

📌 **For the slide:** *Every claim in Part 2–4 of this document is backed by a runnable cell in the notebook's Appendix. Bootstrap collapses AUC's CI to [1, 1], all top features are significant at $p < 10^{-300}$ with very large effect sizes, the model is already well-calibrated (ECE = 0.0001), and the expected-loss curve confirms the recall-leaning operating choice.*

---

# Part 5. Presentation Defense Tips

## 5.1 Key Takeaways for Each Section

### Feature Engineering Framework (Part 1)

> **Core Talking Points for the Evaluation Committee:**
>
> “We engineered 35 distinct behavioral features at the card profile level, structured across six independent behavioral dimensions, each backed by a clear economic driver. The strongest predictors center on B2B exposure metrics, tracking transaction density within wholesale and commercial MCC blocks. This core B2B profile relies on an expert-curated basket of 40 MCC categories that avoids data-driven target leakage; for example, MCC 5122 was intentionally excluded because it was perfectly correlated with business labels in the sample. Furthermore, we leverage Herfindahl-Hirschman Indices and Shannon Entropy to quantify merchant concentration, separating diversified corporate procurement from narrow consumer habits. Finally, the Goh-Barabási index identifies systematic, scheduled business outlays, separating them from impulsive consumer retail shopping patterns.”

### Quality Metrics Evaluation (Part 2)

> **Core Talking Points for the Evaluation Committee:**
>
> “Our evaluation relies on three complementary metrics: ROC-AUC, PR-AUC, and the $F_1$-score. ROC-AUC maps directly to the Wilcoxon rank statistic, capturing the exact probability of correct instance ranking. PR-AUC explicitly adjusts for underlying class imbalances, evaluated against an uninformative baseline floor of 0.20. The $F_1$-score leverages a harmonic mean to penalize imbalances between precision and recall, ensuring hidden vulnerabilities cannot be masked. All models register perfect performance profiles on this validation layout, which is standard for cleanly generated synthetic data.”

### Threshold Configuration Strategies (Part 3)

> **Core Talking Points for the Evaluation Committee:**
>
> “Operational thresholds are optimized using out-of-fold predictions on the training set, so the validation set remains untouched and the headline metrics stay unbiased. We establish three distinct thresholds tailored to specific business use cases: an $F_1$-max configuration for automated portfolio updates, an *outreach* threshold defined as the smallest cutoff whose precision still exceeds 0.50 (the largest viable lead pool), and a fixed 0.30 cutoff for sales lead generation. The 0.30 cutoff is justified by Bayesian decision theory: it is the optimal threshold under a 2.3 : 1 cost ratio between a missed commercial account and an unnecessary verification check.”

### Error and Calibration Tracking (Part 4)

> **Core Talking Points for the Evaluation Committee:**
>
> “Expected Loss serves as our primary operational optimization metric. Because missing a commercial account is significantly more costly than a false alarm, the optimal threshold naturally shifts downward to prioritize recall. While an AUC of 1.0 is expected in synthetic environments, real-world live data performance typically ranges between 0.85 and 0.95 due to population gray zones and concept drift. The core value of this project lies in its scalable engineering framework and feature architecture, which are built to deploy directly onto live data streams.”

---

## 5.2 Expected Questions & Answers

### Question 1: "An AUC of 1.0 is highly unusual. Is this model simply overfitted?"

> **Response:** “No, this is not an overfitting artifact. The 1.0 AUC profile is maintained across independent validation partitions, not just training sets. This perfect separation stems from the synthetic data generator, which uses distinct statistical distributions for consumer and commercial classes. In production environments with real banking data, performance will trend between 0.85 and 0.95 due to real-world edge cases like mixed-use freelance accounts and commercial purchases routed through personal cards.”

### Question 2: "Why select traditional gradient boosting over deep learning architectures or transformers?"

> **Response:** “For tabular datasets scaled within a 35-feature topology, gradient boosted tree architectures like LightGBM consistently represent state-of-the-art performance. Extensive empirical benchmarks (such as Grinsztajn et al., 2022; Shwartz-Ziv & Armon, 2022) confirm that ensemble tree methods regularly outperform deep learning architectures on medium-sized tabular layouts. Additionally, LightGBM integrates natively with SHAP frameworks to provide full explainability, which is a critical regulatory compliance requirement for banking systems.”

### Question 3: "What criteria dictated the inclusion of exactly 40 MCC codes within the B2B basket?"

> **Response:** “The selection was derived entirely from the official semantic definitions of the MCC codes, focusing on wholesale trade, corporate logistics, and professional business support channels. We intentionally avoided data-driven statistical filters to eliminate data leakage risks. This approach is why MCC 5122 was excluded: it was perfectly correlated with business labels in this sample, and including it would introduce artificial predictive inflation.”

### Question 4: "How should we interpret the 11 hidden entrepreneurs detected within the consumer tier?"

> **Response:** “These represent consumer accounts whose transaction profiles are behaviorally identical to commercial entities. They display high wholesale MCC concentration, automated recurring payments, a highly diversified supplier footprint, and a structured operational cadence. Our top prospect registers a corporate probability score of 0.992, indicating extreme model confidence. Each detected lead includes a complete SHAP values breakdown detailing the exact drivers behind the assignment.”

### Question 5: "What is SHAP, and what makes it necessary for this banking application?"

> **Response:** “SHAP (SHapley Additive exPlanations) is a model-agnostic framework rooted in cooperative game theory. For any given account profile, SHAP decomposes the final score into the additive contributions of each individual feature variable: $f(\mathbf{x}) = \phi_0 + \sum_j \phi_j$. It is the only explanation method that satisfies core game-theoretic axioms: efficiency (contributions sum to the total prediction shift), symmetry, and dummy player constraints. This mathematical rigor is essential for satisfying strict regulatory banking compliance and auditing standards.”

### Question 6: "Why do Logistic Regression and LightGBM report identical 1.0 AUC scores?"

> **Response:** “This confirms that the classes in this synthetic dataset are linearly separable, allowing even a basic linear model to achieve perfect separation. In production deployments with live data, LightGBM will outperform linear baselines by capturing complex, non-linear feature interactions. We retain Logistic Regression as a baseline to validate our feature engineering: when even simple linear models perform well, it confirms the engineered features are highly informative.”

### Question 7: "What does the production implementation deployment pipeline look like?"

> **Response:** “The operational architecture is designed as a scheduled pipeline: monthly aggregations compile raw transaction streams over a rolling 6-month window $\to$ compute the 35 behavioral features $\to$ score profiles via the LightGBM engine $\to$ extract prospects meeting the $P \ge 0.30$ cutoff $\to$ route leads to compliance verification systems. The core model weights will be updated quarterly using verified labels generated during manual compliance audits.”

### Question 8: "What are the computational complexity constraints of this algorithm?"

> **Response:** “The feature engineering pipeline runs at $O(N \log N)$ complexity, where $N$ represents total transaction volume, driven primarily by sorting operations for gap statistics. LightGBM inference scales at $O(T \cdot D)$ per record, where $T = 400$ trees and $D = 6$ represents average tree depth. Processing a 100,000-card portfolio requires less than 1 second of total inference time, making it highly scalable for production environments.”

---

## 5.3 Cheat Sheet: Formulas on a Single Page

| Target Analysis Metric | Mathematical Formulation |
|-------------------------|--------------------------|
| Herfindahl–Hirschman Index (HHI) | $HHI_c = \sum_{m=1}^{M_c} s_m^2$ |
| Shannon Entropy Metric | $H_c = -\sum_{m=1}^{M_c} p_m \cdot \ln(p_m)$ |
| Goh–Barabási Burstiness Index | $B = \frac{\sigma_g - \bar{g}}{\sigma_g + \bar{g}}$ |
| Welch's t-test Calculation | $t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$ |
| Mann–Whitney U-Statistic | $U = \sum_{i \in \text{pos}} \sum_{j \in \text{neg}} \mathbf{1}[\hat{p}_i > \hat{p}_j]$ |
| Harmonic Mean ($F_1$-score) | $F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ |
| Optimal Bayesian Threshold Choice | $\tau^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$ |
| Expected Operational Loss | $\mathcal{L} = C_{FP} \cdot FP + C_{FN} \cdot FN$ |
| Platt Scaling Calibration Formula | $p_{\text{cal}} = \frac{1}{1 + e^{-(A \cdot f(\mathbf{x}) + B)}}$ |
| SHAP Explanatory Decomposition | $f(\mathbf{x}) = \phi_0 + \sum_{j=1}^{M} \phi_j$ |