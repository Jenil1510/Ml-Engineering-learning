# 🚀 02 - Exploratory Data Analysis (EDA): Bank Marketing Campaign

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jenil1510/Ml-Engineering-learning/blob/main/02_exploratory_data_analysis/eda_bank_marketing.ipynb)

An end-to-end Exploratory Data Analysis project on **41,188 real-world retail banking contacts** across 21 client, campaign, and macroeconomic dimensions.

---

## 📋 Dataset Specifications
- **Source:** UCI Machine Learning Repository / Moro et al. (Portuguese retail bank marketing campaigns).
- **Volume:** **41,188 rows** (exceeds the 10,000 row requirement by 4x).
- **Dimensions:** **21 columns** (exceeds the 10 column requirement by 2x).
- **Target Variable:** `y` (Binary: `yes` subscribed a term deposit, `no` did not subscribe).
- **Storage Location:** [data/bank_additional_full.csv](file:///home/innvonix/Ml-Engineering-learning/02_exploratory_data_analysis/data/bank_additional_full.csv)

### Data Dictionary
| Category | Feature | Type | Description |
| :--- | :--- | :--- | :--- |
| **Demographics** | `age` | Numeric | Client age (years) |
| | `job` | Categorical | Employment sector (`admin.`, `blue-collar`, `technician`, `services`, `management`, `retired`, `student`, etc.) |
| | `marital` | Categorical | Marital status (`married`, `single`, `divorced`, `unknown`) |
| | `education` | Categorical | Education level (`basic.4y`, `basic.6y`, `basic.9y`, `high.school`, `university.degree`, etc.) |
| **Financial Health** | `default` | Categorical | Credit in default? (`no`, `yes`, `unknown`) |
| | `housing` | Categorical | Has housing loan? (`no`, `yes`, `unknown`) |
| | `loan` | Categorical | Has personal loan? (`no`, `yes`, `unknown`) |
| **Campaign Dynamics** | `contact` | Categorical | Contact communication type (`cellular`, `telephone`) |
| | `month` | Categorical | Last contact month of year (`mar`, `apr`, ..., `nov`, `dec`) |
| | `day_of_week` | Categorical | Last contact day of the week (`mon`, `tue`, `wed`, `thu`, `fri`) |
| | `duration` | Numeric | Last contact call duration in seconds |
| | `campaign` | Numeric | Number of contacts performed during this campaign for this client |
| | `pdays` | Numeric | Days passed after client was last contacted from prior campaign (`999` = never contacted) |
| | `previous` | Numeric | Number of contacts performed before this campaign |
| | `poutcome` | Categorical | Outcome of previous marketing campaign (`failure`, `nonexistent`, `success`) |
| **Macroeconomic** | `emp.var.rate` | Numeric | Employment variation rate (quarterly indicator) |
| | `cons.price.idx` | Numeric | Consumer price index (monthly indicator) |
| | `cons.conf.idx` | Numeric | Consumer confidence index (monthly indicator) |
| | `euribor3m` | Numeric | Euribor 3-month interest rate (daily indicator) |
| | `nr.employed` | Numeric | Number of employees (quarterly indicator) |
| **Target** | `y` | Categorical | Has the client subscribed a term deposit? (`yes`, `no`) |

---

## 🧹 Data Cleaning Summary
1. **Deduplication:** Identified and eliminated **12 exact duplicate records**, reducing the dataset from 41,188 to 41,176 unique records.
2. **Missing Value Architecture:**
   - Missing data was encoded as the string `'unknown'`.
   - `default`: 8,596 unknowns (20.88%).
   - `education`: 1,730 unknowns (4.20%).
   - `housing` & `loan`: 990 unknowns (2.40%).
   - `job`: 330 unknowns (0.80%).
   - `marital`: 80 unknowns (0.19%).
   - **Decision:** Rather than dropping rows (which would eliminate over 25% of data and introduce severe selection bias), `'unknown'` was treated as an explicit, informative category.
3. **Feature Transformations:**
   - Converted target `y` into `target_binary` (0/1) for seamless subgroup conversion calculations.
   - Decoupled `pdays` into `previously_contacted` binary indicator ($pdays \ne 999$).
   - Binned continuous `age` into 6 demographic cohorts (`<25`, `25-34`, `35-44`, `45-54`, `55-64`, `65+`).
   - Converted `duration` into `duration_min` for human interpretability.
   - Binned `campaign` attempts into intensity brackets (`1 contact`, `2-3`, `4-6`, `>6`).

---

## 📊 Exploratory Visualizations & Key Observations

```
Distribution of Conversion Rates across Key Segments:
┌────────────────────────────────────────────────────────┐
│ Overall Baseline Subscription Rate: 11.3%              │
├────────────────────────────────────────────────────────┤
│ Age Cohorts:                                           │
│  • Senior (65+):             ███████████████████ 45.6% │
│  • Gen Z (<25):              ██████████ 24.0%          │
│  • Middle-aged (35-44):      ████ 8.8%                 │
├────────────────────────────────────────────────────────┤
│ Occupations:                                           │
│  • Student:                  █████████████ 31.4%       │
│  • Retired:                  ██████████ 25.2%          │
│  • Blue-collar:              ███ 6.9%                  │
├────────────────────────────────────────────────────────┤
│ Contact Medium:                                        │
│  • Cellular:                 ██████ 14.7%              │
│  • Fixed Telephone:          ██ 5.2%                   │
├────────────────────────────────────────────────────────┤
│ Interest Rate Climate (Euribor 3M):                    │
│  • Low (<1.5%):              ██████████ 23.9%          │
│  • High (>4.5%):             ██ 4.5%                   │
└────────────────────────────────────────────────────────┘
```

---

## 💡 Key Insights and Observations

1. **The U-Shaped Age Distribution:**
   - Both ends of the age spectrum convert at significantly higher rates than the middle.
   - Seniors (65+) convert at **45.6%** (4x baseline) and youth (<25) at **24.0%** (2x baseline), whereas customers in prime working years (35–54) convert at only ~8.8%.
2. **Campaign Fatigue & Diminishing Returns:**
   - Calling a lead 1 time converts at **13.0%**.
   - By contact 4–6, conversion drops to **6.7%**.
   - Calling more than 6 times collapses conversion to **3.7%**. Persisting beyond 3 attempts produces negative ROI.
3. **Communication Channel Efficacy:**
   - Cellular outreach converts at **14.7%** compared to only **5.2%** for fixed landline telephones (nearly **3x more effective**).
4. **Macroeconomic Sensitivity:**
   - Customer willingness to subscribe is heavily dependent on macroeconomic conditions. During periods when the 3-month Euribor benchmark is low (<1.5%), conversion spikes to **23.9%**. When Euribor climbs above 4.5%, conversion drops to **4.5%**.
5. **Call Duration as Engagement Signal:**
   - Median duration for successful subscriptions is **7.5 minutes** vs **2.2 minutes** for rejections (3.4x longer).
   - *Data Leakage Warning:* While duration is a powerful explanatory variable during EDA, it cannot be used as an input feature for lead scoring models prior to placing a call.

---

## 🎯 Recommendations Based on Findings

1. **Target Persona Alignment:**
   - Structure customized deposit products: Wealth preservation and guaranteed returns for **retirees (65+)**, and beginner-friendly high-yield savings plans for **students (<25)**.
2. **Channel & Contact Policy Reform:**
   - Mandate cellular outreach exclusively; phase out landline call lists.
   - Institute an automated CRM hard stop: **Maximum 3 contact attempts** per prospect per campaign.
3. **Macro-Adaptive Campaign Timing:**
   - Synchronize marketing blitzes with monetary easing cycles and low-rate periods (Euribor < 2.0%) or high-liquidity months (March, September, October).
4. **Machine Learning Model Formulation:**
   - Address the 89:11 class imbalance using cost-sensitive learning or SMOTE.
   - Exclude `duration` from pre-call prediction features to prevent data leakage.

---

## 🚀 How to Run the Notebook

### Google Colab (1-Click Run)
Click the [Open In Colab](https://colab.research.google.com/github/Jenil1510/Ml-Engineering-learning/blob/main/02_exploratory_data_analysis/eda_bank_marketing.ipynb) badge. The notebook includes an automated dataset download fallback.

### Locally
```bash
source .venv/bin/activate
jupyter lab 02_exploratory_data_analysis/eda_bank_marketing.ipynb
```
