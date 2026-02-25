# Model Comparison – Operational Threshold Analysis (v2)

#### 

#### Objective

####

Evaluate models under operational constraint:

Recall ≥ 0.60

####

#### Models Evaluated

####

Logistic Regression (with log transformation on PAY\_AMT features) with threshold of 0.237

The log transformation was applied to PAY\_AMT features due to extreme skewness, improving linear separability.

Random Forest (baseline configuration) with threshold of 0.26

####

#### Evaluation Method

####

* 5-fold Stratified CV

* Out-of-fold probabilities

* Threshold search maximizing precision subject to recall constraint

* Metrics reported:

    * ROC-AUC

    * PR-AUC

    * Recall

    * Precision

    * flagged_rate

####

#### Results @ Recall ≥ 0.60

####

|Model|Threshold|Recall|Precision|flagged_rate|ROC-AUC|PR-AUC|
|-|-|-|-|-|-|-|
|Logistic Regression w/Log-transformation|0.237|0.6005123568414708|0.4566288529849891|0.2909|0.7470461903389387|0.506013958479012|
|Random Forest baseline|0.26|0.6009644364074744|0.49338117035754053|0.26943333333333336|0.7786132078129503|0.5550162753121383|

####

#### Operational Impact (100k/month scenario)

#### 

* Logistic -> 29.09% flagged -> 29,090 reviews
* RF -> ~26.943% flagged -> 26,943 reviews
* Δ = 2,147 fewer reviews
Impact:
* 2,147 saved reviews
* ~179 hs saved p/month aproximately ~1 FTE

#### 

#### Interpretation

####

* Both models satisfy the recall constraint.
* Random Forest achieves higher precision while flagging fewer cases, indicating better operational efficiency at the same risk coverage level.
* Logistic is simpler & more interpretable.
* Final choice depends on compliance vs cost trade-off.

####

#### Policy Scenarios

####

##### Scenario A — Operational constraint: Recall ≥ 0.60

Both models satisfy the recall requirement. However:

* Random Forest achieves higher precision (0.493 vs 0.457).

* It flags fewer cases (26.9% vs 29.1%).

* It produces ~647 fewer false positives on the full dataset.

Decision:
Under a minimum recall constraint, Random Forest is operationally superior: it maintains the same risk coverage while reducing manual workload and false positives.

##### Scenario B — Operational constraint: Precision ≥ 0.50

Both models meet the precision requirement.

* Random Forest

    * Higher recall (0.589 vs 0.555)

    * Detects ~222 more true defaults

    * Slightly higher flagged_rate (+1.5%)

* Logistic Regression

    * Slightly lower operational volume

    * Slightly fewer false positives

Decision:
If the cost of missing a default (FN) is high, Random Forest is preferable due to higher recall.
If operational capacity or false-positive cost is the primary constraint, Logistic Regression remains acceptable.

##### Scenario C — Operational constraint: Precision ≥ 0.60

* Random Forest

    * Substantially higher recall (0.476 vs 0.386)

    * Detects ~597 more defaults

    * Higher flagged_rate (+3.3%)

* Logistic Regression

    * Lower volume

    * Significantly lower recall

Decision:
When high precision is required but risk detection remains important, Random Forest clearly dominates by capturing significantly more defaults at moderate operational cost increase.

####

#### Executive Recommendation

####

Under recall-driven policies (≥ 0.60 or ≥ 0.70), Random Forest is technically superior, delivering higher precision and lower operational volume.

Under precision-driven policies, the optimal choice depends on the relative cost of false negatives versus false positives.

If regulatory interpretability is mandatory, Logistic Regression remains a defensible baseline.
If operational efficiency and risk capture are prioritized, Random Forest provides measurable business gains.