# Model Comparison

### 

### Objective



Evaluate and compare two models for the Credit Default classification problem using:



Primary metric: ROC-AUC (ranking performance)



Operational metrics: Precision@20% and Recall@20%



The dataset used is the UCI Credit Default dataset.



### 

### Models Evaluated



Logistic Regression (with log transformation on PAY\_AMT features)



Random Forest (baseline configuration)



The log transformation was applied to PAY\_AMT features due to extreme skewness, improving linear separability.



### 

### Cross-Validation Results

##### 

|Model|CV ROC/AUC mean|CV std|Precision@20%|Recall@20%|
|-|-|-|-|-|
|Logistic Regression w/Log-transformation|0.7470421388730627|0.005015243690172761|0.55|0.49736247174076864|
|Random Forest baseline|0.7797949865296616|0.005263561652388726|0.565|0.5109269027882442|

##### 



##### 

#### Logistic Regression – Analysis

##### 

##### Advantages

* Interpretable coefficients
* Well-calibrated probabilities
* Stable and reproducible
* Lower computational cost
* Fully documented and script-based

##### 

##### Limitations

* Limited ability to model non-linear interactions
* Appears to be approaching its linear capacity limit







### Random Forest – Analysis



##### Advantages

* Captures non-linear relationships and interactions
* Significant ROC-AUC improvement (+0.033 absolute increase)
* Improved Precision@20% and Recall@20%
* Similar cross-validation stability



##### Limitations

* Less interpretable
* Requires additional tools (e.g., SHAP) for explainability
* Higher computational cost at scale
* Not yet fully production-hardened





### Conclusion



The Random Forest demonstrates a substantial improvement in ranking performance while maintaining stability across folds, suggesting that the problem exhibits non-linear structure. However, the Logistic Regression model remains a strong and defensible baseline due to its interpretability and production readiness. Final model selection would depend on regulatory requirements, interpretability constraints, and business priorities.

