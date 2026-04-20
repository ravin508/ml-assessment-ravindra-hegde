B1. Problem Formulation:

(a) Formulate this as a machine learning problem: 

This is formulated as a supervised regression problem (specifically a “predict-then-optimize” setup).

Target variable: Monthly items sold (sales volume / quantity) at the store-month level under the promotion that was actually run.
Candidate input features:
Store attributes: size, location type (urban / semi-urban / rural), average monthly footfall, competition density, customer demographics (e.g., average age, income band).
Promotion type (categorical: Flat Discount, BOGO, Free Gift with Purchase, Category-Specific Offer, Loyalty Points Bonus).
Temporal / calendar features: month / season, proportion or count of weekend days, festival / holiday flags (aggregated per month).
Historical / lagged features: previous month’s sales volume, previous promotion type.
Interactions: promotion type × location type, promotion type × month, etc.


Justification for regression: The business goal is to maximise a continuous numeric outcome (number of items sold). A regression model predicts the expected sales volume for a given store-month and each possible promotion. At inference time we generate five predictions per store-month (one per promotion) and simply choose the promotion with the highest predicted volume. Classification would only tell us “which is best” without quantifying the expected lift, while regression directly supports the optimisation objective and handles varying store baselines naturally.


(b) Why items sold (sales volume) is a more reliable target than total sales revenue:
Revenue is confounded by pricing mechanics that differ across promotions:

Flat Discount and Category-Specific Offer directly reduce price per item.
BOGO and Free Gift with Purchase increase unit volume but may keep the “paid” price the same or add gift cost.
Loyalty Points Bonus may not reduce price at all.

Thus a promotion that drives high volume can paradoxically show lower revenue. Items sold directly measures the core operational goal (moving more product, clearing inventory, driving traffic) without these price distortions.

Broader principle: In real-world ML projects the target variable must be chosen so that it (1) aligns as directly as possible with the true business objective and (2) is minimally confounded by external factors that the model cannot control. Optimising a noisy proxy (revenue) can lead to Goodhart’s-Law problems where the model “games” the proxy rather than the underlying goal.

(c) Alternative modelling strategy to a single global model:

Proposed strategy: Build either (i) store-specific models (one model per store) or (ii) a single global model with store-level heterogeneity (store ID as a categorical feature + explicit interaction terms between promotion type and store characteristics, or store embeddings / hierarchical random effects).
Justification: The problem statement explicitly states that “stores in different locations respond very differently to the same promotion” because of variation in customer demographics, competition density, footfall patterns, and local culture. A single global model averages these heterogeneous treatment effects and will therefore recommend sub-optimal promotions for many stores. Store-specific or heterogeneity-aware models learn local response functions, improving both predictive accuracy and business lift. With 36 months × 50 stores ≈ 1 800 rows, per-store models are feasible if regularised (e.g., LightGBM with early stopping); the interaction/hierarchical approach is a good middle ground when data per store is limited.

_____________________________________________________________

B2: Data and EDA Strategy

(a) Joining the tables and grain of the modelling dataset:

Join logic (performed in SQL or pandas):

transactions JOIN calendar ON transactions.date = calendar.date → adds weekend / festival flags to every transaction.
transactions (now with flags) LEFT JOIN promotion_details ON the linking key (promotion_id or store_id + month) → attaches the promotion type that was active that month.
The enriched transaction table JOIN store_attributes ON store_id → brings in static store features.

Grain of the final modelling dataset: one row = one store-month.
Aggregations performed before modelling (GROUP BY store_id, year_month):

SUM(quantity) → target = monthly items sold.
COUNT(*) → number of transactions (optional diagnostic).
Calendar flags → SUM(weekend_flag) / COUNT(*) = proportion of weekend days; MAX(festival_flag) or count of festival days.
Promotion type → MODE() or MAX() (assumed one dominant promotion per store-month).
Store attributes are constant per store and are carried forward.

The resulting dataset has ~1 800 rows (50 stores × 36 months) and is ready for feature engineering.


(b) EDA before building a model:

I would perform the following four analyses (plus supporting tables):

Sales volume distribution by promotion type and location type (boxplots / violin plots, faceted by urban/semi-urban/rural).
Look for: Which promotions drive highest median volume and variance in each location segment.
Influence: Strong interaction effects → create explicit promo × location features or justify segmented modelling.
Time-series plots of monthly sales volume per store (or aggregated by location type).
Look for: Seasonality, trends, spikes around festivals, and any structural breaks.
Influence: Add month/season features or Fourier terms; decide whether to include trend terms or use time-series cross-validation.
Correlation heatmap + pair-plots among numerical features and target.
Look for: Multicollinearity (e.g., store size vs footfall), strongest predictors of volume, zero-inflation in sales.
Influence: Feature selection / removal of redundant variables; decide on scaling or transformations (log target if skewed).
Promotion assignment frequency and balance (stacked bar chart of promotion type by store and by month).
Look for: Whether promotions were assigned randomly or biased toward certain stores/months; coverage of each promo type.
Influence: If severely imbalanced, consider causal methods (propensity weighting) or re-balancing; also guides whether “No Promotion” should be treated as a sixth baseline category.

These insights directly feed feature engineering (interactions, seasonal encodings) and modelling choices (segmentation vs global model).

(c) Impact of 80 % no-promotion transactions and mitigation steps:
Impact: The model can become biased toward the “no-promotion” baseline because it has far more training examples. It will under-estimate the incremental lift of the five active promotions and may default to recommending no action even when a promotion would be beneficial.
Steps to address:

Explicitly treat “No Promotion” as a sixth category in the promotion feature (if present in data).
Apply class / sample weighting during training so promotion months are not under-represented.
Use targeted oversampling of promotion periods or down-sampling of no-promo periods.
Model the lift (sales volume minus a no-promo baseline) as an alternative target when possible.
Evaluate performance metrics only on promotion months or use uplift-style metrics to ensure the model learns true promotional effects.

_________________________________________________________

B3: Model Evaluation and Deployment

(a) Train-test split, why random is inappropriate, and evaluation metrics
Train-test split: Time-based (chronological) split — train on the first ~24–28 months, test on the most recent 8–12 months (or use rolling-window / walk-forward validation). For hyper-parameter tuning use time-series cross-validation (expanding or sliding windows).
Why random split is inappropriate: The data has strong temporal structure (seasonality, trends, festival effects, evolving customer behaviour). A random split would leak future information into training and produce overly optimistic performance that does not reflect real deployment (predicting the future). It also breaks the causal ordering required for realistic recommendation evaluation.
Evaluation metrics (and business interpretation):

RMSE / MAE on predicted vs actual items sold → measures raw prediction accuracy in units the business cares about (extra items moved).
Mean Absolute Percentage Error (MAPE) or % Lift → relative error; communicates expected % increase in volume when the model’s recommendation is followed versus historical baseline.
Recommendation success rate (in test set): proportion of months where the model’s chosen promotion would have produced higher actual volume than the promotion that was actually run (requires counterfactual simulation).
Business simulation: total extra items sold across the test period if the retailer had followed model recommendations vs actual historical promotions.

(b) Investigating why the model recommends different promotions for the same store in different months
Use local feature attribution (SHAP values) or partial dependence plots for the specific store-month predictions.
Investigation steps:

For December (Loyalty Points Bonus) and March (Flat Discount) predictions of Store 12, extract the top SHAP contributors.
Compare the contribution of month/season features, festival flag, lagged sales, and the interaction terms (promo × month).
Example insight: December may have a high positive SHAP from “festival_flag + Loyalty Points” because loyal customers stock up during festive season; March may show strong response to “Flat Discount” because it is a post-festival clearance month with price-sensitive impulse buyers.

Communication to marketing team:

One-page dashboard showing the two predictions side-by-side with SHAP waterfall plots.
Partial-dependence or interaction plots highlighting how the same promotion’s effect changes across months for that store.
Simple English summary: “In December the model expects Loyalty Points to drive +18 % more volume because of festive repeat-buying; in March a straight discount better stimulates price-driven traffic.”

(c) End-to-end deployment process
Saving the model: Persist the final trained model (and preprocessing pipeline) using joblib / pickle or, preferably, MLflow / BentoML for versioning and reproducibility.
Monthly inference workflow (run at the start of every month):

Ingest latest data: updated store attributes (if any), calendar for the new month, and any new historical transactions up to the previous month.
For each of the 50 stores create five candidate rows (one per promotion type).
Apply the same preprocessing pipeline (encoding, scaling, feature interactions, lag creation using the most recent historical data).
Generate predictions → select the promotion with the highest predicted items sold per store.
Output a simple table / API payload: store_id | recommended_promotion | expected_volume_lift.

Monitoring & retraining triggers:

Feature drift detection (Kolmogorov-Smirnov or population stability index on input distributions month-over-month).
Prediction vs actual performance tracking once the month ends (compare predicted volume of chosen promo vs realised volume).
Business KPI monitoring: actual % lift in items sold versus a simple baseline (e.g., last year’s same month).
Concept drift alerts (e.g., sudden change in promotion effectiveness).
Automatic retraining threshold: if MAE on the latest rolling window exceeds a pre-defined tolerance (or lift falls below target), or every 6–12 months regardless. Retraining uses the same time-based CV on all available data.

This closed-loop system ensures recommendations remain fresh without requiring manual intervention each month.




