import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
df = pd.read_excel('files/CLEANED DATA KATHY/Master Data.xlsx')
df_numeric = df.drop(columns=['Country Name'])

# Prepare independent and dependent variables
y = df_numeric['Public Debt']
X = df_numeric.drop(columns=['Public Debt'])


# Eliminate features with high VIF
def eliminate_high_vif_features(X, threshold=9.0):
    while True:
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        max_vif = vif_data["VIF"].max()
        if max_vif > threshold:
            drop_feature = vif_data.loc[vif_data["VIF"].idxmax(), "feature"]
            print(f"Removing due to high VIF: {drop_feature} (VIF={max_vif:.2f})")
            X = X.drop(columns=[drop_feature])
        else:
            break
    return X


X_vif_filtered = eliminate_high_vif_features(X)
print(X_vif_filtered.columns)


# Eliminate features with high p-values from regression
def eliminate_high_p_values(X, y, threshold=0.05):
    X_ = X.copy()
    while True:
        X_with_const = sm.add_constant(X_)
        model = sm.OLS(y, X_with_const).fit()
        p_values = model.pvalues.drop("const")
        max_p = p_values.max()
        if max_p > threshold:
            drop_feature = p_values.idxmax()
            print(f"Removing due to high p-value: {drop_feature} (p={max_p:.4f})")
            X_ = X_.drop(columns=[drop_feature])
        else:
            break
    return X_, model


X_final, final_model = eliminate_high_p_values(X_vif_filtered, y)
print("\n=== Final Regression Summary ===")
print(final_model.summary())

# Check multicollinearity among remaining variables
print("\n=== Multicollinearity Check for Remaining Variables ===")

# 1. VIF Check
vif_data = pd.DataFrame()
vif_data["feature"] = X_final.columns
vif_data["VIF"] = [variance_inflation_factor(X_final.values, i) for i in range(X_final.shape[1])]
print("\nVIF for Remaining Variables:")
print(vif_data.sort_values("VIF", ascending=False))

# SBreusch-Pagan test for homoscedasticity
X_with_const = sm.add_constant(X_final)
predictions = final_model.predict(X_with_const)
residuals = y - predictions
standardized_residuals = residuals / np.sqrt(final_model.scale)
bp_test = het_breuschpagan(residuals, X_with_const)
bp_lm, bp_lm_pvalue, bp_fvalue, bp_f_pvalue = bp_test

print("\nBreusch-Pagan Test for Heteroscedasticity:")
print(f"LM Statistic: {bp_lm:.4f}")
print(f"LM P-value: {bp_lm_pvalue:.4f}")
print(f"F Statistic: {bp_fvalue:.4f}")
print(f"F P-value: {bp_f_pvalue:.4f}")
print(f"Null Hypothesis: Homoscedasticity (constant variance)")
print(
    f"Conclusion: {'Reject null hypothesis (heteroscedasticity present)' if bp_lm_pvalue < 0.05 else 'Fail to reject null hypothesis (homoscedasticity assumed)'}")

# Apply log transformation to the dependent variable (if needed)
y_log_transformed = np.log(y)

# Fit the regression model with the log-transformed dependent variable
log_model = sm.OLS(y_log_transformed, sm.add_constant(X_final)).fit()

# Print the summary of the log-transformed model
print("\n=== Log-Transformed Regression Model Summary ===")
print(log_model.summary())

# 1. Get residuals and fitted values from the log-transformed model
log_residuals = log_model.resid
log_fitted_values = log_model.fittedvalues

# 2. Add a constant term to X_final for the intercept in the regression
X_with_const = sm.add_constant(X_final)

# 3. Perform the Breusch-Pagan test for heteroscedasticity using the residuals from the log-transformed model
bp_test = het_breuschpagan(log_residuals, X_with_const)

# Extract the test statistics
bp_lm, bp_lm_pvalue, bp_fvalue, bp_f_pvalue = bp_test

# 4. Print the results of the Breusch-Pagan test
print("\nBreusch-Pagan Test for Heteroscedasticity after Log Transformation:")
print(f"LM Statistic: {bp_lm:.4f}")
print(f"LM P-value: {bp_lm_pvalue:.4f}")
print(f"F Statistic: {bp_fvalue:.4f}")
print(f"F P-value: {bp_f_pvalue:.4f}")
print(f"Null Hypothesis: Homoscedasticity (constant variance)")

# Conclusion based on p-value
if bp_lm_pvalue < 0.05:
    print(f"Conclusion: Reject null hypothesis (heteroscedasticity present)")
else:
    print(f"Conclusion: Fail to reject null hypothesis (homoscedasticity assumed)")

X_reduced = X_final.drop(columns=['Net population migration', "Total labor force"])
log_model_reduced = sm.OLS(y_log_transformed, sm.add_constant(X_reduced)).fit()
print(log_model_reduced.summary())

# --- Standardized Coefficients (Feature Importance from OLS) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reduced)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_reduced.columns)

log_model_std = sm.OLS(y_log_transformed, sm.add_constant(X_scaled_df)).fit()
std_coeffs = pd.DataFrame({
    "Feature": X_scaled_df.columns,
    "Standardized Coefficient": log_model_std.params[1:]
}).sort_values(by="Standardized Coefficient", key=abs, ascending=False)

print("\n=== Standardized Coefficients (OLS Feature Importance) ===")
print(std_coeffs)

# --- Random Forest Feature Importances ---
rf = RandomForestRegressor(random_state=0)
rf.fit(X_reduced, y_log_transformed)
rf_importances = pd.DataFrame({
    "Feature": X_reduced.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n=== Random Forest Feature Importances ===")
print(rf_importances)

# --- Visualization ---
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.barplot(x="Standardized Coefficient", y="Feature", data=std_coeffs)
plt.title("OLS Standardized Coefficients")

plt.subplot(1, 2, 2)
sns.barplot(x="Importance", y="Feature", data=rf_importances)
plt.title("Random Forest Importances")
plt.tight_layout()
plt.show()

# Get final variables and prepare cluster input
final_vars = X_reduced.columns.tolist()
df_cluster = df[['Country Name', 'Public Debt'] + final_vars].dropna()

# Standardize selected variables
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(df_cluster[final_vars])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_cluster_scaled)

# Attach cluster labels
df_kmeans = df.loc[df_cluster.index].copy()
df_kmeans['Cluster'] = kmeans_labels

# Show cluster assignment
print("\nCluster assignment using KMEANS:")
print(df_kmeans[['Country Name', 'Cluster']])

silhouette = silhouette_score(X_cluster_scaled, kmeans_labels)
print("Silhouette Score:", silhouette)

print("Cluster Centroids:", kmeans.cluster_centers_)

# Filter to Cluster 0
cluster_0 = df_kmeans[df_kmeans['Cluster'] == 0]

# Sort by Public Debt (ascending) and get top 5
top5_low_debt = cluster_0.sort_values(by='Public Debt').head(5)

# Display result
print("Top 5 countries in Cluster 0 with the lowest public debt:")
print(top5_low_debt[['Country Name', 'Public Debt']])

