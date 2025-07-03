import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz

print("🔍 Loading dataset...")

# ✅ Load from local file
data = pd.read_csv("heart.csv")

print("✅ Dataset loaded successfully!")
print(data.head())

# Features and target
X = data.drop('target', axis=1)
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree
print("\n🌲 Training Decision Tree...")
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)

# Export Decision Tree as PDF
print("📄 Exporting Decision Tree as PDF...")
dot_data = export_graphviz(dt_model, out_file=None,
                           feature_names=X.columns,
                           class_names=["No Disease", "Disease"],
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree")  # Creates 'decision_tree.pdf'

# Evaluate Decision Tree
y_pred_dt = dt_model.predict(X_test)
print("\n📊 Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# Random Forest
print("\n🌲🌲 Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("📊 Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Plot feature importance
print("\n📈 Plotting feature importance...")
importances = rf_model.feature_importances_
feat_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_importance.plot(kind='bar', color='skyblue')
plt.title("Feature Importance from Random Forest")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# Cross-validation
print("\n🔁 Cross-validating models...")
dt_cv = cross_val_score(dt_model, X, y, cv=5)
rf_cv = cross_val_score(rf_model, X, y, cv=5)

print("Decision Tree CV Accuracy:", np.round(dt_cv.mean(), 3))
print("Random Forest CV Accuracy:", np.round(rf_cv.mean(), 3))

input("\n✅ Task complete. Press Enter to exit...")
