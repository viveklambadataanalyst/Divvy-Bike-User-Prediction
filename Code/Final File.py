import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import gc

#data loading
file_202409 = pd.read_csv("D:\\OneDrive - Cape Breton University\\Cape Breton University\\Semester 3 Subjects Material\\MGSC-5126-20-Data Mining\\Project\\Data\\September_divvy_tripdata.csv")
file_202410 = pd.read_csv("D:\\OneDrive - Cape Breton University\\Cape Breton University\\Semester 3 Subjects Material\\MGSC-5126-20-Data Mining\\Project\\Data\\October_divvy_tripdata.csv")
file_202411 = pd.read_csv("D:\\OneDrive - Cape Breton University\\Cape Breton University\\Semester 3 Subjects Material\\MGSC-5126-20-Data Mining\\Project\\Data\\November_divvy_tripdata.csv")
file_202412 = pd.read_csv("D:\\OneDrive - Cape Breton University\\Cape Breton University\\Semester 3 Subjects Material\\MGSC-5126-20-Data Mining\\Project\\Data\\December_divvy_tripdata.csv")
file_202501 = pd.read_csv("D:\\OneDrive - Cape Breton University\\Cape Breton University\\Semester 3 Subjects Material\\MGSC-5126-20-Data Mining\\Project\\Data\\January 2025_divvy_tripdata.csv")
combined_data = pd.concat([file_202409, file_202410, file_202411, file_202412, file_202501], ignore_index=True)

#data cleaning and preparation
combined_data.isnull().sum()
combined_data.dropna(inplace=True)
combined_data["start_date"] = pd.to_datetime(combined_data["started_at"]).dt.date
combined_data["end_date"] = pd.to_datetime(combined_data["ended_at"]).dt.date
combined_data["start_time"] = pd.to_datetime(combined_data["started_at"])
combined_data["end_time"] = pd.to_datetime(combined_data["ended_at"])
combined_data = combined_data.drop(columns=["started_at", "ended_at", "start_lat", "start_lng", "end_lat", "end_lng"]).reset_index(drop=True)
combined_data["start_day"] = pd.to_datetime(combined_data["start_date"]).dt.day_name()
combined_data["end_day"] = pd.to_datetime(combined_data["end_date"]).dt.day_name()
if combined_data["ride_id"].isnull().sum() == 0 and combined_data["ride_id"].nunique() == len(combined_data):
    combined_data.set_index("ride_id", inplace=True)
combined_data["time_diff_sec"] = combined_data["end_time"] - combined_data["start_time"]
combined_data["ride_duration"] = combined_data["time_diff_sec"].dt.total_seconds() / 60
combined_data = combined_data.loc[combined_data["ride_duration"] > 0].reset_index(drop=True)
combined_data["date_week"] = pd.to_datetime(combined_data["start_date"]).dt.weekday #(0=Monday, 6=Sunday

#idenitfying and dealing with outliers
Q1 = combined_data["ride_duration"].quantile(0.25)
Q3 = combined_data["ride_duration"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
median = combined_data["ride_duration"].median()
outliers = combined_data[(combined_data["ride_duration"] < lower_bound) | (combined_data["ride_duration"] > upper_bound)]
print(outliers.sort_values(by="ride_duration", ascending=True))
combined_data = combined_data[(combined_data["ride_duration"] < upper_bound) & (combined_data["ride_duration"] > lower_bound)]
print(combined_data.sort_values(by="ride_duration", ascending=False))
print(combined_data.info())
print(combined_data.describe())
print("Start Station ID:\n", combined_data["start_station_id"].nunique())
print("Start Station Name:\n", combined_data["start_station_name"].nunique())
print("End Station ID:\n", combined_data["end_station_id"].nunique())
print("End Station Name:\n", combined_data["end_station_name"].nunique())

#plotting outliers in boxplot
plt.figure(figsize=(6, 4))
ax = sns.boxplot(y=combined_data["ride_duration"], color="lightblue")
ax.text(0, Q1, f"Q1: {Q1:.2f}", ha="left", va="center", fontsize=10, color="blue")
ax.text(0, Q3, f"Q3: {Q3:.2f}", ha="left", va="center", fontsize=10, color="blue")
ax.text(0, median, f"Median: {median:.2f}", ha="left", va="center", fontsize=10, color="green")
ax.text(0, lower_bound, f"Lower Bound: {lower_bound:.2f}", ha="right", va="center", fontsize=10, color="red")
ax.text(0, upper_bound, f"Upper Bound: {upper_bound:.2f}", ha="right", va="center", fontsize=10, color="red")
plt.title("Boxplot for Outliers in Time Differance")
plt.ylabel("ride_duration")
plt.show()

#plotting time difference in minute
plt.figure(figsize=(8, 5))
sns.kdeplot(combined_data["ride_duration"], shade=True, color="skyblue")
plt.axvline(combined_data["ride_duration"].median(), color='red', linestyle="--", label=f"Median: {combined_data["ride_duration"].median():.2f}")
plt.xlabel("Ride Duration (Minutes)")
plt.ylabel("Frequency")
plt.title("KDE Plot of Ride Duration in Minutes")
plt.legend()
plt.show()

# Rideable Type Distribution
plt.figure(figsize=(8,6))
ax=sns.countplot(x='rideable_type', data=combined_data)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=10, fmt='%d', padding=3)
plt.title('Distribution of Rideable Types')
plt.xlabel('Rideable Type')
plt.ylabel('Number of Rides')
plt.show()

# Membership Type Distribution
plt.figure(figsize=(6,6))
ax=sns.countplot(x='member_casual', data=combined_data)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=10, fmt='%d', padding=3)
plt.title('Distribution of Membership Status')
plt.xlabel('Membership Status')
plt.ylabel('Number of Rides')
plt.show()

# Ride Duration by Member Type
plt.figure(figsize=(10,6))
ax=sns.boxplot(x='member_casual', y='ride_duration', data=combined_data)
for i, member_type in enumerate(combined_data['member_casual'].unique()):
    subset = combined_data[combined_data['member_casual'] == member_type]['ride_duration']
    Q1 = np.percentile(subset, 25)  # First quartile
    Q3 = np.percentile(subset, 75)  # Third quartile
    Median = np.median(subset)      # Median
    IQR = Q3 - Q1                   # Interquartile range
    Lower_Bound = Q1 - 1.5 * IQR
    Upper_Bound = Q3 + 1.5 * IQR
    
    ax.text(i, Q1, f'Q1: {Q1:.1f}', ha='center', va='bottom', fontsize=10, color='blue')
    ax.text(i, Q3, f'Q3: {Q3:.1f}', ha='center', va='top', fontsize=10, color='blue')
    ax.text(i, Median, f'Median: {Median:.1f}', ha='center', va='center', fontsize=10, fontweight='bold', color='red')
    ax.text(i, Lower_Bound, f'LB: {Lower_Bound:.1f}', ha='center', va='bottom', fontsize=10, color='green')
    ax.text(i, Upper_Bound, f'UB: {Upper_Bound:.1f}', ha='center', va='top', fontsize=10, color='green')
plt.title('Ride Duration by Member Type')
plt.xlabel('Member Type')
plt.ylabel('Ride Duration (Minutes)')
plt.show()

# Create a crosstab of 'day_of_week' and 'member_casual'
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
combined_data['date_week'] = pd.Categorical(combined_data['date_week'], categories=day_order, ordered=True)
rides_by_day = pd.crosstab(combined_data['date_week'], combined_data['member_casual'])

#Plotting the bar graph
plt.figure(figsize=(10, 6))
ax = rides_by_day.plot(kind='bar', color=['skyblue', 'lightgreen'], width=0.8)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=10, fmt='%d', padding=3)
plt.title('Trips by Day of the Week: Member vs Casual', fontsize=14)
plt.xlabel('Day of the Week', fontsize=12)
plt.ylabel('Number of Rides', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Membership Status')
plt.tight_layout()
plt.show()

#creating list of months for plotting
combined_data['month'] = pd.to_datetime(combined_data['start_date']).dt.month_name()
combined_data = combined_data[combined_data['month'] != 'August']
month_order = ["September", "October", "November", "December", "January"]
combined_data['month'] = pd.Categorical(combined_data['month'], categories=month_order, ordered=True)
rides_by_month = pd.crosstab(combined_data['month'], combined_data['member_casual'])

#Plotting the line graph
plt.figure(figsize=(10, 6))
ax = rides_by_month.plot(kind='line', marker='o', color=['skyblue', 'lightgreen'], linewidth=2)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=10, fmt='%d', padding=3)
for line in ax.lines:
    for x, y in zip(line.get_xdata(), line.get_ydata()):
        ax.text(x, y, f'{int(y)}', color='black', ha='center', va='bottom', fontsize=10)
plt.title('Trips by Month: Member vs Casual (Excluding August)', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of Rides', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Membership Status')
plt.tight_layout()
plt.show()

#Top 20 Start Stations
plt.figure(figsize=(12,6))
ax=combined_data['start_station_name'].value_counts().head(20).plot(kind='bar')
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=10, fmt='%d', padding=3)
plt.title('Top 20 Start Stations')
plt.ylabel('Number of Rides')
plt.xlabel('Station Name')
plt.show()

#Top 20 End Stations
plt.figure(figsize=(12,6))
ax=combined_data['end_station_name'].value_counts().head(20).plot(kind='bar', color='lightgreen')
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=10, fmt='%d', padding=3)
plt.title('Top 20 End Stations')
plt.xlabel('Station Name')
plt.ylabel('Number of Rides')
plt.xticks(rotation=45)
plt.show()

#Count rides by start station and membership status
rides_by_station_member_casual = pd.crosstab(combined_data['start_station_name'], combined_data['member_casual'])
top_stations = rides_by_station_member_casual.sum(axis=1).sort_values(ascending=False).head(10)
rides_by_top_stations = rides_by_station_member_casual.loc[top_stations.index]
 
#Plotting the grouped bar chart
plt.figure(figsize=(12, 6))
ax=rides_by_top_stations.plot(kind='bar', width=0.8)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=10, fmt='%d', padding=3)
plt.title('Top 10 Start Stations: Member vs Casual', fontsize=14)
plt.xlabel('Station Name', fontsize=12)
plt.ylabel('Number of Rides', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Membership Status', labels=['Casual', 'Member'])
plt.tight_layout()
plt.show()

#Count rides by end station and membership status
rides_by_station_member_casual = pd.crosstab(combined_data['end_station_name'], combined_data['member_casual'])
top_stations = rides_by_station_member_casual.sum(axis=1).sort_values(ascending=False).head(10)
rides_by_top_stations = rides_by_station_member_casual.loc[top_stations.index]
 
#Plotting the grouped bar chart
plt.figure(figsize=(12, 6))
ax=rides_by_top_stations.plot(kind='bar', width=0.8)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=10, fmt='%d', padding=3)
plt.title('Top 10 End Stations: Member vs Casual', fontsize=14)
plt.xlabel('Station Name', fontsize=12)
plt.ylabel('Number of Rides', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Membership Status', labels=['Casual', 'Member'])
plt.tight_layout()
plt.show()

#Top 10 Start Stations: Rideable Type
rides_by_station_rideable_type = pd.crosstab(combined_data['start_station_name'], combined_data['rideable_type'])
top_stations = rides_by_station_rideable_type.sum(axis=1).sort_values(ascending=False).head(10)
rides_by_top_stations = rides_by_station_rideable_type.loc[top_stations.index]
ax=rides_by_top_stations.plot(kind='bar', figsize=(12, 6), width=0.8)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=10, fmt='%d', padding=3)
plt.title('Top 10 Start Stations by Rideable Type')
plt.xlabel('Station Name')
plt.ylabel('Number of Rides')
plt.legend(title='Rideable Type')
plt.xticks(rotation=45)
plt.show()

#Top 10 End Stations: Rideable Type
rides_by_station_rideable_type = pd.crosstab(combined_data['end_station_name'], combined_data['rideable_type'])
top_stations = rides_by_station_rideable_type.sum(axis=1).sort_values(ascending=False).head(10)
rides_by_top_stations = rides_by_station_rideable_type.loc[top_stations.index]
ax=rides_by_top_stations.plot(kind='bar', figsize=(12, 6), width=0.8)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=10, fmt='%d', padding=3)
plt.title('Top 10 End Stations by Rideable Type')
plt.xlabel('Station Name')
plt.ylabel('Number of Rides')
plt.legend(title='Rideable Type')
plt.xticks(rotation=45)
plt.show()

#splitting train test and unseen data from main dataframe
date_threshold = dt.strptime("2024-12-31", "%Y-%m-%d").date()
combined_data_train_test = combined_data[combined_data["start_date"] <= date_threshold].reset_index(drop=True)
combined_data_unseen = combined_data[combined_data["start_date"] > date_threshold].reset_index(drop=True)
print(combined_data_train_test.head())
print(combined_data_train_test.columns)
print(combined_data_train_test["start_date"].max())
print(combined_data_train_test["start_date"].min())
print(combined_data_unseen.head())
print(combined_data_unseen.columns)
print(combined_data_unseen["start_date"].max())
print(combined_data_unseen["start_date"].min())

#logistic regression
#feature selection
features_lr = ['date_week', 'ride_duration']
target_lr = 'member_casual'

#encoding target variable
label_encoder_lr = LabelEncoder()
combined_data_train_test[target_lr] = label_encoder_lr.fit_transform(combined_data_train_test[target_lr])  # 0: Casual, 1: Member
le_name_mapping = dict(zip(label_encoder_lr.classes_, label_encoder_lr.transform(label_encoder_lr.classes_)))
print(le_name_mapping)

#train test split
X = combined_data_train_test[features_lr]
y = combined_data_train_test[target_lr]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train model
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)

# Predict
y_pred = logistic_regression.predict(X_test)
joblib.dump(logistic_regression, "logistic_regression.pkl")

#reverse transformation of encoded target variable
y_pred_lables = label_encoder_lr.inverse_transform(y_pred)

#Evaluate train test model
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Logistic Regression Report:\n", classification_report(y_test, y_pred))
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#Confusion Matrix Visualization for train test data
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Casual", "Member"], yticklabels=["Casual", "Member"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - Logistic Regression")
plt.show()

#prediciting uneen data
X_unseen = combined_data_unseen[features_lr]
logistic_regression = joblib.load("logistic_regression.pkl")
y_pred_unseen = logistic_regression.predict(X_unseen)
y_unseen_pred_lables = label_encoder_lr.inverse_transform(y_pred_unseen)
combined_data_unseen["predicted_member_casual"] = y_unseen_pred_lables
print(combined_data_unseen[["member_casual", "predicted_member_casual"]].head(20))
y_unseen_actual = label_encoder_lr.transform(combined_data_unseen["member_casual"])

#Evaluate unseen model
print("Unseen Logistic Regression Accuracy:", accuracy_score(y_unseen_actual, y_pred_unseen))
print("Unseen Logistic Regression Report:\n", classification_report(y_unseen_actual, y_pred_unseen))
print("Unseen Logistic Regression Confusion Matrix:\n", confusion_matrix(y_unseen_actual, y_pred_unseen))

#Confusion Matrix Visualization for train test data
cm = confusion_matrix(y_unseen_actual, y_pred_unseen)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Casual", "Member"], yticklabels=["Casual", "Member"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - Logistic Regression")
plt.show()

#KNN Classification
# Standardize features
X = combined_data_train_test[['date_week', 'ride_duration']]
y = combined_data_train_test["member_casual"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Find best k value using Elbow Method - as per errors
error_rates = []
k_range = range(1, 31)
for k in k_range:
    knn1 = KNeighborsClassifier(n_neighbors=k)
    knn1.fit(X_train, y_train)
    y_pred1 = knn1.predict(X_test)
    error = 1 - accuracy_score(y_test, y_pred1)
    error_rates.append(error)

plt.figure(figsize=(8, 5))
plt.plot(k_range, error_rates, marker="o", linestyle="dashed", color="b")
plt.xlabel("Number of Neighbors(k)")
plt.ylabel("Error Rate")
plt.title("Elbow Method for Optimal k")
plt.show()

# Find best k value using Elbow Method - as per accuracy
k_values = range(1, 31)
accuracy_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

# Plot accuracy vs. k
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracy_scores, marker="o", linestyle="-", color="b")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.title("KNN Hyperparameter Tuning - Selecting Best k")
plt.xticks(k_values)
plt.grid(True)
plt.show()

best_k1 = k_range[np.argmin(error_rates)]
print("Best k value for lowest Error:", best_k1)
print("Least error:", max(error_rates))

best_k2 = k_values[np.argmax(accuracy_scores)]
print("Best k value for highest Error:", best_k2)
print("Best accuracy:", max(accuracy_scores))

best_k = min(best_k1, best_k2)

# Train final KNN model with best k
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
joblib.dump(knn_best, "knn_model.pkl")
# Predict
y_pred_best = knn_best.predict(X_test)

# Evaluate Model
accuracy_best = accuracy_score(y_test, y_pred_best)
print("Final KNN Accuracy with Best k:", accuracy_best)
print("KNN Report:\n", classification_report(y_test, y_pred_best))
print(confusion_matrix(y_test, y_pred_best))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Casual", "Member"], yticklabels=["Casual", "Member"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - KNN (k={best_k})")
plt.show()

#prediciting unseen data
X_unseen = combined_data_unseen[['date_week', 'ride_duration']]
X_unseen_scaled = scaler.transform(X_unseen)

knn = joblib.load("knn_model.pkl")
y_pred_unseen = knn.predict(X_unseen_scaled)
combined_data_unseen["predicted_member_casual"] = y_pred_unseen
print(combined_data_unseen[["member_casual", "predicted_member_casual"]].head(20))

#Evaluate unseen model
print("Unseen KNN Accuracy:", accuracy_score(combined_data_unseen["member_casual"], y_pred_unseen))
print("Unseen KNN Report:\n", classification_report(combined_data_unseen["member_casual"], y_pred_unseen))
print("Unseen KNN Confusion Matrix:\n", confusion_matrix(combined_data_unseen["member_casual"], y_pred_unseen))

#Confusion Matrix Visualization for unsen
cm = confusion_matrix(combined_data_unseen["member_casual"], y_pred_unseen)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Casual", "Member"], yticklabels=["Casual", "Member"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - KNN")
plt.show()

#Getting the important features
# Feature selection
features = ['start_station_id', 'end_station_id', 'date_week', 'ride_duration']
target = 'member_casual'

#encoding target variable
label_encoder = LabelEncoder()
combined_data_train_test[target] = label_encoder.fit_transform(combined_data_train_test[target])  # 0: Casual, 1: Member
le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(le_name_mapping)

#encoding features
label_encoders = {}
for col in ['start_station_id', 'end_station_id']: #start_station_id and end_station_id are stringtype
    le = LabelEncoder()
    combined_data_train_test[col] = le.fit_transform(combined_data_train_test[col])  # Convert categorical values to numeric
    label_encoders[col] = le

#train test split
X = combined_data_train_test[features]
y = combined_data_train_test[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#XGBoost Classification
#Train XGB model
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb_clf.fit(X_train, y_train)

#XGB feature importance
xgb_clf_feature_importance = pd.Series(xgb_clf.feature_importances_, index=features).sort_values(ascending=False)
print(xgb_clf_feature_importance)

#plotting feature importane for XGB
xgb_clf_feature_importance.plot(kind='bar', title='XGBoost Feature Importance', color='orange')
plt.show()

#Using features as per the results of Feature Importane for RF and XGB
updated_features_xgb = ["date_week", "ride_duration"] #date_week is more than 0.4 for XGB,  ride_duration is about 0.2 and end_station_id, start_station_id are little below 0.2
target_xgb = "member_casual"

#train test split
X = combined_data_train_test[updated_features_xgb]
y = combined_data_train_test[target_xgb]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train XGB model
new_xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
new_xgb_clf.fit(X_train, y_train)

#XGB Prediction
y_pred_xgb_new = new_xgb_clf.predict(X_test)
joblib.dump(new_xgb_clf, "xgb_model.pkl")

#XGB Model Evaluation
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb_new))
print("XGBoost Report:\n", classification_report(y_test, y_pred_xgb_new))
print("XGBoost Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb_new))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred_xgb_new)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Casual", "Member"], yticklabels=["Casual", "Member"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - XGB")
plt.show()

#prediciting uneen data
X_unseen = combined_data_unseen[updated_features_xgb]
new_xgb_clf = joblib.load("xgb_model.pkl")
y_pred_unseen = new_xgb_clf.predict(X_unseen)
y_unseen_pred_lables = label_encoder.inverse_transform(y_pred_unseen)
combined_data_unseen["predicted_member_casual"] = y_unseen_pred_lables
print(combined_data_unseen[["member_casual", "predicted_member_casual"]].head(20))
y_unseen_actual = label_encoder.transform(combined_data_unseen["member_casual"])

#Evaluate unseen model
print("Unseen XGB Classification Accuracy:", accuracy_score(y_unseen_actual, y_pred_unseen))
print("Unseen XGB Classification Report:\n", classification_report(y_unseen_actual, y_pred_unseen))
print("Unseen XGB Classification Confusion Matrix:\n", confusion_matrix(y_unseen_actual, y_pred_unseen))

#Confusion Matrix Visualization for train test data
cm = confusion_matrix(y_unseen_actual, y_pred_unseen)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Casual", "Member"], yticklabels=["Casual", "Member"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - XGB Classification")
plt.show()

#Getting the important features
#random forest classification
# Feature selection
features = ['start_station_id', 'end_station_id', 'date_week', 'ride_duration']
target = 'member_casual'

#encoding target variable
label_encoder_rf = LabelEncoder()
combined_data_train_test[target] = label_encoder_rf.fit_transform(combined_data_train_test[target])  # 0: Casual, 1: Member
le_name_mapping = dict(zip(label_encoder_rf.classes_, label_encoder_rf.transform(label_encoder_rf.classes_)))
print(le_name_mapping)

#encoding features
label_encoders = {}
for col in ['start_station_id', 'end_station_id']: #start_station_id and end_station_id are stringtype
    le = LabelEncoder()
    combined_data_train_test[col] = le.fit_transform(combined_data_train_test[col])
    label_encoders[col] = le

#train test split
X = combined_data_train_test[features]
y = combined_data_train_test[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train Randomg forest model
rf_clf = RandomForestClassifier(n_estimators=20, warm_start=True, random_state=42)
for i in range(20, 101, 10):  #Adding 20 trees at a time
    rf_clf.n_estimators = i
    rf_clf.fit(X_train, y_train)

#RF feature importance
rf_clf_feature_importance = pd.Series(rf_clf.feature_importances_, index=features).sort_values(ascending=False)
print(rf_clf_feature_importance)

#plotting feature importane for RF
rf_clf_feature_importance.plot(kind='bar', title='Random Forest Feature Importance', color='skyblue')
plt.show()

#Using features as per the results of Feature Importane for RF and XGB
updated_features_rf = ["ride_duration", "start_station_id", "end_station_id"] #ride_duration is more than 0.4 for RF, end_station_id, start_station_id are little above 0.2 and date_week is less than 0.1
target_rf = "member_casual"

#train test split
X = combined_data_train_test[updated_features_rf]
y = combined_data_train_test[target_rf]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train Random forest model
new_rf_clf = RandomForestClassifier(n_estimators=20, warm_start=True, random_state=42)
for i in range(20, 101, 10):  #Adding 20 trees at a time
    new_rf_clf.n_estimators = i
    new_rf_clf.fit(X_train, y_train)

#Random Forest Prediction
y_pred_rf_new = new_rf_clf.predict(X_test)
joblib.dump(new_rf_clf, "rf_model.pkl")

#Random Forest Model Evaluation
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf_new))
print("RF Report:\n", classification_report(y_test, y_pred_rf_new))
print("RF Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf_new))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred_rf_new)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Casual", "Member"], yticklabels=["Casual", "Member"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - RF")
plt.show()

#prediciting uneen data
X_unseen = combined_data_unseen[updated_features_rf]
gc.collect()
new_rf_clf = joblib.load("rf_model.pkl", mmap_mode="r")
y_pred_unseen = new_rf_clf.predict(X_unseen)
y_unseen_pred_lables = label_encoder_rf.inverse_transform(y_pred_unseen)
combined_data_unseen["predicted_member_casual"] = y_unseen_pred_lables
print(combined_data_unseen[["member_casual", "predicted_member_casual"]].head(20))
y_unseen_actual = label_encoder_rf.transform(combined_data_unseen["member_casual"])

#Evaluate unseen model
print("Unseen RF Classification Accuracy:", accuracy_score(y_unseen_actual, y_pred_unseen))
print("Unseen RF Classification Report:\n", classification_report(y_unseen_actual, y_pred_unseen))
print("Unseen RF Classification Confusion Matrix:\n", confusion_matrix(y_unseen_actual, y_pred_unseen))

#Confusion Matrix Visualization for train test data
cm = confusion_matrix(y_unseen_actual, y_pred_unseen)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Casual", "Member"], yticklabels=["Casual", "Member"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - RF Classification")
plt.show()