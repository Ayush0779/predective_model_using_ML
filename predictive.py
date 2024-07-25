import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv(r"C:\Users\Ayush Gupta\Desktop\machine failure.csv")

# Preprocess the dataset
df = pd.get_dummies(df, columns=['Type'],drop_first=True)
df.drop(columns=['UDI', 'Product ID'], inplace=True)
df.isnull().sum()
df.dropna(inplace=True)

# Split the data into features and target
X = df.drop(columns=['Machine failure'])
y = df['Machine failure']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
gbm_model = RandomForestClassifier(n_estimators=4,max_features=10,random_state=30)
gbm_model.fit(X_train, y_train)
y.unique()
gbm_model.feature_importances_
gbm_model.n_features_
gbm_model.criterion
# Make predictions and calculate the accuracy
y_pred = gbm_model.predict(X_test)
accuracy_score(y_pred,y_test)
confusion_matrix(y_pred,y_test)
X_train.shape,X_test.shape

# Save the trained model
#joblib.dump(gbm_model, r'C:\Users\856ma\Documents\FlaskApp\trained_model.pkl')

# Generate and save graphs
# Predicted vs Actual Values


# Feature Importances
feature_importances = gbm_model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig('feature_importances.png')
plt.show()
rep=classification_report(y_test,y_pred)
print(rep)
