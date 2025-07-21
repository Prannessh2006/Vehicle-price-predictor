import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

df = pd.read_csv('dataset.csv')

print("Dataset shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=30, kde=True)
plt.title('Price Distribution')
plt.show()

df['age'] = 2024 - df['year']
df = df.drop(['name', 'description'], axis=1)

df['mileage'] = df['mileage'].fillna(df['mileage'].median())
df['cylinders'] = df['cylinders'].fillna(0).astype(int)

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numerical_features = ['year', 'mileage', 'age', 'doors']
categorical_features = [
    'make', 'model', 'fuel', 'transmission', 'trim', 
    'body', 'exterior_color', 'interior_color', 'drivetrain'
]

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(
        n_estimators=200, 
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\nModel Performance:\nRMSE: {rmse:.2f}\nR²: {r2:.4f}")

feature_names = numerical_features + list(
    model.named_steps['preprocessor']
    .named_transformers_['cat']
    .named_steps['onehot']
    .get_feature_names_out(categorical_features)
)

importances = model.named_steps['regressor'].feature_importances_
sorted_idx = importances.argsort()

plt.figure(figsize=(10, 15))
plt.barh(np.array(feature_names)[sorted_idx][-20:], importances[sorted_idx][-20:])
plt.title("Top 20 Important Features")
plt.show()

joblib.dump(model, 'vehicle_price_predictor.pkl')

def predict_price(make, model, year, fuel, transmission, trim, body, 
                 doors, exterior_color, interior_color, drivetrain, mileage):
    data = {
        'make': [make],
        'model': [model],
        'year': [year],
        'fuel': [fuel],
        'transmission': [transmission],
        'trim': [trim],
        'body': [body],
        'doors': [doors],
        'exterior_color': [exterior_color],
        'interior_color': [interior_color],
        'drivetrain': [drivetrain],
        'mileage': [mileage],
        'age': [2024 - year]
    }
    df_input = pd.DataFrame(data)
    loaded_model = joblib.load('vehicle_price_predictor.pkl')
    return loaded_model.predict(df_input)[0]
