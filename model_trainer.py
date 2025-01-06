# pylint: disable=E0401, W0401
from libraries import *

data = "data/updated_pollution_dataset.csv"

df = pd.read_csv(data)
df.columns = df.columns.str.lower()
df.rename(columns={'air quality': 'air_quality'}, inplace=True)

# Feature selection
features = [
    'temperature',
    'humidity',
    'pm2.5',
    'no2',
    'so2',
    'co',
    'proximity_to_industrial_areas',
    'population_density'
]

df_full_train, df_test = train_test_split(df, test_size=0.2,
                                          random_state=42,
                                          stratify=df['air_quality'])
df_train, df_val = train_test_split(df_full_train,
                                    test_size=0.25,
                                    random_state=42,
                                    stratify=df_full_train['air_quality'])

train_target = df_train.pop('air_quality')
val_target = df_val.pop('air_quality')
test_target = df_test.pop('air_quality')

# Save the train, validation and test datasets
df_train.to_csv('data/train.csv', index=False)
df_val.to_csv('data/val.csv', index=False)
df_test.to_csv('data/test.csv', index=False)

scaler = StandardScaler()
le = LabelEncoder()

def preprocess(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, features: list) -> Union[pd.DataFrame, pd.Series]: 

    scaler.fit(df_train[features])
    X_train = scaler.transform(df_train[features])
    X_val = scaler.transform(df_val[features])
    X_test = scaler.transform(df_test[features])

    le.fit(train_target)
    y_train = le.transform(train_target)
    y_val = le.transform(val_target)
    y_test = le.transform(test_target)

    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = preprocess(df_train,
                                                            df_val,
                                                            df_test,
                                                            features)

def initiate_params():
    params = {
        'Logistic_Regression': {
            'model': LogisticRegression(),
            'params': {
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1],
            'solver': ['newton-cg', 'liblinear']
            }
        },
        'Decision_Tree': {
            'model': DecisionTreeClassifier(),
            'params': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [3, 5, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 5, 10]
            }
        },
        'Random_Forest': {
            'model': RandomForestClassifier(),
            'params': {
            'n_estimators': [50, 100],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [None, 10, 20],
            'criterion': ['gini', 'entropy'],
            'n_jobs': [4]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(),
            'params': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.6, 0.8],
            'colsample_bytree': [0.6, 0.8],
            'n_jobs': [4]
            }
        },
        'SVM': {
            'model': SVC(),
            'params': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf'],
            'gamma': ['scale', 'auto']
            }
        }
    }

    return params


def random_search_hyperparameter_tuning(X, y, params):
    
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=20)
    results = []

    for key, values in tqdm(params.items(), desc="Hyperparameter Tuning"):
        random_search = RandomizedSearchCV(
            values['model'],
            values['params'],
            cv=kf,
            return_train_score=False,
            refit=True,
            n_jobs=-1)
        random_search.fit(X, y)

        results.append({
            'model_name': key,
            'best_score': random_search.best_score_,
            'best_param': random_search.best_params_,
        })

    results_df = pd.DataFrame(results, columns=['model_name', 'best_score', 'best_param'])
    

    return results_df

# Model hyper parameter tuning
print('Starting Hyperparameter tuning')
params = initiate_params()
results = random_search_hyperparameter_tuning(X_train, y_train, params)
print('Hyperparameter tuning completed')

best_model = results.sort_values(by='best_score', ascending=False).iloc[0]

model_names = ['Logistic_Regression', 'Decision_Tree', 'Random_Forest', 'XGBoost', 'SVM']

model = params[best_model['model_name']]['model']
model.set_params(**best_model['best_param'])

print(f"Best model is: {type(model)}\nand the parameters are: {best_model['best_param']}")

model.fit(X_train, y_train)
y_pred_val = model.predict(X_val)

print(classification_report(y_val, y_pred_val))

# Save the model, scaler and label encoder
with open('model/model.pkl', 'wb') as model_file, \
     open('model/scaler.pkl', 'wb') as scaler_file, \
     open('model/label_encoder.pkl', 'wb') as le_file:
    pickle.dump(model, model_file)
    pickle.dump(scaler, scaler_file)
    pickle.dump(le, le_file)