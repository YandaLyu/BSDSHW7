import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

df_train = pd.read_csv('/Users/lyuyanda/Downloads/spaceship-titanic/train.csv')
df_test = pd.read_csv('/Users/lyuyanda/Downloads/spaceship-titanic/test.csv')

def preprocesing(df):
    df = df.drop('Name', axis=1)
    df = df.drop('PassengerId', axis=1)
    impute_vals = {'Age': df_train['Age'].mean(),
                   'RoomService': df_train['RoomService'].mean(),
                   'FoodCourt': df_train['FoodCourt'].mean(),
                   'ShoppingMall': df_train['ShoppingMall'].mean(),
                   'Spa': df_train['Spa'].mean(),
                   'VRDeck': df_train['VRDeck'].mean(),
                   'HomePlanet':'other',
                   'Destination':'other',
                   'CryoSleep':False,
                   'VIP':False,
                   }
    df = df.fillna(impute_vals)
    df['Deck'] = df['Cabin'].apply(lambda x: str(x).split('/')[0] if len(str(x).split('/')) == 3 else 'other')
    df['Side'] = df['Cabin'].apply(lambda x: str(x).split('/')[2] if len(str(x).split('/')) == 3 else 'other')
    df = df.drop('Cabin', axis=1)
    df = pd.get_dummies(df, columns=['HomePlanet', 'Destination', 'Deck', 'Side'])
    df['CryoSleep'] = df['CryoSleep'].astype(bool)
    df['VIP'] = df['VIP'].astype(bool)
    return df

df_train = preprocesing(df_train)
PassengerId = df_test['PassengerId']
df_test = preprocesing(df_test)

y = df_train['Transported']
X = df_train.drop('Transported', axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
acc_val = np.mean(y_pred == y_val)
print(f'acc_val: {acc_val}')

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
acc_val = np.mean(y_pred == y_val)
print(f'acc_val: {acc_val}')

clf = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
acc_val = np.mean(y_pred == y_val)
print(f'acc_val: {acc_val}')

result = pd.DataFrame({
    'PassengerId':PassengerId,
    'Transported':clf.predict(df_test).astype(bool)
})
result.to_csv('submission.csv',index=False)