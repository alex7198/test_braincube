from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

#Partie 1
#Lecture csv.
data=pd.read_csv("depenses_anonymes.csv")
#Création DataFrames et split.
X=pd.DataFrame(data['salaire'],columns=['salaire'])
Y=pd.DataFrame(data['depenses'],columns=['depenses'])
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=42)
#Construction du résultat.
df_result=Y_test.copy()
df_result['salaire']=X_test['salaire']
#Entrainement + prédictions
reg = LinearRegression().fit(X_train, Y_train)
df_result['prediction_depenses']=reg.predict(X_test)
#On replace dans l'ordre.
df_result=df_result.reindex(columns=['salaire','depenses','depenses_predit'])
#Export csv
df_result.to_csv('predictions_1.csv',index=False)

#Partie 2
#Ajout colonne age.
X_train=X_train.join(data['age'])
X_test=X_test.join(data['age'])
#Construction du résultat.
df_result=Y_test.copy()
df_result[['age','salaire']]=X_test[['age','salaire']]
#Entrainement + prédictions
reg = LinearRegression().fit(X_train, Y_train)
df_result['prediction2_depense']=reg.predict(X_test)
#On replace dans l'ordre.
df_result=df_result.reindex(columns=['age','salaire','depenses','depenses_predit'])
#Export csv
df_result.to_csv('predictions_2.csv',index=False)