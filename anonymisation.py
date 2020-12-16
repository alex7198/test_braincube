import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

#Lecture données
df=pd.read_csv("depenses.csv")
#Anonymisation avec LabelEncoder.
le = preprocessing.LabelEncoder()
df['nom']=le.fit_transform(df["nom"])
df['ville']=le.fit_transform(df["ville"])
#Ajout 'id' et 'ville' avant la valeur comme demandé.
df['nom']=df['nom'].apply(lambda x: 'id'+str(x))
df['ville']=df['ville'].apply(lambda x: 'ville'+str(x))
#Normalisation sur les colonnes numériques.
ct = ColumnTransformer([("normalisation", StandardScaler(),['age', 'salaire','depenses'])],remainder='passthrough')
df=pd.DataFrame(ct.fit_transform(df),index=df.index,columns=["age","salaire","depenses","nom","ville"])
#On replace dans l'ordre.
df=df.reindex(columns=['nom','ville','age','salaire','depenses'])
#Export csv.
df.to_csv("depenses_anonymes.csv",index=False)