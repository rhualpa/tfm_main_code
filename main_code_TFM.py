#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from collections import Counter
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


with open('C:/Users/Ronald/OneDrive/Escritorio/UEM/TFM/table_content.html', 'r', encoding='utf-8') as file:
    html_content = file.read()

df_list = pd.read_html(html_content)

df = pd.concat(df_list)

new_headers = df.columns.get_level_values(1)

df.columns = new_headers

df = df[df.iloc[:, 0] != 'RL']

df


# In[3]:


#Eliminar la columna RL que es innecesaria
df = df.drop(columns=['RL'])

#Dejamos fuera a los porteros del análisis
df = df[df['Posc'] != 'PO']

df = df.reset_index(drop=True)
df.index = df.index + 1

#Acortar la nacionalidad
df['País'] = df['País'].str[-3:]

#Formatear el campo edad
df['Edad'] = df['Edad'].str[:2]

#Eliminamos columnas innesesarias
df = df.iloc[:, :-6]
df = df.drop(columns=['Nacimiento'])
df = df.drop(columns=['90 s'])
df = df.drop(columns=['G-TP'])
df = df.drop(columns=['G+A'])

#Formateamos el nombre de los campos
df.columns = ['jugador','pais','posicion','equipo','edad','p_jugados','p_titular', 'minutos','goles','asistencias', 'pnl_exitosos', 'pnl_errados', 'amarillas', 'rojas']

#Asignamos los tipos de datos correctos
df['edad'] = df['edad'].astype(int)
df['p_jugados'] = df['p_jugados'].astype(int)
df['p_titular'] = df['p_titular'].astype(int)
df['minutos'] = df['minutos'].astype(int)
df['goles'] = df['goles'].astype(int)
df['asistencias'] = df['asistencias'].astype(int)
df['pnl_exitosos'] = df['pnl_exitosos'].astype(int)
df['pnl_errados'] = df['pnl_errados'].astype(int)
df['amarillas'] = df['amarillas'].astype(int)
df['rojas'] = df['rojas'].astype(int)

df


# In[4]:


#Revisión de nulos o en blanco

nulos_por_columna = df.isnull().sum()
print(nulos_por_columna)

en_blanco_por_columna = df.applymap(lambda x: x == "").sum()
print(en_blanco_por_columna)


# In[5]:


scaler = MinMaxScaler()

columnas_a_normalizar = ['p_jugados','p_titular', 'minutos','goles','asistencias', 'pnl_exitosos', 'pnl_errados', 'amarillas', 'rojas']

df_norm = df.copy()

df_norm[columnas_a_normalizar] = scaler.fit_transform(df[columnas_a_normalizar])

df_norm


# In[6]:


#Una vez normalizado debemos elegir correctamente las variables que sean más relevantes para el análisis

# Crear la métrica compuesta para el MVP
df_norm['mvp_score'] = (df_norm['goles'] * 0.13 +
                   df_norm['asistencias'] * 0.13 +
                   df_norm['pnl_exitosos'] * 0.11 -
                   df_norm['pnl_errados'] * 0.10 + 
                   df_norm['minutos'] * 0.11 + 
                   df_norm['p_jugados'] * 0.13 + 
                   df_norm['p_titular'] * 0.13 - 
                   df_norm['amarillas'] * 0.07 - 
                   df_norm['rojas'] * 0.09)

df_norm


# In[7]:


#Ordenamos el df para ver cómo sería el ranking según esta evaluación subjetiva
df_norm.sort_values(by='mvp_score', ascending=False).head(10)


# In[8]:


#Ahora veremos la selección de variables:

#Fijaremos una semilla
seed = 42
np.random.seed(seed)

X = df_norm.drop(columns=['jugador', 'pais', 'posicion', 'equipo', 'mvp_score'])
y = df_norm['mvp_score']

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


rf = RandomForestRegressor(n_estimators=100, random_state=42)

rfe = RFE(estimator=rf, n_features_to_select=5)


# In[10]:


rfe.fit(X_train, y_train)


# In[11]:


selected_features_rfe = X.columns[rfe.support_].tolist()
print(f"Elegido por RFE: {selected_features_rfe}")


# In[12]:


# Inicializar KNN
knn = KNeighborsRegressor(n_neighbors=5)

# Usar SFS para seleccionar variables
sfs_knn = SFS(knn,
              k_features=5, 
              forward=False,
              floating=False,
              scoring='r2',
              cv=5)
sfs_knn.fit(X_train, y_train)


# In[13]:


# Obtener las características seleccionadas por SFS con KNN
selected_features_knn = list(sfs_knn.k_feature_names_)
print(f"Elegido por KNN: {selected_features_knn}")


# In[14]:


# Usar SFS para seleccionar variables con RFR
sfs_rf = SFS(rf,
             k_features=5,
             forward=False,
             floating=False,
             scoring='r2',
             cv=5)
sfs_rf.fit(X_train, y_train)


# In[15]:


# Obtener las variables
selected_features_rf = list(sfs_rf.k_feature_names_)
print(f"Elegido por RandomForestRegressor: {selected_features_rf}")


# In[16]:


# Combinar todas las variables seleccionadas
all_selected_features = selected_features_rfe + selected_features_knn + selected_features_rf

# Contar las apariciones de cada una
feature_counts = Counter(all_selected_features)

# Mostrar las variables más comunes
most_common_features = feature_counts.most_common()
print("Variables mas comunes:", most_common_features)


# In[17]:


columnas_elegidas = ['jugador','pais','posicion','equipo','edad','p_jugados','p_titular','goles','asistencias','rojas']

df_filtrado = df_norm[columnas_elegidas].copy()

df_filtrado


# In[18]:


#Ahora realizamos otro cálculo del mvp score tomando en cuenta los nuevos pesos y los campos elegidos

# Crear la métrica compuesta para el MVP
df_filtrado['mvp_score'] = (df_filtrado['goles'] * 0.21 +
                   df_filtrado['asistencias'] * 0.21 +
                   df_filtrado['p_jugados'] * 0.21 + 
                   df_filtrado['p_titular'] * 0.21 -
                   df_filtrado['rojas'] * 0.16)

df_filtrado


# In[19]:


#Ordenamos el df para ver cómo sería el ranking según esta otra evaluación
df_filtrado.sort_values(by='mvp_score', ascending=False).head(10)


# In[20]:


#Inicio de creación y entrenamiento con modelos de ML

X = df_filtrado.drop(columns=['jugador', 'pais', 'posicion', 'equipo', 'mvp_score'])
y = df_filtrado['mvp_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[21]:


# Definir el modelo de Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

rf_regressor.fit(X_train, y_train)

y_pred = rf_regressor.predict(X_test)


# In[22]:


mse_rfr = mean_squared_error(y_test, y_pred)
r2_rfr = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse_rfr)
print("R-squared (R2):", r2_rfr)


# In[23]:


#Parámetros de Grid Search para el SVR
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'epsilon': [0.1, 0.2, 0.5, 0.3, 0.05]
}


# In[24]:


svr = SVR(kernel='rbf')


# In[25]:


grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, scoring='r2', cv=5, verbose=2, n_jobs=-1)


# In[26]:


# Entrenar el modelo usando Grid Search
grid_search.fit(X_train, y_train)


# In[27]:


# Obtener los mejores parámetros
best_params = grid_search.best_params_
print(f"Mejores parametros encontrados: {best_params}")


# In[28]:


# Usar el mejor modelo encontrado para hacer la predicción
best_svr = grid_search.best_estimator_
y_pred_2 = best_svr.predict(X_test)


# In[29]:


# Calcular métricas de rendimiento
mse_svr = mean_squared_error(y_test, y_pred_2)
r2_svr = r2_score(y_test, y_pred_2)

print(f'Mean Squared Error (MSE): {mse_svr}')
print(f'R-squared (R2): {r2_svr}')


# In[30]:


# Entrenar y evaluar GradientBoostingRegressor

#Aplicamos GridSearch
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

gbr = GradientBoostingRegressor(random_state=42)

grid_search_gbr = GridSearchCV(estimator=gbr, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)

grid_search_gbr.fit(X_train, y_train)

best_params_gbr = grid_search_gbr.best_params_
best_estimator_gbr = grid_search_gbr.best_estimator_

print(f"Best parameters found: {best_params_gbr}")


y_pred_gbr = best_estimator_gbr.predict(X_test)


# In[31]:


mse_gbr = mean_squared_error(y_test, y_pred_gbr)
r2_gbr = r2_score(y_test, y_pred_gbr)

print(f'Mean Squared Error (MSE): {mse_gbr}')
print(f'R-squared (R2): {r2_gbr}')


# In[32]:


# Crear un DataFrame para comparar las métricas
comparison_df = pd.DataFrame({
    'Model': ['RFR', 'SVR', 'GBR'],
    'MSE': [mse_rfr, mse_svr, mse_gbr],
    'R2': [r2_rfr, r2_svr, r2_gbr]
})


# In[33]:


# Graficar la comparación de las métricas
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Gráfico de barras para MSE
ax[0].bar(comparison_df['Model'], comparison_df['MSE'], color=['blue', 'green', 'red'])
ax[0].set_title('Comparación de MSE')
ax[0].set_ylabel('Mean Squared Error (MSE)')

# Gráfico de barras para R2
ax[1].bar(comparison_df['Model'], comparison_df['R2'], color=['blue', 'green', 'red'])
ax[1].set_title('Comparación de R2')
ax[1].set_ylabel('R-squared (R2)')

plt.tight_layout()
plt.show()


# In[34]:


#Comparamos los 3 modelos con el total de los datos
y_pred_all_rfr = rf_regressor.predict(X)
y_pred_all_svr = best_svr.predict(X)
y_pred_all_gbr = best_estimator_gbr.predict(X)

mse_all_rfr = mean_squared_error(y, y_pred_all_rfr)
r2_all_rfr = r2_score(y, y_pred_all_rfr)

mse_all_svr = mean_squared_error(y, y_pred_all_svr)
r2_all_svr = r2_score(y, y_pred_all_svr)

mse_all_gbr = mean_squared_error(y, y_pred_all_gbr)
r2_all_gbr = r2_score(y, y_pred_all_gbr)

# Crear un DataFrame para comparar las métricas
comparison_df_2 = pd.DataFrame({
    'Model': ['RFR', 'SVR', 'GBR'],
    'MSE': [mse_all_rfr, mse_all_svr, mse_all_gbr],
    'R2': [r2_all_rfr, r2_all_svr, r2_all_gbr]
})


# In[35]:


# Graficar la comparación de las métricas
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Gráfico de barras para MSE
ax[0].bar(comparison_df_2['Model'], comparison_df_2['MSE'], color=['blue', 'green', 'red'])
ax[0].set_title('Comparación de MSE')
ax[0].set_ylabel('Mean Squared Error (MSE)')

# Gráfico de barras para R2
ax[1].bar(comparison_df_2['Model'], comparison_df_2['R2'], color=['blue', 'green', 'red'])
ax[1].set_title('Comparación de R2')
ax[1].set_ylabel('R-squared (R2)')

plt.tight_layout()
plt.show()


# In[45]:


# Crear un DataFrame con las predicciones y los jugadores correspondientes
predictions_df = pd.DataFrame({'Jugador': df['jugador'], 
                               'Pais': df['pais'],
                               'Posicion': df['posicion'],
                               'Equipo': df['equipo'],
                               'Edad': df['edad'],
                               'P_Jugados': df['p_jugados'],
                               'P_Titular': df['p_titular'],
                               'Goles': df['goles'],
                               'Asistencias': df['asistencias'],
                               'Rojas': df['rojas'],
                               'MVP_Score_or': df_filtrado['mvp_score'],
                               'MVP_Score_pred': y_pred_all_gbr})

# Ordenar el DataFrame por las predicciones en orden descendente
predictions_df = predictions_df.sort_values(by='MVP_Score_pred', ascending=False)


# In[46]:


# Mostrar las primeras filas del DataFrame para ver los jugadores con los mayores mvp_score predichos
predictions_df = predictions_df.reset_index(drop=True)
predictions_df.index = predictions_df.index + 1
predictions_df.head(10)


# In[49]:


predictions_df.head(40)


# In[51]:


jugadores_seleccionados = predictions_df.loc[predictions_df['Jugador'].isin(['Andy Polo', 'Martín Cauteruccio', 'Hernán Barcos', 'Bernardo Cuesta'])]

jugadores_seleccionados


# In[ ]:




