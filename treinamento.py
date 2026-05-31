from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

def load_data():
    path = 'C:/codigos/python/projetos com python/casas_sao_leo/data/housing_sp_city.csv'
    data = pd.read_csv(path, encoding='latin1')
    sp_features = [
    'area_util',
    'banheiros',
    'suites',
    'quartos',
    'vagas_garagem',
    'taxa_condominio',
    'iptu_ano'
]
    # print(data[sp_features].isna().sum())
    for feature in sp_features:
        data[feature] = data[feature].fillna(data[feature].median())
    # print(data[sp_features].isna().sum())
    X = data[sp_features]
    y = data['preco_venda']
    return X, y

def train_model(train_X, val_X, train_y, val_y, max_depth):
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(train_X, train_y)
    for nome, importancia in zip(train_X.columns, model.feature_importances_):
        print(nome, importancia)
    y_pred = model.predict(val_X)
    mae = mean_absolute_error(val_y, y_pred)
    return mae

if __name__ == "__main__":
    erro_possivel = float('inf')
    melhor_depth = None
    X, y = load_data()
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=42)
    max_depth = [10, 50, 100, 200, 300, 400, 500]
    train_model(train_X, val_X, train_y, val_y, max_depth=10)
    for numeroMelhor in max_depth:
        erro = train_model(train_X, val_X, train_y, val_y, numeroMelhor)

        print(f'Max Depth: {numeroMelhor}, Mean Absolute Error: {erro}')

        if erro <= erro_possivel:
            erro_possivel = erro
            melhor_depth = numeroMelhor

    print(f'Melhor MAE: {erro_possivel}')
    print(f'Melhor Max Depth: {melhor_depth}')


