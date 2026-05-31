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

def final_model(X, y, max_depth):
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X, y)
    return model

if __name__ == "__main__":
    X, y = load_data()
    max_depth = 10
    model = final_model(X, y, max_depth)
    area_util = float(input("Digite o valor de area util: "))
    banheiros = int(input("Digite o numero de banheiros: "))
    suites = int(input("Digite o numero de suites: "))
    quartos = int(input("Digite o numero de quartos: "))
    vagas_garagem = int(input("Digite o numero de vagas na garagem: "))
    taxa_condominio = float(input("Digite o valor da taxa de condominio: "))
    iptu_ano = float(input("Digite o valor do iptu do ano: "))

    nova_casa = pd.DataFrame({
    'area_util': [area_util],
    'banheiros': [banheiros],
    'suites': [suites],
    'quartos': [quartos],
    'vagas_garagem': [vagas_garagem],
    'taxa_condominio': [taxa_condominio],
    'iptu_ano': [iptu_ano]
    })

    preco_venda = model.predict(nova_casa)
    print(f'O preço de venda estimado para a nova casa é: R$ {preco_venda[0]:.2f}')
