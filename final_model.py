from typing import final
from sklearn import model_selection
import sys
import io
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')


def load_data():
    path = 'C:/codigos/python/projetos com python/casas_sao_leo/data/housing_sp_city.csv'

    data = pd.read_csv(path, encoding='latin1')

    data = data[data['tipo_anuncio'] == 'Venda']
    data = data[data['periodicidade'] == 'MONTHLY']
    
    bairros = sorted(data['bairro'].dropna().unique())
    tipos_imovel = sorted(data['tipo_imovel'].dropna().unique())

    data = pd.get_dummies(
        data,
        columns=[
            'tipo_imovel',
        ]
    )

    numeric_features = [
        'area_util',
        'banheiros',
        'suites',
        'quartos',
        'vagas_garagem',
        'taxa_condominio',
        'iptu_ano'
    ]

    for feature in numeric_features:
        data[feature] = data[feature].fillna(
            data[feature].median()
        )

    X = data.drop(columns=[
        'preco_venda',
        'logradouro',
        'numero',
        'bairro',
        'cep',
        'cidade',
        'anuncio_criado',
        'preco_aluguel',
        'tipo_anuncio',
        'periodicidade'
    ])

    y = data['preco_venda']

    return (
        X,
        y,
        bairros,
        tipos_imovel
    )


def final_model_and_mae(X, y):

    train_X, val_X, train_y, val_y = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    erro = float('inf')
    best_tree_size = None
    candidate_max_leaf_nodes = [5, 10, 15, 20, 25, 30, 40, 50]
    
    print("\n=== TESTANDO PROFUNDIDADES ===")
    
    for candidate in candidate_max_leaf_nodes:
        model = RandomForestRegressor(n_estimators=200, max_depth=candidate, random_state=42)
        model.fit(train_X, train_y)
        preds_val = model.predict(val_X)
        erro_do_candidato = mean_absolute_error(val_y, preds_val)
        
        print(f"Max depth: {candidate}  MAE: {erro_do_candidato:,.2f}")
        
        if erro_do_candidato < erro:
            erro = erro_do_candidato
            best_tree_size = candidate
            
    print(f"\nMelhor profundidade (max_depth): {best_tree_size} | Melhor MAE: {erro:,.2f}")
            
    # Treinando o modelo final com a melhor profundidade
    model = RandomForestRegressor(n_estimators=200, max_depth=best_tree_size, random_state=42)
    model.fit(train_X, train_y)
    
    return model, erro

   

if __name__ == "__main__":

    (
        X,
        y,
        bairros,
        tipos_imovel
    ) = load_data()

    

    print("\n=== TIPOS DE IMÓVEL DISPONÍVEIS ===")
    for i, t in enumerate(tipos_imovel, 1):
        print(f"  {i}. {t}")

    tipo_imovel = input(
        "\nDigite o tipo do imóvel exatamente como aparece acima: "
    ).strip()

    area_util = float(
        input("\nDigite a área útil: ")
    )

    banheiros = int(
        input("Digite o número de banheiros: ")
    )

    suites = int(
        input("Digite o número de suítes: ")
    )

    quartos = int(
        input("Digite o número de quartos: ")
    )

    vagas_garagem = int(
        input("Digite o número de vagas na garagem: ")
    )

    taxa_condominio = float(
        input("Digite a taxa de condomínio: ")
    )

    iptu_ano = float(
        input("Digite o IPTU anual: ")
    )


    nova_casa = {
        'area_util': area_util,
        'banheiros': banheiros,
        'suites': suites,
        'quartos': quartos,
        'vagas_garagem': vagas_garagem,
        'taxa_condominio': taxa_condominio,
        'iptu_ano': iptu_ano
    }

    for coluna in X.columns:

        if coluna.startswith('tipo_imovel_'):
            nova_casa[coluna] = 0

        if coluna.startswith('periodicidade_'):
            nova_casa[coluna] = 0

    coluna_tipo_imovel = f'tipo_imovel_{tipo_imovel}'

    if coluna_tipo_imovel not in X.columns:
        print("\nTipo de imóvel inválido.")
        exit()

    nova_casa[coluna_tipo_imovel] = 1

    nova_casa = pd.DataFrame([nova_casa])

    nova_casa = nova_casa.reindex(
        columns=X.columns,
        fill_value=0
    )

    model, mae = final_model_and_mae(X, y)


    pred_y = model.predict(nova_casa)

    print(
        f"\nPreço estimado de venda: "
        f"R$ {pred_y[0]:,.2f}"
    )