import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

import lightgbm as lgb


print(">>> Passo 1: Carregando e preparando os dados...")

colunas = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]
# URL do dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

# Carregar dados
try:
    df = pd.read_csv(url, header=None, names=colunas, sep=r',\s', na_values='?', engine='python')
    print("Dados carregados com sucesso!")
except Exception as e:
    print(f"Erro ao carregar os dados: {e}")
    exit()

# Separar features (X) e alvo (y)
X = df.drop('income', axis=1)
y = df['income']

# Binarizar o alvo: >50K -> 1, <=50K -> 0
y = y.apply(lambda x: 1 if x == '>50K' else 0)

# Dividir os dados em treino e teste ANTES do pré-processamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


print("\n>>> Passo 2: Definindo o pipeline de pré-processamento...")

# Identificar colunas numéricas e categóricas a partir do X_train
numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

print(f"Features Numéricas: {numeric_features}")
print(f"Features Categóricas: {categorical_features}")

# Criar os pipelines de transformação
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Juntar os transformers em um único pré-processador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)
print("Pipeline de pré-processamento criado com sucesso!")


print("\n--- Treinando Modelo Final: LightGBM (GPU com sua RX 7600) ---")

final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', lgb.LGBMClassifier(device='gpu', random_state=42))
])

try:
    # Treinar
    final_model.fit(X_train, y_train)
    print("Modelo treinado com sucesso!")

    # Avaliar
    y_pred = final_model.predict(X_test)
    print("\nRelatório de Classificação (LightGBM - GPU):")
    print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))
    
    # Plotar Matriz de Confusão
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['<=50K', '>50K'], ax=ax)
    ax.set_title('Matriz de Confusão - Modelo Final (LightGBM - GPU)')

    plt.savefig('matriz_confusao.png')

    # Salvar o modelo
    print("\n>>> Salvando o modelo final em um arquivo...")
    nome_arquivo = 'modelo_renda.joblib'
    joblib.dump(final_model, nome_arquivo)
    print(f"Modelo salvo com sucesso no arquivo: {nome_arquivo}")

except Exception as e:
    print(f"\nERRO ao treinar ou salvar o modelo com GPU: {e}")
    print("Verifique se os drivers da AMD e o OpenCL estão corretamente instalados.")
    print("Para mais informações, consulte a documentação do LightGBM sobre instalação com suporte a GPU.")


print("\n>>> Fim do processo. Exibindo gráfico...")
plt.tight_layout()
plt.show()