# ml-pca-svm-classification

Projeto de machine learning clássico para classificação de câncer de mama usando redução de dimensionalidade com PCA e classificador SVM.

## Objetivo

O repositório demonstra um pipeline supervisionado completo:

- carregamento de dados via Kaggle Hub;
- remoção de identificadores;
- separação treino/teste estratificada;
- imputação de valores ausentes;
- codificação de variáveis categóricas quando necessário;
- padronização das features;
- redução de dimensionalidade com PCA;
- busca de hiperparâmetros de SVM com GridSearchCV;
- avaliação com acurácia, matriz de confusão e relatório de classificação;
- exportação dos artefatos com `joblib`.

## Conteúdo

- `training.ipynb`: notebook principal de treinamento e avaliação.
- `artifacts/pca_svm_pipeline.joblib`: pipeline completo treinado.
- `artifacts/pca.joblib`: etapa de PCA extraída do pipeline.
- `artifacts/svm.joblib`: classificador SVM extraído do pipeline.
- `requirements.txt`: dependências.

## Base de Dados

O notebook baixa a base `neurocipher/breast-cancer-dataset` via Kaggle Hub:

```python
kagglehub.dataset_download("neurocipher/breast-cancer-dataset")
```

A coluna alvo esperada é `diagnosis`. A coluna `id`, quando presente, é removida antes do treinamento para evitar que identificadores entrem como feature.

O arquivo `data.csv` local não foi publicado porque contém dados de contexto médico e identificadores numéricos. A reprodução deve usar o download direto da fonte indicada no notebook.

## Como Executar

Instale as dependências:

```bash
pip install -r requirements.txt
```

Abra o notebook:

```bash
jupyter notebook training.ipynb
```

## Uso dos Artefatos

O arquivo mais importante para reutilização é `artifacts/pca_svm_pipeline.joblib`, pois contém pré-processamento, PCA e SVM no mesmo pipeline.

Exemplo conceitual:

```python
import joblib

model = joblib.load("artifacts/pca_svm_pipeline.joblib")
predictions = model.predict(X)
```

## Limitações

Este repositório está completo como estudo de treinamento, mas não inclui API, dashboard ou script de inferência dedicado. Para portfólio/produção, os próximos passos seriam adicionar `predict.py`, exemplos de entrada e testes de carregamento do artefato.
