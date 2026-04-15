# ml-pca-svm-classification

Projeto de machine learning classico para classificacao de cancer de mama usando reducao de dimensionalidade com PCA e classificador SVM.

## Objetivo

O repositorio demonstra um pipeline supervisionado completo:

- carregamento de dados via Kaggle Hub;
- remocao de identificadores;
- separacao treino/teste estratificada;
- imputacao de valores ausentes;
- codificacao de variaveis categoricas quando necessario;
- padronizacao das features;
- reducao de dimensionalidade com PCA;
- busca de hiperparametros de SVM com GridSearchCV;
- avaliacao com acuracia, matriz de confusao e relatorio de classificacao;
- exportacao dos artefatos com `joblib`.

## Conteudo

- `training.ipynb`: notebook principal de treinamento e avaliacao.
- `artifacts/pca_svm_pipeline.joblib`: pipeline completo treinado.
- `artifacts/pca.joblib`: etapa de PCA extraida do pipeline.
- `artifacts/svm.joblib`: classificador SVM extraido do pipeline.
- `requirements.txt`: dependencias.

## Base de Dados

O notebook baixa a base `neurocipher/breast-cancer-dataset` via Kaggle Hub:

```python
kagglehub.dataset_download("neurocipher/breast-cancer-dataset")
```

A coluna alvo esperada e `diagnosis`. A coluna `id`, quando presente, e removida antes do treinamento para evitar que identificadores entrem como feature.

O arquivo `data.csv` local nao foi publicado porque contem dados de contexto medico e identificadores numericos. A reproducao deve usar o download direto da fonte indicada no notebook.

## Como Executar

Instale as dependencias:

```bash
pip install -r requirements.txt
```

Abra o notebook:

```bash
jupyter notebook training.ipynb
```

## Uso dos Artefatos

O arquivo mais importante para reutilizacao e `artifacts/pca_svm_pipeline.joblib`, pois contem pre-processamento, PCA e SVM no mesmo pipeline.

Exemplo conceitual:

```python
import joblib

model = joblib.load("artifacts/pca_svm_pipeline.joblib")
predictions = model.predict(X)
```

## Limitacoes

Este repositorio esta completo como estudo de treinamento, mas nao inclui API, dashboard ou script de inferencia dedicado. Para portifolio/producao, os proximos passos seriam adicionar `predict.py`, exemplos de entrada e testes de carregamento do artefato.
