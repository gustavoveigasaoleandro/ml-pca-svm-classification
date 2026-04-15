# ml-pca-svm-classification

Projeto de machine learning classico para classificacao usando reducao de dimensionalidade com PCA e classificador SVM.

## Objetivo

O notebook `training.ipynb` demonstra um pipeline supervisionado com:

- carregamento de dados via `kagglehub`;
- separacao treino/teste estratificada;
- imputacao de valores ausentes;
- codificacao de variaveis categoricas quando necessario;
- padronizacao com `StandardScaler`;
- reducao de dimensionalidade com `PCA`;
- otimizacao de hiperparametros de `SVC` com `GridSearchCV`;
- avaliacao com acuracia, matriz de confusao e relatorio de classificacao;
- exportacao dos artefatos treinados com `joblib`.

## Base de dados

O notebook baixa a base `neurocipher/breast-cancer-dataset` via Kaggle Hub:

```python
kagglehub.dataset_download("neurocipher/breast-cancer-dataset")
```

Pelo codigo, a coluna alvo esperada e `diagnosis`, e a coluna `id` e removida antes do treino quando existe. As demais colunas numericas descrevem medidas extraidas de exames de cancer de mama, como raio, textura, perimetro, area, suavidade, concavidade e simetria.

O arquivo `data.csv` local nao foi incluido nesta versao publicada. Mesmo aparentando ser uma amostra de uma base publica, ele contem dados de contexto medico e identificadores numericos, entao a reproducao do treino deve baixar a base diretamente da fonte indicada no notebook.

## Artefatos

O diretorio `artifacts/` contem:

- `pca_svm_pipeline.joblib`: pipeline completo treinado;
- `pca.joblib`: etapa de PCA extraida do pipeline;
- `svm.joblib`: classificador SVM extraido do pipeline.

## Como executar

1. Instale as dependencias:

```bash
pip install -r requirements.txt
```

2. Abra o notebook:

```bash
jupyter notebook training.ipynb
```

3. Execute as celulas para baixar a base, treinar o pipeline e gerar os artefatos.

## Cuidados de publicacao

Esta versao remove:

- ambiente virtual local;
- `data.csv` local;
- arquivos temporarios do macOS/Jupyter.

Nao foram encontrados tokens, chaves de API ou credenciais no material publicado.
