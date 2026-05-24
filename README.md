# Trabalho_IA — Projeto 8 (NSL-KDD): Random Forest vs SVM

Este repositório contém uma implementação completa do **Projeto 8: Classificação de Ataques de Rede com Random Forest e SVM**, com foco didático para apresentação.

## Estrutura

- `notebooks/Projeto8_AtaquesRede_RF_SVM.ipynb` — notebook principal (explicado e comentado)
- `src/data.py` — carregamento e preparação do NSL-KDD
- `src/models.py` — pipelines, modelos e grids de hiperparâmetros
- `src/evaluation.py` — métricas, matriz de confusão e tempos
- `data/raw/` — colocar os arquivos brutos do dataset aqui
- `data/processed/` — espaço para dados processados
- `reports/figures/` — figuras exportadas

## Dataset (NSL-KDD)

Dataset oficial: https://www.unb.ca/cic/datasets/nsl.html

Baixe os arquivos e coloque em `data/raw/` com estes nomes:

- `KDDTrain+.txt`
- `KDDTest+.txt`

> O notebook lê diretamente esses arquivos de `data/raw/`.

## Ambiente

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

## Como rodar

```bash
jupyter notebook notebooks/Projeto8_AtaquesRede_RF_SVM.ipynb
```

## O que é implementado

- Comparativo entre **Random Forest** e **SVM**
- Pré-processamento completo com `ColumnTransformer` + `Pipeline`
  - Categóricas: imputação + one-hot
  - Numéricas: imputação e padronização (para SVM)
- Seleção de características com `SelectKBest(mutual_info_classif)` (comparada com `passthrough`)
- Tratamento de desbalanceamento com `class_weight` e experimento opcional de SMOTE (fallback automático)
- Busca de hiperparâmetros via `GridSearchCV`
- Métricas exigidas:
  - acurácia
  - precisão/recall/F1 por classe
  - matriz de confusão
  - tempo de treinamento e inferência
