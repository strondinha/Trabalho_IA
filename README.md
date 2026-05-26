# Projeto de Machine Learning para Cibersegurança

## Visão geral
Este projeto tem como foco a **classificação de ataques de rede** em cenários de alta dimensionalidade, utilizando dados tabulares dos datasets **NSL-KDD** ou **CICIDS2017**. A proposta é comparar dois algoritmos clássicos de aprendizado supervisionado:

- **Random Forest** (ênfase em interpretabilidade via importância de atributos)
- **SVM (Support Vector Machine)** (ênfase na avaliação de diferentes kernels)

## Escopo acadêmico
O trabalho foi estruturado para apoiar atividades de estudo e pesquisa em Inteligência Artificial aplicada à Segurança da Informação, cobrindo:

1. Exploração inicial de dados de tráfego de rede.
2. Pré-processamento para dados numéricos/categóricos e tratamento de alta dimensionalidade.
3. Balanceamento de classes com técnicas como **SMOTE** ou ajuste de **class weights**.
4. Treinamento e comparação inicial entre Random Forest e SVM.
5. Avaliação por métricas de classificação multiclasse e análise de matriz de confusão.
6. Medição de tempo de treinamento e inferência para análise de custo computacional.

## Objetivos de aprendizado
Ao utilizar este projeto, espera-se desenvolver:

- Entendimento sobre o pipeline de Machine Learning em problemas de cibersegurança.
- Capacidade de preparar dados tabulares de alta dimensionalidade.
- Aplicação prática de redução/seleção de características.
- Comparação técnica entre modelos baseados em árvores e margens máximas.
- Interpretação de métricas por classe em cenários desbalanceados.

## Estrutura do projeto

```text
.
├── data/
│   ├── raw/                # Dados originais (não processados)
│   └── processed/          # Dados após pré-processamento
├── docs/                   # Documentação complementar
├── notebooks/
│   └── 01_exploracao_e_modelagem.ipynb
├── src/                    # Código-fonte modular do projeto
├── requirements.txt
└── README.md
```

## Requisitos
Instale as dependências listadas em `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Execução no Google Colab
O Colab reinicia o ambiente a cada nova sessão (inclusive em outro computador).  
Por isso, o notebook `notebooks/Projeto8_AtaquesRede_RF_SVM.ipynb` agora inclui uma seção **Setup Colab** no topo para automatizar:

- detecção de execução no Colab;
- clone do repositório em `/content/Trabalho_IA` (se necessário);
- instalação de dependências com `requirements.txt`;
- configuração do caminho para os imports `from src...`.

Abra diretamente no Colab:

https://colab.research.google.com/github/strondinha/Trabalho_IA/blob/main/notebooks/Projeto8_AtaquesRede_RF_SVM.ipynb

Para o dataset, mantenha os arquivos em `data/raw/` dentro do projeto no Colab:

- `KDDTrain+.txt`
- `KDDTest+.txt`

O notebook traz duas opções para isso:
1. upload manual (`files.upload()`);
2. cópia via Google Drive (`drive.mount()` + `copy`).

> Observação: o dataset **não** deve ser versionado no Git.

## Como executar
1. Coloque o dataset escolhido em `data/raw/`.
2. Abra o notebook inicial:
   ```bash
   jupyter notebook notebooks/Projeto8_AtaquesRede_RF_SVM.ipynb
   ```
3. Siga as células para:
   - carregar os dados;
   - pré-processar e reduzir dimensionalidade;
   - tratar desbalanceamento;
   - treinar Random Forest e SVM;
   - avaliar métricas e tempos.

## Próximos passos sugeridos
- Inserir validação cruzada estratificada.
- Realizar ajuste de hiperparâmetros (GridSearch/RandomizedSearch).
- Comparar desempenho em diferentes recortes de features.
- Registrar experimentos para reprodutibilidade.
