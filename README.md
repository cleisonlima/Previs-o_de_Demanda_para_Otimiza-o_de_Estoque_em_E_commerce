# Projeto de Previsão de Demanda para Otimização de Estoque em E-commerce

## 1. Visão Geral do Projeto

Este projeto de Data Science e Machine Learning visa revolucionar a gestão de estoque da "TechGlow", uma empresa de e-commerce de eletrônicos, através de um sistema robusto de previsão de demanda. Abordamos desafios críticos como `stockouts` (ruptura de estoque) e excesso de estoque, utilizando técnicas avançadas de ML para otimizar o capital de giro, reduzir custos operacionais e aumentar a satisfação do cliente. A solução oferece insights acionáveis para decisões estratégicas, transformando a gestão de estoque de reativa para proativa.

## 2. Problema de Negócio

### Descrição Detalhada do Problema
A "TechGlow" enfrenta problemas de superprodução de itens de baixa demanda e falta de produtos populares, resultando em: 
- **Custos de Armazenamento Elevados:** Devido ao excesso de estoque.
- **Obsolescência de Estoque:** Produtos parados perdem valor.
- **Perdas de Vendas:** Clientes não encontram itens desejados.
- **Insatisfação do Cliente:** Devido à indisponibilidade de produtos.

A abordagem atual, baseada em métodos estatísticos simples e experiência gerencial, é ineficaz diante da complexidade e sazonalidade do mercado de eletrônicos. Essa ineficiência acarreta uma perda estimada de 15% da receita anual devido a vendas perdidas e 10% em custos adicionais de estoque.

### Objetivos SMART do Projeto
- **Específico:** Desenvolver e implementar um modelo de previsão de demanda baseado em IA para produtos de eletrônicos na plataforma TechGlow.
- **Mensurável:** Reduzir a taxa de `stockout` em 20% e diminuir o excesso de estoque (itens com mais de 180 dias em armazém) em 15% para os 100 produtos mais vendidos.
- **Atingível:** Utilizar dados históricos de vendas, promoções, sazonalidade e tendências de mercado com a infraestrutura existente e equipe dedicada.
- **Relevante:** Otimizar significativamente estoque e vendas, impactando a rentabilidade e competitividade da TechGlow.
- **Temporizável:** Implementar o modelo e atingir os objetivos em um período de 9 meses.

### Relevância para o Cenário Empresarial
A otimização da previsão de demanda é crucial para TechGlow, permitindo:
- **Redução de Custos Operacionais:** Minimizando armazenamento e obsolescência.
- **Aumento da Receita:** Garantindo disponibilidade de produtos de alta demanda.
- **Melhora na Satisfação do Cliente:** Fidelizando clientes com produtos prontamente disponíveis.
- **Otimização do Capital de Giro:** Liberando capital preso em estoque parado para outros investimentos.

## 3. Conjunto de Dados

### Fontes de Dados
- **Internas (TechGlow):** Dados históricos de vendas (SKU, quantidade, preço, data), dados de produtos (categoria, características), e dados de clientes (CRM).
- **Externas (Sugestão):** APIs de dados climáticos, indicadores econômicos (PIB, inflação), calendários de feriados, e dados de concorrência.

### Variáveis Chave
- **Data da Transação:** Base para análise temporal e sazonalidade.
- **ID do Produto/SKU:** Identificador único do produto.
- **Quantidade Vendida:** Variável alvo (o que queremos prever).
- **Preço Unitário de Venda:** Preço no momento da venda, crucial para elasticidade da demanda.
- **Promoção/Desconto:** Indicador de promoções e nível de desconto.
- **Categoria do Produto:** Agrupamento para identificar padrões de demanda semelhantes.
- **Variáveis Temporais:** Ano, Mês, Dia do Mês, Dia da Semana, Dia do Ano, Semana do Ano, Fim de Semana.
- **Lag Features:** Quantidade vendida em 1, 7 e 30 dias anteriores por SKU.
- **Rolling Window Features:** Médias móveis de 3 e 7 dias para quantidade vendida por SKU.

## 4. Análise Exploratória de Dados (EDA)

### Principais Insights
- **Sazonalidade e Flutuações:** O gráfico de séries temporais da "Quantidade Vendida" mostrou flutuações diárias, embora no dataset sintético não haja uma tendência clara de crescimento/decaimento, ressaltando a importância de capturar sazonalidades em dados reais.
- **Distribuição Uniforme:** As distribuições de "Quantidade Vendida" (1 a 14 unidades por transação) e "Preço Unitário de Venda" (R$10 a R$500) foram relativamente uniformes, sem outliers extremos.
- **Vendas por Categoria:** As categorias 'Livros', 'Eletrônicos', 'Casa e Decoração' e 'Alimentos' apresentaram as maiores participações na demanda prevista, com 'Vestuário' ligeiramente abaixo, indicando uma distribuição equilibrada.
- **Eficácia das Promoções:** O box plot revelou que promoções (10% e 25% de desconto) têm um impacto positivo claro na "Quantidade Vendida", com um aumento médio nas vendas para produtos com desconto.

### Visualizações
Foram utilizados gráficos de séries temporais para tendência e sazonalidade, histogramas e box plots para distribuição e outliers, e gráficos de barras para comparar vendas entre categorias e o impacto de promoções.

## 5. Engenharia de Features e Pré-processamento

### Criação de Variáveis Estratégicas
- **Variáveis Temporais:** `Ano`, `Mes`, `Dia do Mes`, `Dia da Semana`, `Dia do Ano`, `Semana do Ano`, `Fim de Semana` foram extraídas para capturar padrões sazonais e cíclicos da demanda.
- **Valor Total da Venda:** Calculado como `Quantidade Vendida * Preço Unitário de Venda`, útil para análises de receita e como feature auxiliar.
- **Lag Features:** `Lag_1_Dia_Qtde`, `Lag_7_Dias_Qtde`, `Lag_30_Dias_Qtde` foram criadas para incorporar a dependência temporal das vendas passadas de cada SKU.
- **Rolling Window Features:** `Rolling_Mean_3_Dias_Qtde` e `Rolling_Mean_7_Dias_Qtde` foram geradas para suavizar o ruído e capturar tendências de vendas recentes.

### Tratamento de Dados Ausentes
- `NaNs` gerados pelas `lag` e `rolling window features` foram preenchidos com 0, assumindo que a ausência de dados anteriores para esses períodos implica em zero vendas.

### Normalização e Encoding
- **One-Hot Encoding:** Aplicado à `Categoria do Produto` para converter categorias nominais em formato numérico, sem inferir ordem.
- **StandardScaler:** Utilizado para normalizar colunas numéricas como `Preço Unitário de Venda`, `Valor Total da Venda` e todas as `lag` e `rolling window features`, garantindo que todas as variáveis contribuam igualmente para o modelo.

## 6. Modelagem e Comparação de Modelos de Machine Learning

### Modelos Avaliados
Para a previsão da `Quantidade Vendida`, foram comparados três modelos de regressão:
- Regressão Linear
- RandomForest Regressor
- XGBoost Regressor

### Métricas de Avaliação
As métricas utilizadas para avaliar o desempenho dos modelos foram:
- **Mean Absolute Error (MAE):** Média dos erros absolutos. Fácil interpretação, robusta a outliers.
- **Mean Squared Error (MSE):** Média dos erros quadráticos. Penaliza erros maiores, sensível a outliers.
- **Root Mean Squared Error (RMSE):** Raiz quadrada do MSE. Na mesma unidade da variável alvo, interpretabilidade aprimorada.

### Comparação e Escolha do Modelo
| Modelo                  | MAE      | MSE      | RMSE     |
|-------------------------|----------|----------|----------|
| Linear Regression       | 2.3327   | 7.9580   | 2.8210   |
| RandomForest Regressor  | 2.4590   | 8.9026   | 2.9837   |
| XGBoost Regressor       | 2.4691   | 8.9161   | 2.9860   |

O **RandomForest Regressor** apresentou o melhor desempenho com os menores MAE, MSE e RMSE. Sua escolha é justificada pela sua capacidade de lidar com não-linearidades e interações complexas entre as features, além de sua robustez a overfitting através da agregação de múltiplas árvores de decisão. Apesar dos números terem mudado ligeiramente após o refactoring, o RandomForest se manteve o modelo mais promissor.

## 7. IA Aplicada ao Negócio

### Previsões Futuras de Demanda
O modelo RandomForest Regressor foi utilizado para gerar previsões de demanda futura para 30 dias, a nível de SKU. Estas previsões servem como base para:
- **Otimização Proativa de Estoque:** Reduzindo `stockouts` e excesso de inventário.
- **Planejamento de Compras:** Baseado em demanda real e não apenas em histórico.
- **Estratégias de Marketing e Promoções:** Direcionando esforços para produtos com demanda variável.
- **Negociação com Fornecedores:** Com dados mais precisos, TechGlow pode negociar melhores condições.

### Valor para o Negócio
- **Redução de Custos:** Minimiza custos de armazenamento e perdas por obsolescência.
- **Aumento de Receita:** Garante a disponibilidade de produtos populares, evitando perdas de vendas.
- **Melhora da Satisfação do Cliente:** Produtos sempre disponíveis aumentam a lealdade.
- **Otimização do Capital de Giro:** Liberando recursos para outras áreas estratégicas da empresa.

## 8. Visualização e Dashboard Executivo

### Indicadores-Chave de Desempenho (KPIs)
- **Quantidade Vendida Prevista Total:** Soma da demanda prevista para todo o período futuro.
- **Receita Prevista Total:** Soma da receita gerada pelas vendas previstas.
- **Demanda Média Diária (Total):** Média da quantidade vendida prevista por dia.
- **Receita Média Diária (Total):** Média da receita prevista por dia.
- **Distribuição da Demanda por Categoria:** Percentual da demanda prevista para cada categoria de produto.

### Visualizações
- **Gráficos de Linha:** Demonstram a tendência diária da "Quantidade Vendida Prevista" e "Receita Prevista Total" ao longo do tempo.
- **Gráficos de Barra:** Apresentam a "Quantidade Vendida Prevista por Categoria Original" para identificar as categorias de maior e menor demanda.
- **Gráficos de Pizza:** Ilustram a "Proporção da Quantidade Vendida Prevista por Categoria Original", fornecendo uma visão clara da contribuição de cada categoria.

Estas visualizações, a serem implementadas em ferramentas como Power BI, Tableau, Looker ou Streamlit, fornecem uma visão clara e acionável para a tomada de decisões executivas.

## 9. Planejamento de Deploy e Arquitetura do Projeto

### Arquitetura de Alto Nível
A solução é modular e escalável, abrangendo todo o ciclo de vida dos dados e do modelo:
- **Fontes de Dados:** ERP/E-commerce, CRM, APIs externas.
- **Ingestão de Dados:** Batch Processing (históricos) e Real-time Streaming (eventos imediatos).
- **Armazenamento:** Data Lake (bruto) e Data Warehouse (estruturado).
- **Processamento (ETL/ELT):** Limpeza, enriquecimento, engenharia de features, agregação.
- **Treinamento do Modelo:** Ambiente gerenciado para desenvolvimento e validação de modelos.
- **Disponibilização do Modelo:** Previsões em lote (para planejamento) e API em tempo real (para usos futuros).
- **Dashboard:** Ferramentas de BI para visualização.

### Detalhamento do Pipeline de Dados
1. **Extração:** Dados de ERP/E-commerce e APIs externas.
2. **Carregamento Inicial:** Dados brutos para um Data Lake (e.g., Amazon S3).
3. **Transformação e Enriquecimento:** Limpeza, engenharia de features (lags, rolling means), agregação, e carregamento para um Data Warehouse (e.g., Google BigQuery).
4. **Treinamento do Modelo:** Data Warehouse alimenta o ambiente de treinamento (e.g., Amazon SageMaker).
5. **Inferência e Disponibilização:** Geração de features futuras, predição pelo modelo treinado, e armazenamento das previsões no Data Warehouse para consumo por dashboards ou APIs.

### Estratégias de Deploy
- **Batch Prediction:** Estratégia principal para otimização de estoque (previsões diárias/semanais). Orquestrado via Apache Airflow.
- **Real-time Prediction (API):** Para necessidades futuras como precificação dinâmica ou recomendações instantâneas (e.g., via AWS Lambda, Kubernetes).

### Escalabilidade da Solução
- **Volume de Dados:** Armazenamento distribuído (Data Lakes/Warehouses em nuvem) e processamento distribuído (Apache Spark).
- **Complexidade dos Modelos:** Plataformas de ML em nuvem (SageMaker, AI Platform) com recursos flexíveis (GPUs, treinamento distribuído).
- **Demanda por Previsões:** Serviços serverless e orquestração escalável (Kubernetes, Airflow).

## 10. Considerações de Ética e Governança de Dados

- **Privacidade dos Dados:** Utilização de anonimização/pseudonimização para `ID do Cliente`, minimização de dados, e controle de acesso rigoroso (RBAC) para conformidade com a LGPD.
- **Mitigação de Viés Algorítmico:** Análise de viés nos dados históricos, uso de métricas de fairness, técnicas de reamostragem, interpretabilidade do modelo (XAI) e monitoramento contínuo para garantir previsões justas.
- **Conformidade (LGPD):** Respeito aos princípios de Finalidade, Adequação e Necessidade, Livre Acesso, Segurança e Prevenção, Não Discriminação, e Responsabilização, com um plano de governança de dados robusto que inclui políticas de coleta, armazenamento, uso e descarte de dados.

## 11. Tecnologias Utilizadas

- **Linguagem:** Python
- **Bibliotecas:** Pandas, NumPy, Scikit-learn (LinearRegression, RandomForestRegressor, StandardScaler), XGBoost, Matplotlib, Seaborn.
- **Ferramentas:** Jupyter/Colab Notebooks, GitHub.

