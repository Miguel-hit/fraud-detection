# Detecção de Fraude em Cartões de Crédito com Machine Learning e XAI

Pipeline end-to-end desenvolvido em Python para detecção de transações fraudulentas a partir do dataset de cartões de crédito da Kaggle / TensorFlow (`creditcard.csv`). O projeto cobre desde a engenharia de recursos e o tratamento de desbalanceamento severo até a calibração de limiar de corte (*threshold tuning*) e explicabilidade caixa-cinza com **SHAP**.

---

## 1. O Problema e a Armadilha da Acurácia no Desbalanceamento Extremo

O dataset possui **284.807 transações**, das quais apenas **492 são fraudes** — o que representa míseros **0,17%** do total contra **99,83%** de transações legítimas.

### Por que o desbalanceamento muda a forma de avaliar?
* **A Falácia da Acurácia**: Se criarmos um modelo "ingênuo" (*dummy*) que prevê que toda e qualquer transação é legítima ($0$), ele terá **99,83% de acurácia**, mas deixará passar 100% das fraudes. Esse modelo seria um desastre financeiro.
* **Recall (Sensibilidade)**: É a métrica mais crítica no antifraude. Responde a: *"De todas as fraudes que realmente aconteceram, quantas nós pegamos?"*. Perder uma fraude (Falso Negativo) gera estorno, prejuízo financeiro e perda de confiança.
* **Precisão**: Responde a: *"Quando o modelo apitou fraude, quantas realmente eram?"*. Baixa precisão gera Falsos Positivos, sobrecarregando a mesa de análise manual e gerando atrito com clientes legítimos.
* **F1-Score e PR-AUC (AUPRC)**: Enquanto a curva ROC-AUC pode apresentar uma pontuação artificialmente alta devido ao volume massivo de verdadeiros negativos, a **Curva Precisão-Recall (PR-AUC)** é a métrica padrão-ouro para classes raras, pois avalia o equilíbrio entre precisão e recall sem se iludir com a classe majoritária.

---

## 2. Preparação e Engenharia de Recursos

1. **Transformação Logarítmica (`log1p`)**: A coluna `Amount` possui uma distribuição com assimetria extrema à direita e valores discrepantes. A transformação $log(1 + x)$ comprime a escala e reduz a distorção causada por compras de valores excepcionais.
2. **RobustScaler em vez de StandardScaler**: As variáveis `Amount` e `Time` foram padronizadas utilizando o `RobustScaler`, que subtrai a mediana e divide pelo intervalo interquartil (IQR). Isso impede que outliers extremos contaminem o escalonamento, diferentemente do `StandardScaler` (que utiliza média e desvio padrão).
3. **Divisão Estratificada (`stratify=y`)**: Uma partição padrão aleatória poderia alocar desproporcionalmente as poucas 492 fraudes. O `train_test_split` com `stratify=y` (80% treino e 20% teste) garante que ambos os conjuntos possuam exatamente 0,17% de fraudes (98 fraudes no teste de 56.962 amostras).
4. **Isolamento de Dados**: Nenhuma técnica de reamostragem ou escalonamento foi "vazada" (*data leakage*) para o conjunto de teste. O teste permaneceu intacto com a distribuição original do mundo real.

---

## 3. Comparação de Desempenho entre Modelos

Foram comparadas três abordagens com pesos de classe balanceados para compensar a proporção de 1:578:

| Modelo | Recall (Fraude) | Precisão (Fraude) | F1-Score | PR-AUC (Avg Precision) | ROC-AUC | Fraudes Detectadas (TP) | Fraudes Perdidas (FN) | Alarmes Falsos (FP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regressão Logística** (*Balanced*) | ~91.8% | ~06.0% | ~0.11 | ~0.72 | ~0.97 | 90 | 8 | 1.400+ |
| **Random Forest** (*Balanced*) | ~82.6% | ~85.3% | ~0.84 | ~0.86 | ~0.96 | 81 | 17 | 14 |
| **XGBoost** (*scale_pos_weight*) | **~85.7%** | **~87.5%** | **~0.86** | **~0.88** | **~0.98** | **84** | **14** | **12** |

*Observação: A Regressão Logística maximiza o recall às custas de disparar mais de mil alarmes falsos, o que inviabilizaria a operação diária. O **XGBoost** entregou o melhor compromisso de negócio entre capturar fraudes e evitar atrito com bons clientes.*

---

## 4. Otimização do Limiar de Decisão e Explicabilidade (SHAP)

### Calibração do Limiar de Decisão (*Threshold Tuning*)
O limiar convencional de classificação ($p = 0.5$) não é mandatório em cenários de risco assíncrono:
* Foi realizada uma varredura ao longo da curva de probabilidade do modelo campeão (XGBoost).
* Um limiar entre **0.35 e 0.42** aumentou o **Recall para ~88%** com queda quase desprezível na precisão, permitindo capturar mais fraudes sem saturar o canal de atendimento.

### O que o SHAP (*SHapley Additive exPlanations*) Revelou:
1. **Importância Global (*Beeswarm Plot*)**:
   * As componentes latentes **`V14`**, **`V10`**, **`V12`** e **`V4`** foram as que mais ditaram o risco de fraude.
   * Valores fortemente negativos de `V14`, `V10` e `V12` aumentam expressivamente a probabilidade de fraude (correlação negativa).
   * Valores elevados de `V4` e `V11` empurram a probabilidade na direção da anomalia.
2. **Diagnóstico Local (*Waterfall Plot*)**:
   * Para transações individuais bloqueadas, o gráfico em cascata decompôs a pontuação final a partir do valor base $E[f(X)]$, permitindo justificar para auditorias e clientes exatamente quais anomalias comportamentais geraram o bloqueio preventivo.

---

## 5. O que Mudou em Relação ao que a Expert Fez?

| Etapa / Conceito | Abordagem Sugerida pela Expert | O que foi Implementado / Evoluído no Notebook |
| :--- | :--- | :--- |
| **Escalonamento de Variáveis** | `StandardScaler` tradicional aplicado em `Amount` e `Time`. | Substituição por **`np.log1p()` + `RobustScaler`**, evitando que outliers multimilionários distorçam a média e variância do escalonador. |
| **Técnicas de Balanceamento** | Menção ou uso direto de *oversampling* / *undersampling* isolados. | Priorização de **pesos algorítmicos** (`class_weight='balanced'` e `scale_pos_weight`) para manter os dados 100% reais, deixando módulos de SMOTE e Undersample preparados exclusivamente para o treino. |
| **Modelagem** | Baseline linear com expansão para Random Forest. | Adição do **XGBoost com `scale_pos_weight` calculado dinamicamente**, superando os modelos em PR-AUC e tempo de convergência. |
| **Métricas de Sucesso** | Leitura de Matriz de Confusão e ROC-AUC. | Destaque prioritário para a **Curva Precisão-Recall (PR-AUC)**, que é a métrica estatisticamente correta para o desbalanceamento de 0,17%. |
| **Ajuste de Limiar (*Threshold*)** | Sugerido como evolução conceitual para pipelines futuros. | **Implementado e plotado diretamente no código**, com cálculo automatizado do limiar que maximiza o $F_1$ e comparação antes/depois. |
| **Explicabilidade** | Importância de features básica (*MDI / Gini* nativo do scikit-learn). | Implementação do **SHAP TreeExplainer**, trazendo explicabilidade teórica sólida (valores Shapley), gráfico global de dispersão (*beeswarm*) e auditoria de transação individual (*waterfall*). |

---

## Como Executar
1. Instale os requisitos: `pip install -r requirements.txt` (ou instale: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `imblearn`, `seaborn`, `matplotlib`).
2. Abra o arquivo `credit_card_fraud_detection.ipynb` no Jupyter Lab, VS Code ou Google Colab.
3. Execute as células sequencialmente. O download da base é feito de forma automática diretamente no notebook.