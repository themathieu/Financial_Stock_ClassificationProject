# Machine Learning Pipeline Project: 200+ Financial Indicators Stock Performance Prediction

Questo progetto implementa una pipeline completa di Machine Learning per la classificazione predittiva delle performance azionarie statunitensi a partire dai bilanci societari (SEC 10-K filings).

Il lavoro è sviluppato per il corso di **Machine Learning** (SUPSI) nel rispetto delle specifiche descritte in `Progetto.pdf`.

---

## 1. Descrizione del Dataset e Contesto di Business

### Origine dei Dati
Il dataset utilizzato raccoglie oltre 200 indicatori finanziari per le società quotate sui mercati azionari statunitensi (NYSE, NASDAQ, AMEX) nel quinquennio **2014–2018**. I dati sono stati estratti tramite l'API di [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs/) e derivano direttamente dai bilanci annuali ufficiali (Form 10-K).

### Obiettivo di Modellazione
L'obiettivo è comprendere se le metriche fondamentali di bilancio di un anno fiscale $T$ contengano un segnale predittivo sufficiente a prevedere la performance del prezzo del titolo nell'anno successivo $T+1$.

- **Problema di Machine Learning**: Classificazione binaria supervisionata.
- **Variabile Target (`Class`)**:
  - `Class = 1`: Il prezzo del titolo azionario registra un rendimento positivo nell'anno solare successivo ($\text{Price Var} > 0\%$). Rappresenta i titoli da acquistare all'inizio dell'anno e vendere a fine anno.
  - `Class = 0`: Il prezzo del titolo azionario registra un rendimento nullo o negativo nell'anno successivo ($\text{Price Var} \le 0\%$).

### Dimensione dei Dati
- **Volumi**: ~3.800–4.300 aziende per anno, per un totale di oltre 20.000 campioni sui 5 anni (2014–2018).
- **Spazio delle Feature**: 225 colonne originali, comprendenti:
  - Ticker aziendale (identificativo univoco).
  - Variabile categorica `Sector` (es. *Technology, Healthcare, Financial Services, Industrials, Consumer Defensive*, ecc.).
  - Variabili numeriche di bilancio (Stato Patrimoniale, Conto Economico, Rendiconto Finanziario, indicatori di crescita e multipli di mercato come P/E, P/B, EV/EBITDA, Debt/Equity, ROE, FCF).
  - Variabile di leakage `[ANNO] PRICE VAR [%]` (da rimuovere).
  - Target `Class`.

---

## 2. Metodologia di Validazione Temporale ("Out-of-Time")

Per garantire la massima correttezza scientifica ed evitare il fenomeno del **Data Leakage temporale** (imparare dal futuro o contaminare distribuzioni macroeconomiche distinte):
1. **Training & Validation Set (Anni 2014–2017)**:
   - I primi 4 anni storici (~17.500 osservazioni) sono utilizzati per l'addestramento, il preprocessing e la validazione incrociata.
   - Per il tuning degli iperparametri viene impiegata una validazione temporale/rolling (`TimeSeriesSplit` o fold per anno solare), rispettando la freccia del tempo.
2. **Test Set Finale Out-of-Sample (Anno 2018)**:
   - L'ultimo anno (~4.300 osservazioni) è isolato rigorosamente come banco di prova "cieco".
   - Non viene mai osservato durante il preprocessing o il tuning, simulando la reale decisione di un investitore a inizio 2018.

---

## 3. Selezione delle Feature con LASSO ($L_1$ Regularization)

Dato l'elevato numero di feature (~220) a fronte di forte multicollinearità contabile e rumore statistico, il primo passo di modellazione prevede una selezione delle feature tramite **LASSO (Logistic Regression con penalizzazione $L_1$)**:

### Perché LASSO?
- **Sparsità**: A differenza della penalità $L_2$ (Ridge), $L_1$ azzera esattamente i coefficienti delle feature non informative o ridondanti, riducendo lo spazio a un sottoinsieme ristretto di variabili chiave.
- **Efficienza computazionale**: Riduce la dimensionalità per modelli sensibili alla distanza (come KNN) e accelera il tuning.
- **Interpretabilità**: Permette di ispezionare quali indicatori di bilancio hanno il maggior peso predittivo.

### Calibrazione Operativa ed Esito della Selezione
- **Problema di Underfitting Iniziale**: Una regolarizzazione $L_1$ con penalità eccessivamente severa e filtro a soglia media riduceva lo spazio a sole 7 feature, causando un grave impoverimento informativo sui bilanci.
- **Soluzione Adottata**: Utilizzo del solutore stabile `liblinear` impostando `max_features=50` e ordinamento per coefficiente assoluto decrescente.
- **Risultato Ottenuto**: Riduzione da **229 feature trasformate a 50 feature selezionate (compressione del 78.2%)**, garantendo il bilanciamento ideale tra riduzione del rumore e copertura di tutti i pilastri contabili.

### Coerenza Economica e Macroeconomica dei Risultati
L'ispezione dei coefficienti stimati da LASSO sul quadriennio 2014–2017 evidenzia una notevole coerenza con la storia economica del periodo:
1. **Forte penalizzazione del settore energetico (`Sector_Energy = -0.497`)**: Riflette fedelmente il crollo del prezzo del petrolio greggio (WTI) crollato da oltre 100\$ a meno di 30\$ al barile tra il 2014 e inizio 2016.
2. **Premio per i settori difensivi e a dividendo (`Sector_Utilities = +0.714`, `Sector_Financial Services = +0.380`, `Sector_Real Estate = +0.329`)**: In un contesto di tassi di interesse a zero (ZIRP) della Federal Reserve, il mercato ha cercato rendimento sicuro nei flussi di cassa regolari di utility e canoni immobiliari.
3. **Solidità dei fondamentali microeconomici**: Indicatori come `3Y Dividend per Share Growth (+0.127)`, `5Y Dividend per Share Growth (+0.125)`, `10Y Net Income Growth (+0.082)` e `Gross Margin (+0.030)` mostrano coefficienti positivi stabili, confermando che le società con flussi di cassa reali e vantaggi competitivi duraturi (*economic moat*) sovraperformano con maggiore probabilità.

### Limiti e Trade-off noti (Discussione critica)
- **Variabili collineari**: In presenza di gruppi di feature fortemente correlate (es. `EBIT` vs `Operating Income`), LASSO tende a selezionarne una arbitrariamente, azzerando le altre.
- **Relazioni non-lineari**: Essendo un modello lineare, LASSO potrebbe sottovalutare feature con impatti non-lineari (es. andamenti a "U" sui livelli ottimali di indebitamento).
- **Sensibilità alla scala**: Richiede una robusta standardizzazione preliminare per evitare distorsioni da outlier estremi.

---

## 4. Struttura del Repository

```text
Project/
├── data/
│   ├── 2014_Financial_Data.csv
│   ├── 2015_Financial_Data.csv
│   ├── 2016_Financial_Data.csv
│   ├── 2017_Financial_Data.csv
│   └── 2018_Financial_Data.csv
├── Financial_Stock_Classification_Pipeline.ipynb  # Notebook principale end-to-end
└── README.md                                     # Documentazione metodologica
```
