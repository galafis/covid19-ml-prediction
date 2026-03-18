# Data Directory

Esta pasta contém os dados utilizados no pipeline de predição de COVID-19.

## Fonte dos Dados / Data Source

**Fonte primária:** [Our World in Data (OWID) — COVID-19 Dataset](https://github.com/owid/covid-19-data/tree/master/public/data)

- Licença: CC BY 4.0
- Atualização: diária (durante a pandemia)
- Cobertura: 200+ países, Janeiro 2020 – presente
- Responsável: Hannah Ritchie, Edouard Mathieu, Lucas Rodés-Guirao et al.

**Fallback sintético:** quando o download falha, o módulo `src/data_loader.py` gera dados sintéticos realistas via simulação de ondas epidemiológicas gaussianas (semente aleatória fixa = 42, totalmente reprodutível). Esses dados **NÃO representam nenhum país real**.

## Estrutura de Pastas

```
data/
├── raw/                 # Dados brutos baixados do OWID
│   └── owid-covid-data.csv
├── processed/           # Feature matrix gerada pelo pipeline
│   └── features.parquet
└── README.md            # Este arquivo
```

## Dicionário de Dados / Data Dictionary

| Coluna | Tipo | Descrição | Unidade |
|---|---|---|---|
| `iso_code` | str | Código ISO 3166-1 alpha-3 do país | — |
| `continent` | str | Continente | — |
| `location` | str | Nome do país ou região | — |
| `date` | date | Data da observação | YYYY-MM-DD |
| `total_cases` | float | Total acumulado de casos confirmados | Casos |
| `new_cases` | float | Novos casos confirmados no dia | Casos/dia |
| `total_deaths` | float | Total acumulado de mortes | Óbitos |
| `new_deaths` | float | Novas mortes no dia | Óbitos/dia |
| `total_vaccinations` | float | Total acumulado de doses aplicadas | Doses |
| `people_vaccinated` | float | Pessoas com ao menos 1 dose | Pessoas |
| `population` | float | População do país | Pessoas |
| `gdp_per_capita` | float | PIB per capita em dólares internacionais | USD (PPP, 2017) |
| `hospital_beds_per_thousand` | float | Leitos hospitalares por 1.000 habitantes | Leitos/1.000 |

## Features Engineered

O módulo `src/feature_engineering.py` acrescenta as seguintes colunas à matrix de features:

| Feature | Descrição |
|---|---|
| `new_cases_7d_avg` | Média móvel de 7 dias de novos casos |
| `new_cases_lag_7` | Novos casos com defasagem de 7 dias |
| `cases_growth_rate_7d` | Taxa de crescimento semanal de casos |
| `doubling_time_7d` | Tempo de duplicação estimado em dias |
| `cases_per_million` | Casos por milhão de habitantes |
| `vacc_pct_population` | Percentual da população vacinada |

## Privacidade / Privacy

- Nenhum dado individual ou sensível é utilizado.
- Os dados são agregados por país e por dia.
- Dados sintéticos são marcados com prefixo `CC` no código ISO.

## Reprodutibilidade

Para baixar os dados originais:

```bash
make data
```

Ou manualmente:

```bash
wget -O data/raw/owid-covid-data.csv \
  https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv
```

## Referência

> Mathieu, E., Ritchie, H., Ortiz-Ospina, E. et al. (2021). *A global database of COVID-19 vaccinations.* Nature Human Behaviour. https://doi.org/10.1038/s41562-021-01122-8
