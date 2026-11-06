# CLAUDE.md — Projeto DataClean Co.: Tratamento de Dados de Clientes

## Visão geral

Sistema de tratamento de dados em Python, implementado do zero, que lê o arquivo `dados_clientes.csv` (histórico de clientes e suas compras) e o deixa limpo, consistente e pronto para análise exploratória e visualizações.

Contexto: atuamos como analista de dados da empresa fictícia **DataClean Co.**. O time de negócios enviou um arquivo com valores ausentes, duplicatas, outliers e inconsistências. A missão é garantir que o dataset seja confiável.

## Idioma e estilo

- Responder, comentar o código e nomear mensagens de saída **em português (Brasil)**.
- Nomes de variáveis e funções em `snake_case`, preferencialmente em português, coerentes com as colunas do CSV.
- Código simples e didático, com comentários curtos explicando cada etapa do tratamento. É um projeto de aprendizado, então clareza vale mais que sofisticação.

## Stack

- Python 3.10+
- `pandas` e `numpy` (tratamento)
- `matplotlib` e `seaborn` (visualizações opcionais)
- `scikit-learn` (opcional, para `MinMaxScaler` / `StandardScaler`; pode ser feito só com pandas)
- `openpyxl` (opcional, para o relatório `.xlsx`)

## Estrutura de arquivos

```
.
├── CLAUDE.md
├── dados_clientes.csv          # entrada (não modificar)
├── tratamento_dados.py         # script principal (ou tratamento_dados.ipynb)
├── dados_clientes_limpo.csv    # saída opcional
├── relatorio_tratamento.xlsx   # saída opcional (antes x depois)
└── graficos/                   # saída opcional (PNGs das visualizações)
```

## Como executar

1. Colocar `dados_clientes.csv` no mesmo diretório do script.
2. Instalar dependências: `pip install pandas numpy matplotlib seaborn openpyxl scikit-learn`
3. Rodar: `python tratamento_dados.py` (ou executar as células no Jupyter/Google Colab).

No Colab, o CSV precisa ser enviado antes (upload na aba de arquivos ou `files.upload()`).

## Modelo de dados real (`dados_clientes.csv`)

O arquivo tem **30 linhas e 9 colunas**. Atenção: os nomes reais diferem um pouco do enunciado, e o código deve usar os **nomes reais**.

| Coluna | Tipo esperado | Observação |
|---|---|---|
| `id_cliente` | inteiro | **Não é único** (ex.: 1006, 1010, 1011 e 1018 aparecem várias vezes com nomes diferentes) |
| `nome` | string | Nomes se repetem entre ids diferentes |
| `idade` | float → inteiro | Tem nulos (lida como float) |
| `genero` | string | Valores: Masculino, Feminino, Outro |
| `renda_anual` | float | No enunciado aparece como `renda`. Tem nulo e valores extremos (500000, 750000) |
| `categoria_produto` | string | Eletrônicos, Casa, Moda, Esporte, Beleza |
| `valor_compra` | float | No enunciado aparece como `valor_gasto`. Tem nulos e valores extremos (15000, 20000) |
| `data_compra` | string → datetime | **Formatos misturados** e valores `nan` em texto |
| `regiao` | string | Norte, Nordeste, Centro-Oeste, Sudeste, Sul |

Não existe coluna `produto` neste arquivo (só `categoria_produto`). Não inventar essa coluna.

### Problemas conhecidos nos dados (verificar no código, não assumir)

- **Nulos:** `idade`, `renda_anual`, `valor_compra` e `data_compra`.
- **Datas inconsistentes:** coexistem `2024-06-22` (ISO), `2024/03/10` (barras com ano primeiro), `15/02/2024` (dia/mês/ano) e a string `nan`.
- **Outliers prováveis:** `renda_anual` em 500000 e 750000; `valor_compra` em 15000 e 20000, bem acima do restante.
- **Duplicatas:** como `id_cliente` não é único, **não** usar só o id para detectar duplicatas. Verificar duplicatas de linha completa e, depois, discutir critérios mais amplos (ex.: mesmo `nome` + `data_compra` + `valor_compra`). Reportar quantas foram encontradas, mesmo que sejam zero.
- **Possíveis inconsistências de domínio:** idades mínimas de 18 anos, repetição de valores idênticos de renda e de compra em linhas distintas (pode indicar preenchimento anterior ou dado sintético). Apenas sinalizar, sem corrigir sem justificativa.

## Requisitos do sistema (etapas, nesta ordem)

Organizar o código em funções, uma por etapa, chamadas em sequência no `main()`.

### 1. Carregar os dados
- Ler `dados_clientes.csv` com `pd.read_csv`.
- Tratar erro de arquivo não encontrado com mensagem clara.

### 2. Informações básicas
- Mostrar as 5 primeiras linhas (`head()`).
- Exibir número total de registros e de colunas (`shape`).
- Exibir o tipo de dado de cada coluna (`dtypes` / `info()`).

### 3. Dados ausentes
- Identificar colunas com nulos (contagem e percentual).
- Converter a string `"nan"` em `data_compra` para nulo real (`np.nan`) antes de contar.
- Decidir e **justificar** a estratégia por coluna. Sugestão:
  - `idade`: preencher com a **mediana**.
  - `renda_anual` e `valor_compra`: preencher com a **mediana** (mais robusta a outliers que a média).
  - `data_compra`: não inventar data; remover as linhas sem data **ou** manter como nulo e documentar a escolha.
- Mostrar a contagem de nulos antes e depois.

### 4. Duplicatas
- Verificar com `duplicated()` e mostrar quantas existem.
- Remover mantendo o registro mais relevante (ex.: o mais recente por `data_compra`, ou o mais completo, com menos nulos). Documentar o critério.
- Mostrar o número de linhas antes e depois.

### 5. Outliers
- Aplicar o método do **IQR** (quartis) nas colunas numéricas (`idade`, `renda_anual`, `valor_compra`): `Q1 - 1.5*IQR` e `Q3 + 1.5*IQR`.
- Gerar **boxplots** para identificação visual.
- Decidir por coluna se o valor é **removido, substituído (capping/winsorização) ou mantido**, e justificar. Com apenas 30 linhas, preferir **capping** a remoção, para não perder dados.
- Mostrar quantos outliers foram detectados e o que foi feito.

### 6. Normalização e encoding
- **Normalização** das colunas numéricas (`idade`, `renda_anual`, `valor_compra`): Min-Max (0 a 1) e/ou Z-score. Criar colunas novas (ex.: `renda_anual_norm`) em vez de sobrescrever as originais.
- **Encoding** das categóricas:
  - `genero`, `categoria_produto`, `regiao`: One-Hot Encoding (`pd.get_dummies`) ou Label Encoding, com justificativa.
  - `nome` e `id_cliente` **não** entram no encoding (identificadores, não variáveis preditoras).

### 7. Datas
- Converter `data_compra` para `datetime` lidando com os **três formatos** (ex.: normalizar `/` e `-`, e tratar `dd/mm/yyyy` separadamente, com `pd.to_datetime` e `errors="coerce"`). Conferir que nenhuma data válida virou nula por engano.
- Criar colunas derivadas: `ano`, `mes`, `dia`, `dia_da_semana` (e, se quiser, `trimestre`).

## Opcional (desafio extra)

### Visualizações (salvar em `graficos/`)
- Histograma da distribuição de `renda_anual`.
- Boxplot de `valor_compra` por `categoria_produto`.
- Scatterplot de `idade` x `valor_compra`.
- Heatmap de correlação entre as variáveis numéricas.

### Exportações e relatório
- Exportar o dataset final limpo para `dados_clientes_limpo.csv` (`index=False`, `encoding="utf-8"`).
- Criar `relatorio_tratamento.xlsx` (ou `.pdf`) com o **antes e depois**: shape, nulos, duplicatas, outliers e estatísticas descritivas.

## Regras de trabalho

- **Nunca sobrescrever** `dados_clientes.csv`. Trabalhar sobre uma cópia (`df = df_original.copy()`).
- Manter o `df_original` para permitir o comparativo antes/depois.
- Cada etapa deve **imprimir um resumo legível** do que foi feito (ex.: "Nulos em idade: 2 → 0").
- Não usar valores mágicos soltos: constantes (nome do arquivo, fator do IQR = 1.5) no topo do script.
- Toda decisão de tratamento (preencher, remover, manter) deve ter um comentário com o motivo.
- Não alterar valores de forma silenciosa: se algo for corrigido, registrar quantas linhas foram afetadas.
- Reprodutibilidade: o script deve rodar do início ao fim sem intervenção manual.
- Se o código for um notebook, separar as etapas em células com títulos em markdown.

## Critérios de qualidade (checklist final)

- [ ] Carrega o CSV e mostra `head(5)`, shape e dtypes
- [ ] Nulos tratados em todas as colunas, com justificativa
- [ ] Duplicatas verificadas (e removidas, se houver) com critério documentado
- [ ] Outliers detectados por IQR + boxplot e tratados com justificativa
- [ ] Colunas numéricas normalizadas e categóricas codificadas
- [ ] `data_compra` convertida corretamente (3 formatos) e colunas `ano`, `mes`, `dia_da_semana` criadas
- [ ] Sem nulos inesperados no dataset final
- [ ] (Opcional) gráficos, CSV limpo e relatório antes/depois gerados
