"""
DataClean Co. — Tratamento de dados de clientes.

Lê o arquivo dados_clientes.csv e aplica, em sequência:
  1. Carregamento dos dados
  2. Informações básicas
  3. Tratamento de dados ausentes
  4. Tratamento de duplicatas
  5. Detecção e tratamento de outliers (IQR + boxplot)
  6. Normalização e encoding
  7. Conversão de datas e colunas derivadas

O arquivo original nunca é modificado: todo o trabalho é feito sobre uma cópia.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # backend sem janela: o script roda do início ao fim sem intervenção
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes (nada de valores mágicos soltos no código)
# ---------------------------------------------------------------------------
ARQUIVO_ENTRADA = "dados_clientes.csv"
PASTA_GRAFICOS = "graficos"
ARQUIVO_BOXPLOT = os.path.join(PASTA_GRAFICOS, "boxplots_outliers.png")

FATOR_IQR = 1.5  # fator clássico de Tukey para o método do IQR

COLUNAS_NUMERICAS = ["idade", "renda_anual", "valor_compra"]
COLUNAS_CATEGORICAS = ["genero", "categoria_produto", "regiao"]
COLUNAS_IDENTIFICADORAS = ["id_cliente", "nome"]  # não entram no encoding
COLUNA_DATA = "data_compra"

# Critério amplo de duplicata: mesma pessoa, mesma data e mesmo valor de compra
CRITERIO_DUPLICATA_AMPLO = ["nome", "data_compra", "valor_compra"]

# Colunas em que os outliers do IQR são mantidos (valores plausíveis, não erros)
COLUNAS_MANTER_OUTLIERS = ["idade"]

# Formatos de data encontrados no arquivo (regex que identifica -> formato do strptime)
FORMATOS_DATA = {
    r"^\d{4}-\d{2}-\d{2}$": "%Y-%m-%d",  # 2024-06-22 (ISO)
    r"^\d{4}/\d{2}/\d{2}$": "%Y/%m/%d",  # 2024/03/10 (ano primeiro, com barras)
    r"^\d{2}/\d{2}/\d{4}$": "%d/%m/%Y",  # 15/02/2024 (dia/mês/ano)
}

DIAS_DA_SEMANA = {
    0: "segunda-feira",
    1: "terça-feira",
    2: "quarta-feira",
    3: "quinta-feira",
    4: "sexta-feira",
    5: "sábado",
    6: "domingo",
}

LARGURA_TITULO = 70


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------
def imprimir_titulo(texto):
    """Imprime um cabeçalho para separar visualmente as etapas."""
    print("\n" + "=" * LARGURA_TITULO)
    print(texto)
    print("=" * LARGURA_TITULO)


def converter_datas(serie):
    """
    Converte uma série de textos com formatos de data misturados para datetime.

    Cada formato é tratado separadamente (com format explícito), para evitar que
    o pandas "adivinhe" errado e troque dia por mês em 15/02/2024.
    Valores que não casam com nenhum formato conhecido viram NaT (errors="coerce").
    """
    texto = serie.astype("string").str.strip()
    resultado = pd.Series(pd.NaT, index=serie.index, dtype="datetime64[ns]")
    for padrao, formato in FORMATOS_DATA.items():
        mascara = texto.str.match(padrao).fillna(False).astype(bool)
        resultado[mascara] = pd.to_datetime(texto[mascara], format=formato, errors="coerce")
    return resultado


def limites_iqr(serie):
    """Calcula os limites inferior e superior do método do IQR."""
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    return q1, q3, iqr, q1 - FATOR_IQR * iqr, q3 + FATOR_IQR * iqr


# ---------------------------------------------------------------------------
# Etapa 1 — Carregar os dados
# ---------------------------------------------------------------------------
def carregar_dados(caminho=ARQUIVO_ENTRADA):
    imprimir_titulo("ETAPA 1 — CARREGAMENTO DOS DADOS")
    try:
        df_original = pd.read_csv(caminho)
    except FileNotFoundError:
        print(f"ERRO: arquivo '{caminho}' não encontrado.")
        print("Coloque o CSV no mesmo diretório do script e rode novamente.")
        sys.exit(1)

    print(f"Arquivo '{caminho}' carregado com sucesso.")
    return df_original


# ---------------------------------------------------------------------------
# Etapa 2 — Informações básicas
# ---------------------------------------------------------------------------
def informacoes_basicas(df):
    imprimir_titulo("ETAPA 2 — INFORMAÇÕES BÁSICAS")

    print("Primeiras 5 linhas:")
    print(df.head(5).to_string())

    linhas, colunas = df.shape
    print(f"\nTotal de registros: {linhas}")
    print(f"Total de colunas:   {colunas}")

    print("\nTipo de dado de cada coluna:")
    print(df.dtypes.to_string())

    print("\nResumo (info):")
    df.info()

    # Conferência extra: id_cliente não é único, então não serve como chave
    ids_repetidos = df["id_cliente"].value_counts()
    ids_repetidos = ids_repetidos[ids_repetidos > 1]
    print(f"\nid_cliente únicos: {df['id_cliente'].nunique()} de {linhas} linhas")
    print(f"ids que aparecem mais de uma vez: {sorted(ids_repetidos.index.tolist())}")


# ---------------------------------------------------------------------------
# Etapa 3 — Dados ausentes
# ---------------------------------------------------------------------------
def tratar_ausentes(df):
    imprimir_titulo("ETAPA 3 — DADOS AUSENTES")

    # Padroniza espaços nas colunas de texto antes de tudo (evita "Sul " ≠ "Sul")
    for coluna in COLUNAS_CATEGORICAS + ["nome", COLUNA_DATA]:
        antes = df[coluna].copy()
        df[coluna] = df[coluna].astype("string").str.strip()
        alteradas = int((antes.astype("string") != df[coluna]).fillna(False).sum())
        if alteradas:
            print(f"Espaços removidos em '{coluna}': {alteradas} linha(s)")

    # A string "nan" (e variações) em data_compra deve virar nulo real.
    # Obs.: o read_csv já converte "nan" para nulo por padrão; esta linha garante o
    # comportamento mesmo se o arquivo vier com "NaN", "NAN" etc. lidos como texto.
    mascara_nan_texto = df[COLUNA_DATA].str.lower().isin(["nan", "none", "null", ""]).fillna(False)
    qtd_nan_texto = int(mascara_nan_texto.sum())
    df.loc[mascara_nan_texto, COLUNA_DATA] = pd.NA
    print(f"Strings 'nan' convertidas em nulo real em '{COLUNA_DATA}': {qtd_nan_texto}")

    # Diagnóstico: contagem e percentual de nulos por coluna
    nulos_antes = df.isna().sum()
    percentual = (nulos_antes / len(df) * 100).round(1)
    tabela = pd.DataFrame({"nulos": nulos_antes, "percentual_%": percentual})
    print("\nNulos por coluna (antes do tratamento):")
    print(tabela[tabela["nulos"] > 0].to_string())

    # Guarda quantos nulos cada linha tinha ORIGINALMENTE. Isso é usado na etapa 4
    # para escolher o registro "mais completo" caso existam duplicatas.
    df["_nulos_originais"] = df[COLUNAS_NUMERICAS + [COLUNA_DATA]].isna().sum(axis=1)

    # --- idade: mediana ---
    # Motivo: distribuição de idades pode ser assimétrica; a mediana não é puxada
    # por extremos. Depois arredondamos e convertemos para inteiro (idade é discreta).
    mediana_idade = df["idade"].median()
    df["idade"] = df["idade"].fillna(mediana_idade).round().astype(int)
    print(f"\nidade: preenchida com a mediana ({mediana_idade:.1f}) e convertida para inteiro")

    # --- renda_anual e valor_compra: mediana ---
    # Motivo: estas colunas têm outliers fortes (500000/750000 e 15000/20000).
    # A média seria inflada por eles; a mediana é robusta.
    for coluna in ["renda_anual", "valor_compra"]:
        mediana = df[coluna].median()
        media = df[coluna].mean()
        df[coluna] = df[coluna].fillna(mediana)
        print(f"{coluna}: preenchida com a mediana ({mediana:,.2f}); "
              f"a média seria {media:,.2f}, distorcida pelos outliers")

    # --- data_compra: mantida como nula ---
    # Motivo: não existe base para inventar uma data de compra. Remover as linhas
    # descartaria ~13% de um dataset de apenas 30 registros (e com elas valores de
    # compra, renda, região etc. que são válidos). Por isso a data fica nula e as
    # colunas derivadas (ano, mês, ...) também ficarão nulas nessas linhas.
    print(f"{COLUNA_DATA}: MANTIDA como nula ({int(nulos_antes[COLUNA_DATA])} linhas) — "
          "não inventamos datas e não removemos linhas válidas no resto")

    # Resumo antes -> depois
    nulos_depois = df.drop(columns="_nulos_originais").isna().sum()
    print("\nResumo de nulos (antes → depois):")
    for coluna in df.columns.drop("_nulos_originais"):
        if nulos_antes[coluna] > 0:
            print(f"  Nulos em {coluna}: {int(nulos_antes[coluna])} → {int(nulos_depois[coluna])}")
    return df


# ---------------------------------------------------------------------------
# Etapa 4 — Duplicatas
# ---------------------------------------------------------------------------
def tratar_duplicatas(df):
    imprimir_titulo("ETAPA 4 — DUPLICATAS")
    linhas_antes = len(df)
    colunas_dados = df.columns.drop("_nulos_originais")

    # 4.1 Duplicatas de linha completa (todas as colunas iguais)
    dup_completas = int(df.duplicated(subset=colunas_dados).sum())
    print(f"Duplicatas de linha completa: {dup_completas}")

    # 4.2 Apenas id_cliente — mostrado só para ilustrar por que NÃO serve como critério
    dup_por_id = int(df.duplicated(subset=["id_cliente"]).sum())
    print(f"Linhas com id_cliente repetido: {dup_por_id} "
          "(NÃO usado como critério: o mesmo id aparece com nomes diferentes)")

    # 4.3 Critério amplo: mesmo nome + mesma data + mesmo valor de compra.
    # As datas são comparadas já convertidas (2024/03/10 e 2024-03-10 são a mesma data).
    chave = pd.DataFrame({
        "nome": df["nome"],
        "data_compra": converter_datas(df[COLUNA_DATA]),
        "valor_compra": df["valor_compra"],
    })
    # Linhas sem data não entram neste critério: duas datas nulas não provam
    # que foi a mesma compra.
    chave_valida = chave.dropna(subset=["data_compra"])
    mascara_amplo = chave_valida.duplicated(subset=CRITERIO_DUPLICATA_AMPLO, keep=False)
    dup_amplas = int(chave_valida.duplicated(subset=CRITERIO_DUPLICATA_AMPLO).sum())
    print(f"Duplicatas pelo critério amplo {CRITERIO_DUPLICATA_AMPLO}: {dup_amplas}")
    if mascara_amplo.any():
        print(df.loc[mascara_amplo[mascara_amplo].index].to_string())

    # Critério de remoção: entre registros duplicados, mantém o MAIS COMPLETO
    # (menos nulos no arquivo original — ou seja, menos valores imputados).
    # Ordenação estável: em caso de empate, fica o que aparece primeiro no arquivo.
    df = df.sort_values("_nulos_originais", kind="stable")
    df = df[~df.duplicated(subset=colunas_dados)]  # remove duplicatas completas
    chave = chave.loc[df.index]
    remover_amplo = chave.duplicated(subset=CRITERIO_DUPLICATA_AMPLO) & chave["data_compra"].notna()
    df = df[~remover_amplo].sort_index()

    linhas_depois = len(df)
    print(f"\nCritério de remoção: manter o registro mais completo (menos nulos originais)")
    print(f"Linhas: {linhas_antes} → {linhas_depois} ({linhas_antes - linhas_depois} removida(s))")

    return df.drop(columns="_nulos_originais")


# ---------------------------------------------------------------------------
# Etapa 5 — Outliers
# ---------------------------------------------------------------------------
def tratar_outliers(df):
    imprimir_titulo("ETAPA 5 — OUTLIERS (MÉTODO DO IQR)")

    # Boxplots ANTES do tratamento, para identificação visual
    os.makedirs(PASTA_GRAFICOS, exist_ok=True)
    fig, eixos = plt.subplots(1, len(COLUNAS_NUMERICAS), figsize=(14, 4))
    for eixo, coluna in zip(eixos, COLUNAS_NUMERICAS):
        eixo.boxplot(df[coluna], whis=FATOR_IQR)
        eixo.set_title(coluna)
    fig.suptitle("Boxplots antes do tratamento de outliers")
    fig.tight_layout()
    fig.savefig(ARQUIVO_BOXPLOT, dpi=100)
    plt.close(fig)
    print(f"Boxplots salvos em '{ARQUIVO_BOXPLOT}'\n")

    for coluna in COLUNAS_NUMERICAS:
        q1, q3, iqr, limite_inf, limite_sup = limites_iqr(df[coluna])
        abaixo = df[coluna] < limite_inf
        acima = df[coluna] > limite_sup
        qtd = int((abaixo | acima).sum())

        print(f"{coluna}: Q1={q1:,.2f}  Q3={q3:,.2f}  IQR={iqr:,.2f}  "
              f"limites=[{limite_inf:,.2f}, {limite_sup:,.2f}]")
        print(f"  Outliers detectados: {qtd}")
        if qtd:
            print(f"  Valores: {sorted(df.loc[abaixo | acima, coluna].round(2).tolist())}")

        if not qtd:
            print("  Ação: nenhuma (sem outliers)")
        elif coluna in COLUNAS_MANTER_OUTLIERS:
            # Decisão para idade: MANTER. O IQR marca 62 anos só porque as idades
            # estão concentradas entre 31 e 42; mas 62 é uma idade perfeitamente
            # plausível para um cliente. Cortar para 58 seria alterar um dado real.
            print(f"  Ação: mantidos ({qtd} valor(es) plausível(is); outlier estatístico, não erro)")
        else:
            # Decisão para renda_anual e valor_compra: CAPPING (winsorização).
            # Motivo: com só 30 linhas, remover registros custa caro e os demais
            # campos dessas linhas são válidos. O capping mantém a linha, mas impede
            # que um valor extremo domine médias, normalização e correlações.
            df[coluna] = df[coluna].clip(lower=limite_inf, upper=limite_sup)
            print(f"  Ação: capping nos limites do IQR → {qtd} linha(s) alterada(s)")
    return df


# ---------------------------------------------------------------------------
# Etapa 6 — Normalização e encoding
# ---------------------------------------------------------------------------
def normalizar_e_codificar(df):
    imprimir_titulo("ETAPA 6 — NORMALIZAÇÃO E ENCODING")
    colunas_antes = df.shape[1]

    # Normalização: criamos colunas NOVAS, preservando as originais.
    #  - Min-Max (0 a 1): bom para comparar escalas e para algoritmos baseados em distância.
    #  - Z-score: média 0 e desvio 1 (ddof=0, igual ao StandardScaler do scikit-learn).
    for coluna in COLUNAS_NUMERICAS:
        minimo, maximo = df[coluna].min(), df[coluna].max()
        media, desvio = df[coluna].mean(), df[coluna].std(ddof=0)
        df[f"{coluna}_norm"] = (df[coluna] - minimo) / (maximo - minimo)
        df[f"{coluna}_z"] = (df[coluna] - media) / desvio
        print(f"{coluna}: {coluna}_norm em [{df[f'{coluna}_norm'].min():.2f}, "
              f"{df[f'{coluna}_norm'].max():.2f}] | {coluna}_z média="
              f"{df[f'{coluna}_z'].mean():.2f}, desvio={df[f'{coluna}_z'].std(ddof=0):.2f}")

    # Encoding: One-Hot para genero, categoria_produto e regiao.
    # Motivo: são categorias NOMINAIS (sem ordem). Label Encoding (0, 1, 2...)
    # criaria uma ordem falsa (ex.: "Sul" > "Norte"). Com poucas categorias,
    # o One-Hot gera poucas colunas e não tem esse problema.
    # As colunas originais são mantidas para leitura humana e para os gráficos.
    # id_cliente e nome NÃO entram: são identificadores, não variáveis preditoras.
    print()
    for coluna in COLUNAS_CATEGORICAS:
        print(f"{coluna}: {df[coluna].nunique()} categorias → {sorted(df[coluna].dropna().unique())}")
    dummies = pd.get_dummies(df[COLUNAS_CATEGORICAS], prefix=COLUNAS_CATEGORICAS, dtype=int)
    df = pd.concat([df, dummies], axis=1)

    print(f"\nColunas: {colunas_antes} → {df.shape[1]} "
          f"(+{len(COLUNAS_NUMERICAS) * 2} de normalização, +{dummies.shape[1]} de one-hot)")
    return df


# ---------------------------------------------------------------------------
# Etapa 7 — Datas
# ---------------------------------------------------------------------------
def tratar_datas(df):
    imprimir_titulo("ETAPA 7 — DATAS")
    texto_original = df[COLUNA_DATA].copy()

    print("Formatos encontrados em data_compra:")
    for padrao, formato in FORMATOS_DATA.items():
        qtd = int(texto_original.str.match(padrao).fillna(False).sum())
        print(f"  {formato:<10} → {qtd} linha(s)")
    print(f"  (nulo)     → {int(texto_original.isna().sum())} linha(s)")

    df[COLUNA_DATA] = converter_datas(texto_original)

    # Conferência: nenhuma data que existia como texto pode ter virado nula
    perdidas = texto_original.notna() & df[COLUNA_DATA].isna()
    print(f"\nTipo: {texto_original.dtype} → {df[COLUNA_DATA].dtype}")
    print(f"Datas válidas: {int(texto_original.notna().sum())} → {int(df[COLUNA_DATA].notna().sum())}")
    if perdidas.any():
        print(f"ATENÇÃO: {int(perdidas.sum())} data(s) não reconhecida(s): "
              f"{texto_original[perdidas].tolist()}")
    else:
        print("Conferência OK: nenhuma data válida virou nula na conversão")

    # Colunas derivadas. Usamos Int64 (inteiro que aceita nulo) porque as linhas
    # sem data foram mantidas propositalmente na etapa 3.
    df["ano"] = df[COLUNA_DATA].dt.year.astype("Int64")
    df["mes"] = df[COLUNA_DATA].dt.month.astype("Int64")
    df["dia"] = df[COLUNA_DATA].dt.day.astype("Int64")
    df["trimestre"] = df[COLUNA_DATA].dt.quarter.astype("Int64")
    df["dia_da_semana"] = df[COLUNA_DATA].dt.day_of_week.map(DIAS_DA_SEMANA).astype("string")
    print("Colunas criadas: ano, mes, dia, trimestre, dia_da_semana")

    print("\nDatas distintas após a conversão:")
    print(df[COLUNA_DATA].value_counts(dropna=False).sort_index().to_string())
    return df


# ---------------------------------------------------------------------------
# Resumo final
# ---------------------------------------------------------------------------
def resumo_final(df_original, df):
    imprimir_titulo("RESUMO FINAL (ANTES × DEPOIS)")
    print(f"Shape: {df_original.shape} → {df.shape}")
    # Compara só as 9 colunas originais; as derivadas de data herdam os nulos de data_compra
    colunas_orig = list(df_original.columns)
    print(f"Nulos nas colunas originais: {int(df_original.isna().sum().sum())} → "
          f"{int(df[colunas_orig].isna().sum().sum())}")

    nulos_finais = df.isna().sum()
    nulos_finais = nulos_finais[nulos_finais > 0]
    print("\nNulos restantes (esperados — linhas sem data mantidas de propósito):")
    print(nulos_finais.to_string())

    print("\nEstatísticas das colunas numéricas (antes):")
    print(df_original[COLUNAS_NUMERICAS].describe().round(2).to_string())
    print("\nEstatísticas das colunas numéricas (depois):")
    print(df[COLUNAS_NUMERICAS].describe().round(2).to_string())

    print("\nPrimeiras 5 linhas do dataset tratado (colunas principais):")
    colunas_principais = list(df_original.columns) + ["ano", "mes", "dia_da_semana"]
    print(df[colunas_principais].head(5).to_string())


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    df_original = carregar_dados()
    df = df_original.copy()  # trabalhamos SEMPRE sobre a cópia

    informacoes_basicas(df)
    df = tratar_ausentes(df)
    df = tratar_duplicatas(df)
    df = tratar_outliers(df)
    df = normalizar_e_codificar(df)
    df = tratar_datas(df)
    resumo_final(df_original, df)
    return df_original, df


if __name__ == "__main__":
    main()
