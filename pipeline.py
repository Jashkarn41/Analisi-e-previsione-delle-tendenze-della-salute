import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

CARTELLA_DATI = Path(__file__).resolve().parent / "notebook" / "data"
CARTELLA_OUTPUT = Path(__file__).resolve().parent / "notebook" / "output"

FILE_T1 = "T1_hlth_cd_aro__custom_24003598_linear_2_0.csv"
FILE_T2 = "T2_demo_mlexpec__custom_24024898_linear_2_0.csv"
FILE_T3 = "T3_hlth_ehis_cd1e__custom_24003937_linear_2_0.csv"
FILE_T5_HLY = "T5.1_hlth_hlye__custom_23928152_linear_2_0.csv"
FILE_T5_PROJ = "T5.2_proj_stp25__custom_23927919_linear_2_0.csv"
FILE_T6_ATTIVITA = "T6.1_hlth_ehis_pe9e$defaultview_linear_2_0.csv"
FILE_T6_FUMO = "T6.2_sdg_03_30__custom_24025495_linear_2_0.csv"
FILE_T6_ALCOL = "T6.3_hlth_ehis_al1e__custom_24027848_linear_2_0.csv"
FILE_T7_REDDITO = "T7.1_hlth_silc_11__custom_23966893_linear_2_0.csv"
FILE_T7_ISTRUZIONE = "T7.2_hlth_silc_02__custom_23967814_linear_2_0.csv"
FILE_T8_SPESA = "T8.1_hlth_sha11_hc__custom_23969546_linear_2_0.csv"

FILE_T4_SPERANZA = "T4.1_demo_r_mlifexp__custom_24078981_linear_2_0.csv"
FILE_T4_MORTALITA = "T4.2_hlth_cd_asdr2__custom_24079748_linear_2_0.csv"

MACROREGIONI = {
    "SE": "Europa settentrionale", "FI": "Europa settentrionale", "DK": "Europa settentrionale",
    "EE": "Europa settentrionale", "LV": "Europa settentrionale", "LT": "Europa settentrionale",
    "IE": "Europa settentrionale",
    "DE": "Europa occidentale", "FR": "Europa occidentale", "NL": "Europa occidentale",
    "BE": "Europa occidentale", "LU": "Europa occidentale", "AT": "Europa occidentale",
    "IT": "Europa meridionale", "ES": "Europa meridionale", "PT": "Europa meridionale",
    "EL": "Europa meridionale", "MT": "Europa meridionale", "CY": "Europa meridionale",
    "HR": "Europa meridionale",
    "PL": "Europa orientale", "CZ": "Europa orientale", "SK": "Europa orientale",
    "HU": "Europa orientale", "RO": "Europa orientale", "BG": "Europa orientale",
    "SI": "Europa orientale",
}

GENERE_REGIONI = {
    "Females": "Femmine", "Males": "Maschi", "Total": "Totale",
    "F: Females": "Femmine", "M: Males": "Maschi", "T: Total": "Totale",
}

PAESI = {
    "AT": "Austria", "BE": "Belgio", "BG": "Bulgaria", "CY": "Cipro", "CZ": "Cechia",
    "DE": "Germania", "DK": "Danimarca", "EE": "Estonia", "EL": "Grecia", "ES": "Spagna",
    "FI": "Finlandia", "FR": "Francia", "HR": "Croazia", "HU": "Ungheria", "IE": "Irlanda",
    "IS": "Islanda", "IT": "Italia", "LI": "Liechtenstein", "LT": "Lituania", "LU": "Lussemburgo",
    "LV": "Lettonia", "MT": "Malta", "NL": "Paesi Bassi", "NO": "Norvegia", "PL": "Polonia",
    "PT": "Portogallo", "RO": "Romania", "SE": "Svezia", "SI": "Slovenia", "SK": "Slovacchia",
    "TR": "Turchia", "RS": "Serbia", "CH": "Svizzera",
}

PAESI_UE = {
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU",
    "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK",
}

AGGREGATI = {
    "European Free Trade Association": "Associazione europea di libero scambio",
    "Euro area \u2013 20 countries (2023-2025)": "Area euro - 20 paesi (2023-2025)",
    "Euro area \u2013 21 countries (from 2026)": "Area euro - 21 paesi (dal 2026)",
    "European Economic Area (EU27 - 2007-2013 and IS, LI, NO)": "Spazio economico europeo (UE27 - 2007-2013 e IS, LI, NO)",
    "European Economic Area (EU28 - 2013-2020 and IS, LI, NO)": "Spazio economico europeo (UE28 - 2013-2020 e IS, LI, NO)",
    "European Union - 27 countries (2007-2013)": "Unione europea - 27 paesi (2007-2013)",
    "European Union - 27 countries (from 2020)": "Unione europea - 27 paesi (dal 2020)",
    "European Union - 28 countries (2013-2020)": "Unione europea - 28 paesi (2013-2020)",
}

GENERE = {"F": "Femmine", "M": "Maschi", "T": "Totale"}

PATOLOGIE = {
    "Certain infectious and parasitic diseases (A00-B99)": "Alcune malattie infettive e parassitarie (A00-B99)",
    "Neoplasms": "Tumori",
    "Mental and behavioural disorders (F00-F99)": "Disturbi mentali e comportamentali (F00-F99)",
    "Diseases of the respiratory system (J00-J99)": "Malattie del sistema respiratorio (J00-J99)",
    "Diseases of the digestive system (K00-K93)": "Malattie del sistema digerente (K00-K93)",
    "Diseases of the skin and subcutaneous tissue (L00-L99)": "Malattie della cute e del tessuto sottocutaneo (L00-L99)",
    "Diseases of the musculoskeletal system and connective tissue (M00-M99)": "Malattie del sistema muscoloscheletrico e del tessuto connettivo (M00-M99)",
    "Diseases of the genitourinary system (N00-N99)": "Malattie dell'apparato genitourinario (N00-N99)",
    "Pregnancy, childbirth and the puerperium (O00-O99)": "Gravidanza, parto e puerperio (O00-O99)",
    "Certain conditions originating in the perinatal period (P00-P96)": "Alcune condizioni morbose che hanno origine nel periodo perinatale (P00-P96)",
    "Congenital malformations, deformations and chromosomal abnormalities (Q00-Q99)": "Malformazioni e deformazioni congenite e anomalie cromosomiche (Q00-Q99)",
    "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified (R00-R99)": "Sintomi, segni e risultati anomali di esami clinici e di laboratorio, non classificati altrove (R00-R99)",
    "COVID-19, virus identified": "COVID-19, virus identificato",
    "External causes of morbidity and mortality (V01-Y89)": "Cause esterne di morbosità e mortalità (V01-Y89)",
}

PROBLEMI_SALUTE = {
    "Allergy": "Allergia",
    "Arthrosis": "Artrosi",
    "Asthma": "Asma",
    "Chronic depression": "Depressione cronica",
    "Chronic lower respiratory diseases (excluding asthma)": "Malattie croniche delle basse vie respiratorie (esclusa asma)",
    "Cirrhosis of the liver": "Cirrosi epatica",
    "Coronary heart disease or angina pectoris": "Cardiopatia coronarica o angina pectoris",
    "Diabetes": "Diabete",
    "Heart attack or chronic consequences of heart attack": "Infarto o sue conseguenze croniche",
    "High blood lipids": "Lipidi elevati nel sangue",
    "High blood pressure": "Ipertensione",
    "Kidney problems": "Problemi renali",
    "Low back disorder or other chronic back defect": "Disturbi lombari o altri difetti cronici della schiena",
    "Neck disorder or other chronic neck defect": "Disturbi cervicali o altri difetti cronici del collo",
    "Stroke or chronic consequences of stroke": "Ictus o sue conseguenze croniche",
    "Urinary incontinence, problems in controlling the bladder": "Incontinenza urinaria",
}

ETA_SILC = {
    "Y16-24": "16-24 anni", "Y25-34": "25-34 anni", "Y35-44": "35-44 anni", "Y45-54": "45-54 anni",
    "Y55-64": "55-64 anni", "Y65-74": "65-74 anni", "Y75-84": "75-84 anni", "Y_GE85": "85 anni e oltre",
}

QUINTILI = {
    "QU1": "1° quintile (reddito più basso)", "QU2": "2° quintile", "QU3": "3° quintile",
    "QU4": "4° quintile", "QU5": "5° quintile (reddito più alto)",
}

NUMERO_QUINTILE = {"QU1": 1, "QU2": 2, "QU3": 3, "QU4": 4, "QU5": 5}

ISTRUZIONE = {
    "ED0-2": "Bassa (ISCED 0-2)",
    "ED3_4": "Media (ISCED 3-4)",
    "ED5-8": "Alta (ISCED 5-8)",
}

COLONNE_ISTRUZIONE = {
    "Bassa (ISCED 0-2)": "salute_cattiva_istr_bassa",
    "Media (ISCED 3-4)": "salute_cattiva_istr_media",
    "Alta (ISCED 5-8)": "salute_cattiva_istr_alta",
}

SALUTE = {
    "VBAD": "Molto cattiva", "BAD": "Cattiva", "FAIR": "Discreta",
    "GOOD": "Buona", "VGOOD": "Molto buona",
}

FLAG_SILC = {
    "b": "interruzione nella serie storica",
    "bu": "interruzione nella serie storica, bassa affidabilità",
    "e": "stimato",
    "m": "valore mancante, il dato non può esistere",
    "n": "non significativo",
    "p": "provvisorio",
    "pu": "provvisorio, bassa affidabilità",
    "u": "bassa affidabilità",
}

FUNZIONI_SANITA = {
    "TOT_HC": "Spesa sanitaria corrente totale",
    "HC1": "Cure curative",
    "HC11": "Cure curative in regime di ricovero",
    "HC11_21": "Cure curative e riabilitative in regime di ricovero",
    "HC1_2": "Cure curative e riabilitative totali",
    "HC2": "Cure riabilitative",
    "HC21": "Cure riabilitative in regime di ricovero",
    "HC22": "Cure riabilitative in day-hospital",
    "HC23": "Cure riabilitative ambulatoriali",
    "HC24": "Cure riabilitative domiciliari",
    "HC3": "Assistenza a lungo termine (long-term care)",
    "HC31": "Assistenza a lungo termine in regime di ricovero",
    "HC32": "Assistenza a lungo termine semi-residenziale",
    "HC33": "Assistenza a lungo termine ambulatoriale",
    "HC34": "Assistenza a lungo termine domiciliare",
    "HC41": "Servizi di laboratorio",
    "HC42": "Servizi di diagnostica per immagini",
    "HC43": "Trasporto pazienti e soccorso",
    "HC5": "Beni medici non specificati per funzione",
    "HC51": "Farmaci e altri beni medici non durevoli",
    "HC52": "Apparecchi terapeutici e altri beni medici durevoli",
    "HC6": "Assistenza preventiva e prevenzione",
    "HC7": "Governance e amministrazione del sistema sanitario",
    "HCCOV1": "Costi di trattamento legati al Covid-19",
    "HCCOV2": "Costi Covid-19 per test e tracciamento",
    "HCCOV3": "Costi Covid-19 per vaccinazione",
    "HCCOV4": "Costi Covid-19 per beni medici",
    "HCCOV5": "Altri costi sanitari legati al Covid-19",
    "HCRI1": "Spesa farmaceutica totale (TPE)",
    "HCRI_EMS": "Cure di emergenza",
    "HCRI_MHC": "Assistenza per la salute mentale",
}

FUNZIONI_CHIAVE = ["HC1", "HC3", "HC51", "HC6", "HC7"]

FLAG_SPESA = {
    "p": "provvisorio",
    "b": "interruzione nella serie storica",
    "d": "differenza di definizione",
    "bd": "interruzione nella serie e differenza di definizione",
}

CHIAVI_COMUNI = ["codice_paese", "anno", "sesso", "classe_eta"]

ETICHETTA_PAESE = "Geopolitical entity (reporting)"


def leggi(nome_file, cartella=CARTELLA_DATI):
    df = pd.read_csv(Path(cartella) / nome_file)
    df.columns = df.columns.str.strip()
    for colonna in df.columns:
        if pd.api.types.is_string_dtype(df[colonna]):
            df[colonna] = df[colonna].str.strip()
    return df


def nome_paese(df):
    return df["geo"].map(PAESI).fillna(df[ETICHETTA_PAESE].replace(AGGREGATI))


def traduci_eta(etichetta):
    if pd.isna(etichetta):
        return etichetta
    testo = str(etichetta).strip()
    regole = [
        (r"^Less than (\d+) years?$", r"Meno di \1 anno"),
        (r"^From (\d+) to (\d+) years$", r"Da \1 a \2 anni"),
        (r"^(\d+) years or over$", r"\1 anni o più"),
        (r"^1 year$", "1 anno"),
        (r"^(\d+) years$", r"\1 anni"),
        (r"^Total$", "Totale"),
    ]
    for schema, sostituto in regole:
        if re.match(schema, testo):
            return re.sub(schema, sostituto, testo)
    return testo


def traduci(serie, dizionario, nome):
    tradotta = serie.map(dizionario)
    sconosciuti = serie[tradotta.isna() & serie.notna()].unique()
    if len(sconosciuti) > 0:
        raise ValueError(f"Codici non presenti nel dizionario per '{nome}': {list(sconosciuti)}")
    return tradotta


def controlla_percentuali(df, colonna, nome):
    fuori = ((df[colonna] < 0) | (df[colonna] > 100)).sum()
    if fuori > 0:
        raise ValueError(f"[{nome}] {fuori} percentuali fuori dall'intervallo 0-100")


def controlla_duplicati(df, chiavi, nome):
    duplicati = df.duplicated(subset=chiavi).sum()
    if duplicati > 0:
        raise ValueError(f"[{nome}] {duplicati} righe duplicate sulle chiavi {chiavi}")


def tabella_decessi(cartella=CARTELLA_DATI):
    df = leggi(FILE_T1, cartella)
    tabella = pd.DataFrame({
        "codice_paese": df["geo"],
        "paese": nome_paese(df),
        "anno": df["TIME_PERIOD"].astype(int),
        "codice_genere": df["sex"],
        "genere": df["sex"].map(GENERE),
        "fascia_eta": df["Age class"].map(traduci_eta),
        "codice_patologia": df["icd10"],
        "patologia": df["International Statistical Classification of Diseases and Related Health Problems (ICD-10)"].replace(PATOLOGIE),
        "numero_decessi": pd.to_numeric(df["OBS_VALUE"], errors="coerce"),
    })
    return tabella.reset_index(drop=True)


def tabella_speranza_vita(cartella=CARTELLA_DATI):
    df = leggi(FILE_T2, cartella)
    tabella = pd.DataFrame({
        "codice_paese": df["geo"],
        "paese": nome_paese(df),
        "anno": df["TIME_PERIOD"].astype(int),
        "codice_genere": df["sex"],
        "genere": df["sex"].map(GENERE),
        "classe_eta": df["Age class"].map(traduci_eta),
        "speranza_vita": pd.to_numeric(df["OBS_VALUE"], errors="coerce"),
    })
    return tabella.reset_index(drop=True)


def tabella_malattie_croniche(cartella=CARTELLA_DATI):
    df = leggi(FILE_T3, cartella)
    if "isced11" in df.columns:
        df = df[df["isced11"] == "TOTAL"]
    tabella = pd.DataFrame({
        "codice_paese": df["geo"],
        "paese": nome_paese(df),
        "anno": df["TIME_PERIOD"].astype(int),
        "codice_genere": df["sex"],
        "genere": df["sex"].map(GENERE),
        "fascia_eta": df["Age class"].map(traduci_eta),
        "codice_problema": df["hlth_pb"],
        "problema_salute": df["Health problems"].replace(PROBLEMI_SALUTE),
        "pct_persone": pd.to_numeric(df["OBS_VALUE"], errors="coerce"),
    })
    return tabella.reset_index(drop=True)


def tabella_anni_vita_sana(cartella=CARTELLA_DATI):
    df = leggi(FILE_T5_HLY, cartella)
    tabella = pd.DataFrame({
        "codice_paese": df["geo"],
        "paese": nome_paese(df),
        "anno": df["TIME_PERIOD"].astype(int),
        "codice_genere": df["sex"],
        "genere": df["sex"].map(GENERE),
        "indicatore_salute": df["Health and life expectancy"],
        "valore": pd.to_numeric(df["OBS_VALUE"], errors="coerce"),
    })
    return tabella.dropna(subset=["valore"]).reset_index(drop=True)


def tabella_proiezioni_demografiche(cartella=CARTELLA_DATI):
    df = leggi(FILE_T5_PROJ, cartella)
    tabella = pd.DataFrame({
        "codice_paese": df["geo"],
        "paese": nome_paese(df),
        "anno": df["TIME_PERIOD"].astype(int),
        "indicatore_demografico": df["Demographic indicator"],
        "valore": pd.to_numeric(df["OBS_VALUE"], errors="coerce"),
    })
    return tabella.dropna(subset=["valore"]).reset_index(drop=True)


def ultimo_anno(df):
    return df[df["TIME_PERIOD"] == df["TIME_PERIOD"].max()]


def estrai_indicatore(df, nome_valore):
    return df.rename(columns={
        "geo": "codice_paese",
        "sex": "codice_genere",
        "OBS_VALUE": nome_valore,
    })[["codice_paese", "codice_genere", nome_valore]].dropna()


def tabella_stile_di_vita(cartella=CARTELLA_DATI):
    attivita = leggi(FILE_T6_ATTIVITA, cartella)
    attivita = ultimo_anno(attivita[
        (attivita["isced11"] == "TOTAL")
        & (attivita["age"] == "TOTAL")
        & (attivita["physact"] == "MV_AERO")
    ])
    fumo = leggi(FILE_T6_FUMO, cartella)
    fumo = ultimo_anno(fumo[fumo["age"] == "Y_GE15"])
    alcol = leggi(FILE_T6_ALCOL, cartella)
    alcol = ultimo_anno(alcol[(alcol["age"] == "TOTAL") & (alcol["frequenc"] == "DAY")])

    chiavi = ["codice_paese", "codice_genere"]
    tabella = estrai_indicatore(attivita, "attivita_fisica_pct")
    tabella = tabella.merge(estrai_indicatore(fumo, "fumo_pct"), on=chiavi, how="outer")
    tabella = tabella.merge(estrai_indicatore(alcol, "alcol_quotidiano_pct"), on=chiavi, how="outer")
    tabella = tabella[tabella["codice_paese"].isin(PAESI_UE)].copy()

    colonne_valori = ["attivita_fisica_pct", "fumo_pct", "alcol_quotidiano_pct"]
    for colonna in colonne_valori:
        tabella[colonna] = pd.to_numeric(tabella[colonna], errors="coerce")

    tabella["paese"] = tabella["codice_paese"].map(PAESI)
    tabella["genere"] = tabella["codice_genere"].map(GENERE)
    tabella["punteggio_salute"] = (
        tabella["attivita_fisica_pct"] - tabella["fumo_pct"] - tabella["alcol_quotidiano_pct"]
    )
    tabella["ranking"] = (
        tabella.groupby("codice_genere")["punteggio_salute"]
        .rank(method="first", ascending=False)
        .astype("Int64")
    )
    tabella["punteggio_salute"] = tabella["punteggio_salute"].round(1)
    tabella = tabella.sort_values(["codice_genere", "ranking"], na_position="last")
    return tabella[[
        "ranking", "codice_paese", "paese", "codice_genere", "genere",
        "attivita_fisica_pct", "fumo_pct", "alcol_quotidiano_pct", "punteggio_salute",
    ]].reset_index(drop=True)


def tabella_reddito_salute(cartella=CARTELLA_DATI, escludi_bassa_affidabilita=False):
    df = leggi(FILE_T7_REDDITO, cartella)
    tabella = pd.DataFrame({
        "codice_paese": df["geo"],
        "paese": traduci(df["geo"], PAESI, "geo"),
        "anno": df["TIME_PERIOD"].astype(int),
        "sesso": traduci(df["sex"], GENERE, "sex"),
        "classe_eta": traduci(df["age"], ETA_SILC, "age"),
        "numero_quintile": traduci(df["quant_inc"], NUMERO_QUINTILE, "quant_inc"),
        "quintile": traduci(df["quant_inc"], QUINTILI, "quant_inc"),
        "pct_malattia_lunga_durata": pd.to_numeric(df["OBS_VALUE"], errors="coerce"),
        "flag": df["OBS_FLAG"],
    })
    tabella = tabella.dropna(subset=["pct_malattia_lunga_durata"]).copy()
    tabella["descrizione_flag"] = tabella["flag"].map(FLAG_SILC)
    tabella["bassa_affidabilita"] = tabella["flag"].fillna("").str.contains("u")
    if escludi_bassa_affidabilita:
        tabella = tabella[~tabella["bassa_affidabilita"]].copy()
    tabella["numero_quintile"] = tabella["numero_quintile"].astype(int)
    tabella["classe_eta"] = pd.Categorical(tabella["classe_eta"], categories=list(ETA_SILC.values()), ordered=True)
    tabella["quintile"] = pd.Categorical(tabella["quintile"], categories=list(QUINTILI.values()), ordered=True)
    controlla_percentuali(tabella, "pct_malattia_lunga_durata", "reddito")
    controlla_duplicati(tabella, CHIAVI_COMUNI + ["numero_quintile"], "reddito")
    return tabella.reset_index(drop=True)


def tabella_istruzione_salute(cartella=CARTELLA_DATI, escludi_bassa_affidabilita=False):
    df = leggi(FILE_T7_ISTRUZIONE, cartella)
    tabella = pd.DataFrame({
        "codice_paese": df["geo"],
        "paese": traduci(df["geo"], PAESI, "geo"),
        "anno": df["TIME_PERIOD"].astype(int),
        "sesso": traduci(df["sex"], GENERE, "sex"),
        "classe_eta": traduci(df["age"], ETA_SILC, "age"),
        "livello_istruzione": traduci(df["isced11"], ISTRUZIONE, "isced11"),
        "livello_salute": traduci(df["levels"], SALUTE, "levels"),
        "pct_salute_percepita": pd.to_numeric(df["OBS_VALUE"], errors="coerce"),
        "flag": df["OBS_FLAG"],
    })
    tabella = tabella.dropna(subset=["pct_salute_percepita"]).copy()
    tabella["descrizione_flag"] = tabella["flag"].map(FLAG_SILC)
    tabella["bassa_affidabilita"] = tabella["flag"].fillna("").str.contains("u")
    if escludi_bassa_affidabilita:
        tabella = tabella[~tabella["bassa_affidabilita"]].copy()
    tabella["classe_eta"] = pd.Categorical(tabella["classe_eta"], categories=list(ETA_SILC.values()), ordered=True)
    tabella["livello_istruzione"] = pd.Categorical(tabella["livello_istruzione"], categories=list(ISTRUZIONE.values()), ordered=True)
    tabella["livello_salute"] = pd.Categorical(tabella["livello_salute"], categories=list(SALUTE.values()), ordered=True)
    controlla_percentuali(tabella, "pct_salute_percepita", "istruzione")
    controlla_duplicati(tabella, CHIAVI_COMUNI + ["livello_istruzione", "livello_salute"], "istruzione")
    return tabella.reset_index(drop=True)


def tabella_reddito_largo(reddito):
    largo = reddito.pivot_table(
        index=CHIAVI_COMUNI, columns="numero_quintile",
        values="pct_malattia_lunga_durata", aggfunc="first", observed=True,
    )
    largo.columns = [f"malattia_q{q}" for q in largo.columns]
    largo["divario_malattia_q1_q5"] = (largo["malattia_q1"] - largo["malattia_q5"]).round(1)
    return largo.reset_index()


def salute_cattiva_per_gruppo(istruzione):
    cattiva = (
        istruzione[istruzione["livello_salute"].isin(["Molto cattiva", "Cattiva"])]
        .groupby(CHIAVI_COMUNI + ["livello_istruzione"], observed=True)["pct_salute_percepita"]
        .agg(["sum", "count"])
        .reset_index()
    )
    cattiva = cattiva[cattiva["count"] == 2].drop(columns="count")
    return cattiva.rename(columns={"sum": "pct_salute_cattiva"})


def tabella_istruzione_larga(istruzione):
    cattiva = salute_cattiva_per_gruppo(istruzione)
    largo = cattiva.pivot_table(
        index=CHIAVI_COMUNI, columns="livello_istruzione",
        values="pct_salute_cattiva", aggfunc="first", observed=True,
    )
    largo.columns = [COLONNE_ISTRUZIONE[c] for c in largo.columns]
    largo = largo.round(1)
    largo["divario_salute_cattiva_bassa_alta"] = (
        largo["salute_cattiva_istr_bassa"] - largo["salute_cattiva_istr_alta"]
    ).round(1)
    return largo.reset_index()


def tabella_reddito_istruzione_integrata(reddito_largo, istruzione_larga):
    integrato = reddito_largo.merge(istruzione_larga, on=CHIAVI_COMUNI, how="inner")
    integrato.insert(1, "paese", integrato["codice_paese"].map(PAESI))
    return integrato


def tabella_spesa_sanitaria(cartella=CARTELLA_DATI):
    df = leggi(FILE_T8_SPESA, cartella)
    tabella = pd.DataFrame({
        "codice_paese": df["geo"],
        "paese": df["geo"].map(PAESI),
        "anno": df["TIME_PERIOD"].astype(int),
        "codice_funzione": df["icha11_hc"],
        "funzione_sanitaria": df["icha11_hc"].map(FUNZIONI_SANITA),
        "spesa_milioni_euro": pd.to_numeric(df["OBS_VALUE"], errors="coerce"),
        "flag": df["OBS_FLAG"],
    })
    tabella["descrizione_flag"] = tabella["flag"].map(FLAG_SPESA).fillna("dato standard")
    tabella = tabella.dropna(subset=["spesa_milioni_euro"])
    if (tabella["spesa_milioni_euro"] < 0).any():
        raise ValueError("[spesa] presenti valori di spesa negativi")
    controlla_duplicati(tabella, ["codice_paese", "anno", "codice_funzione"], "spesa")
    return tabella.reset_index(drop=True)


def anno_riferimento(spesa, anno=2022):
    anni = spesa.loc[spesa["codice_funzione"] == "TOT_HC", "anno"].unique()
    return anno if anno in anni else int(anni.max())


def tabella_classifica_spesa(spesa, anno=2022):
    anno_rif = anno_riferimento(spesa, anno)
    classifica = spesa[(spesa["codice_funzione"] == "TOT_HC") & (spesa["anno"] == anno_rif)]
    classifica = classifica[["codice_paese", "paese", "anno", "spesa_milioni_euro"]]
    classifica = classifica.sort_values("spesa_milioni_euro", ascending=False).reset_index(drop=True)
    classifica.insert(0, "posizione", np.arange(1, len(classifica) + 1))
    return classifica


def tabella_spesa_per_funzione(spesa, anno=2022):
    anno_rif = anno_riferimento(spesa, anno)
    funzioni = spesa[(spesa["anno"] == anno_rif) & (spesa["codice_funzione"].isin(FUNZIONI_CHIAVE))]
    return funzioni[[
        "codice_paese", "paese", "anno", "codice_funzione", "funzione_sanitaria", "spesa_milioni_euro",
    ]].reset_index(drop=True)


def tabella_spesa_salute(classifica, istruzione):
    anno_rif = int(classifica["anno"].iloc[0])
    cattiva = salute_cattiva_per_gruppo(istruzione[istruzione["anno"] == anno_rif])
    salute = (
        cattiva.groupby("codice_paese", as_index=False)["pct_salute_cattiva"]
        .mean()
        .rename(columns={"pct_salute_cattiva": "pct_salute_cattiva_media"})
    )
    salute["pct_salute_cattiva_media"] = salute["pct_salute_cattiva_media"].round(1)
    return classifica.merge(salute, on="codice_paese", how="left")

def separa_codice_nome(serie):
    testo = serie.astype(str).str.strip()
    parti = testo.str.split(":", n=1, expand=True)
    if parti.shape[1] == 1:
        return testo, testo
    codice = parti[0].str.strip()
    nome = parti[1].str.strip().fillna(codice)
    return codice, nome


def regioni_speranza_vita(cartella=CARTELLA_DATI):
    df = leggi(FILE_T4_SPERANZA, cartella)
    df = df[df["age: Age class"].astype(str).str.contains("Less than 1 year|Y_LT1", na=False)]
    codice, nome = separa_codice_nome(df["geo: Geopolitical entity (reporting)"])
    return pd.DataFrame({
        "codice_regione": codice,
        "nome_regione": nome,
        "anno": pd.to_numeric(df["TIME_PERIOD: Time"], errors="coerce"),
        "genere": df["sex: Sex"],
        "indicatore": "Speranza di vita alla nascita",
        "valore": pd.to_numeric(df["OBS_VALUE: Observation value"], errors="coerce"),
        "fonte": "demo_r_mlifexp",
    })


def regioni_mortalita(cartella=CARTELLA_DATI):
    df = leggi(FILE_T4_MORTALITA, cartella)
    causa = "International Statistical Classification of Diseases and Related Health Problems (ICD-10)"
    df = df[
        (df["Age class"].astype(str) == "Total")
        & (df[causa].astype(str).str.contains("All causes", na=False))
    ]
    codice, nome = separa_codice_nome(df[ETICHETTA_PAESE])
    return pd.DataFrame({
        "codice_regione": codice,
        "nome_regione": nome,
        "anno": pd.to_numeric(df["TIME_PERIOD"], errors="coerce"),
        "genere": df["Sex"],
        "indicatore": "Tasso mortalità standardizzato",
        "valore": pd.to_numeric(df["OBS_VALUE"], errors="coerce"),
        "fonte": "hlth_cd_asdr2",
    })


def tabella_disparita_regionali(cartella=CARTELLA_DATI):
    tabella = pd.concat([regioni_speranza_vita(cartella), regioni_mortalita(cartella)], ignore_index=True)
    tabella = tabella[tabella["codice_regione"].str.len() == 4]
    tabella = tabella.dropna(subset=["valore", "anno"]).copy()
    tabella["codice_paese"] = tabella["codice_regione"].str[:2]
    tabella = tabella[tabella["codice_paese"].isin(PAESI_UE)].copy()
    tabella["paese"] = tabella["codice_paese"].map(PAESI)
    tabella["macroregione"] = tabella["codice_paese"].map(MACROREGIONI)
    tabella["genere"] = tabella["genere"].replace(GENERE_REGIONI)
    tabella["anno"] = tabella["anno"].astype(int)
    return tabella[[
        "codice_paese", "paese", "macroregione", "codice_regione", "nome_regione",
        "anno", "genere", "indicatore", "valore", "fonte",
    ]].reset_index(drop=True)

def esegui_pipeline(cartella=CARTELLA_DATI, anno_spesa=2022, escludi_bassa_affidabilita=False):
    reddito = tabella_reddito_salute(cartella, escludi_bassa_affidabilita)
    istruzione = tabella_istruzione_salute(cartella, escludi_bassa_affidabilita)
    reddito_largo = tabella_reddito_largo(reddito)
    istruzione_larga = tabella_istruzione_larga(istruzione)
    spesa = tabella_spesa_sanitaria(cartella)
    classifica = tabella_classifica_spesa(spesa, anno_spesa)

    tabelle = {
        "decessi": tabella_decessi(cartella),
        "speranza_vita": tabella_speranza_vita(cartella),
        "malattie_croniche": tabella_malattie_croniche(cartella),
        "anni_vita_sana": tabella_anni_vita_sana(cartella),
        "proiezioni_demografiche": tabella_proiezioni_demografiche(cartella),
        "stile_di_vita": tabella_stile_di_vita(cartella),
        "reddito_salute": reddito,
        "istruzione_salute": istruzione,
        "reddito_largo": reddito_largo,
        "istruzione_larga": istruzione_larga,
        "reddito_istruzione_integrato": tabella_reddito_istruzione_integrata(reddito_largo, istruzione_larga),
        "spesa_sanitaria": spesa,
        "classifica_spesa": classifica,
        "spesa_per_funzione": tabella_spesa_per_funzione(spesa, anno_spesa),
        "spesa_salute": tabella_spesa_salute(classifica, istruzione),
        "disparita_regionali": tabella_disparita_regionali(cartella),
    }
    tabelle["dim_paese"] = tabella_dim_paese(tabelle)
    return tabelle

def tabella_dim_paese(tabelle):
    sorgenti = ["decessi", "speranza_vita", "spesa_sanitaria", "reddito_salute", "malattie_croniche", "anni_vita_sana"]
    dim = pd.concat([tabelle[n][["codice_paese", "paese"]] for n in sorgenti])
    dim = dim.dropna(subset=["codice_paese"]).drop_duplicates("codice_paese")
    dim["ue"] = dim["codice_paese"].isin(PAESI_UE)
    return dim.sort_values("paese").reset_index(drop=True)


def prepara_per_sql(df):
    pronto = df.copy()
    for colonna in pronto.select_dtypes(include="category").columns:
        pronto[colonna] = pronto[colonna].astype(object)
    return pronto


def esporta_csv(tabelle, cartella_output=CARTELLA_OUTPUT):
    cartella_output = Path(cartella_output)
    cartella_output.mkdir(parents=True, exist_ok=True)
    for nome, df in tabelle.items():
        prepara_per_sql(df).to_csv(cartella_output / f"{nome}.csv", index=False, encoding="utf-8")


def carica_mysql(tabelle, url_connessione, if_exists="replace"):
    motore = create_engine(url_connessione)
    with motore.begin() as connessione:
        for nome, df in tabelle.items():
            prepara_per_sql(df).to_sql(nome, connessione, if_exists=if_exists, index=False, chunksize=5000)
    motore.dispose()


if __name__ == "__main__":
    tabelle = esegui_pipeline()
    esporta_csv(tabelle)
    url_mysql = os.getenv("MYSQL_URL")
    if url_mysql:
        carica_mysql(tabelle, url_mysql)
