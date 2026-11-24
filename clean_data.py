"""
Clean the CDC NVSS Natality public-use files (2024, 2025) for the
preterm-birth project.

Input : Nat<year>PublicUS*.txt  (fixed-width, ~4.9 GB each, NOT in git)
Output: clean/natality_<year>.parquet

Run:    python clean_data.py
Load:   pd.read_parquet("clean/natality_2024.parquet")
"""

import numpy as np
import pandas as pd
from pathlib import Path

# "parquet" (~48 MB, keeps dtypes, small enough for GitHub),
# "csv.gz" (~72 MB) or "csv" (~430 MB, too big for GitHub)
SAVE_AS = "parquet"

YEARS = [2024, 2025]


# 1.  WHICH BYTES TO READ        name: (start position, how many characters)

# From the NCHS "User Guide to the 2024/2025 Natality Public Use File".
# 2024 and 2025 use the same positions. The guide says records are 1330
# characters; the actual files are 1345 with Windows line endings. The extra
# characters are at the end, past everything we read, so it does not matter.
# Positions were checked against the real 2024 file (every field gives
# sensible values). DO NOT change these numbers -- being off by one silently
# ruins a column.

COLUMNS = {
    # --- year and location
    "DOB_YY":      (9, 4),      # birth year
    "RESTATUS":    (104, 1),    # residence status (4 = foreign resident)
    # --- mother's demographics
    "MAGER":       (75, 2),     # age in years
    "MRACEHISP":   (117, 1),    # race / Hispanic origin
    "DMAR":        (120, 1),    # marital status (blank = not reported)
    "MEDUC":       (124, 1),    # education
    # --- past pregnancies
    "PRIORLIVE":   (171, 2),    # prior live births
    "PRIORDEAD":   (173, 2),    # prior births now dead
    "PRIORTERM":   (175, 2),    # prior terminations
    "ILLB_R":      (198, 3),    # months since last live birth
    # --- prenatal care
    "PRECARE":     (224, 2),    # month care began
    "PREVIS":      (238, 2),    # total number of visits
    # --- smoking
    "CIG_0":       (253, 2),    # cigarettes/day before pregnancy
    "CIG_1":       (255, 2),    # ... 1st trimester
    "CIG_2":       (257, 2),    # ... 2nd trimester
    "CIG_3":       (259, 2),    # ... 3rd trimester
    # --- body size
    "M_Ht_In":     (280, 2),    # height in inches
    "BMI":         (283, 4),    # pre-pregnancy BMI
    "PWgt_R":      (292, 3),    # pre-pregnancy weight (lb)
    "DWgt_R":      (299, 3),    # weight at delivery (lb)
    "WTGAIN":      (304, 2),    # weight gained (lb)
    # --- risk factors (all coded Y / N / U)
    "RF_PDIAB":    (313, 1),    # pre-pregnancy diabetes
    "RF_GDIAB":    (314, 1),    # gestational diabetes
    "RF_PHYPE":    (315, 1),    # pre-pregnancy hypertension
    "RF_GHYPE":    (316, 1),    # gestational hypertension
    "RF_EHYPE":    (317, 1),    # hypertension w/ eclampsia
    "RF_PPTERM":   (318, 1),    # previous preterm birth
    "RF_INFTR":    (325, 1),    # infertility treatment
    "RF_CESAR":    (331, 1),    # previous cesarean
    # --- the birth
    "DPLURAL":     (454, 1),    # 1=single 2=twin 3=triplet 4+
    "SEX":         (475, 1),    # M / F
    "COMBGEST":    (490, 2),    # gestation, LMP-based  (NOT what we use)
    "OEGEST_COMB": (499, 2),    # gestation, obstetric estimate  <-- we use this
    "DBWT":        (504, 4),    # birth weight in grams
}

# 2.  VALID RANGES -- anything outside becomes missing

# Every "unknown" code sits outside the real range (99, 999, 99.9, 9999),
# so one rule handles them all. This is the most important step in the file:
# left alone, a mother with unknown BMI enters your model as BMI = 99.9.

VALID = {
    "MAGER": (12, 50),      "PRIORLIVE": (0, 98),   "PRIORDEAD": (0, 98),
    "PRIORTERM": (0, 98),   "PRECARE": (0, 10),     "PREVIS": (0, 98),
    "CIG_0": (0, 98),       "CIG_1": (0, 98),       "CIG_2": (0, 98),
    "CIG_3": (0, 98),       "M_Ht_In": (30, 78),    "BMI": (13.0, 69.9),
    "PWgt_R": (75, 375),    "DWgt_R": (100, 400),   "WTGAIN": (0, 98),
    "COMBGEST": (17, 47),   "OEGEST_COMB": (17, 47),"DBWT": (227, 8165),
}

# these use a specific code for "unknown" instead of an out-of-range number
UNKNOWN_CODE = {"MRACEHISP": "8", "DMAR": "9", "MEDUC": "9"}


def number(text, field):
    "Text -> number, with out-of-range 'unknown' codes set to missing."
    value = pd.to_numeric(text.str.strip(), errors="coerce")
    low, high = VALID[field]
    return value.where(value.between(low, high))


def category(text, field):
    "Coded category -> number, with the documented unknown code removed."
    stripped = text.str.strip()
    stripped = stripped.mask((stripped == UNKNOWN_CODE[field]) | (stripped == ""))
    return pd.to_numeric(stripped, errors="coerce")


def yes_no(text):
    "Y -> 1, N -> 0, U or blank -> missing."
    return text.str.strip().str.upper().map({"Y": 1.0, "N": 0.0})


# 3.  BUILD THE CLEAN COLUMNS

def clean(raw):
    df = pd.DataFrame(index=raw.index)

    # the outcome
    # OEGEST_COMB is the obstetric estimate, the CDC standard since 2014.
    # COMBGEST is the older LMP-based measure and runs ~1.7 points higher,
    # so which one you pick changes your headline number. We keep both so
    # you can show the difference, but model on gest_weeks.
    df["gest_weeks"] = number(raw["OEGEST_COMB"], "OEGEST_COMB")
    df["gest_weeks_lmp"] = number(raw["COMBGEST"], "COMBGEST")
    df["preterm"] = (df["gest_weeks"] < 37).astype(float)
    df["preterm_type"] = pd.cut(
        df["gest_weeks"],
        bins=[0, 28, 32, 34, 37, 99], right=False,
        labels=["extremely", "very", "moderate", "late", "term"],
    )

    # used for filtering
    df["year"] = pd.to_numeric(raw["DOB_YY"].str.strip(), errors="coerce")
    df["plurality"] = pd.to_numeric(raw["DPLURAL"].str.strip(), errors="coerce")
    df["residence_status"] = pd.to_numeric(raw["RESTATUS"].str.strip(), errors="coerce")

    # mother's demographics
    df["mother_age"] = number(raw["MAGER"], "MAGER")
    df["age_group"] = pd.cut(df["mother_age"], bins=[0, 20, 25, 30, 35, 40, 99],
                             right=False,
                             labels=["<20", "20-24", "25-29", "30-34", "35-39", "40+"])
    df["race_hispanic"] = category(raw["MRACEHISP"], "MRACEHISP")
    df["education"] = category(raw["MEDUC"], "MEDUC")
    # ~11% blank in 2024 (not reported) -> missing
    df["married"] = category(raw["DMAR"], "DMAR").map({1: 1.0, 2: 0.0})

    # body size
    df["bmi"] = number(raw["BMI"], "BMI")
    df["bmi_group"] = pd.cut(df["bmi"], bins=[0, 18.5, 25, 30, 35, 40, 99],
                             right=False,
                             labels=["underweight", "normal", "overweight",
                                     "obese_1", "obese_2", "obese_3"])
    df["height_in"] = number(raw["M_Ht_In"], "M_Ht_In")
    df["weight_prepreg"] = number(raw["PWgt_R"], "PWgt_R")

    # smoking
    df["cigs_before"] = number(raw["CIG_0"], "CIG_0")
    df["cigs_tri1"] = number(raw["CIG_1"], "CIG_1")
    df["cigs_tri2"] = number(raw["CIG_2"], "CIG_2")
    df["cigs_tri3"] = number(raw["CIG_3"], "CIG_3")
    df["smoked_before"] = (df["cigs_before"] > 0).astype(float).where(
        df["cigs_before"].notna())
    trimesters = df[["cigs_tri1", "cigs_tri2", "cigs_tri3"]]
    df["smoked_during"] = (trimesters.max(axis=1) > 0).astype(float).where(
        trimesters.notna().any(axis=1))

    # past pregnancies
    df["prior_births"] = number(raw["PRIORLIVE"], "PRIORLIVE")
    df["prior_deaths"] = number(raw["PRIORDEAD"], "PRIORDEAD")
    df["prior_terminations"] = number(raw["PRIORTERM"], "PRIORTERM")
    df["first_baby"] = (df["prior_births"] == 0).astype(float).where(
        df["prior_births"].notna())
    # ILLB_R: 888 means "no earlier live birth" (real info, not missing),
    # 999 means unknown. So 888 rows get a missing interval but first_baby=1.
    # ~41% missing for this reason: do NOT dropna() on this column.
    months = pd.to_numeric(raw["ILLB_R"].str.strip(), errors="coerce")
    df["months_since_last_birth"] = months.where(months.between(4, 300))

    # risk factors
    df["diabetes_prepreg"] = yes_no(raw["RF_PDIAB"])
    df["hypertension_prepreg"] = yes_no(raw["RF_PHYPE"])
    df["prior_preterm"] = yes_no(raw["RF_PPTERM"])
    df["prior_cesarean"] = yes_no(raw["RF_CESAR"])
    df["infertility_treatment"] = yes_no(raw["RF_INFTR"])
    # these three happen DURING pregnancy, so they are not pre-pregnancy risks
    df["diabetes_gestational"] = yes_no(raw["RF_GDIAB"])
    df["hypertension_gestational"] = yes_no(raw["RF_GHYPE"])
    df["hypertension_eclampsia"] = yes_no(raw["RF_EHYPE"])

    # prenatal care
    # CAREFUL: the start month also leaks the outcome. A baby born at 32 weeks
    # cannot have a first visit in month 9, so late starts look "protective"
    # (2024: month 9 -> 1.1% preterm, month 10 -> 0%). For modeling prefer
    # prenatal_first_tri (care began in months 1-3), which every pregnancy in
    # the data (>= 17 weeks) had the chance to reach.
    df["prenatal_start_month"] = number(raw["PRECARE"], "PRECARE")
    df["no_prenatal_care"] = (df["prenatal_start_month"] == 0).astype(float).where(
        df["prenatal_start_month"].notna())
    df["prenatal_first_tri"] = df["prenatal_start_month"].between(1, 3).astype(float).where(
        df["prenatal_start_month"].notna())
    df["prenatal_trimester"] = pd.cut(
        df["prenatal_start_month"].replace(0, np.nan),
        bins=[1, 4, 7, 11], right=False, labels=["first", "second", "third"])

    # careful: these grow with pregnancy length
    # A pregnancy that ends at 30 weeks had 10 fewer weeks to collect visits
    # and gain weight, so these partly MEASURE the outcome. Keep them out of
    # your main models; use them to show the leakage effect (H6).
    df["prenatal_visits"] = number(raw["PREVIS"], "PREVIS")
    df["weight_gain"] = number(raw["WTGAIN"], "WTGAIN")
    df["weight_delivery"] = number(raw["DWgt_R"], "DWgt_R")

    # about the baby (results of the birth, never predictors)
    df["baby_sex"] = raw["SEX"].str.strip()
    df["birth_weight_g"] = number(raw["DBWT"], "DBWT")

    return df


# smaller dtypes: the file is 3.6 million rows, so this matters
# (all values fit in Int16: max is birth weight 8165)
INT_COLS = [
    "year", "plurality", "residence_status", "gest_weeks", "gest_weeks_lmp",
    "preterm", "mother_age", "race_hispanic", "education", "married",
    "height_in", "cigs_before", "cigs_tri1", "cigs_tri2", "cigs_tri3",
    "smoked_before", "smoked_during", "prior_births", "prior_deaths",
    "prior_terminations", "first_baby", "diabetes_prepreg",
    "hypertension_prepreg", "prior_preterm", "prior_cesarean",
    "infertility_treatment", "diabetes_gestational", "hypertension_gestational",
    "hypertension_eclampsia", "prenatal_start_month", "no_prenatal_care",
    "prenatal_first_tri", "prenatal_visits", "weight_gain",
    "weight_prepreg", "weight_delivery", "birth_weight_g",
    "months_since_last_birth",
]


def shrink(df):
    for col in INT_COLS:
        df[col] = df[col].astype("Int16")
    df["bmi"] = df["bmi"].astype("float32")
    df["baby_sex"] = df["baby_sex"].astype("category")
    return df


# 4.  READ ONE FILE

def clean_file(year, folder="."):
    "Clean one year and return a DataFrame. Finds Nat<year>*.txt itself."
    matches = sorted(Path(folder).glob(f"Nat{year}*.txt"))
    if not matches:
        here = Path(folder).resolve()
        found = [f.name for f in here.iterdir() if f.is_file()][:12]
        raise FileNotFoundError(
            f"No Nat{year}*.txt found in:\n  {here}\n"
            f"That folder contains: {found}\n"
            f"Download it from https://www.cdc.gov/nchs/data_access/vitalstatsonline.htm "
            f"or call clean_file({year}, folder=r\"C:\\path\\to\\your\\files\")"
        )
    path = matches[0]
    size_gb = path.stat().st_size / 1e9
    print(f"[{year}] reading {path.name}  ({size_gb:.1f} GB) ...", flush=True)

    names = list(COLUMNS)
    positions = [(s - 1, s - 1 + w) for s, w in COLUMNS.values()]

    pieces, total = [], 0
    reader = pd.read_fwf(path, colspecs=positions, names=names, dtype=str,
                         header=None, chunksize=400_000,
                         encoding="latin-1", keep_default_na=False)
    for chunk in reader:
        total += len(chunk)
        df = clean(chunk)

        # THE FILTER: US residents, singletons, known gestational age
        df = df[(df["residence_status"] != 4)
                & (df["plurality"] == 1)
                & df["gest_weeks"].notna()]

        pieces.append(shrink(df))
        print(f"       {total:>10,} records read", flush=True)

    data = pd.concat(pieces, ignore_index=True)
    del pieces

    rate = data["preterm"].mean()
    print(f"[{year}] records in file      {total:,}")
    print(f"[{year}] kept after filtering {len(data):,}  "
          f"(US residents, singletons, known gestation)")
    print(f"[{year}] preterm              {data['preterm'].sum():,}  ({rate:.2%})")
    if 0.085 <= rate <= 0.090:
        print(f"[{year}] CHECK PASSED -- matches the published ~8.8%")
    else:
        print(f"[{year}] CHECK FAILED -- expected about 8.8%, got {rate:.2%}")

    # save so you never have to run this again
    outdir = Path(folder) / "clean"
    outdir.mkdir(exist_ok=True)
    if SAVE_AS == "csv":
        out = outdir / f"natality_{year}.csv"
    elif SAVE_AS == "csv.gz":
        out = outdir / f"natality_{year}.csv.gz"
    else:
        out = outdir / f"natality_{year}.parquet"
    if out.suffix == ".parquet":
        data.to_parquet(out, index=False, compression="zstd")
    else:
        data.to_csv(out, index=False)
    print(f"[{year}] saved                {out}  "
          f"({out.stat().st_size/1e6:,.0f} MB)\n", flush=True)
    return data


# 5.  RUN BOTH YEARS

if __name__ == "__main__":
    data = {}
    for year in YEARS:
        try:
            data[year] = clean_file(year)
        except FileNotFoundError as e:
            print(f"[{year}] SKIPPED -- {e}\n")

    print("=" * 60)
    for year, df in data.items():
        print(f"DONE.  data_{year}: {df.shape}")
    print("Train your models on 2024, test them on 2025.")
    print("=" * 60)
