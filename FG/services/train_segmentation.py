from FG.core.utils.log_and_progress import tracker_log_and_progress
from FG.services.segmentation_db_service import fetch_segmentation_data
from FG.core.utils.log_and_progress import tracker_log_and_progress
import logging
from fastapi import HTTPException
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import joblib
import logging
from datetime import datetime
from FG.core.config import file_path, THRESHOLD_MB,MODEL_PATH

logger = logging.getLogger(__name__)


CATEGORY_MAP = {
    # Domestic
    "DS-I (D)": "Domestic",
    "DS-II (D)": "Domestic",
    "DS-III (D)": "Domestic",
    "HAR-GHAR-NAL": "Domestic",
    "Kutir Jyoti Rural": "Domestic",
    "Kutir Jyoti Urban": "Domestic",
    "LT-Domestic": "Domestic",

    # Industrial
    "HT-INDUSTRIAL": "Industrial",
    "LT-INDUSTRIAL": "Industrial",
    "HTS-I(11KV)": "Industrial",
    "HTS-II(33KV)": "Industrial",
    "HTS-III(132KV)": "Industrial",
    "HTS-V(400KV)": "Industrial",
    "HTSS(132KV/220KV)": "Industrial",
    "HT-Cold Storage": "Industrial",
    "IAS-I": "Industrial",
    "IAS-I (UM)": "Industrial",
    "IAS-II (D)": "Industrial",

    # Commercial / Non-Domestic
    "NDS-I (D)": "Commercial",
    "NDS-II (D) (Upto 0.5 KW)": "Commercial",
    "NDS-II (D) (Upto 70 KW)": "Commercial",
    "NON-DOMESTIC": "Commercial",
    "HT-GENERAL": "Commercial",
    "SS-I (D)": "Commercial",
    "SS-II": "Commercial",
    "PUBLIC WATER WORKS": "Commercial",
    "PWW (D)": "Commercial",
    "RTS": "Commercial",
    "STREET LIGHT SERVICES": "Commercial",

    # Agriculture
    "IRRIGATION & AGRICULTURE SERVICE": "Agriculture",

    # Electric Vehicles
    "LT-EV": "EV",
    "LT-EV CHARGING STATIONS": "EV",
    "HT-EV": "EV",

    # High Tension IS
    "HTIS-I(11KV)": "Industrial",
    "HTIS-II(33KV)": "Industrial",
    "HTIS-III(132KV)": "Industrial",
    "HTIS-IV(220KV)": "Industrial",
    "HTIS–V(400KV)": "Industrial",
    "HTS-I (OM 11KV)": "Industrial",
    "HTS-I (OM 33KV)": "Industrial",
    "HTSS-11KV/33KV": "Industrial",
}

async def load_data(section_id: str,task_id: str,start_date: str,end_date: str):
    try:
        logging.info("File not found. Fetching from Oracle.")
        df = await fetch_segmentation_data(section_id,task_id,start_date,end_date)
        if df is None:
            tracker_log_and_progress(task_id, "fetch_segmentation_data() returned None","failed")
            raise Exception("fetch_segmentation_data() returned None")

        logging.info(f"Return Fetched {len(df)} rows from Oracle")
        tracker_log_and_progress(task_id, f"Return Fetched {len(df)} rows from Oracle")
        file_size = df.memory_usage(deep=True).sum() / (1024 ** 2)

        # Compute memory usage only for Pandas DataFrame
        if isinstance(df, pd.DataFrame):
            file_size = df.memory_usage(deep=True).sum() / (1024 ** 2)
        else:
            # Spark DataFrame – estimate size or set as unknown
            file_size = None

        return df, file_size
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise HTTPException(status_code=500, detail="Failed to load data")

def run_segmentation_pipeline(df, file_size, task_id, model_name):
    """
    Run consumer segmentation pipeline using Pandas or PySpark 
    based on file size (similar to demand forecasting logic).
    
    Args:
        df (DataFrame): Consumer-level input data.
        file_size (float): Size of dataset in MB.
        task_id (str): Celery task ID for logging/tracking.
        model_name (str): Model name to store/reuse.
    """
    try:
        logger.info(f"[Segmentation] Task {task_id} started for model={model_name}")
        logger.info(f"[Segmentation] Input size={file_size:.2f} MB, rows={len(df)}")

        # Step 1: Add required features (seasonality, payment behavior, load band, etc.)
        df = add_segmentation_features(df)

        # Step 2: Decide whether to use Pandas or PySpark
        model_file = os.path.join(MODEL_PATH, f"{model_name}_segmentation.pkl")

        if os.path.exists(model_file):
            logger.info(f"[Segmentation] Existing model found at {model_file}, attempting incremental update...")
            model = joblib.load(model_file)

            #if model.get("engine") == "pandas":
            #    model = update_segmentation_pandas(df, model)
            #else:
            #    model = update_segmentation_pyspark(df, model)

        else:
            logger.info("[Segmentation] Training fresh segmentation model...")
            if file_size < THRESHOLD_MB:
                segmented_df, artifacts = train_segmentation_pandas(df)
                #artifacts["engine"] = "pandas"
            else:
                raise NotImplementedError("PySpark segmentation not implemented yet")

        # Step 3: Save artifacts only (not the full DF)
        #joblib.dump(artifacts, model_file)
        logger.info(f"[Segmentation] Model saved at {model_file}")

        # Step 4: Return segmented data (final labels are already in segmented_df)
        segmented_df["final_segment"] = segmented_df["SEG_HYBRID"]

        logger.info(f"[Segmentation] Task {task_id} completed successfully")
        return segmented_df

    except Exception as e:
        logger.error(f"[Segmentation] Task {task_id} failed: {str(e)}", exc_info=True)
        raise

def add_segmentation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds features needed for hybrid segmentation and maps category_name → broad_category.
    Expected (case-insensitive accepted):
      - category_name (string like "LT-Domestic")
      - BILL_UNITS, BILL_AMOUNT, AMOUNT_COLLECTED, LOAD_IN_KW
      - optional: LOAD_FACTOR, PEAK_DEMAND_KW, ARREAR_RATIO, SUBSIDY_SHARE,
                  COLLECTION_RATIO, PAYMENT_LAG_DAYS, GEO_TAG (Urban/Rural)
    """
    df = df.copy()

    # ---- normalize column names (tolerate case differences) ----
    col_map = {c.lower(): c for c in df.columns}
    def get(col):  # fetch by case-insensitive name if present
        return col_map.get(col.lower(), col)

    # Ensure numeric types where needed
    num_cols = [
        "BILL_UNITS", "BILL_AMOUNT", "AMOUNT_COLLECTED", "LOAD_IN_KW",
        "LOAD_FACTOR", "PEAK_DEMAND_KW", "ARREAR_TOTAL", "SUBSIDY",
        "SUBSIDY_SHARE", "ARREAR_RATIO", "COLLECTION_RATIO",
        "PAYMENT_LAG_DAYS", "ENERGY_CHG", "FIXED_CHG","BILL_UNITS", 
        "BILL_AMOUNT", "AMOUNT_COLLECTED", "ARREARS","BILL_MONTH","BILL_YEAR","SUBSIDY"
    ]
    for c in num_cols:
        if get(c) in df.columns:
            df[get(c)] = pd.to_numeric(df[get(c)], errors="coerce")

    # ---- Map category_name → broad_category using your map ----
    # accept "category_name" or "CATEGORY_NAME"
    cat_col = get("cat_code")
    if cat_col not in df.columns:
        raise ValueError("Expected 'category_name' column with values like 'LT-Domestic', 'HT-INDUSTRIAL', etc.")
    df["broad_category"] = df[cat_col].map(CATEGORY_MAP).fillna("Other")

    # ---- Revenue per kWh (guard /0) ----
    bu = get("BILL_UNITS"); ba = get("BILL_AMOUNT")
    if bu in df.columns and ba in df.columns:
        df["REV_PER_KWH"] = np.where(df[bu] > 0, df[ba] / df[bu], np.nan)
    else:
        df["REV_PER_KWH"] = np.nan

    # ---- Load band from LOAD_IN_KW ----
    def load_band(x):
        try:
            x = float(x)
        except:
            return "Unknown"
        if x < 2:      return "<2 kW"
        if x <= 10:    return "2–10 kW"
        return ">10 kW"

    lik = get("LOAD_IN_KW")
    df["SEG_LOAD_BAND"] = df[lik].apply(load_band) if lik in df.columns else "Unknown"

    # ---- Geography tag if provided (else Unknown) ----
    geo_col = get("GEO_TAG") if get("GEO_TAG") in df.columns else (get("geography") if get("geography") in df.columns else None)
    df["GEO_TAG"] = df[geo_col].fillna("Unknown") if geo_col else "Unknown"

    # ---- Rule label (Tariff/Category + Geo + Load band) ----
    df["SEG_RULE"] = df["broad_category"] + " | " + df["GEO_TAG"] + " | " + df["SEG_LOAD_BAND"]

    # Optional: light winsorization to stabilize features
    def winsorize(s, p=0.01):
        s = s.copy()
        lo, hi = s.quantile(p), s.quantile(1 - p)
        return s.clip(lower=lo, upper=hi)

    for c in ["BILL_UNITS", "LOAD_IN_KW", "LOAD_FACTOR", "PEAK_DEMAND_KW",
              "ARREAR_RATIO", "SUBSIDY_SHARE", "COLLECTION_RATIO", "PAYMENT_LAG_DAYS", "REV_PER_KWH"]:
        if get(c) in df.columns and df[get(c)].notna().any():
            df[get(c)] = winsorize(df[get(c)].astype(float), p=0.01)

    return df

def train_segmentation_pandas(df: pd.DataFrame, k_choices=(3, 4, 5, 6), random_state: int = 42):
    """
    Runs ML clustering on behavioral/financial features and produces:
      - CLUSTER_ID
      - CLUSTER_LABEL (human readable)
      - SEG_HYBRID = SEG_RULE + " | " + CLUSTER_LABEL

    Returns:
      segmented_df, artifacts_dict
    """
    print(" training Pandas-based segmentation...")
    df = df.copy()
    # Ensure features exist (call add_segmentation_features if not run)
    for need in ["broad_category", "SEG_RULE", "REV_PER_KWH"]:
        if need not in df.columns:
            df = add_segmentation_features(df)
            break

    # Case-insensitive fetch
    col_map = {c.lower(): c for c in df.columns}
    def get(col): return col_map.get(col.lower(), col)

    # Feature set (all optional-safe; missing ones are filled with medians)
    feature_cols = [
        get("BILL_UNITS"),
        get("LOAD_FACTOR"),
        get("PEAK_DEMAND_KW"),
        get("SUBSIDY_SHARE"),
        get("ARREAR_RATIO"),
        get("COLLECTION_RATIO"),
        get("PAYMENT_LAG_DAYS"),
        "REV_PER_KWH",
        get("LOAD_IN_KW"),
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    if not feature_cols:
        raise ValueError("No usable numeric feature columns found for clustering.")

    feat = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    feat = feat.fillna(feat.median(numeric_only=True))  # simple impute

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(feat)

    # Choose k by silhouette; fallback to k=4 if tiny/degenerate
    best_k, best_score, best_model = None, -1, None
    for k in k_choices:
        if len(df) >= k + 1 and np.isfinite(X).all():
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = km.fit_predict(X)
            if len(set(labels)) > 1:
                try:
                    score = silhouette_score(X, labels)
                except Exception:
                    score = -1
                if score > best_score:
                    best_k, best_score, best_model = k, score, km

    if best_model is None:
        best_model = KMeans(n_clusters=4, random_state=random_state, n_init=10).fit(X)

    df["CLUSTER_ID"] = best_model.predict(X)

    # Cluster profiling (means)
    profile_cols = [c for c in ["BILL_UNITS","LOAD_FACTOR","ARREAR_RATIO","COLLECTION_RATIO",
                                "PAYMENT_LAG_DAYS","SUBSIDY_SHARE","REV_PER_KWH"] if get(c) in df.columns or c=="REV_PER_KWH"]
    prof = df.groupby("CLUSTER_ID")[profile_cols].mean(numeric_only=True).reset_index()

    # Percentiles for labeling (global)
    q = {}
    for c in profile_cols:
        s = df[c].dropna().astype(float)
        q[c] = {'p25': (s.quantile(0.25) if len(s) else 0), 'p75': (s.quantile(0.75) if len(s) else 0)}

    def label_cluster(row):
        tags = []
        # Usage
        if "BILL_UNITS" in row and row["BILL_UNITS"] >= q["BILL_UNITS"]["p75"]: tags.append("High usage")
        elif "BILL_UNITS" in row and row["BILL_UNITS"] <= q["BILL_UNITS"]["p25"]: tags.append("Low usage")
        # Efficiency
        if "LOAD_FACTOR" in row and row["LOAD_FACTOR"] >= q["LOAD_FACTOR"]["p75"]: tags.append("High LF")
        elif "LOAD_FACTOR" in row and row["LOAD_FACTOR"] <= q["LOAD_FACTOR"]["p25"]: tags.append("Low LF")
        # Collections / Payments
        if "COLLECTION_RATIO" in row and row["COLLECTION_RATIO"] <= q["COLLECTION_RATIO"]["p25"]: tags.append("Low collection")
        if "PAYMENT_LAG_DAYS" in row and row["PAYMENT_LAG_DAYS"] >= q["PAYMENT_LAG_DAYS"]["p75"]: tags.append("Late payers")
        if "ARREAR_RATIO" in row and row["ARREAR_RATIO"] >= q["ARREAR_RATIO"]["p75"]: tags.append("High arrears")
        if "SUBSIDY_SHARE" in row and row["SUBSIDY_SHARE"] >= q["SUBSIDY_SHARE"]["p75"]: tags.append("Subsidy-dependent")
        if not tags:
            return "Balanced"
        return ", ".join(tags)

    prof["CLUSTER_LABEL"] = prof.apply(label_cluster, axis=1)
    df = df.merge(prof[["CLUSTER_ID", "CLUSTER_LABEL"]], on="CLUSTER_ID", how="left")

    # Hybrid label = rule + ML
    df["SEG_HYBRID"] = df["SEG_RULE"] + " | " + df["CLUSTER_LABEL"]

    # Artifacts (so you can persist if needed)
    artifacts = {
        "scaler": scaler,
        "kmeans": best_model,
        "k": int(getattr(best_model, "n_clusters", 4)),
        "silhouette": float(best_score) if best_score is not None else None,
        "cluster_profile": prof.to_dict(orient="records"),
        "feature_cols": feature_cols,
    }

    return df, artifacts


