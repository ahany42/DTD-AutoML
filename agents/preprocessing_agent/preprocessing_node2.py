"""
Preprocessing Node for LangGraph Orchestrator - LLM Policy Driven
=================================================================

This version uses a conservative, LLM-guided policy while protecting dataset quality.
The LLM (Hugging Face by default) is asked to decide strategy. Guardrails prevent destructive actions.
"""

import json
import os
import re
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

try:
    from huggingface_hub import InferenceClient
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False
    InferenceClient = None


PREPROCESSING_CONFIG = {
    "default_input_path": "Datasets/",
    "default_output_path": "output/",
    "test_size": 0.2,
    "random_state": 42,
    "use_llm": True,
    "llm_final_decision": True,
    "safe_mode": True,
    "max_label_categories": 30,
    "hash_features": 8,
    "max_row_drop_fraction": 0.02,
    "max_outlier_clip_quantile": 0.01,
    "target_metric_priority": "f1",
    "llm_provider": "gemini",
    "hf_model": "Qwen/Qwen2.5-14B-Instruct",
    "hf_api_key": "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",
    "hf_api_key_env": "HUGGINGFACE_API_KEY",
    "gemini_model": "gemini-2.5-flash",
    "gemini_api_key": "AIzaSyBhpkQJ6jy8NEl2LBxvKJTLBepFkmnukdM",
    "gemini_api_key_env": "GEMINI_API_KEY",
}


class PreprocessingState(TypedDict):
    dataset_path: str
    target_column: str
    output_folder: str
    test_size: float
    random_state: int
    use_llm: bool
    X_train_path: Optional[str]
    X_test_path: Optional[str]
    y_train_path: Optional[str]
    y_test_path: Optional[str]
    summary_path: Optional[str]
    column_actions_path: Optional[str]
    policy_path: Optional[str]
    evidence_path: Optional[str]
    status: str
    error: Optional[str]


class PreprocessingNode:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = PREPROCESSING_CONFIG.copy()
        if config:
            self.config.update(config)
        self.llm_provider = str(self.config.get(
            "llm_provider", "huggingface")).lower()
        # Try direct key first, then environment variable
        self.hf_api_key = self.config.get("hf_api_key") or os.getenv(
            self.config.get("hf_api_key_env", "HUGGINGFACE_API_KEY"))
        self.gemini_api_key = self.config.get("gemini_api_key") or os.getenv(
            self.config.get("gemini_api_key_env", "GEMINI_API_KEY"))

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            dataset_path = Path(state["dataset_path"])
            target_col = state["target_column"]
            output_folder = Path(
                state.get("output_folder", self.config["default_output_path"]))
            test_size = float(state.get("test_size", self.config["test_size"]))
            random_state = int(
                state.get("random_state", self.config["random_state"]))
            use_llm = bool(state.get("use_llm", self.config["use_llm"]))

            output_folder.mkdir(parents=True, exist_ok=True)
            df = pd.read_csv(dataset_path)
            self._validate_target_column(df, target_col)

            evidence = self._build_evidence(df, target_col)
            default_policy = self._default_policy(df, target_col, evidence)

            llm_policy = None
            if use_llm and self._has_llm_credentials():
                llm_policy = self._llm_decide_policy(evidence, target_col)

            policy = self._merge_and_validate_policy(
                default_policy=default_policy,
                llm_policy=llm_policy,
                evidence=evidence,
                total_rows=len(df),
            )

            X, y, metadata = self._preprocess_with_policy(
                df, target_col, policy)
            metadata["llm_policy_used"] = bool(llm_policy)

            # will be deleted but for now we need the full dataset for the next node, so we save it here.
            if len(X) != len(y):
                raise ValueError(f"Sync Error: X has {len(X)} rows, y has {len(y)} rows.")
            full_df = X.copy()
            full_df[target_col] = y.reset_index(drop=True)
            full_path = output_folder / "full_preprocessed.csv"
            full_df.to_csv(full_path, index=False)
            # X_train, X_test, y_train, y_test = train_test_split(
            #     X,
            #     y,
            #     test_size=test_size,
            #     random_state=random_state,
            #     stratify=y if y.nunique() <= 20 else None,
            # )
            # =========================
            # SAFE STRATIFIED SPLIT
            # =========================

            stratify_target = None

            if y.nunique() <= 20:
                class_counts = y.value_counts()

                if class_counts.min() < 2:
                    print("⚠️ Rare classes detected (<2 samples). Dropping them...")
                    valid_classes = class_counts[class_counts >= 2].index
                    mask = y.isin(valid_classes)

                    X = X[mask]
                    y = y[mask]

                    stratify_target = y
                else:
                    stratify_target = y

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_target,
            )

            X_train, y_train, imbalance_meta = self._apply_imbalance_method(
                X_train,
                y_train,
                method=str(policy["imbalance"]["method"]),
                random_state=random_state,
            )
            metadata["imbalance"] = imbalance_meta

            paths = self._save_outputs(
                output_folder=output_folder,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                metadata=metadata,
                policy=policy,
                evidence=evidence,
                dataset_name=dataset_path.name,
                target_col=target_col,
                test_size=test_size,
                random_state=random_state,
                use_llm=use_llm,
            )

            state.update(
                {
                    "X_train_path": str(paths["X_train"]),
                    "X_test_path": str(paths["X_test"]),
                    "y_train_path": str(paths["y_train"]),
                    "y_test_path": str(paths["y_test"]),
                    "full_dataset_path": str(full_path),
                    "summary_path": str(paths["summary"]),
                    "column_actions_path": str(paths["column_actions"]),
                    "policy_path": str(paths["policy"]),
                    "evidence_path": str(paths["evidence"]),
                    "status": "success",
                    "error": None,
                    "output_folder": str(output_folder),
                }
            )
            return state

        except Exception as e:
            state.update({"status": "failed", "error": str(e)})
            return state

    def _validate_target_column(self, df: pd.DataFrame, target_col: str) -> None:
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found. Available: {', '.join(df.columns)}"
            )

    def _has_llm_credentials(self) -> bool:
        if self.llm_provider == "huggingface":
            return bool(self.hf_api_key)
        if self.llm_provider == "gemini":
            return bool(self.gemini_api_key)
        return False

    def _call_huggingface(self, prompt: str) -> Optional[str]:
        if not self.hf_api_key:
            return None

        if not HAS_HF_HUB:
            return None

        try:
            model = str(self.config.get(
                "hf_model", "Qwen/Qwen2.5-14B-Instruct"))
            client = InferenceClient(api_key=self.hf_api_key)
            response = client.text_generation(
                model=model,
                prompt=prompt,
                max_new_tokens=900,
                temperature=0.0,
            )
            return str(response) if response else None
        except Exception as e:
            return None

    def _call_gemini(self, prompt: str) -> Optional[str]:
        if not self.gemini_api_key:
            return None
        model = str(self.config.get("gemini_model", "gemini-2.5-flash"))
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key=" + self.gemini_api_key
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0},
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError):
            return None

        try:
            parsed = json.loads(body)
            return (
                parsed.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text")
            )
        except (json.JSONDecodeError, IndexError, KeyError, TypeError):
            return None

    def _call_llm(self, prompt: str) -> Optional[str]:
        if self.llm_provider == "huggingface":
            return self._call_huggingface(prompt)
        if self.llm_provider == "gemini":
            return self._call_gemini(prompt)
        return None

    def _parse_llm_json(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start: end + 1])
        except json.JSONDecodeError:
            return None

    def _build_evidence(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        y = df[target_col]
        class_counts = y.value_counts(dropna=False).to_dict()
        min_count = min(class_counts.values()) if class_counts else 0
        max_count = max(class_counts.values()) if class_counts else 0
        imbalance_ratio = float(
            max_count / max(min_count, 1)) if class_counts else 1.0

        columns: Dict[str, Dict[str, Any]] = {}
        for col in df.columns:
            if col == target_col:
                continue
            s = df[col]
            non_null = s.dropna()
            if pd.api.types.is_numeric_dtype(s):
                numeric_ratio = 1.0
                numeric_conv = pd.to_numeric(s, errors="coerce")
            elif non_null.empty:
                numeric_ratio = 0.0
                numeric_conv = pd.to_numeric(s, errors="coerce")
            else:
                numeric_conv = pd.to_numeric(non_null, errors="coerce")
                numeric_ratio = float(numeric_conv.notna().mean())

            datetime_ratio = 0.0
            if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
                sample_text = " ".join(non_null.astype(
                    str).head(8).tolist()).lower()
                # Avoid expensive datetime parsing for clear non-date text columns.
                if re.search(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec", sample_text):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        dt_conv = pd.to_datetime(
                            non_null, errors="coerce") if not non_null.empty else pd.Series(dtype="datetime64[ns]")
                    datetime_ratio = float(
                        dt_conv.notna().mean()) if not non_null.empty else 0.0

            top_values = s.astype(str).value_counts(dropna=False).head(10)
            columns[col] = {
                "dtype": str(s.dtype),
                "missing_ratio": float(s.isna().mean()),
                "unique_count": int(s.nunique(dropna=True)),
                "unique_ratio": float(s.nunique(dropna=True) / max(len(s), 1)),
                "numeric_parse_ratio": numeric_ratio,
                "datetime_parse_ratio": datetime_ratio,
                "sample_values": s.dropna().astype(str).head(8).tolist(),
                "top_values": {str(k): int(v) for k, v in top_values.items()},
                "numeric_stats": {
                    "mean": float(numeric_conv.mean()) if numeric_conv.notna().any() else None,
                    "std": float(numeric_conv.std()) if numeric_conv.notna().any() else None,
                    "q1": float(numeric_conv.quantile(0.25)) if numeric_conv.notna().any() else None,
                    "q3": float(numeric_conv.quantile(0.75)) if numeric_conv.notna().any() else None,
                },
            }

        return {
            "rows": int(len(df)),
            "columns": int(len(df.columns) - 1),
            "duplicate_rows": int(df.duplicated().sum()),
            "target": {
                "name": target_col,
                "n_classes": int(y.nunique(dropna=False)),
                "class_counts": {str(k): int(v) for k, v in class_counts.items()},
                "imbalance_ratio": imbalance_ratio,
            },
            "metric_priority": self.config["target_metric_priority"],
            "columns_profile": columns,
        }

    def _default_policy(self, df: pd.DataFrame, target_col: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
        columns_policy: Dict[str, Dict[str, Any]] = {}
        for col in df.columns:
            if col == target_col:
                continue
            profile = evidence["columns_profile"][col]
            is_numeric_dtype = pd.api.types.is_numeric_dtype(df[col])
            dtype_guess = "numeric" if is_numeric_dtype or profile[
                "numeric_parse_ratio"] > 0.85 else "categorical"
            if profile["datetime_parse_ratio"] > 0.85 and not is_numeric_dtype:
                dtype_guess = "datetime"
            drop = profile["missing_ratio"] > 0.6 or profile["unique_ratio"] > 0.99
            encoding = "none" if dtype_guess == "numeric" else "label"
            if dtype_guess == "categorical" and profile["unique_count"] > self.config["max_label_categories"]:
                encoding = "hash"
            columns_policy[col] = {
                "drop": bool(drop),
                "dtype": dtype_guess,
                "missing": "median" if dtype_guess == "numeric" else "mode",
                "outlier": "keep",
                "encoding": encoding,
                "reason": "default_policy",
            }

        imbalance_method = "none"
        if evidence["target"]["imbalance_ratio"] >= 3.0:
            imbalance_method = "class_weight"

        return {
            "duplicates": {"action": "drop_exact", "reason": "safe_default"},
            "columns": columns_policy,
            "feature_selection": {"method": "variance", "threshold": 0.0},
            "feature_creation": {"method": "datetime_parts"},
            "dimensionality_reduction": {"method": "none", "n_components": "auto"},
            "scaling": {"method": "standard"},
            "normalization": {"method": "none"},
            "imbalance": {"method": imbalance_method, "reason": "default_by_ratio"},
        }

    def _llm_decide_policy(self, evidence: Dict[str, Any], target_col: str) -> Optional[Dict[str, Any]]:
        prompt = (
            "You are the final preprocessing policy authority. "
            "Return only JSON with keys: duplicates, columns, feature_selection, feature_creation, "
            "dimensionality_reduction, scaling, normalization, imbalance. "
            "Allowed methods only: duplicates.action=drop_exact|keep; "
            "columns.dtype=numeric|categorical|datetime|text; "
            "columns.missing=median|mean|mode|constant|indicator|keep; "
            "columns.outlier=keep|clip|log_transform; "
            "columns.encoding=none|label|hash|onehot|frequency; "
            "feature_selection.method=none|variance; "
            "feature_creation.method=none|datetime_parts; "
            "dimensionality_reduction.method=none|pca; "
            "scaling.method=none|standard|minmax|robust|quantile|power; "
            "normalization.method=none|l1|l2|max; "
            "imbalance.method=none|class_weight|oversample|undersample. "
            "Do not remove more than 20 percent of features. "
            f"Target column is '{target_col}'. "
            f"Evidence: {json.dumps(evidence)}"
        )
        raw = self._call_llm(prompt)
        if not raw:
            return None
        parsed = self._parse_llm_json(raw)
        if not parsed:
            return None
        return parsed

    def _merge_and_validate_policy(
        self,
        default_policy: Dict[str, Any],
        llm_policy: Optional[Dict[str, Any]],
        evidence: Dict[str, Any],
        total_rows: int,
    ) -> Dict[str, Any]:
        policy = default_policy
        if llm_policy and self.config["llm_final_decision"]:
            policy = {**default_policy, **llm_policy}
            if "columns" in llm_policy:
                merged_cols = default_policy["columns"].copy()
                if isinstance(llm_policy["columns"], dict):
                    merged_cols.update(llm_policy["columns"])
                policy["columns"] = merged_cols

        allowed_scaling = {"none", "standard",
                           "minmax", "robust", "quantile", "power"}
        allowed_norm = {"none", "l1", "l2", "max"}
        allowed_imb = {"none", "class_weight", "oversample", "undersample"}
        allowed_enc = {"none", "label", "hash", "onehot", "frequency"}
        allowed_dtype = {"numeric", "categorical", "datetime", "text"}
        allowed_missing = {"median", "mean",
                           "mode", "constant", "indicator", "keep"}
        allowed_outlier = {"keep", "clip", "log_transform"}

        if policy.get("scaling", {}).get("method") not in allowed_scaling:
            policy["scaling"] = {"method": "standard"}
        if policy.get("normalization", {}).get("method") not in allowed_norm:
            policy["normalization"] = {"method": "none"}
        if policy.get("imbalance", {}).get("method") not in allowed_imb:
            policy["imbalance"] = {"method": "none",
                                   "reason": "invalid_replaced"}

        safe_notes: List[str] = []
        col_policy = policy.get("columns", {})
        dropped_count = 0
        for col, decisions in col_policy.items():
            if decisions.get("dtype") not in allowed_dtype:
                decisions["dtype"] = default_policy["columns"][col]["dtype"]
            if decisions.get("encoding") not in allowed_enc:
                decisions["encoding"] = default_policy["columns"][col]["encoding"]
            if decisions.get("missing") not in allowed_missing:
                decisions["missing"] = default_policy["columns"][col]["missing"]
            if decisions.get("outlier") not in allowed_outlier:
                decisions["outlier"] = "keep"
            if bool(decisions.get("drop")):
                dropped_count += 1

            if self.config["safe_mode"] and decisions.get("outlier") not in {"keep", "clip", "log_transform"}:
                decisions["outlier"] = "keep"
                safe_notes.append(f"{col}: unsafe outlier action replaced")

        total_features = max(len(col_policy), 1)
        if dropped_count / total_features > 0.35:
            safe_notes.append(
                "Too many columns dropped by policy; capping drops to protect dataset")
            kept = 0
            for col in sorted(col_policy.keys()):
                if col_policy[col].get("drop"):
                    if kept / total_features < 0.35:
                        kept += 1
                    else:
                        col_policy[col]["drop"] = False

        if policy.get("duplicates", {}).get("action") not in {"drop_exact", "keep"}:
            policy["duplicates"] = {
                "action": "drop_exact", "reason": "invalid_replaced"}

        policy["safeguards"] = {
            "safe_mode": bool(self.config["safe_mode"]),
            "notes": safe_notes,
            "max_row_drop_fraction": self.config["max_row_drop_fraction"],
            "rows": total_rows,
            "target_imbalance_ratio": evidence["target"]["imbalance_ratio"],
        }
        return policy

    def _preprocess_with_policy(
        self,
        df: pd.DataFrame,
        target_col: str,
        policy: Dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        steps_status = {
            "missing_values": "handled",
            "outliers": "handled",
            "duplicates": "handled",
            "type_conversion": "handled",
            "categorical_encoding": "handled",
            "feature_selection": "handled",
            "feature_creation": "handled",
            "dimensionality_reduction": "handled",
            "normalization": "handled",
            "scaling": "handled",
            "imbalance": "handled_after_split",
        }

        df_work = df.copy()
        duplicate_rows_removed = 0
        if policy["duplicates"]["action"] == "drop_exact":
            duplicate_rows_removed = int(df_work.duplicated().sum())
            df_work = df_work.drop_duplicates().copy()

        y = df_work[target_col].copy()
        feature_frames: List[pd.DataFrame] = []
        numeric_cols: List[str] = []
        categorical_cols: List[str] = []
        dropped_cols: List[str] = []
        column_actions: Dict[str, Any] = {}

        for col in [c for c in df_work.columns if c != target_col]:
            s = df_work[col]
            p = policy["columns"].get(col, {})
            if bool(p.get("drop", False)):
                dropped_cols.append(col)
                column_actions[col] = {"action": "drop",
                                       "reason": p.get("reason", "policy_drop")}
                continue

            dtype_choice = p.get("dtype", "categorical")
            missing_method = p.get("missing", "mode")
            outlier_method = p.get("outlier", "keep")
            encoding_method = p.get("encoding", "none")

            # Type conversion
            if dtype_choice == "numeric":
                converted = pd.to_numeric(s, errors="coerce")
            elif dtype_choice == "datetime":
                converted = pd.to_datetime(s, errors="coerce")
            else:
                converted = s.astype("string")

            # Missing values
            if dtype_choice == "numeric":
                if missing_method == "median":
                    fill_val = float(converted.median()
                                     ) if converted.notna().any() else 0.0
                    converted = converted.fillna(fill_val)
                elif missing_method == "mean":
                    fill_val = float(
                        converted.mean()) if converted.notna().any() else 0.0
                    converted = converted.fillna(fill_val)
                elif missing_method == "constant":
                    converted = converted.fillna(0.0)
                elif missing_method == "indicator":
                    indicator = converted.isna().astype(int)
                    feature_frames.append(pd.DataFrame(
                        {f"{col}__missing": indicator}, index=df_work.index))
                    fill_val = float(converted.median()
                                     ) if converted.notna().any() else 0.0
                    converted = converted.fillna(fill_val)
                else:
                    converted = converted.fillna(
                        float(converted.median()) if converted.notna().any() else 0.0)
            elif dtype_choice == "datetime":
                converted = converted.fillna(pd.Timestamp("1970-01-01"))
            else:
                if missing_method in {"mode", "keep"}:
                    mode_val = converted.mode(dropna=True)
                    fill_val = str(
                        mode_val.iloc[0]) if not mode_val.empty else "missing"
                    converted = converted.fillna(fill_val)
                elif missing_method == "constant":
                    converted = converted.fillna("missing")
                elif missing_method == "indicator":
                    indicator = converted.isna().astype(int)
                    feature_frames.append(pd.DataFrame(
                        {f"{col}__missing": indicator}, index=df_work.index))
                    converted = converted.fillna("missing")
                else:
                    converted = converted.fillna("missing")

            # Outliers (numeric only)
            if dtype_choice == "numeric":
                if outlier_method == "clip":
                    q = float(self.config["max_outlier_clip_quantile"])
                    lo, hi = converted.quantile(q), converted.quantile(1.0 - q)
                    converted = converted.clip(lower=lo, upper=hi)
                elif outlier_method == "log_transform":
                    converted = np.sign(converted) * \
                        np.log1p(np.abs(converted))

            # Feature creation
            if dtype_choice == "datetime":
                if policy.get("feature_creation", {}).get("method") == "datetime_parts":
                    feature_frames.append(
                        pd.DataFrame(
                            {
                                f"{col}__year": converted.dt.year.astype(int),
                                f"{col}__month": converted.dt.month.astype(int),
                                f"{col}__day": converted.dt.day.astype(int),
                            },
                            index=df_work.index,
                        )
                    )
                    column_actions[col] = {
                        "type": "datetime",
                        "action": "datetime_parts",
                        "missing": missing_method,
                        "reason": p.get("reason", "policy"),
                    }
                    continue
                feature_frames.append(pd.DataFrame(
                    {col: converted.astype("int64")}, index=df_work.index))
                numeric_cols.append(col)
                column_actions[col] = {
                    "type": "datetime", "action": "timestamp", "reason": p.get("reason", "policy")}
                continue

            if dtype_choice == "numeric":
                feature_frames.append(pd.DataFrame(
                    {col: converted}, index=df_work.index))
                numeric_cols.append(col)
                column_actions[col] = {
                    "type": "numeric",
                    "missing": missing_method,
                    "outlier": outlier_method,
                    "encoding": "none",
                    "reason": p.get("reason", "policy"),
                }
                continue

            # Categorical / text encoding
            cleaned = converted.astype(str)
            if encoding_method == "hash":
                hasher = FeatureHasher(
                    n_features=int(self.config["hash_features"]), input_type="string"
                )
                hashed = hasher.transform([[val] for val in cleaned]).toarray()
                hash_cols = [f"{col}__hash_{i}" for i in range(
                    self.config["hash_features"])]
                feature_frames.append(pd.DataFrame(
                    hashed, columns=hash_cols, index=df_work.index))
            elif encoding_method == "onehot":
                feature_frames.append(pd.get_dummies(
                    cleaned, prefix=col, dtype=int))
            elif encoding_method == "frequency":
                freq = cleaned.value_counts(normalize=True)
                feature_frames.append(pd.DataFrame(
                    {f"{col}__freq": cleaned.map(freq)}, index=df_work.index))
            elif encoding_method == "label":
                categories = sorted(cleaned.unique().tolist())
                mapping = {cat: idx for idx, cat in enumerate(categories)}
                feature_frames.append(pd.DataFrame(
                    {col: cleaned.map(mapping).astype(int)}, index=df_work.index))
            else:
                feature_frames.append(pd.DataFrame(
                    {col: cleaned}, index=df_work.index))
                categorical_cols.append(col)

            column_actions[col] = {
                "type": "categorical",
                "missing": missing_method,
                "encoding": encoding_method,
                "reason": p.get("reason", "policy"),
            }

        if not feature_frames:
            raise ValueError("No features left after preprocessing policy")

        X = pd.concat(feature_frames, axis=1)

        # Feature selection
        fs_method = policy.get("feature_selection", {}).get("method", "none")
        if fs_method == "variance":
            constant_cols = [
                c for c in X.columns if X[c].nunique(dropna=False) <= 1]
            if constant_cols:
                X = X.drop(columns=constant_cols)
        elif fs_method == "none":
            steps_status["feature_selection"] = "skipped"

        # Scaling
        scaler = None
        scaling_method = policy.get("scaling", {}).get("method", "standard")
        numeric_X_cols = [
            c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        if numeric_X_cols and scaling_method != "none":
            if scaling_method == "standard":
                scaler = StandardScaler()
            elif scaling_method == "minmax":
                scaler = MinMaxScaler()
            elif scaling_method == "robust":
                scaler = RobustScaler()
            elif scaling_method == "quantile":
                scaler = QuantileTransformer(
                    output_distribution="normal", random_state=42)
            elif scaling_method == "power":
                scaler = PowerTransformer()
            if scaler is not None:
                X[numeric_X_cols] = scaler.fit_transform(X[numeric_X_cols])
        else:
            steps_status["scaling"] = "skipped"

        # Normalization
        norm_method = policy.get("normalization", {}).get("method", "none")
        if norm_method in {"l1", "l2", "max"} and not X.empty:
            norm = Normalizer(norm=norm_method)
            X = pd.DataFrame(norm.fit_transform(
                X), columns=X.columns, index=X.index)
        else:
            steps_status["normalization"] = "skipped"

        # Dimensionality reduction
        dr_method = policy.get("dimensionality_reduction",
                               {}).get("method", "none")
        if dr_method == "pca":
            if X.shape[1] > 2:
                n_components = min(
                    max(2, int(np.sqrt(X.shape[1]))), X.shape[1])
                pca = PCA(n_components=n_components, random_state=42)
                reduced = pca.fit_transform(X)
                X = pd.DataFrame(
                    reduced,
                    columns=[f"pca_{i+1}" for i in range(reduced.shape[1])],
                    index=X.index,
                )
            else:
                steps_status["dimensionality_reduction"] = "skipped_not_enough_features"
        else:
            steps_status["dimensionality_reduction"] = "skipped"

        metadata = {
            "column_actions": column_actions,
            "dropped_columns": dropped_cols,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "duplicates_removed": duplicate_rows_removed,
            "steps_status": steps_status,
            "scaler": {
                "method": scaling_method,
                "columns": numeric_X_cols,
                "means": scaler.mean_.tolist() if hasattr(scaler, "mean_") else [],
                "scales": scaler.scale_.tolist() if hasattr(scaler, "scale_") else [],
            },
        }

        return X, y.loc[X.index], metadata

    def _apply_imbalance_method(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        method: str,
        random_state: int,
    ) -> tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        counts = y_train.value_counts()
        if counts.empty:
            return X_train, y_train, {"method": "none", "status": "skipped_empty_target"}

        min_count = int(counts.min())
        max_count = int(counts.max())
        ratio = float(max_count / max(min_count, 1))
        meta = {
            "method": method,
            "before_counts": {str(k): int(v) for k, v in counts.items()},
            "imbalance_ratio": ratio,
            "status": "handled",
        }

        if method == "none" or ratio < 1.5:
            meta["status"] = "skipped_not_needed"
            return X_train, y_train, meta

        if method == "class_weight":
            class_weights = {str(c): float(max_count / max(int(v), 1))
                             for c, v in counts.items()}
            meta["class_weights"] = class_weights
            meta["status"] = "handled_class_weight_metadata_only"
            return X_train, y_train, meta

        if method not in {"oversample", "undersample"}:
            meta["status"] = "skipped_unknown_method"
            return X_train, y_train, meta

        combined = X_train.copy()
        combined["__target__"] = y_train.values
        groups = [g for _, g in combined.groupby("__target__")]

        if method == "oversample":
            target_n = max_count
            rebalanced = []
            for g in groups:
                if len(g) < target_n:
                    sampled = g.sample(n=target_n - len(g),
                                       replace=True, random_state=random_state)
                    g = pd.concat([g, sampled], axis=0)
                rebalanced.append(g)
            out = pd.concat(rebalanced, axis=0).sample(
                frac=1.0, random_state=random_state)
        else:
            # Conservative undersampling: do not keep less than 60% of original training rows.
            target_n = min_count
            projected_rows = target_n * len(groups)
            min_rows_allowed = int(0.6 * len(combined))
            if projected_rows < min_rows_allowed:
                meta["status"] = "skipped_undersample_too_aggressive"
                return X_train, y_train, meta
            rebalanced = []
            for g in groups:
                if len(g) > target_n:
                    g = g.sample(n=target_n, replace=False,
                                 random_state=random_state)
                rebalanced.append(g)
            out = pd.concat(rebalanced, axis=0).sample(
                frac=1.0, random_state=random_state)

        X_out = out.drop(columns=["__target__"])
        y_out = out["__target__"]
        meta["after_counts"] = {str(k): int(v)
                                for k, v in y_out.value_counts().items()}
        return X_out, y_out, meta

    def _save_outputs(
        self,
        output_folder: Path,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        metadata: Dict[str, Any],
        policy: Dict[str, Any],
        evidence: Dict[str, Any],
        dataset_name: str,
        target_col: str,
        test_size: float,
        random_state: int,
        use_llm: bool,
    ) -> Dict[str, Path]:
        X_train_path = output_folder / "X_train.csv"
        X_test_path = output_folder / "X_test.csv"
        y_train_path = output_folder / "y_train.csv"
        y_test_path = output_folder / "y_test.csv"
        summary_path = output_folder / "preprocessing_summary.json"
        column_actions_path = output_folder / "column_actions.json"
        policy_path = output_folder / "llm_policy.json"
        evidence_path = output_folder / "evidence_snapshot.json"

        X_train.to_csv(X_train_path, index=False)
        X_test.to_csv(X_test_path, index=False)
        y_train.to_csv(y_train_path, index=False)
        y_test.to_csv(y_test_path, index=False)

        summary = {
            "dataset": dataset_name,
            "target_column": target_col,
            "rows": len(X_train) + len(X_test),
            "features": int(X_train.shape[1]),
            "dropped_columns": metadata["dropped_columns"],
            "duplicates_removed": metadata["duplicates_removed"],
            "test_size": test_size,
            "random_state": random_state,
            "llm": {
                "enabled": use_llm,
                "policy_used": bool(metadata.get("llm_policy_used", False)),
                "provider": self.llm_provider,
                "model": self.config.get("hf_model", "Qwen/Qwen2.5-14B-Instruct")
                if self.llm_provider == "huggingface"
                else self.config.get("gemini_model", "gemini-2.5-flash"),
                "env_var": self.config.get("hf_api_key_env", "HUGGINGFACE_API_KEY")
                if self.llm_provider == "huggingface"
                else self.config.get("gemini_api_key_env", "GEMINI_API_KEY"),
                "final_decision_enabled": bool(self.config.get("llm_final_decision", True)),
            },
            "steps_status": metadata["steps_status"],
            "imbalance": metadata.get("imbalance", {}),
            "scaling": metadata["scaler"],
            "safeguards": policy.get("safeguards", {}),
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(column_actions_path, "w", encoding="utf-8") as f:
            json.dump(metadata["column_actions"], f, indent=2)
        with open(policy_path, "w", encoding="utf-8") as f:
            json.dump(policy, f, indent=2)
        with open(evidence_path, "w", encoding="utf-8") as f:
            json.dump(evidence, f, indent=2)

        return {
            "X_train": X_train_path,
            "X_test": X_test_path,
            "y_train": y_train_path,
            "y_test": y_test_path,
            "summary": summary_path,
            "column_actions": column_actions_path,
            "policy": policy_path,
            "evidence": evidence_path,
        }


def preprocessing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    node = PreprocessingNode()
    return node.run(state)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python preprocessing_node.py <dataset_path> <target_column>")
        sys.exit(1)

    state = {
        "dataset_path": sys.argv[1],
        "target_column": sys.argv[2],
        "output_folder": sys.argv[3] if len(sys.argv) > 3 else "output",
    }

    result = preprocessing_node(state)

    if result["status"] == "success":
        print("Preprocessing successful")
        print(f"Output folder: {result['output_folder']}")
        print(f"X_train: {result['X_train_path']}")
        print(f"Summary: {result['summary_path']}")
        print(f"Policy: {result['policy_path']}")
        print(f"Evidence: {result['evidence_path']}")
    else:
        print(f"Preprocessing failed: {result['error']}")
        sys.exit(1)
