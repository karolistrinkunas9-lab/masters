"""
Simplified AI Inference Service
FastAPI with custom model endpoints and fuzzy inference systems
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import json
import logging
import re
from setfit import SetFitModel
from huggingface_hub import hf_hub_download
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.linear_model import LogisticRegression
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Custom Model Head (required for unpickling the SetFit model)
# ============================================================================

class WeightedBinaryRelevanceHead:
    """
    Custom classification head for SetFit model.
    Must be defined before loading the model to avoid pickle errors.
    """
    def __init__(self, zero_row_weight: float = 0.3, class_weight: str = "balanced", **lr_kwargs):
        self.zero_row_weight = float(zero_row_weight)
        self.class_weight = class_weight
        self.lr_kwargs = dict(max_iter=1000, **lr_kwargs)
        self.estimators_ = None

    def fit(self, X_emb: np.ndarray, y_multi):
        Y = np.asarray(y_multi, dtype=int)
        n_labels = Y.shape[1]
        has_pos = (Y.sum(axis=1) > 0).astype(float)
        sample_weight = np.where(has_pos > 0, 1.0, self.zero_row_weight)
        self.estimators_ = []
        for j in range(n_labels):
            yj = Y[:, j]
            clf = LogisticRegression(class_weight=self.class_weight, **self.lr_kwargs)
            clf.fit(X_emb, yj, sample_weight=sample_weight)
            self.estimators_.append(clf)
        return self

    def predict(self, X_emb: np.ndarray):
        preds = [est.predict(X_emb) for est in self.estimators_]
        return np.vstack(preds).T

    def predict_proba(self, X_emb: np.ndarray):
        probs = [est.predict_proba(X_emb)[:, 1] for est in self.estimators_]
        return np.vstack(probs).T


# ============================================================================
# Word-level Contribution Utilities (Occlusion-based Explanations)
# ============================================================================

def _to_numpy_probs(y_prob):
    """Convert probabilities to numpy array."""
    return np.asarray(y_prob, dtype=np.float32)


def _logit(p, eps=1e-12):
    """Convert probabilities to logits."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def word_level_contributions(model, text: str, label_names, thresholds, top_k=6):
    """
    Leave-one-word-out occlusion: for each token, remove it and measure
    the drop in logits for each label. Positive numbers mean the word
    increased the label's log-odds.
    
    Returns: 
        - explanations: dict[label] -> list[(word, contribution)]
        - base_probs: numpy array of base probabilities
    """
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return {}, np.array([])

    base_probs = _to_numpy_probs(model.predict_proba([text]))[0]
    base_logits = _logit(base_probs)

    # Create N texts with token i removed
    occ_texts = [" ".join(tokens[:i] + tokens[i+1:]) for i in range(len(tokens))]
    occ_probs = _to_numpy_probs(model.predict_proba(occ_texts))
    occ_logits = _logit(occ_probs)

    # Contribution of token i to each label = base_logit - occ_logit[i]
    contrib_matrix = (base_logits[None, :] - occ_logits)  # shape (N_tokens, n_labels)

    # Aggregate by normalized lowercased tokens (skip tiny/non-alnum)
    agg = {}
    for i, tok in enumerate(tokens):
        key = tok.lower()
        if len(key) <= 1 or not re.search(r"[A-Za-z0-9]", key):
            continue
        vec = contrib_matrix[i, :]
        if key in agg:
            agg[key] += vec
        else:
            agg[key] = vec.copy()

    # Keep top_k positive contributors for labels that are predicted (above threshold)
    explanations = {}
    for j, lab in enumerate(label_names):
        if base_probs[j] < thresholds[j]:
            continue
        scores = [(w, float(v[j])) for w, v in agg.items()]
        scores = [(w, s) for w, s in scores if s > 0]
        scores.sort(key=lambda x: x[1], reverse=True)
        explanations[lab] = scores[:top_k]
    
    return explanations, base_probs


# FastAPI app will be created after lifespan function is defined (see below)

# Global model cache
models_cache = {}


# ============================================================================
# Request/Response Models
# ============================================================================

class RequirementQualityRequest(BaseModel):
    texts: List[str] = Field(..., description="List of requirement texts to classify")
    model_name: str = Field(default="Hulyyy/req-quality-setfit-hintaware-dict")


class LabelPrediction(BaseModel):
    text: str
    predicted_labels: List[str]
    probabilities: Dict[str, float]
    all_probabilities: Dict[str, float]


class RequirementQualityResponse(BaseModel):
    success: bool
    predictions: List[LabelPrediction]
    model_name: str
    error: Optional[str] = None


class RequirementModificationRequest(BaseModel):
    original_requirement: str
    modification_instruction: str
    model_name: str = Field(default="Qwen/Qwen3-4B-Instruct-2507")
    max_new_tokens: Optional[int] = Field(default=512)
    temperature: Optional[float] = Field(default=0.7, description="Temperature for sampling (higher = more random)")
    top_p: Optional[float] = Field(default=0.9, description="Nucleus sampling parameter")
    top_k: Optional[int] = Field(default=50, description="Top-k sampling parameter")
    do_sample: Optional[bool] = Field(default=True, description="Enable sampling for non-deterministic output")


class RequirementModificationResponse(BaseModel):
    success: bool
    original_requirement: str
    modification_instruction: str
    modified_requirement: str
    model_name: str
    error: Optional[str] = None


class RequirementAnalysisRequest(BaseModel):
    original_requirement: str
    modification_instruction: str
    modified_requirement: str = Field(..., description="The modified requirement to evaluate")
    model_name: str = Field(default="Qwen/Qwen3-4B-Instruct-2507")
    max_new_tokens: Optional[int] = Field(default=256)
    temperature: Optional[float] = Field(default=0.2, description="Temperature for sampling")
    top_p: Optional[float] = Field(default=0.9, description="Top-p for nucleus sampling")
    quant_4bit: Optional[bool] = Field(default=False, description="Use 4-bit quantization to save memory")
    force_reload: Optional[bool] = Field(default=False, description="Force reload model (clear cache first)")


class RequirementAnalysisResponse(BaseModel):
    success: bool
    preservation_correctness: float
    change_correctness: float
    analysis_text: str
    model_name: str
    error: Optional[str] = None


class ComprehensiveRequirementAnalysisRequest(BaseModel):
    original_requirement: str
    modification_instruction: str
    model_name: str = Field(default="Qwen/Qwen3-4B-Instruct-2507")
    quality_model_name: str = Field(default="Hulyyy/req-quality-setfit-hintaware-dict")
    max_new_tokens: Optional[int] = Field(default=512)
    temperature: Optional[float] = Field(default=0.7, description="Temperature for sampling (use 0.1 for deterministic)")
    top_p: Optional[float] = Field(default=0.9, description="Nucleus sampling parameter")
    top_k: Optional[int] = Field(default=50, description="Top-k sampling parameter")
    do_sample: Optional[bool] = Field(default=True, description="Enable sampling (set False for deterministic)")


class ComprehensiveRequirementAnalysisResponse(BaseModel):
    success: bool
    modified_requirement: str
    preservation_correctness: float
    change_correctness: float
    detected_issues: List[str]
    comments: List[str]
    model_name: str
    error: Optional[str] = None


# Fuzzy Inference System Models
class DefectSeverityRequest(BaseModel):
    subjective: float = Field(..., ge=0, le=1, description="Subjective score (0-1)")
    ambiguous: float = Field(..., ge=0, le=1, description="Ambiguous score (0-1)")
    nonverifiable: float = Field(..., ge=0, le=1, description="Nonverifiable score (0-1)")
    negative: float = Field(..., ge=0, le=1, description="Negative score (0-1)")
    vague: float = Field(..., ge=0, le=1, description="Vague score (0-1)")


class DefectSeverityResponse(BaseModel):
    success: bool
    defect_severity: float
    defect_severity_label: str
    inputs: Dict[str, float]
    error: Optional[str] = None


class CorrectnessRequest(BaseModel):
    preservation_correctness: float = Field(..., ge=0, le=1, description="Preservation correctness (0-1)")
    change_correctness: float = Field(..., ge=0, le=1, description="Change correctness (0-1)")


class CorrectnessResponse(BaseModel):
    success: bool
    correctness: float
    correctness_label: str
    correctness_three_level_label: Optional[str] = None  # Low/Average/High
    inputs: Dict[str, float]
    error: Optional[str] = None


class RequirementQualityFuzzyRequest(BaseModel):
    defect_severity: float = Field(..., ge=0, le=1, description="Defect severity (0-1)")
    correctness: float = Field(..., ge=0, le=1, description="Correctness (0-1)")


class RequirementQualityFuzzyResponse(BaseModel):
    success: bool
    requirement_quality: float
    requirement_quality_label: str
    inputs: Dict[str, float]
    error: Optional[str] = None


# ============================================================================
# Fuzzy System Initialization
# ============================================================================

# Initialize fuzzy systems on startup
fuzzy_systems = {}

def apply_rule_weight(rule, weight):
    """
    Simulate rule weight by returning multiple instances of the same rule.
    Higher weight = more instances = more influence on output.
    
    Weight to repetitions mapping:
    - 0.5: 1 instance
    - 0.6: 1 instance
    - 0.7: 2 instances
    - 0.75: 2 instances
    - 0.8: 2 instances
    - 0.85: 2 instances
    - 0.9: 3 instances
    - 0.95: 3 instances
    """
    if weight >= 0.9:
        repetitions = 3
    elif weight >= 0.7:
        repetitions = 2
    else:
        repetitions = 1
    
    return [rule] * repetitions

def initialize_defect_severity_fis():
    """Initialize DefectSeverity FIS based on defectSeverity.fis"""
    # Inputs
    subjective = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'subjective')
    ambiguous = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'ambiguous')
    nonverifiable = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'nonverifiable')
    negative = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'negative')
    vague = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'vague')
    
    # Output
    defect_severity = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'defect_severity')
    
    # Membership functions for inputs (zmf, trimf, smf)
    subjective['low'] = fuzz.zmf(subjective.universe, 0.25, 0.55)
    subjective['average'] = fuzz.trimf(subjective.universe, [0.4, 0.55, 0.7])
    subjective['high'] = fuzz.smf(subjective.universe, 0.65, 0.85)
    
    ambiguous['low'] = fuzz.zmf(ambiguous.universe, 0.2, 0.5)
    ambiguous['average'] = fuzz.trimf(ambiguous.universe, [0.35, 0.5, 0.65])
    ambiguous['high'] = fuzz.smf(ambiguous.universe, 0.55, 0.75)
    
    nonverifiable['low'] = fuzz.zmf(nonverifiable.universe, 0.2, 0.5)
    nonverifiable['average'] = fuzz.trimf(nonverifiable.universe, [0.35, 0.5, 0.65])
    nonverifiable['high'] = fuzz.smf(nonverifiable.universe, 0.55, 0.75)
    
    negative['low'] = fuzz.zmf(negative.universe, 0.25, 0.55)
    negative['average'] = fuzz.trimf(negative.universe, [0.4, 0.55, 0.7])
    negative['high'] = fuzz.smf(negative.universe, 0.6, 0.8)
    
    vague['low'] = fuzz.zmf(vague.universe, 0.2, 0.5)
    vague['average'] = fuzz.trimf(vague.universe, [0.35, 0.5, 0.65])
    vague['high'] = fuzz.smf(vague.universe, 0.55, 0.75)
    
    # Output membership functions
    defect_severity['low'] = fuzz.trapmf(defect_severity.universe, [0, 0, 0.2, 0.4])
    defect_severity['average'] = fuzz.trimf(defect_severity.universe, [0.35, 0.5, 0.65])
    defect_severity['high'] = fuzz.trapmf(defect_severity.universe, [0.6, 0.8, 1, 1])
    
    # Rules (from defectSeverity.fis) with weight simulation
    rules = []
    
    # weight=0.7 rules (2 instances each)
    rules.extend(apply_rule_weight(ctrl.Rule(ambiguous['high'], defect_severity['high']), 0.7))
    rules.extend(apply_rule_weight(ctrl.Rule(vague['high'], defect_severity['high']), 0.7))
    rules.extend(apply_rule_weight(ctrl.Rule(nonverifiable['high'], defect_severity['high']), 0.7))
    
    # weight=0.75 rules (2 instances each)
    rules.extend(apply_rule_weight(ctrl.Rule(ambiguous['average'] & vague['average'], defect_severity['high']), 0.75))
    rules.extend(apply_rule_weight(ctrl.Rule(ambiguous['average'] & nonverifiable['average'], defect_severity['high']), 0.75))
    rules.extend(apply_rule_weight(ctrl.Rule(nonverifiable['average'] & vague['average'], defect_severity['high']), 0.75))
    
    # weight=0.6 rules (1 instance each)
    rules.extend(apply_rule_weight(ctrl.Rule(ambiguous['average'] & nonverifiable['low'] & vague['low'], defect_severity['average']), 0.6))
    rules.extend(apply_rule_weight(ctrl.Rule(ambiguous['low'] & nonverifiable['low'] & vague['average'], defect_severity['average']), 0.6))
    rules.extend(apply_rule_weight(ctrl.Rule(ambiguous['low'] & nonverifiable['average'] & vague['low'], defect_severity['average']), 0.6))
    
    # weight=0.5 rules (1 instance each)
    rules.extend(apply_rule_weight(ctrl.Rule(subjective['high'] & ambiguous['low'] & nonverifiable['low'] & vague['low'], defect_severity['average']), 0.5))
    rules.extend(apply_rule_weight(ctrl.Rule(ambiguous['low'] & nonverifiable['low'] & negative['high'] & vague['low'], defect_severity['average']), 0.5))
    
    # weight=0.85 rule (2 instances)
    rules.extend(apply_rule_weight(ctrl.Rule(subjective['low'] & ambiguous['low'] & nonverifiable['low'] & negative['low'] & vague['low'], defect_severity['low']), 0.85))
    
    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)


def initialize_correctness_fis():
    """
    Initialize Correctness FIS based on correctness.fis
    
    Rules from FIS file:
    3 3, 3 (0.9) : 1  -> High & High -> High (weight 0.9)
    3 2, 3 (0.8) : 1  -> High & Average -> High (weight 0.8)
    2 3, 3 (0.8) : 1  -> Average & High -> High (weight 0.8)
    2 2, 2 (0.7) : 1  -> Average & Average -> Average (weight 0.7)
    1 1, 1 (0.95) : 1 -> Low & Low -> Low (weight 0.95)
    1 0, 1 (0.95) : 1 -> Low & (any) -> Low (weight 0.95) [0 means "don't care"]
    0 1, 1 (0.95) : 1 -> (any) & Low -> Low (weight 0.95) [0 means "don't care"]
    """
    # Inputs
    preservation = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'preservation')
    change = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'change')
    
    # Output
    correctness = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'correctness')
    
    # Membership functions - matching correctness.fis
    # Input1: PreservationCorrectness
    preservation['low'] = fuzz.zmf(preservation.universe, 0.25, 0.55)
    preservation['average'] = fuzz.trimf(preservation.universe, [0.4, 0.55, 0.7])
    preservation['high'] = fuzz.smf(preservation.universe, 0.65, 0.85)
    
    # Input2: ChangeCorrectness
    change['low'] = fuzz.zmf(change.universe, 0.25, 0.55)
    change['average'] = fuzz.trimf(change.universe, [0.4, 0.55, 0.7])
    change['high'] = fuzz.smf(change.universe, 0.65, 0.85)
    
    # Output1: Correctness
    correctness['low'] = fuzz.trapmf(correctness.universe, [0, 0, 0.2, 0.4])
    correctness['average'] = fuzz.trimf(correctness.universe, [0.35, 0.5, 0.65])
    correctness['high'] = fuzz.trapmf(correctness.universe, [0.6, 0.8, 1, 1])
    
    # Rules with weights matching correctness.fis
    rules = []
    
    # Rule 3 3, 3 (0.9): High & High -> High
    rules.extend(apply_rule_weight(ctrl.Rule(preservation['high'] & change['high'], correctness['high']), 0.9))
    
    # Rule 3 2, 3 (0.8): High & Average -> High
    rules.extend(apply_rule_weight(ctrl.Rule(preservation['high'] & change['average'], correctness['high']), 0.8))
    
    # Rule 2 3, 3 (0.8): Average & High -> High
    rules.extend(apply_rule_weight(ctrl.Rule(preservation['average'] & change['high'], correctness['high']), 0.8))
    
    # Rule 2 2, 2 (0.7): Average & Average -> Average
    rules.extend(apply_rule_weight(ctrl.Rule(preservation['average'] & change['average'], correctness['average']), 0.7))
    
    # Rule 1 1, 1 (0.95): Low & Low -> Low
    rules.extend(apply_rule_weight(ctrl.Rule(preservation['low'] & change['low'], correctness['low']), 0.95))
    
    # Rule 1 0, 1 (0.95): Low & (any) -> Low
    # In skfuzzy, we use OR to represent "any" condition
    rules.extend(apply_rule_weight(ctrl.Rule(preservation['low'], correctness['low']), 0.95))
    
    # Rule 0 1, 1 (0.95): (any) & Low -> Low
    rules.extend(apply_rule_weight(ctrl.Rule(change['low'], correctness['low']), 0.95))
    
    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)


def initialize_requirement_quality_fis():
    """Initialize RequirementQuality FIS based on requirementQuality.fis"""
    # Inputs
    defect_sev = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'defect_severity')
    correct = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'correctness')
    
    # Output
    req_quality = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'requirement_quality')
    
    # Membership functions
    defect_sev['low'] = fuzz.trimf(defect_sev.universe, [0, 0.25, 0.5])
    defect_sev['average'] = fuzz.trimf(defect_sev.universe, [0.35, 0.5, 0.65])
    defect_sev['high'] = fuzz.trimf(defect_sev.universe, [0.5, 0.75, 1])
    
    correct['low'] = fuzz.trimf(correct.universe, [0, 0.25, 0.5])
    correct['average'] = fuzz.trimf(correct.universe, [0.35, 0.5, 0.65])
    correct['high'] = fuzz.trimf(correct.universe, [0.5, 0.75, 1])
    
    req_quality['very_low'] = fuzz.trapmf(req_quality.universe, [0, 0, 0.1, 0.25])
    req_quality['low'] = fuzz.trimf(req_quality.universe, [0.15, 0.3, 0.45])
    req_quality['average'] = fuzz.trimf(req_quality.universe, [0.35, 0.5, 0.65])
    req_quality['high'] = fuzz.trimf(req_quality.universe, [0.55, 0.7, 0.85])
    req_quality['very_high'] = fuzz.trapmf(req_quality.universe, [0.75, 0.9, 1, 1])
    
    # Rules (from requirementQuality.fis)
    rules = [
        ctrl.Rule(defect_sev['high'] & correct['low'], req_quality['very_low']),
        ctrl.Rule(defect_sev['high'] & correct['average'], req_quality['low']),
        ctrl.Rule(defect_sev['high'] & correct['high'], req_quality['low']),
        ctrl.Rule(defect_sev['average'] & correct['low'], req_quality['very_low']),
        ctrl.Rule(defect_sev['average'] & correct['average'], req_quality['average']),
        ctrl.Rule(defect_sev['average'] & correct['high'], req_quality['average']),
        ctrl.Rule(defect_sev['low'] & correct['low'], req_quality['low']),
        ctrl.Rule(defect_sev['low'] & correct['average'], req_quality['high']),
        ctrl.Rule(defect_sev['low'] & correct['high'], req_quality['very_high']),
    ]
    
    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)


def get_label(value: float, thresholds: Dict[str, tuple]) -> str:
    """Determine label based on value and thresholds"""
    for label, (low, high) in thresholds.items():
        if low <= value <= high:
            return label
    return "unknown"


# ============================================================================
# Helper Functions
# ============================================================================

def load_setfit_model(model_name: str):
    """Load SetFit model with config and thresholds"""
    if model_name in models_cache:
        return models_cache[model_name]
    
    logger.info(f"Loading SetFit model: {model_name}")
    
    # Load model (WeightedBinaryRelevanceHead must be defined before this)
    model = SetFitModel.from_pretrained(model_name)
    
    # Load config (supports both naming conventions)
    cfg_path = None
    for fname in ("config_labels.json", "config_label.json"):
        try:
            cfg_path = hf_hub_download(repo_id=model_name, filename=fname)
            break
        except Exception:
            pass
    
    if cfg_path is None:
        raise FileNotFoundError(f"Config file not found in {model_name}")
    
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    # Parse labels (ordered by id)
    id2label = {int(k): v for k, v in cfg["id2label"].items()}
    label_names = [id2label[i] for i in sorted(id2label.keys())]
    num_labels = len(label_names)
    
    # Load thresholds if provided, else default 0.5
    thresholds = np.full(num_labels, 0.5, dtype=float)
    thr_file = cfg.get("thresholds_path") or cfg.get("label_thresholds.npy")
    if thr_file:
        try:
            thr_path = hf_hub_download(repo_id=model_name, filename=thr_file)
            thresholds = np.load(thr_path)
            logger.info(f"Loaded thresholds from {thr_file}")
        except Exception as e:
            logger.warning(f"Could not load thresholds, using defaults: {e}")
    
    model_data = {
        "model": model,
        "id2label": id2label,
        "label_names": label_names,
        "thresholds": thresholds
    }
    
    models_cache[model_name] = model_data
    logger.info(f"Model loaded with {num_labels} labels: {', '.join(label_names)}")
    return model_data


# Qwen3 system prompt for requirement evaluation
QWEN_SYSTEM_PROMPT = (
    "You are a strict, analytical evaluator of software requirement modifications.\n"
    "Compare the Original, the Requested modification, and the Modified requirement.\n\n"
    "Scoring rules:\n"
    "- preservation_correctness (PC): Measures how much of the original meaning and intent are preserved outside the requested change.\n"
    "  • 1.0 = fully preserved, no unintended meaning change.\n"
    "  • 0.-0.9 = minor rewording or stylistic edits that don't change meaning.\n"
    "  • 0.5 = roughly half of the original preserved, moderate drift or extra content.\n"
    "  • 0.2-0.4 = major scope change or loss of important meaning.\n"
    "  • 0.0 = original meaning lost or contradicted.\n"
    "- change_correctness (CC): Measures how precisely the requested modification was implemented.\n"
    "  • 1.0 = fully and exactly applied as requested.\n"
    "  • 0.8-0.9 = almost correct, small omission or minor misinterpretation.\n"
    "  • 0.5 = partially correct, only some aspects applied.\n"
    "  • 0.2-0.4 = mostly wrong or incomplete.\n"
    "  • 0.0 = no change applied or opposite meaning introduced.\n"
    "If the requested change isn't applied, set CC close to 0. If unrelated parts are changed or removed, lower PC accordingly.\n\n"
    "Output policy:\n"
    "Think privately in <THINK>…</THINK>, then output ONLY a <FINAL>{JSON}</FINAL>.\n"
    'The JSON must be exactly: {"preservation_correctness": X.XX, "change_correctness": Y.YY} with both floats in [0,1]. No extra text.'
)


def build_user_input(original: str, instruction: str, modified: str) -> str:
    """Build user input prompt for Qwen evaluation"""
    return (
        "Task: Score two criteria in [0,1] and return ONLY the JSON in <FINAL>.\n\n"
        f"Original:\n<<<{original}>>>\n\n"
        f"Requested modification:\n<<<{instruction}>>>\n\n"
        f"Modified:\n<<<{modified}>>>\n\n"
        "Think privately first, then output the final JSON only.\n"
        "<THINK>\n</THINK>\n<FINAL>"
    )


def parse_scores(generated_text: str):
    """Parse preservation_correctness and change_correctness from Qwen output"""
    # Prefer JSON inside <FINAL>…</FINAL>; fallback to first JSON object containing the keys.
    m = re.search(r"<FINAL>\s*(\{.*?\})\s*</?FINAL>?", generated_text, flags=re.S)
    blob = m.group(1) if m else None
    
    if not blob:
        # Fallback: try to find JSON object with nested structure support
        # Find the first opening brace
        json_start = generated_text.find('{')
        if json_start == -1:
            # Last resort: try simple regex (won't handle nested structures)
            m2 = re.search(r"\{[^{}]*\"preservation_correctness\"[^{}]*\"change_correctness\"[^{}]*\}", generated_text, flags=re.S)
            if not m2:
                raise ValueError(f"No JSON found in model output:\n{generated_text}")
            blob = m2.group(0)
        else:
            # Find matching closing brace by counting braces (handles nested structures)
            brace_count = 0
            json_end = json_start
            for i in range(json_start, len(generated_text)):
                if generated_text[i] == '{':
                    brace_count += 1
                elif generated_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            blob = generated_text[json_start:json_end]
    
    obj = json.loads(blob)
    pc = float(obj["preservation_correctness"])
    cc = float(obj["change_correctness"])
    if not (0.0 <= pc <= 1.0 and 0.0 <= cc <= 1.0):
        raise ValueError(f"Scores out of range: {obj}")
    return {"preservation_correctness": pc, "change_correctness": cc}


def clear_model_cache(model_name: Optional[str] = None):
    """Clear model cache. If model_name is provided, only clear that model's cache entries."""
    if model_name:
        # Clear all cache entries for this model (with different quantization settings)
        keys_to_remove = [key for key in models_cache.keys() if key.startswith(f"{model_name}_")]
        for key in keys_to_remove:
            logger.info(f"Clearing cached model: {key}")
            # Try to free memory if it's a model tuple
            cached_item = models_cache.pop(key, None)
            if cached_item and isinstance(cached_item, tuple) and len(cached_item) >= 2:
                # Delete model to free GPU memory
                try:
                    if hasattr(cached_item[1], 'to'):
                        del cached_item[1]
                    if hasattr(cached_item[0], 'to'):
                        del cached_item[0]
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Error freeing memory for {key}: {e}")
        logger.info(f"Cleared {len(keys_to_remove)} cache entries for model: {model_name}")
    else:
        # Clear all cache entries
        cache_size = len(models_cache)
        for key in list(models_cache.keys()):
            cached_item = models_cache.pop(key, None)
            if cached_item and isinstance(cached_item, tuple) and len(cached_item) >= 2:
                # Delete model to free GPU memory
                try:
                    if hasattr(cached_item[1], 'to'):
                        del cached_item[1]
                    if hasattr(cached_item[0], 'to'):
                        del cached_item[0]
                except Exception as e:
                    logger.warning(f"Error freeing memory for {key}: {e}")
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Cleared all {cache_size} cache entries")
    return {"cleared": True, "message": f"Cache cleared for {model_name if model_name else 'all models'}"}


def load_qwen(model_name: str, quant_4bit: bool = False, force_reload: bool = False):
    """Load Qwen tokenizer and model
    
    Args:
        model_name: Name of the model to load
        quant_4bit: Whether to use 4-bit quantization
        force_reload: If True, clear cache and reload the model
    """
    cache_key = f"{model_name}_quant{quant_4bit}"
    
    # If force_reload is True, clear the cache for this model first
    if force_reload:
        logger.info(f"Force reload requested for {model_name}, clearing cache...")
        clear_model_cache(model_name)
    elif cache_key in models_cache:
        logger.info(f"Using cached model: {cache_key}")
        return models_cache[cache_key]
    
    logger.info(f"Loading Qwen model: {model_name} (quant_4bit={quant_4bit})")
    
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.padding_side = "right"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    
    if quant_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=bnb,
            device_map="auto"
        )
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mdl = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    
    models_cache[cache_key] = (tok, mdl)
    logger.info(f"Model {model_name} loaded successfully")
    return tok, mdl


# Lifespan event handler (must be defined after all helper functions)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup - clear model cache and initialize fuzzy systems"""
    logger.info("Starting up - clearing model cache to ensure fresh models...")
    clear_model_cache()  # Clear all cached models on startup
    
    logger.info("Initializing fuzzy systems...")
    try:
        fuzzy_systems['defect_severity'] = initialize_defect_severity_fis()
        fuzzy_systems['correctness'] = initialize_correctness_fis()
        fuzzy_systems['requirement_quality'] = initialize_requirement_quality_fis()
        logger.info("Fuzzy systems initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing fuzzy systems: {e}", exc_info=True)
        logger.warning("Fuzzy systems will be initialized on first use")
    
    yield  # App is running
    
    # Cleanup on shutdown (if needed)
    logger.info("Shutting down...")


# Create FastAPI app with lifespan
app = FastAPI(
    title="AI Inference Service",
    description="Simple API for requirement analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================================
# Endpoints
# ============================================================================

def ensure_fuzzy_system(system_name: str):
    """Ensure fuzzy system is initialized, initialize if needed"""
    if system_name not in fuzzy_systems:
        logger.info(f"Lazy-initializing fuzzy system: {system_name}")
        try:
            if system_name == 'defect_severity':
                fuzzy_systems[system_name] = initialize_defect_severity_fis()
            elif system_name == 'correctness':
                fuzzy_systems[system_name] = initialize_correctness_fis()
            elif system_name == 'requirement_quality':
                fuzzy_systems[system_name] = initialize_requirement_quality_fis()
            logger.info(f"Successfully initialized: {system_name}")
        except Exception as e:
            logger.error(f"Error initializing {system_name}: {e}", exc_info=True)
            raise
    return fuzzy_systems[system_name]


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "name": "AI Inference Service",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "ml_models": [
                "/api/inference/requirement-quality",
                "/api/inference/requirement-analysis",
                "/api/inference/comprehensive-analysis",
                "/api/inference/requirement-modification"
            ],
            "fuzzy_systems": [
                "/api/fuzzy/defect-severity",
                "/api/fuzzy/correctness",
                "/api/fuzzy/requirement-quality"
            ]
        }
    }


@app.get("/health")
def health():
    """Health check"""
    return {"status": "healthy"}


class ClearCacheRequest(BaseModel):
    model_name: Optional[str] = Field(default=None, description="Optional model name to clear. If not provided, clears all models.")


@app.post("/api/cache/clear")
async def clear_cache(request: Optional[ClearCacheRequest] = None):
    """Clear model cache. If model_name is provided, only clear that model's cache."""
    model_name = request.model_name if request else None
    result = clear_model_cache(model_name)
    return {"success": True, **result}


@app.post("/api/inference/requirement-quality", response_model=RequirementQualityResponse)
async def requirement_quality(request: RequirementQualityRequest):
    """
    Analyze requirement quality using SetFit multi-label classification
    
    Example:
    {
        "texts": ["The system shall be fast."]
    }
    """
    try:
        # Load model
        model_data = load_setfit_model(request.model_name)
        model = model_data["model"]
        id2label = model_data["id2label"]
        label_names = model_data["label_names"]
        thresholds = model_data["thresholds"]
        
        # Predict
        probas = np.array(model.predict_proba(request.texts))
        preds = (probas >= thresholds.reshape(1, -1)).astype(int)
        
        # Format results
        results = []
        for text, pred_row, proba_row in zip(request.texts, preds, probas):
            predicted_labels = [label_names[i] for i, v in enumerate(pred_row) if v == 1]
            
            predicted_probs = {
                label_names[i]: float(proba_row[i])
                for i, v in enumerate(pred_row) if v == 1
            }
            
            all_probs = {
                label_names[i]: float(proba_row[i])
                for i in range(len(label_names))
            }
            
            # Log in the requested format: Subjective, Ambiguous, Nonverifiable, Negative, Vague
            label_order = ['Subjective', 'Ambiguous', 'Nonverifiable', 'Negative', 'Vague']
            probabilities_list = []
            for label in label_order:
                # Find the index of this label in label_names
                try:
                    idx = label_names.index(label)
                    probabilities_list.append(float(proba_row[idx]))
                except ValueError:
                    # Label not found, use 0.0
                    probabilities_list.append(0.0)
            
            logger.info(f"Predicted labels: {predicted_labels}")
            logger.info(f"Probabilities: {probabilities_list}")
            
            results.append(LabelPrediction(
                text=text,
                predicted_labels=predicted_labels,
                probabilities=predicted_probs,
                all_probabilities=all_probs
            ))
        
        return RequirementQualityResponse(
            success=True,
            predictions=results,
            model_name=request.model_name
        )
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return RequirementQualityResponse(
            success=False,
            predictions=[],
            model_name=request.model_name,
            error=str(e)
        )


@app.post("/api/inference/requirement-analysis", response_model=RequirementAnalysisResponse)
async def requirement_analysis(request: RequirementAnalysisRequest):
    """
    Evaluate requirement modification using Qwen3-4B-Instruct-2507 model
    
    Qwen provides evaluation scores. It does NOT generate the modified requirement.
    
    Example:
    {
        "original_requirement": "The system must support users.",
        "modification_instruction": "Add three types of users.",
        "modified_requirement": "The system must support three types of users: admin, manager, and regular user."
    }
    
    Returns preservation_correctness and change_correctness scores (0-1) from Qwen
    """
    try:
        # Clear cache for old models if switching to a new model
        # Check if we have a different model cached
        cache_key = f"{request.model_name}_quant{request.quant_4bit}"
        if cache_key not in models_cache:
            # Clear any old Qwen model caches to free memory
            old_keys = [k for k in models_cache.keys() if k.startswith("Qwen/") and k != cache_key]
            if old_keys:
                logger.info(f"Clearing {len(old_keys)} old model cache entries before loading {request.model_name}")
                for old_key in old_keys:
                    models_cache.pop(old_key, None)
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Load tokenizer and model (will use cache if available, or load fresh)
        tok, mdl = load_qwen(request.model_name, quant_4bit=request.quant_4bit, force_reload=request.force_reload or False)
        
        # Build messages with system prompt and user input
        messages = [
            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_input(
                request.original_requirement,
                request.modification_instruction,
                request.modified_requirement
            )},
        ]
        
        # Apply chat template
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        logger.info(f"=== Qwen Evaluation Input ===")
        logger.info(f"Original requirement: '{request.original_requirement}'")
        logger.info(f"Modification instruction: '{request.modification_instruction}'")
        logger.info(f"Modified requirement: '{request.modified_requirement}'")
        logger.info(f"Generation config: temp={request.temperature}, top_p={request.top_p}, max_tokens={request.max_new_tokens}")
        
        # Tokenize and generate
        inputs = tok(prompt, return_tensors="pt").to(mdl.device)
        
        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=True,
                temperature=request.temperature,
                top_p=request.top_p,
            )
        
        # Decode generated text (only the new tokens)
        generated_text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        logger.info(f"Qwen generated text: {generated_text}")
        
        # Parse scores using the parse_scores function
        scores = parse_scores(generated_text)
        
        preservation_correctness = scores["preservation_correctness"]
        change_correctness = scores["change_correctness"]
        
        # Extract explanation from <THINK> tags if present
        explanation = ""
        think_match = re.search(r'<THINK>(.*?)</THINK>', generated_text, flags=re.DOTALL | re.IGNORECASE)
        if think_match:
            explanation = think_match.group(1).strip()
        else:
            explanation = generated_text  # Fallback to full generated text
        
        logger.info(f"Parsed scores from Qwen - Preservation: {preservation_correctness}, Change: {change_correctness}")
        
        return RequirementAnalysisResponse(
            success=True,
            preservation_correctness=preservation_correctness,
            change_correctness=change_correctness,
            analysis_text=explanation,
            model_name=request.model_name
        )
            
    except Exception as e:
        logger.error(f"Error analyzing requirement: {str(e)}", exc_info=True)
        # Return error - no hardcoded fallback values
        return RequirementAnalysisResponse(
            success=False,
            preservation_correctness=0.0,  # Indicate failure with 0
            change_correctness=0.0,  # Indicate failure with 0
            analysis_text=f"Failed to get scores from Qwen model: {str(e)}",
            model_name=request.model_name,
            error=str(e)
        )


@app.post("/api/inference/comprehensive-analysis", response_model=ComprehensiveRequirementAnalysisResponse)
async def comprehensive_requirement_analysis(request: ComprehensiveRequirementAnalysisRequest):
    """
    Comprehensive requirement analysis workflow:
    1. Use Qwen to generate modified requirement from original + modification instruction
    2. Extract preservation_correctness and change_correctness from Qwen output
    3. Detect quality issues in the Qwen-generated modified requirement using SetFit
    4. Return everything together
    
    Example:
    {
        "original_requirement": "The system shall send email notifications.",
        "modification_instruction": "Use in-app notifications instead of email."
    }
    """
    try:
        # Step 1: Load Qwen model
        pipe = load_qwen_pipeline(request.model_name)
        
        # Step 2: Format prompt for Qwen
        prompt = f"""Original: {request.original_requirement}
Modification: {request.modification_instruction}
Modified:"""
        
        messages = [{"role": "user", "content": prompt}]
        
        # Step 3: Qwen generates modified requirement and provides evaluation
        # Build generation config dynamically
        gen_config = {
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.do_sample,
        }
        
        # Only add sampling parameters if sampling is enabled
        if request.do_sample:
            gen_config["temperature"] = request.temperature
            gen_config["top_p"] = request.top_p
            gen_config["top_k"] = request.top_k
        
        result = pipe(messages, **gen_config)
        
        logger.info(f"Qwen comprehensive analysis result: {result}")
        
        if not result or len(result) == 0:
            raise ValueError("Empty result from Qwen model")
        
        full_response = result[0].get("generated_text", "")
        
        # Step 4: Parse Qwen output (modified requirement is in user message, scores in assistant message)
        import re
        import json
        
        # Extract modified requirement from user message (after "Modified:")
        modified_requirement = request.original_requirement  # fallback
        assistant_text = ""
        
        if isinstance(full_response, list):
            # Find user message with the modified requirement
            for msg in full_response:
                if msg.get("role") == "user":
                    user_content = msg.get("content", "")
                    # Extract text after "Modified:"
                    if "Modified:" in user_content:
                        parts = user_content.split("Modified:", 1)
                        if len(parts) > 1:
                            modified_requirement = parts[1].strip()
                            logger.info(f"Extracted modified requirement from user message: {modified_requirement}")
            
            # Find assistant message with JSON scores
            for msg in reversed(full_response):
                if msg.get("role") == "assistant":
                    assistant_text = msg.get("content", "").strip()
                    break
            
            if not assistant_text:
                assistant_text = str(full_response)
        else:
            assistant_text = str(full_response).strip()
        
        logger.info(f"Assistant response text: {assistant_text}")
        
        # Remove <think> tags if present
        clean_text = re.sub(r'<think>.*?</think>', '', assistant_text, flags=re.DOTALL).strip()
        
        # Find JSON object with required fields
        json_match = re.search(r'\{[^\}]*"preservation_correctness"[^\}]*"change_correctness"[^\}]*\}', clean_text, re.DOTALL)
        
        if not json_match:
            raise ValueError(f"Could not find JSON in Qwen output: {clean_text}")
        
        json_str = json_match.group(0)
        logger.info(f"Extracted JSON: {json_str}")
        
        parsed = json.loads(json_str)
        
        # Extract scores (Qwen returns them already in 0-1 range)
        if "preservation_correctness" not in parsed or "change_correctness" not in parsed:
            raise ValueError(f"Missing required fields: {parsed}")
        
        preservation_correctness = float(parsed["preservation_correctness"])
        change_correctness = float(parsed["change_correctness"])
        
        # Extract comments from Qwen
        qwen_comments = []
        if "comments" in parsed:
            comments_data = parsed["comments"]
            if isinstance(comments_data, list):
                qwen_comments = comments_data
            else:
                qwen_comments = [str(comments_data)]
        
        logger.info(f"Extracted modified requirement: {modified_requirement}")
        
        # Step 5: Detect issues in the Qwen-generated modified requirement using SetFit
        try:
            model_data = load_setfit_model(request.quality_model_name)
            model = model_data["model"]
            id2label = model_data["id2label"]
            thresholds = model_data["thresholds"]
            
            probas = np.array(model.predict_proba([modified_requirement]))
            preds = (probas >= thresholds.reshape(1, -1)).astype(int)
            
            detected_issues = [id2label[i] for i, v in enumerate(preds[0]) if v == 1]
            logger.info(f"Detected issues in modified requirement: {detected_issues}")
        except Exception as e:
            logger.error(f"Error detecting issues: {e}", exc_info=True)
            detected_issues = []
        
        return ComprehensiveRequirementAnalysisResponse(
            success=True,
            modified_requirement=modified_requirement,
            preservation_correctness=preservation_correctness,
            change_correctness=change_correctness,
            detected_issues=detected_issues,
            comments=qwen_comments,
            model_name=request.model_name
        )
        
    except Exception as e:
        logger.error(f"Comprehensive analysis error: {str(e)}", exc_info=True)
        return ComprehensiveRequirementAnalysisResponse(
            success=False,
            modified_requirement="",
            preservation_correctness=0.0,
            change_correctness=0.0,
            detected_issues=[],
            comments=[],
            model_name=request.model_name,
            error=str(e)
        )


@app.post("/api/inference/requirement-modification", response_model=RequirementModificationResponse)
async def requirement_modification(request: RequirementModificationRequest):
    """
    Modify a requirement using fine-tuned Qwen model
    
    Example:
    {
        "original_requirement": "Send email notifications.",
        "modification_instruction": "Use SMS instead."
    }
    """
    try:
        # Load model
        pipe = load_qwen_pipeline(request.model_name)
        
        # Format prompt
        prompt = (
            f"Original: {request.original_requirement}\n"
            f"Modification: {request.modification_instruction}\n"
            f"Modified:"
        )
        
        # Use the simplified message format as specified
        messages = [{"role": "user", "content": prompt}]
        
        # Generate using the pipeline with non-deterministic sampling parameters
        # Build generation config dynamically
        gen_config = {
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.do_sample,
        }
        
        # Only add sampling parameters if sampling is enabled
        if request.do_sample:
            gen_config["temperature"] = request.temperature
            gen_config["top_p"] = request.top_p
            gen_config["top_k"] = request.top_k
        
        result = pipe(messages, **gen_config)
        
        # Extract result
        if result and len(result) > 0:
            # The result is a list with generated text
            full_response = result[0].get("generated_text", "")
            
            # If the response is a list of messages (chat format), extract the last assistant message
            if isinstance(full_response, list):
                for msg in reversed(full_response):
                    if msg.get("role") == "assistant":
                        modified_requirement = msg.get("content", "").strip()
                        break
                else:
                    modified_requirement = str(full_response)
            else:
                modified_requirement = str(full_response).strip()
            
            # Clean up if needed - remove the prompt if it's included
            if "Modified:" in modified_requirement:
                parts = modified_requirement.split("Modified:", 1)
                if len(parts) > 1:
                    modified_requirement = parts[1].strip()
            
            return RequirementModificationResponse(
                success=True,
                original_requirement=request.original_requirement,
                modification_instruction=request.modification_instruction,
                modified_requirement=modified_requirement,
                model_name=request.model_name
            )
        else:
            raise ValueError("No response generated")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return RequirementModificationResponse(
            success=False,
            original_requirement=request.original_requirement,
            modification_instruction=request.modification_instruction,
            modified_requirement="",
            model_name=request.model_name,
            error=str(e)
        )


# ============================================================================
# Fuzzy Inference System Endpoints
# ============================================================================

@app.post("/api/fuzzy/defect-severity", response_model=DefectSeverityResponse)
async def calculate_defect_severity(request: DefectSeverityRequest):
    """
    Calculate defect severity using fuzzy inference system
    
    Inputs:
    - subjective: Subjective defect score (0-1)
    - ambiguous: Ambiguity level (0-1)
    - nonverifiable: Non-verifiability level (0-1)
    - negative: Negative phrasing level (0-1)
    - vague: Vagueness level (0-1)
    
    Output:
    - defect_severity: Overall defect severity (0-1)
    
    Example:
    {
        "subjective": 0.3,
        "ambiguous": 0.7,
        "nonverifiable": 0.6,
        "negative": 0.2,
        "vague": 0.8
    }
    """
    try:
        fis = ensure_fuzzy_system('defect_severity')
        
        # Reset the simulation to clear any previous state
        fis.reset()
        
        # Set inputs
        fis.input['subjective'] = request.subjective
        fis.input['ambiguous'] = request.ambiguous
        fis.input['nonverifiable'] = request.nonverifiable
        fis.input['negative'] = request.negative
        fis.input['vague'] = request.vague
        
        # Log inputs (the 5 requirement quality outputs)
        logger.info("=" * 60)
        logger.info("DEFECT SEVERITY FIS CALCULATION")
        logger.info("=" * 60)
        logger.info(f"Input - Subjective: {request.subjective}")
        logger.info(f"Input - Ambiguous: {request.ambiguous}")
        logger.info(f"Input - Non-verifiable: {request.nonverifiable}")
        logger.info(f"Input - Negative: {request.negative}")
        logger.info(f"Input - Vague: {request.vague}")
        
        # Compute
        fis.compute()
        
        defect_severity = float(fis.output['defect_severity'])
        
        # Determine label
        thresholds = {
            "low": (0, 0.4),
            "average": (0.35, 0.65),
            "high": (0.6, 1.0)
        }
        label = get_label(defect_severity, thresholds)
        
        # Log outputs
        logger.info(f"Output - defect_severity (score): {defect_severity}")
        logger.info(f"Output - defect_severity_label: {label}")
        logger.info("=" * 60)
        
        return DefectSeverityResponse(
            success=True,
            defect_severity=defect_severity,
            defect_severity_label=label,
            inputs={
                "subjective": request.subjective,
                "ambiguous": request.ambiguous,
                "nonverifiable": request.nonverifiable,
                "negative": request.negative,
                "vague": request.vague
            }
        )
        
    except Exception as e:
        logger.error(f"Defect severity calculation error: {str(e)}", exc_info=True)
        return DefectSeverityResponse(
            success=False,
            defect_severity=0.0,
            defect_severity_label="error",
            inputs={},
            error=str(e)
        )


@app.post("/api/fuzzy/correctness", response_model=CorrectnessResponse)
async def calculate_correctness(request: CorrectnessRequest):
    """
    Calculate correctness using fuzzy inference system
    
    Inputs:
    - preservation_correctness: How well original requirement is preserved (0-1)
    - change_correctness: How correctly changes were applied (0-1)
    
    Output:
    - correctness: Overall correctness (0-1)
    
    Example:
    {
        "preservation_correctness": 0.8,
        "change_correctness": 0.7
    }
    """
    try:
        fis = ensure_fuzzy_system('correctness')
        
        # Reset the simulation to clear any previous state
        fis.reset()
        
        # Set inputs
        fis.input['preservation'] = request.preservation_correctness
        fis.input['change'] = request.change_correctness
        
        # Compute
        fis.compute()
        
        # Access output - the output should be available after compute
        correctness = float(fis.output['correctness'])
        
        # Determine label using 5 levels (Very low, Low, Average, High, Very high) for Quality of Change
        thresholds_5level = {
            "very_low": (0, 0.25),
            "low": (0.15, 0.45),
            "average": (0.35, 0.65),
            "high": (0.55, 0.85),
            "very_high": (0.75, 1.0)
        }
        label_5level = get_label(correctness, thresholds_5level)
        
        # Determine 3-level label (Low, Average, High) for Correctness display
        thresholds_3level = {
            "low": (0, 0.4),
            "average": (0.35, 0.65),
            "high": (0.6, 1.0)
        }
        label_3level = get_label(correctness, thresholds_3level)
        
        return CorrectnessResponse(
            success=True,
            correctness=correctness,
            correctness_label=label_5level,
            correctness_three_level_label=label_3level,
            inputs={
                "preservation_correctness": request.preservation_correctness,
                "change_correctness": request.change_correctness
            }
        )
        
    except Exception as e:
        logger.error(f"Correctness calculation error: {str(e)}", exc_info=True)
        return CorrectnessResponse(
            success=False,
            correctness=0.0,
            correctness_label="error",
            correctness_three_level_label="low",
            inputs={},
            error=str(e)
        )


@app.post("/api/fuzzy/requirement-quality", response_model=RequirementQualityFuzzyResponse)
async def calculate_requirement_quality_fuzzy(request: RequirementQualityFuzzyRequest):
    """
    Calculate overall requirement quality using fuzzy inference system
    
    This combines defect severity and correctness to produce overall quality score.
    
    Inputs:
    - defect_severity: Defect severity from defect-severity endpoint (0-1)
    - correctness: Correctness from correctness endpoint (0-1)
    
    Output:
    - requirement_quality: Overall quality (0-1)
    
    Example:
    {
        "defect_severity": 0.3,
        "correctness": 0.8
    }
    
    Note: You can chain the other endpoints to get these values, or provide them directly.
    """
    try:
        fis = ensure_fuzzy_system('requirement_quality')
        
        # Reset the simulation to clear any previous state
        fis.reset()
        
        # Set inputs
        fis.input['defect_severity'] = request.defect_severity
        fis.input['correctness'] = request.correctness
        
        # Log inputs
        logger.info("=" * 60)
        logger.info("REQUIREMENT QUALITY FIS CALCULATION")
        logger.info("=" * 60)
        logger.info(f"Input - defect_severity: {request.defect_severity}")
        logger.info(f"Input - correctness: {request.correctness}")
        
        # Compute
        fis.compute()
        
        req_quality = float(fis.output['requirement_quality'])
        
        # Determine label
        thresholds = {
            "very_low": (0, 0.25),
            "low": (0.15, 0.45),
            "average": (0.35, 0.65),
            "high": (0.55, 0.85),
            "very_high": (0.75, 1.0)
        }
        label = get_label(req_quality, thresholds)
        
        # Log outputs
        logger.info(f"Output - requirement_quality (score): {req_quality}")
        logger.info(f"Output - requirement_quality_label: {label}")
        logger.info("=" * 60)
        
        return RequirementQualityFuzzyResponse(
            success=True,
            requirement_quality=req_quality,
            requirement_quality_label=label,
            inputs={
                "defect_severity": request.defect_severity,
                "correctness": request.correctness
            }
        )
        
    except Exception as e:
        logger.error(f"Requirement quality calculation error: {str(e)}", exc_info=True)
        return RequirementQualityFuzzyResponse(
            success=False,
            requirement_quality=0.0,
            requirement_quality_label="error",
            inputs={},
            error=str(e)
        )


# ============================================================================
# Run with: uvicorn app_simple:app --reload --port 8001
# ============================================================================

