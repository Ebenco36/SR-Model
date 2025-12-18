from typing import Dict, List

# Put ONLY the atomic labels you will annotate as spans:
ATOMIC_LABELS: List[str] = [
    "REVIEW_TYPE",
    "STUDY_DESIGN",
    "N_STUDIES",
    "SEARCH_DATE",
    "QUALITY_TOOL",
    "PATHOGEN",
    "CANCER",
    "CONDITION",
    "HPV_TYPE",
    "AGE_GROUP",
    "GENDER",
    "SPECIAL_POP",
    "RISK_GROUP",
    "SAFETY",
    "ACCEPTANCE",
    "EFFICACY",
    "IMMUNOGENICITY",
    "COVERAGE",
    "ECONOMIC",
    "ADMINISTRATION",
    "ETHICAL",
    "LOGISTICS",
    "MODELLING",
    "CLINICAL",
    "LESION",
    "COUNTRY",
    "REGION",
    "WHO_REGION",
    "INCOME_GROUP",
    "VACCINE_TYPE",
    "VACCINE_BRAND",
    "DOSE",
    "ROUTE",
    "PROGRAM",
    "COMPONENT",
    "SCREENING",
    "COMBINATION",
    "BARRIER",
    "FACILITATOR",
    "DATABASE",
    "SEARCH_TERMS",
    "INCLUSION",
    "ANALYSIS",
    "PERIOD",
    "FOLLOWUP",
    "TIMING",
    "SAMPLE_SIZE",
    "PERCENT",
    "COST",
    "QALY",
    "ICER",
    "EFFECT_MEASURE",
    "EFFECT_VALUE",
    "CI",
    "PVALUE",
]

def build_label2id() -> Dict[str, int]:
    # BIO scheme
    labels = ["O"]
    for lab in ATOMIC_LABELS:
        labels.append(f"B-{lab}")
        labels.append(f"I-{lab}")
    return {lab: i for i, lab in enumerate(labels)}
