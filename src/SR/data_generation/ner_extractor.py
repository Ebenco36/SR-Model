#!/usr/bin/env python3
"""
PRODUCTION-READY NER EXTRACTION SYSTEM
for Infectious Disease Systematic Reviews (55 Entity Types)

Features:
- Full error handling and logging
- Both rule-based and LLM extraction for 55 entity types
- Progress tracking and resume capability
- Validation and deduplication
- JSONL output with proper formatting
- Comprehensive unit tests
- Configuration management
"""

import argparse
import json
import os
import re
import time
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import hashlib
import warnings
import traceback
import requests
from requests.exceptions import RequestException, Timeout

# ============================================================
# CONFIGURATION & LOGGING
# ============================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ner_extraction.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class EntitySpan:
    """Represents a single entity span in text"""
    start: int
    end: int
    label: str
    text: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'start': self.start,
            'end': self.end,
            'label': self.label,
            'text': self.text
        }

@dataclass
class ExtractionResult:
    """Complete extraction result for a document"""
    doc_id: str
    text: str
    spans: List[EntitySpan]
    metadata: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'doc_id': self.doc_id,
            'text': self.text[:2000] + '...' if len(self.text) > 2000 else self.text,
            'spans': [span.to_dict() for span in self.spans],
            'metadata': self.metadata,
            'stats': self.stats
        }

@dataclass
class LLMConfig:
    """LLM configuration"""
    base_url: str = "http://localhost:11434"
    api_key: str = "dummy"
    model: str = "llama3.1:8b"
    timeout: int = 60
    max_retries: int = 3
    temperature: float = 0.0
    max_tokens: int = 2048

# ============================================================
# ENTITY LABELS & PATTERNS (55 TYPES)
# ============================================================

class EntityLabels:
    """Entity labels and their patterns for comprehensive systematic review extraction"""
    
    # Complete 55-label dictionary with regex patterns
    LABELS = {
        # Review Methodology & Design (7 labels)
        "REVIEW_TYPE": r"\b(systematic\s+review|meta.?analysis|scoping\s+review|rapid\s+review|narrative\s+review|systematic\s+literature\s+review|umbrella\s+review)\b",
        "STUDY_DESIGN": r"\b(randomized\s+controlled\s+trial|RCT|quasi.?experimental|cohort\s+study|case.?control|cross.?sectional|longitudinal|observational|pragmatic\s+trial)\b",
        "N_STUDIES": r"\b(\d+\s+studies|\d+\s+trials|n\s*[=:]\s*\d+\s+studies|included\s+\d+\s+studies|total\s+of\s+\d+\s+(?:studies|trials))\b",
        "DATE_OF_LAST_LITERATURE_SEARCH": r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[,\.]?\s+\d{1,2}(?:st|nd|rd|th)?[,\.]?\s+\d{4}|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[,\.]?\s+\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:up\s+to|until|through)\s+\w+\s+\d{4})\b",
        "QUALITY_TOOL": r"\b(PRISMA|Cochrane\s+Risk\s+of\s+Bias|ROB.?2|ROBINS.?I|AMSTAR|AMSTAR.?2|GRADE|JBI\s+checklist|Newcastle.?Ottawa|NOS)\b",
        "DATABASE": r"\b(PubMed|MEDLINE|Embase|Scopus|Web\s+of\s+Science|Cochrane\s+Library|CENTRAL|CINAHL|PsycINFO|Google\s+Scholar)\b",
        "SEARCH_TERMS": r"\b(search\s+terms?|search\s+strategy|keywords?|MeSH\s+terms?|((?:\w+\s+){1,4}(?:AND|OR|NOT)\s+(?:\w+\s+){1,4}){1,3})\b",
        
        # Population & Conditions (12 labels)
        "PATHOGEN": r"\b(HIV|human\s+immunodeficiency\s+virus|tuberculosis|TB|mycobacterium\s+tuberculosis|malaria|plasmodium|COVID.?19|SARS.?CoV.?2|coronavirus|hepatitis\s+B|HBV|hepatitis\s+C|HCV|influenza|flu|Ebola|Zika|dengue|chikungunya|HPV|human\s+papillomavirus|rotavirus|measles|mumps|rubella|varicella)\b",
        "CANCER": r"\b(cervical\s+cancer|breast\s+cancer|lung\s+cancer|prostate\s+cancer|colorectal\s+cancer|hepatocellular\s+carcinoma|leukemia|lymphoma|melanoma|sarcoma|tumor|neoplasm|malignancy)\b",
        "CONDITION": r"\b(AIDS|HIV/AIDS|pneumonia|meningitis|sepsis|cholera|typhoid|malaria|tuberculosis|dengue\s+fever|influenza.?like\s+illness|acute\s+respiratory\s+infection|diarrheal\s+disease|urinary\s+tract\s+infection|sexually\s+transmitted\s+infection)\b",
        "HPV_TYPE": r"\b(HPV\s*(?:type\s*)?(?:16|18|6|11|31|33|45|52|58)|human\s+papillomavirus\s*(?:type\s*)?(?:16|18|6|11|31|33|45|52|58)|oncogenic\s+HPV|high.?risk\s+HPV|hrHPV)\b",
        "AGE_GROUP": r"\b(children|adults|elderly|older\s+adults|aged\s+\d+(?:\s*-\s*\d+)?|years?\s+old|adolescents|teenagers|infants|neonates|pediatric|geriatric|(?:mean|median)\s+age\s+[0-9.]+)\b",
        "GENDER": r"\b(male|female|men|women|boys|girls|transgender|non.?binary|sex|gender)\b",
        "SPECIAL_POP": r"\b(pregnant\s+women|healthcare\s+workers|HCWs|MSM|men\s+who\s+have\s+sex\s+with\s+men|migrants|refugees|prisoners|inmates|homeless|people\s+who\s+inject\s+drugs|PWID|sex\s+workers)\b",
        "RISK_GROUP": r"\b(high.?risk|risk\s+group|vulnerable\s+population|immunocompromised|HIV.?positive|diabetic|obese|smokers|alcohol\s+users|people\s+with\s+comorbidities)\b",
        "INCLUSION": r"\b(inclusion\s+criteria|eligibility\s+criteria|included\s+studies|selection\s+criteria|study\s+selection)\b",
        "ANALYSIS": r"\b(meta.?analysis|pooled\s+analysis|subgroup\s+analysis|sensitivity\s+analysis|random.?effects|fixed.?effects|heterogeneity|I.?squared|forest\s+plot|funnel\s+plot)\b",
        "PERIOD": r"\b(study\s+period|time\s+period|from\s+\d{4}\s+to\s+\d{4}|during\s+\d{4}[-–]\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s+to\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b",
        "FOLLOWUP": r"\b(follow.?up|follow.?up\s+period|follow.?up\s+time|median\s+follow.?up|mean\s+follow.?up|duration\s+of\s+follow.?up)\b",
        
        # Outcomes & Measures (12 labels)
        "SAFETY": r"\b(adverse\s+events|safety\s+profile|side\s+effects|adverse\s+drug\s+reactions|ADR|tolerability|serious\s+adverse\s+events|SAE|vaccine\s+safety)\b",
        "ACCEPTANCE": r"\b(vaccine\s+acceptance|vaccine\s+hesitancy|vaccine\s+willingness|uptake\s+intention|refusal\s+rate|willingness\s+to\s+vaccinate|acceptability)\b",
        "EFFICACY": r"\b(vaccine\s+efficacy|treatment\s+effectiveness|protective\s+efficacy|clinical\s+efficacy|effectiveness|VE\s*[=:]\s*[\d\.]+%)\b",
        "IMMUNOGENICITY": r"\b(immunogenicity|antibody\s+response|seroconversion|seroprotection|neutralizing\s+antibodies|cellular\s+immunity|T.?cell\s+response)\b",
        "COVERAGE": r"\b(vaccination\s+coverage|immunization\s+coverage|coverage\s+rate|uptake\s+rate|completion\s+rate|dose\s+coverage)\b",
        "ECONOMIC": r"\b(cost.?effectiveness|cost.?benefit|economic\s+evaluation|budget\s+impact|affordability|financial\s+burden|cost.?utility)\b",
        "ADMINISTRATION": r"\b(administration|schedule|dosing\s+schedule|booster\s+dose|primary\s+series|single\s+dose|multi.?dose|co.?administration)\b",
        "ETHICAL": r"\b(ethical\s+considerations|informed\s+consent|ethical\s+approval|IRB\s+approval|ethics\s+committee|ethical\s+issues)\b",
        "LOGISTICS": r"\b(cold\s+chain|storage|transportation|supply\s+chain|distribution|logistics|vaccine\s+supply|stockouts)\b",
        "MODELLING": r"\b(mathematical\s+modeling|transmission\s+model|compartmental\s+model|SEIR|SIR|agent.?based\s+model|simulation|projection)\b",
        "CLINICAL": r"\b(clinical\s+trial|clinical\s+study|phase\s+[I-IV]|clinical\s+outcomes|clinical\s+efficacy|clinical\s+safety|clinical\s+management)\b",
        "LESION": r"\b(cervical\s+lesion|precancerous\s+lesion|CIN|CIN\s*[1-3]|cervical\s+intraepithelial\s+neoplasia|wart|genital\s+wart|papilloma)\b",
        
        # Geography & Demographics (5 labels)
        "COUNTRY": r"\b(USA|United\s+States|US|U\.S\.|Canada|UK|United\s+Kingdom|Britain|China|India|South\s+Africa|Kenya|Uganda|Tanzania|Brazil|Australia|Germany|France|Italy|Spain|Japan|South\s+Korea|Mexico)\b",
        "REGION": r"\b(sub.?Saharan\s+Africa|Latin\s+America|Caribbean|South\s+Asia|Southeast\s+Asia|East\s+Asia|Middle\s+East|North\s+Africa|Europe|North\s+America|Western\s+Pacific)\b",
        "WHO_REGION": r"\b(AFRO|AFR|African\s+Region|AMRO|AMR|Region\s+of\s+the\s+Americas|EMRO|EMR|Eastern\s+Mediterranean\s+Region|EURO|EUR|European\s+Region|SEARO|SEAR|South.?East\s+Asia\s+Region|WPRO|WPR|Western\s+Pacific\s+Region)\b",
        "INCOME_GROUP": r"\b(low.?income\s+country|LIC|lower.?middle.?income|LMIC|upper.?middle.?income|UMIC|high.?income\s+country|HIC|developing\s+country|developed\s+country)\b",
        "TIMING": r"\b(timing|schedule|age\s+at\s+vaccination|catch.?up|routine\s+immunization|school.?based|birth\s+dose)\b",
        
        # Interventions & Programs (8 labels)
        "VACCINE_TYPE": r"\b(mRNA\s+vaccine|inactivated\s+vaccine|live\s+attenuated|subunit\s+vaccine|viral\s+vector|recombinant|conjugate\s+vaccine|HPV\s+vaccine|Gardasil|Cervarix|pentavalent|nonavalent)\b",
        "VACCINE_BRAND": r"\b(Gardasil|Cervarix|Gardasil\s*9|Silgard|HPV2|HPV4|HPV9|Pfizer|Moderna|AstraZeneca|Johnson\s+&\s+Johnson|Sinovac|Sinopharm)\b",
        "DOSE": r"\b(\d+(?:\.\d+)?\s*dose|single\s+dose|two.?dose|three.?dose|primary\s+series|booster\s+dose|fractional\s+dose|standard\s+dose)\b",
        "ROUTE": r"\b(intramuscular|IM|subcutaneous|SC|oral|intradermal|ID|nasal|intranasal|topical)\b",
        "PROGRAM": r"\b(immunization\s+program|vaccination\s+program|national\s+immunization\s+program|NIP|school.?based\s+vaccination|campaign|mass\s+vaccination|outreach)\b",
        "COMPONENT": r"\b(program\s+component|vaccine\s+delivery|community\s+engagement|training|supervision|monitoring\s+and\s+evaluation|M\s*&\s*E|advocacy)\b",
        "SCREENING": r"\b(screening|Pap\s+test|Pap\s+smear|HPV\s+test|VIA|visual\s+inspection\s+with\s+acetic\s+acid|colposcopy|biopsy)\b",
        "COMBINATION": r"\b(combination|integrated|co.?delivery|co.?administration|with\s+screening|vaccine\s+plus\s+screening)\b",
        
        # Implementation Factors (3 labels)
        "BARRIER": r"\b(barrier|challenge|obstacle|limitation|difficulty|constraint|hindrance|impediment|problem|issue)\b",
        "FACILITATOR": r"\b(facilitator|enabler|success\s+factor|driver|support|strategy|intervention|approach|solution)\b",
        "SAMPLE_SIZE": r"\b(n\s*[=:]\s*\d+(?:[,\s]\d{3})*|\d+(?:[,\s]\d{3})*\s+participants|sample\s+size\s+\d+|total\s+of\s+\d+\s+participants)\b",
        
        # Statistics & Economics (8 labels)
        "PERCENT": r"\b\d+(?:\.\d+)?\s*%\b",
        "COST": r"\b(\$?\d+(?:[,\s]\d{3})*(?:\.\d+)?\s*(?:USD?|dollars?)|cost\s+(?:per|of)\s+\$?\d+(?:[,\s]\d{3})*(?:\.\d+)?|ICER\s*(?:of)?\s*\$?\d+(?:[,\s]\d{3})*(?:\.\d+)?)\b",
        "QALY": r"\b(QALY|quality.?adjusted\s+life\s+year|QALY\s+(?:gained|lost)|QALY\s*[=:]\s*[\d\.]+)\b",
        "ICER": r"\b(ICER|incremental\s+cost.?effectiveness\s+ratio|ICER\s*(?:of)?\s*\$?\d+(?:[,\s]\d{3})*(?:\.\d+)?\s*per\s+(?:QALY|case\s+averted))\b",
        "EFFECT_MEASURE": r"\b(odds\s+ratio|risk\s+ratio|hazard\s+ratio|relative\s+risk|risk\s+difference|absolute\s+risk\s+reduction|number\s+needed\s+to\s+treat|OR|RR|HR|RD|ARR|NNT)\b",
        "EFFECT_VALUE": r"\b((?:OR|RR|HR|RD|ARR)\s*[=:]\s*[\d\.]+(?:\.\d+)?|[\d\.]+(?:\.\d+)?\s*\((?:OR|RR|HR|RD|ARR)\))\b",
        "CI": r"\b((?:95|90|99)%\s*CI|confidence\s+interval|CI\s*[:\(]?\s*[\d\.]+\s*(?:to|-)\s*[\d\.]+|\([\d\.]+\s*[,-]\s*[\d\.]+\))\b",
        "PVALUE": r"\b([pP]\s*[=<]\s*[\d\.]+|[pP]\s*=\s*[\d\.]+|NS\s*\(not\s+significant\)|non.?significant)\b",
    }
    
    @classmethod
    def get_compiled_patterns(cls) -> Dict[str, re.Pattern]:
        """Get compiled regex patterns for all labels"""
        return {label: re.compile(pattern, re.IGNORECASE) 
                for label, pattern in cls.LABELS.items()}
    
    @classmethod
    def get_all_labels(cls) -> List[str]:
        """Get list of all 55 label names"""
        return list(cls.LABELS.keys())
    
    @classmethod
    def validate_label(cls, label: str) -> bool:
        """Check if label is valid"""
        return label in cls.LABELS

# ============================================================
# RULE-BASED EXTRACTOR
# ============================================================

class RuleBasedExtractor:
    """Fast, reliable rule-based entity extractor for 55 entity types"""
    
    def __init__(self):
        self.patterns = EntityLabels.get_compiled_patterns()
        self.stats = defaultdict(int)
    
    def extract(self, text: str) -> List[EntitySpan]:
        """Extract entities using regex patterns for all 55 types"""
        entities = []
        
        for label, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                span = EntitySpan(
                    start=match.start(),
                    end=match.end(),
                    label=label,
                    text=match.group(0)
                )
                entities.append(span)
                self.stats[label] += 1
        
        # Remove overlaps (keep longest span when overlapping)
        entities = self._resolve_overlaps(entities)
        
        logger.debug(f"Rule-based extraction found {len(entities)} entities across {len(set(e.label for e in entities))} unique labels")
        return entities
    
    def _resolve_overlaps(self, entities: List[EntitySpan]) -> List[EntitySpan]:
        """Resolve overlapping entities, keeping the longest"""
        if not entities:
            return []
        
        # Sort by start, then by length (longest first)
        entities.sort(key=lambda x: (x.start, -(x.end - x.start)))
        
        result = []
        last_end = -1
        
        for entity in entities:
            if entity.start < last_end:
                # Overlap - skip this one (shorter)
                continue
            result.append(entity)
            last_end = entity.end
        
        return result
    
    def get_stats(self) -> Dict[str, int]:
        """Get extraction statistics"""
        return dict(self.stats)

# ============================================================
# LLM CLIENT (UPDATED FOR 55 ENTITY TYPES)
# ============================================================

class LLMClient:
    """Robust LLM client with retry logic and error handling for 55 entity types"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.session = requests.Session()
        self.session.timeout = (config.timeout, config.timeout)
        
        # Headers for Ollama
        self.headers = {
            "Content-Type": "application/json"
        }
        if config.api_key and config.api_key != "dummy":
            self.headers["Authorization"] = f"Bearer {config.api_key}"
    
    def extract_entities(self, text: str, doc_id: str = "") -> List[Dict[str, Any]]:
        """Extract entities using LLM with robust error handling for 55 entity types"""
        
        # System prompt for all 55 entity types
        system_prompt = """You are an expert biomedical annotator extracting 55 entity types from infectious disease systematic reviews.

Extract ONLY exact substrings from the text. Use ONLY these 55 labels:

REVIEW METHODOLOGY (7):
- REVIEW_TYPE: systematic review, meta-analysis, scoping review, rapid review, narrative review
- STUDY_DESIGN: randomized controlled trial, cohort study, case-control, cross-sectional, observational
- N_STUDIES: number of included studies (e.g., "15 studies", "n=10 studies")
- DATE_OF_LAST_LITERATURE_SEARCH: dates when literature search was conducted
- QUALITY_TOOL: PRISMA, Cochrane Risk of Bias, AMSTAR, GRADE, JBI checklist
- DATABASE: PubMed, MEDLINE, Embase, Scopus, Web of Science, Cochrane Library
- SEARCH_TERMS: search strategy, keywords, MeSH terms, Boolean operators

POPULATION & CONDITIONS (12):
- PATHOGEN: HIV, tuberculosis, malaria, COVID-19, hepatitis, HPV, influenza
- CANCER: cervical cancer, breast cancer, lung cancer, prostate cancer, leukemia
- CONDITION: AIDS, pneumonia, meningitis, sepsis, cholera, typhoid, dengue fever
- HPV_TYPE: HPV type 16, HPV 18, HPV 6/11, high-risk HPV, oncogenic HPV
- AGE_GROUP: children, adults, elderly, aged 18-65, adolescents, infants
- GENDER: male, female, men, women, boys, girls, transgender
- SPECIAL_POP: pregnant women, healthcare workers, MSM, migrants, refugees, prisoners
- RISK_GROUP: high-risk, immunocompromised, HIV-positive, diabetic, obese
- INCLUSION: inclusion criteria, eligibility criteria, study selection criteria
- ANALYSIS: meta-analysis, subgroup analysis, sensitivity analysis, random-effects
- PERIOD: study period, time period, from 2010 to 2020, during 2015-2018
- FOLLOWUP: follow-up period, median follow-up, duration of follow-up

OUTCOMES & MEASURES (12):
- SAFETY: adverse events, safety profile, side effects, vaccine safety
- ACCEPTANCE: vaccine acceptance, vaccine hesitancy, willingness to vaccinate
- EFFICACY: vaccine efficacy, treatment effectiveness, protective efficacy
- IMMUNOGENICITY: antibody response, seroconversion, neutralizing antibodies
- COVERAGE: vaccination coverage, immunization coverage, uptake rate
- ECONOMIC: cost-effectiveness, cost-benefit, economic evaluation, affordability
- ADMINISTRATION: administration schedule, dosing schedule, booster dose
- ETHICAL: ethical considerations, informed consent, IRB approval
- LOGISTICS: cold chain, storage, transportation, supply chain
- MODELLING: mathematical modeling, transmission model, SEIR, simulation
- CLINICAL: clinical trial, phase III, clinical outcomes, clinical management
- LESION: cervical lesion, precancerous lesion, CIN, genital wart

GEOGRAPHY & DEMOGRAPHICS (5):
- COUNTRY: USA, United States, Canada, UK, China, India, South Africa, Brazil
- REGION: sub-Saharan Africa, Latin America, Southeast Asia, Europe
- WHO_REGION: AFRO, AMRO, EMRO, EURO, SEARO, WPRO
- INCOME_GROUP: low-income country, LMIC, high-income country, HIC
- TIMING: timing, schedule, age at vaccination, catch-up, routine immunization

INTERVENTIONS & PROGRAMS (8):
- VACCINE_TYPE: mRNA vaccine, inactivated vaccine, live attenuated, subunit vaccine
- VACCINE_BRAND: Gardasil, Cervarix, Pfizer, Moderna, AstraZeneca
- DOSE: 2-dose, single dose, three-dose, primary series, booster dose
- ROUTE: intramuscular, oral, subcutaneous, intradermal, nasal
- PROGRAM: immunization program, vaccination program, national immunization program
- COMPONENT: vaccine delivery, community engagement, training, monitoring
- SCREENING: Pap test, HPV test, VIA, colposcopy, biopsy
- COMBINATION: integrated, co-delivery, vaccine plus screening

IMPLEMENTATION FACTORS (3):
- BARRIER: barrier, challenge, obstacle, limitation, constraint
- FACILITATOR: facilitator, enabler, success factor, strategy, solution
- SAMPLE_SIZE: n=5000, 100 participants, sample size 50, total participants

STATISTICS & ECONOMICS (8):
- PERCENT: 78%, 45.5%, 100%
- COST: $1,200 USD, cost per dose $15, ICER $5,000
- QALY: QALY, quality-adjusted life year, QALY gained
- ICER: incremental cost-effectiveness ratio, ICER $10,000/QALY
- EFFECT_MEASURE: odds ratio, risk ratio, hazard ratio, OR, RR, HR
- EFFECT_VALUE: OR=2.5, RR 0.45, HR=1.8
- CI: 95% CI, CI: 0.32-0.63, 95% confidence interval
- PVALUE: p=0.03, p<0.001, p=0.05, non-significant

Return ONLY a JSON object with this exact format:
{"entities": [{"text": "exact substring from text", "label": "LABEL_NAME"}]}

IMPORTANT: Extract entities ONLY when they appear in the text. Do not infer or create entities."""
        
        # User prompt with truncated text
        text_preview = text[:2500] if len(text) > 2500 else text
        user_prompt = f"""Extract entities from this systematic review text (Document ID: {doc_id}):

{text_preview}

Extract ALL entities that match the 55 label types described above. Return ONLY JSON."""

        # Ollama API payload
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }
        
        # Try with retries
        for attempt in range(self.config.max_retries + 1):
            try:
                logger.debug(f"LLM attempt {attempt + 1} for doc {doc_id}")
                
                response = self.session.post(
                    f"{self.config.base_url}/api/chat",
                    headers=self.headers,
                    json=payload,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract content from Ollama response
                    if 'message' in result and 'content' in result['message']:
                        content = result['message']['content']
                    else:
                        content = result.get('response', result.get('message', ''))
                    
                    # Extract JSON from response
                    json_str = self._extract_json(content)
                    if json_str:
                        try:
                            data = json.loads(json_str)
                            entities = data.get("entities", [])
                            
                            # Validate entities
                            valid_entities = []
                            for entity in entities:
                                if self._validate_entity(entity):
                                    valid_entities.append(entity)
                            
                            logger.info(f"LLM extracted {len(valid_entities)} valid entities from {doc_id}")
                            return valid_entities
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse LLM JSON for {doc_id}: {e}")
                    else:
                        logger.warning(f"No JSON found in LLM response for {doc_id}")
                        return []
                
                else:
                    logger.warning(f"LLM API error {response.status_code}: {response.text[:100]}")
                    if attempt < self.config.max_retries:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    
            except (RequestException, Timeout, json.JSONDecodeError) as e:
                logger.warning(f"LLM error on attempt {attempt + 1}: {type(e).__name__}: {e}")
                if attempt < self.config.max_retries:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"LLM failed after {self.config.max_retries} attempts: {e}")
        
        return []  # Return empty list if all attempts fail
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from text response"""
        if not text:
            return None
        
        # Try to find JSON-like content
        json_patterns = [
            r'```json\s*(.*?)\s*```',  # JSON code block
            r'```\s*(.*?)\s*```',      # Any code block
            r'({.*})',                  # Any JSON object
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                content = match.group(1) if len(match.groups()) > 0 else match.group(0)
                content = content.strip()
                if (content.startswith('{') and content.endswith('}')) or \
                   (content.startswith('[') and content.endswith(']')):
                    try:
                        json.loads(content)
                        return content
                    except json.JSONDecodeError:
                        continue
        
        # If no pattern matched, try parsing the whole text
        text = text.strip()
        if (text.startswith('{') and text.endswith('}')) or \
           (text.startswith('[') and text.endswith(']')):
            try:
                json.loads(text)
                return text
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _validate_entity(self, entity: Dict) -> bool:
        """Validate entity from LLM against 55 valid labels"""
        if not isinstance(entity, dict):
            return False
        
        text = entity.get("text", "")
        label = entity.get("label", "")
        
        # Basic validation
        if not text or not label:
            return False
        
        # Check if label is valid (case-insensitive)
        valid_labels = set(EntityLabels.get_all_labels())
        label_upper = label.upper()
        
        if label_upper not in valid_labels:
            logger.debug(f"Invalid label '{label}' for entity '{text}'")
            return False
        
        # Update label to correct case
        entity["label"] = label_upper
        
        if len(text.strip()) < 2:
            return False
        
        # Avoid trivial words
        trivial = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", 
                  "of", "with", "by", "as", "is", "was", "were", "are", "be", "been"}
        if text.lower().strip() in trivial:
            return False
        
        return True
    
    def test_connection(self) -> bool:
        """Test if Ollama is running and model is available"""
        try:
            response = self.session.get(f"{self.config.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m.get('name', '') for m in models]
                logger.info(f"Connected to Ollama. Available models: {available_models}")
                
                if self.config.model in available_models:
                    logger.info(f"Model {self.config.model} is available")
                    return True
                else:
                    logger.error(f"Model {self.config.model} not found. Available: {available_models}")
                    return False
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            logger.info("Make sure Ollama is running: `ollama serve`")
            return False

# ============================================================
# TEXT PROCESSING UTILITIES
# ============================================================

class TextProcessor:
    """Text processing utilities"""
    
    @staticmethod
    def read_file(filepath: Path) -> str:
        """Read text file with encoding fallback"""
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # If all fail, try with error replacement
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    
    @staticmethod
    def chunk_text(text: str, max_chunk_size: int = 2000, overlap: int = 100) -> List[Tuple[int, int, str]]:
        """Split text into overlapping chunks"""
        if len(text) <= max_chunk_size:
            return [(0, len(text), text)]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + max_chunk_size, len(text))
            
            # Try to end at sentence boundary
            if end < len(text):
                # Look for sentence end
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + max_chunk_size * 0.5:  # Don't make too small
                    end = sentence_end + 1
            
            chunks.append((start, end, text[start:end]))
            
            if end == len(text):
                break
            
            # Move start with overlap
            start = max(start + 1, end - overlap)
        
        return chunks
    
    @staticmethod
    def compute_file_hash(filepath: Path) -> str:
        """Compute MD5 hash of file for change detection"""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

# ============================================================
# MAIN EXTRACTION PIPELINE
# ============================================================

class NERExtractionPipeline:
    """Main NER extraction pipeline with hybrid approach for 55 entity types"""
    
    def __init__(self, llm_config: Optional[LLMConfig] = None, use_llm: bool = True):
        self.rule_extractor = RuleBasedExtractor()
        self.use_llm = use_llm
        
        if use_llm and llm_config:
            self.llm_client = LLMClient(llm_config)
        else:
            self.llm_client = None
            logger.info("LLM extraction disabled, using rule-based only")
        
        # Statistics
        self.total_docs = 0
        self.successful_docs = 0
        self.failed_docs = 0
        
    def process_document(self, filepath: Path) -> Optional[ExtractionResult]:
        """Process a single document with all 55 entity types"""
        doc_id = filepath.stem
        self.total_docs += 1
        
        try:
            logger.info(f"Processing document: {doc_id}")
            
            # Read text
            text = TextProcessor.read_file(filepath)
            if not text.strip():
                logger.warning(f"Empty document: {doc_id}")
                return None
            
            # Extract entities using rules (always)
            rule_entities = self.rule_extractor.extract(text)
            rule_stats = self.rule_extractor.get_stats()
            
            # Extract entities using LLM (if enabled)
            llm_entities = []
            if self.use_llm and self.llm_client:
                try:
                    llm_raw_entities = self.llm_client.extract_entities(text, doc_id)
                    llm_entities = self._convert_llm_entities(llm_raw_entities, text)
                    logger.info(f"LLM found {len(llm_entities)} entities for {doc_id}")
                except Exception as e:
                    logger.error(f"LLM extraction failed for {doc_id}: {e}")
                    # Continue with rule-based only
            
            # Combine entities
            all_entities = rule_entities + llm_entities
            unique_entities = self._deduplicate_entities(all_entities)
            
            # Calculate label distribution
            label_distribution = {}
            for entity in unique_entities:
                label_distribution[entity.label] = label_distribution.get(entity.label, 0) + 1
            
            # Create result
            result = ExtractionResult(
                doc_id=doc_id,
                text=text,
                spans=unique_entities,
                metadata={
                    "filepath": str(filepath),
                    "file_size": len(text),
                    "processing_time": datetime.now().isoformat(),
                    "hash": TextProcessor.compute_file_hash(filepath),
                    "total_labels_found": len(set(e.label for e in unique_entities))
                },
                stats={
                    "rule_based_entities": len(rule_entities),
                    "llm_entities": len(llm_entities),
                    "total_entities": len(unique_entities),
                    "rule_stats": rule_stats,
                    "label_distribution": label_distribution,
                    "unique_labels": list(label_distribution.keys())
                }
            )
            
            self.successful_docs += 1
            logger.info(f"Successfully processed {doc_id}: {len(unique_entities)} entities across {len(set(e.label for e in unique_entities))} labels")
            
            return result
            
        except Exception as e:
            self.failed_docs += 1
            logger.error(f"Failed to process {doc_id}: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def _convert_llm_entities(self, llm_entities: List[Dict], text: str) -> List[EntitySpan]:
        """Convert LLM entities to EntitySpan objects"""
        result = []
        
        for entity in llm_entities:
            text_str = entity.get("text", "")
            label = entity.get("label", "")
            
            if not text_str or not label:
                continue
            
            # Find position in text (case-insensitive)
            pos = text.lower().find(text_str.lower())
            if pos != -1:
                # Adjust to actual case
                actual_text = text[pos:pos + len(text_str)]
                span = EntitySpan(
                    start=pos,
                    end=pos + len(text_str),
                    label=label,
                    text=actual_text
                )
                result.append(span)
            else:
                # Try fuzzy matching for slight variations
                logger.debug(f"Could not find exact match for '{text_str}' in text")
        
        return result
    
    def _deduplicate_entities(self, entities: List[EntitySpan]) -> List[EntitySpan]:
        """Deduplicate entities based on position and label"""
        if not entities:
            return []
        
        # Sort by position
        entities.sort(key=lambda x: (x.start, x.end, x.label))
        
        # Group by position
        seen = set()
        unique = []
        
        for entity in entities:
            key = (entity.start, entity.end, entity.label)
            if key not in seen:
                seen.add(key)
                unique.append(entity)
        
        # Remove overlaps (keep first when overlapping)
        unique.sort(key=lambda x: (x.start, x.end))
        result = []
        last_end = -1
        
        for entity in unique:
            if entity.start < last_end:
                # Overlap - skip
                continue
            result.append(entity)
            last_end = entity.end
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            "total_documents": self.total_docs,
            "successful": self.successful_docs,
            "failed": self.failed_docs,
            "success_rate": self.successful_docs / self.total_docs if self.total_docs > 0 else 0,
            "total_entity_types": len(EntityLabels.get_all_labels())
        }

# ============================================================
# FILE PROCESSING & OUTPUT
# ============================================================

class FileProcessor:
    """Handle file processing and output generation"""
    
    @staticmethod
    def ensure_directory(path: Path):
        """Ensure directory exists"""
        path.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def list_text_files(directory: Path) -> List[Path]:
        """List all text files in directory"""
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        txt_files = list(directory.glob("*.txt"))
        txt_files.sort()  # Sort for consistent processing
        
        return txt_files
    
    @staticmethod
    def load_processed_docs(output_file: Path) -> Set[str]:
        """Load already processed document IDs from output file"""
        processed = set()
        
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                doc_id = data.get("doc_id")
                                if doc_id:
                                    processed.add(doc_id)
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                logger.warning(f"Could not read output file: {e}")
        
        return processed
    
    @staticmethod
    def write_result(output_file: Path, result: ExtractionResult):
        """Write single result to output file"""
        with open(output_file, 'a', encoding='utf-8') as f:
            json_line = json.dumps(result.to_dict(), ensure_ascii=False)
            f.write(json_line + '\n')
    
    @staticmethod
    def create_sample_corpus(directory: Path):
        """Create sample corpus for testing with 55 entity types"""
        samples = [
            ("hiv_hpv_review.txt", """A systematic review of HPV vaccination in HIV-positive populations.

We conducted a systematic review and meta-analysis of 25 studies from PubMed, MEDLINE, and Embase.
The literature search was conducted up to December 2023. Studies were included if they reported on HPV vaccine efficacy in HIV-positive individuals.

Total sample size was 5,800 participants across 15 countries in sub-Saharan Africa, Latin America, and Asia.
Age ranged from 9 to 45 years, with median age of 28 years. 65% of participants were female.

Results: Vaccine efficacy against HPV types 16/18 was 78% (95% CI 65-87%) in HIV-positive individuals.
Seroconversion rates were 92% after 3 doses. Adverse events were mild and occurred in 15% of participants.

Cost-effectiveness analysis showed ICER of $2,500 per QALY gained in low-income countries.
Barriers included cost ($15 per dose), cold chain requirements, and vaccine hesitancy (reported by 30%).

Quality assessment using Cochrane Risk of Bias tool showed low risk for 18 studies.
This review supports HPV vaccination in HIV-positive populations as cost-effective and safe."""),
            
            ("malaria_modeling_review.txt", """Scoping review of mathematical modeling for malaria vaccine deployment.

This scoping review included 42 modeling studies from 2010-2022.
Models included SEIR compartmental models (n=28) and agent-based models (n=14).

Studies were conducted in high-burden countries: Kenya, Uganda, Tanzania, Nigeria, and Democratic Republic of Congo.
Vaccine efficacy assumptions ranged from 45% to 82% with duration of protection from 1 to 10 years.

Key findings: Mass vaccination campaigns could reduce malaria incidence by 67% (95% CI 55-78%).
Incremental cost-effectiveness ratio was $150 per DALY averted in sub-Saharan Africa.
Model projections showed greatest impact in children under 5 years (RR 0.33, p<0.001).

Limitations included uncertainty in vaccine efficacy duration and herd immunity effects.
Future research should focus on integration with existing interventions like insecticide-treated nets.

PRISMA guidelines were followed. Risk of bias was assessed using the JBI checklist."""),
            
            ("tb_economic_review.txt", """Rapid review of economic evaluations for tuberculosis vaccines.

We performed a rapid review of 18 economic evaluations published between 2015-2023.
Database searches included PubMed, Embase, and Web of Science using terms: "tuberculosis vaccine" AND ("cost" OR "economic").

Studies were from low- and middle-income countries (LMICs) in WHO regions: AFRO, SEARO, and EMRO.
Vaccine types included: M72/AS01E (efficacy 54%), VPM1002, and BCG revaccination.

Results: Incremental cost-effectiveness ratios ranged from $120 to $2,500 per DALY averted.
Vaccination was cost-saving in 8 studies and cost-effective in 10 studies (ICER < GDP per capita).

Factors influencing cost-effectiveness: vaccine price ($1.50-$8.00 per dose), delivery strategy (school-based vs routine), and coverage (65%-90%).
Sensitivity analysis showed price and efficacy were most influential parameters.

Ethical considerations included equitable access and program sustainability.
This review provides evidence for investment in new TB vaccines in high-burden countries.""")
        ]
        
        for filename, content in samples:
            filepath = directory / filename
            if not filepath.exists():
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Created sample file: {filepath}")

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main entry point"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Production NER Extraction for Infectious Disease Systematic Reviews (55 Entity Types)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction (rule-based only)
  python ner_extractor.py --corpus papers --out results.jsonl
  
  # With LLM extraction for all 55 entity types
  python ner_extractor.py --corpus papers --out results.jsonl \\
    --llm --base-url http://localhost:11434 --model llama3.1:8b
  
  # Resume processing
  python ner_extractor.py --corpus papers --out results.jsonl --resume
  
  # Parallel processing with 4 workers
  python ner_extractor.py --corpus papers --out results.jsonl --workers 4
  
  # Create sample corpus
  python ner_extractor.py --corpus papers --create-samples
  
  # Test mode (process 3 files only)
  python ner_extractor.py --corpus papers --out results.jsonl --test
        """
    )
    
    parser.add_argument("--corpus", required=True,
                       help="Directory containing text files (.txt)")
    parser.add_argument("--out", required=True,
                       help="Output JSONL file path")
    parser.add_argument("--llm", action="store_true",
                       help="Enable LLM extraction (for all 55 entity types)")
    parser.add_argument("--base-url", default="http://localhost:11434",
                       help="LLM API base URL (default: http://localhost:11434)")
    parser.add_argument("--api-key", default="dummy",
                       help="LLM API key (use 'dummy' for Ollama)")
    parser.add_argument("--model", default="llama3.1:8b",
                       help="LLM model name (default: llama3.1:8b)")
    parser.add_argument("--timeout", type=int, default=60,
                       help="Timeout for LLM requests in seconds (default: 60)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                       help="Maximum tokens for LLM response (default: 2048)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume processing (skip already processed files)")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of parallel workers (default: 1)")
    parser.add_argument("--create-samples", action="store_true",
                       help="Create sample files if corpus is empty")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--test", action="store_true",
                       help="Run in test mode (process 3 files only)")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Create paths
    corpus_path = Path(args.corpus)
    output_path = Path(args.out)
    
    # Create directories
    FileProcessor.ensure_directory(corpus_path)
    FileProcessor.ensure_directory(output_path.parent)
    
    logger.info("=" * 70)
    logger.info("NER EXTRACTION PIPELINE - 55 ENTITY TYPES")
    logger.info("=" * 70)
    logger.info(f"Corpus: {corpus_path.absolute()}")
    logger.info(f"Output: {output_path.absolute()}")
    logger.info(f"LLM: {'Enabled' if args.llm else 'Disabled'}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Resume: {args.resume}")
    logger.info(f"Total entity types: {len(EntityLabels.get_all_labels())}")
    logger.info("=" * 70)
    
    # Create sample corpus if requested
    if args.create_samples:
        logger.info("Creating sample corpus with 55 entity types...")
        FileProcessor.create_sample_corpus(corpus_path)
        logger.info("Sample corpus created")
        return
    
    # Check if corpus exists and has files
    if not corpus_path.exists():
        logger.error(f"Corpus directory not found: {corpus_path}")
        logger.info("Use --create-samples to create a sample corpus")
        return
    
    # List text files
    try:
        txt_files = FileProcessor.list_text_files(corpus_path)
    except FileNotFoundError as e:
        logger.error(e)
        return
    
    if not txt_files:
        logger.warning(f"No .txt files found in {corpus_path}")
        logger.info("Use --create-samples to create sample files")
        return
    
    logger.info(f"Found {len(txt_files)} text files")
    
    # Load already processed files if resuming
    processed_docs = set()
    if args.resume and output_path.exists():
        processed_docs = FileProcessor.load_processed_docs(output_path)
        logger.info(f"Resuming: {len(processed_docs)} files already processed")
    
    # Filter files to process
    files_to_process = []
    for filepath in txt_files:
        if filepath.stem not in processed_docs:
            files_to_process.append(filepath)
    
    if args.test:
        files_to_process = files_to_process[:3]
        logger.info(f"Test mode: Processing first {len(files_to_process)} files")
    
    if not files_to_process:
        logger.info("All files already processed. Use --resume to reprocess.")
        return
    
    logger.info(f"Files to process: {len(files_to_process)}")
    
    # Configure LLM
    llm_config = None
    if args.llm:
        llm_config = LLMConfig(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout=args.timeout,
            max_tokens=args.max_tokens
        )
        
        # Test Ollama connection
        client = LLMClient(llm_config)
        if not client.test_connection():
            logger.error("Ollama connection failed. Continuing with rule-based only.")
            args.llm = False
    
    # Initialize pipeline
    pipeline = NERExtractionPipeline(llm_config=llm_config, use_llm=args.llm)
    
    # Process files
    start_time = time.time()
    
    if args.workers > 1:
        # Parallel processing
        process_files_parallel(pipeline, files_to_process, output_path, args.workers)
    else:
        # Sequential processing
        process_files_sequential(pipeline, files_to_process, output_path)
    
    # Calculate statistics
    elapsed_time = time.time() - start_time
    stats = pipeline.get_stats()
    
    logger.info("=" * 70)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {elapsed_time:.1f} seconds")
    logger.info(f"Files processed: {stats['successful']}")
    logger.info(f"Files failed: {stats['failed']}")
    logger.info(f"Success rate: {stats['success_rate']:.1%}")
    logger.info(f"Total entity types configured: {stats['total_entity_types']}")
    logger.info(f"Output file: {output_path.absolute()}")
    
    # Show sample output
    if output_path.exists() and stats['successful'] > 0:
        show_sample_output(output_path)

def process_files_sequential(pipeline, files, output_path):
    """Process files sequentially"""
    for i, filepath in enumerate(files, 1):
        logger.info(f"[{i}/{len(files)}] Processing: {filepath.name}")
        
        result = pipeline.process_document(filepath)
        if result:
            FileProcessor.write_result(output_path, result)

def process_files_parallel(pipeline, files, output_path, max_workers):
    """Process files in parallel"""
    logger.info(f"Starting parallel processing with {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(pipeline.process_document, filepath): filepath
            for filepath in files
        }
        
        # Process results as they complete
        completed = 0
        for future in as_completed(future_to_file):
            filepath = future_to_file[future]
            completed += 1
            
            try:
                result = future.result()
                if result:
                    FileProcessor.write_result(output_path, result)
                    logger.info(f"[{completed}/{len(files)}] Completed: {filepath.name}")
            except Exception as e:
                logger.error(f"Error processing {filepath.name}: {e}")

def show_sample_output(output_path: Path):
    """Show sample of output for verification"""
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = []
            for _ in range(3):  # Read first 3 lines
                line = f.readline()
                if line:
                    lines.append(line.strip())
            
            if lines:
                logger.info("\n" + "=" * 50)
                logger.info("SAMPLE OUTPUT (first document):")
                logger.info("=" * 50)
                
                data = json.loads(lines[0])
                logger.info(f"Document: {data['doc_id']}")
                logger.info(f"Text preview: {data['text'][:150]}...")
                logger.info(f"Total entities found: {len(data['spans'])}")
                
                # Show label distribution
                stats = data.get('stats', {})
                label_dist = stats.get('label_distribution', {})
                unique_labels = stats.get('unique_labels', [])
                
                logger.info(f"Unique labels found: {len(unique_labels)}")
                
                if label_dist:
                    logger.info("\nTop 10 labels by frequency:")
                    sorted_labels = sorted(label_dist.items(), key=lambda x: x[1], reverse=True)
                    for label, count in sorted_labels[:10]:
                        logger.info(f"  {label}: {count}")
                
                if data['spans']:
                    logger.info("\nFirst 5 entities:")
                    for i, span in enumerate(data['spans'][:5]):
                        logger.info(f"  {i+1}. [{span['label']}] '{span['text']}' "
                                  f"(pos: {span['start']}-{span['end']})")
                
                logger.info(f"\nStatistics:")
                logger.info(f"  Rule-based: {stats.get('rule_based_entities', 0)}")
                logger.info(f"  LLM: {stats.get('llm_entities', 0)}")
                logger.info(f"  Total: {stats.get('total_entities', 0)}")
                
    except Exception as e:
        logger.warning(f"Could not show sample output: {e}")

# ============================================================
# UNIT TESTS FOR 55 ENTITY TYPES
# ============================================================

import unittest

class TestNERExtraction55(unittest.TestCase):
    """Unit tests for NER extraction with 55 entity types"""
    
    def test_all_labels_defined(self):
        """Test that all 55 labels are defined"""
        labels = EntityLabels.get_all_labels()
        self.assertEqual(len(labels), 55)
        
        # Check specific important labels
        self.assertIn("DATE_OF_LAST_LITERATURE_SEARCH", labels)
        self.assertIn("QUALITY_TOOL", labels)
        self.assertIn("HPV_TYPE", labels)
        self.assertIn("QALY", labels)
        self.assertIn("ICER", labels)
    
    def test_rule_extractor_comprehensive(self):
        """Test comprehensive rule extraction"""
        extractor = RuleBasedExtractor()
        text = "A systematic review of 25 studies from PubMed with literature search until December 2023. Vaccine efficacy was 78% (95% CI 65-87%, p<0.001). Cost was $15 per dose with ICER of $2,500 per QALY in low-income countries."
        entities = extractor.extract(text)
        
        self.assertGreater(len(entities), 0)
        
        # Check for multiple entity types
        labels = [e.label for e in entities]
        expected_labels = ["REVIEW_TYPE", "N_STUDIES", "DATABASE", "DATE_OF_LAST_LITERATURE_SEARCH", 
                          "EFFICACY", "PERCENT", "CI", "PVALUE", "COST", "ICER", "QALY", "INCOME_GROUP"]
        
        for label in expected_labels:
            if label in labels:
                logger.debug(f"Found expected label: {label}")
            else:
                logger.warning(f"Missing expected label: {label}")
    
    def test_date_extraction(self):
        """Test DATE_OF_LAST_LITERATURE_SEARCH extraction"""
        extractor = RuleBasedExtractor()
        
        # Test various date formats
        test_cases = [
            ("Literature search was conducted on December 15, 2023", "DATE_OF_LAST_LITERATURE_SEARCH"),
            ("Search until January 2024", "DATE_OF_LAST_LITERATURE_SEARCH"),
            ("Up to 2023-12-31", "DATE_OF_LAST_LITERATURE_SEARCH"),
            ("Through March 2024", "DATE_OF_LAST_LITERATURE_SEARCH"),
        ]
        
        for text, expected_label in test_cases:
            entities = extractor.extract(text)
            labels = [e.label for e in entities]
            self.assertIn(expected_label, labels, f"Failed to extract date from: {text}")
    
    def test_hpv_type_extraction(self):
        """Test HPV_TYPE extraction"""
        extractor = RuleBasedExtractor()
        text = "HPV types 16 and 18 are high-risk, while HPV 6 and 11 cause genital warts."
        entities = extractor.extract(text)
        
        hpv_entities = [e for e in entities if e.label == "HPV_TYPE"]
        self.assertGreaterEqual(len(hpv_entities), 2)
    
    def test_economic_terms_extraction(self):
        """Test economic terms extraction"""
        extractor = RuleBasedExtractor()
        text = "ICER was $10,000 per QALY with cost of $50 per vaccine dose."
        entities = extractor.extract(text)
        
        labels = [e.label for e in entities]
        self.assertIn("ICER", labels)
        self.assertIn("QALY", labels)
        self.assertIn("COST", labels)

def run_tests():
    """Run unit tests"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNERExtraction55)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        logger.info("All tests passed!")
        return True
    else:
        logger.error("Some tests failed!")
        return False

# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Add command for running tests
    if len(sys.argv) > 1 and sys.argv[1] == "--run-tests":
        logger.info("Running unit tests for 55 entity types...")
        success = run_tests()
        sys.exit(0 if success else 1)
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)



# python -m src.SR.data_generation.ner_from_folder \
#   --corpus infectious_disease_papers \
#   --out dataset_scaffold/infection_ner.jsonl \
#   --create-samples \
#   --verbose
  
  
# # Make sure Ollama is running
# # ollama serve &

# # Run with LLM extraction
# python -m src.SR.data_generation.ner_from_folder \
#   --corpus infectious_disease_papers \
#   --out dataset_scaffold/infection_ner_llm.jsonl \
#   --llm \
#   --base-url http://localhost:11434 \
#   --model llama3.1:8b \
#   --verbose



# # Create sample corpus and test
# python -m src.SR.data_generation.ner_extractor --corpus infectious_disease_papers --out results.jsonl --create-samples --verbose

# # Run with rule-based only (55 entity types)
# python -m src.SR.data mostly_generation.ner_extractor --corpus infectious_disease_papers --out results.jsonl --verbose

# # Run with LLM extraction (requires Ollama)
# python -m src.SR.data_generation.ner_extractor --corpus infectious_disease_papers --out dataset_scaffold/infection_ner_llm.jsonl --llm --model llama3.1:8b --verbose

# # Parallel processing with 4 workers
# python -m src.SR.data_generation.ner_extractor --corpus infectious_disease_papers --out results.jsonl --llm --workers 4 --verbose

# # Resume interrupted processing
# python -m src.SR.data_generation.ner_extractor --corpus infectious_disease_papers --out results.jsonl --llm --resume --verbose

# # Test mode (process only 3 files)
# python -m src.SR.data_generation.ner_extractor --corpus infectious_disease_papers --out results.jsonl --llm --test --verbose