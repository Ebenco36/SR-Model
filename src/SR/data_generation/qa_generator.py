#!/usr/bin/env python3
"""
QA DATASET GENERATION FROM NER EXTRACTIONS
for Infectious Disease Systematic Reviews

Features:
- Generates QA pairs from NER extraction results
- Multiple question templates per entity type
- Handles multiple answers per document
- Proper SQuAD-style JSONL format
- Balance control and deduplication
- Progress tracking and validation
"""

import argparse
import json
import os
import re
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import random
from datetime import datetime
import hashlib

# ============================================================
# CONFIGURATION & LOGGING
# ============================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('qa_generation.log')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class QAExample:
    """Single QA example in SQuAD format"""
    id: str
    doc_id: str
    question: str
    context: str
    answers: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'doc_id': self.doc_id,
            'question': self.question,
            'context': self.context,
            'answers': self.answers,
            'metadata': self.metadata
        }

@dataclass
class EntityGroup:
    """Group of entities with same label in a document"""
    doc_id: str
    label: str
    spans: List[Dict[str, Any]]
    text: str
    
    def get_unique_texts(self) -> List[str]:
        """Get unique entity texts (case-insensitive)"""
        seen = set()
        unique = []
        for span in self.spans:
            text_lower = span['text'].lower()
            if text_lower not in seen:
                seen.add(text_lower)
                unique.append(span['text'])
        return unique
    
    def get_answer_spans(self) -> Tuple[List[str], List[int]]:
        """Get answer texts and start positions"""
        texts = []
        starts = []
        
        # Sort by position
        sorted_spans = sorted(self.spans, key=lambda x: x['start'])
        
        for span in sorted_spans:
            texts.append(span['text'])
            starts.append(span['start'])
        
        return texts, starts

# ============================================================
# QUESTION TEMPLATES FOR 55 ENTITY TYPES
# ============================================================

class QuestionTemplates:
    """Question templates for all 55 entity types"""
    
    TEMPLATES = {
        # Review Methodology & Design (7 labels)
        "REVIEW_TYPE": [
            "What type of review is this study?",
            "What is the study design type?",
            "Is this a systematic review, meta-analysis, or other type of review?",
            "How would you classify this review?",
            "What review methodology was used?"
        ],
        "STUDY_DESIGN": [
            "What study design was used in the included studies?",
            "What type of study designs were included?",
            "What was the design of the studies reviewed?",
            "Were the studies randomized controlled trials, cohort studies, or other designs?"
        ],
        "N_STUDIES": [
            "How many studies were included in this review?",
            "What is the total number of included studies?",
            "How many studies were analyzed?",
            "What was the study count in this review?"
        ],
        "DATE_OF_LAST_LITERATURE_SEARCH": [
            "When was the literature search conducted?",
            "What was the date of the last literature search?",
            "Up to what date was the literature searched?",
            "When was the search for studies performed?"
        ],
        "QUALITY_TOOL": [
            "What quality assessment tool was used?",
            "Which tool was used to assess study quality?",
            "How was risk of bias assessed?",
            "What quality appraisal tool was employed?"
        ],
        "DATABASE": [
            "Which databases were searched?",
            "Where was the literature search conducted?",
            "What electronic databases were used?",
            "Which bibliographic databases were searched?"
        ],
        "SEARCH_TERMS": [
            "What search terms were used?",
            "What keywords or MeSH terms were used in the search?",
            "What was the search strategy?",
            "How was the literature search constructed?"
        ],
        
        # Population & Conditions (12 labels)
        "PATHOGEN": [
            "Which pathogen was studied?",
            "What infectious agent is the focus of this review?",
            "What pathogen is being investigated?",
            "Which microorganism is the target?"
        ],
        "CANCER": [
            "What type of cancer is studied?",
            "Which cancer is the focus?",
            "What malignancy is being investigated?",
            "Which oncological condition is addressed?"
        ],
        "CONDITION": [
            "What health condition is studied?",
            "Which disease is the focus?",
            "What medical condition is being investigated?",
            "Which health outcome is addressed?"
        ],
        "HPV_TYPE": [
            "What HPV types are studied?",
            "Which HPV genotypes are included?",
            "What types of human papillomavirus are investigated?",
            "Which HPV variants are addressed?"
        ],
        "AGE_GROUP": [
            "What age group was studied?",
            "Which age range was included?",
            "What was the age of participants?",
            "Who was the target population by age?"
        ],
        "GENDER": [
            "What gender was studied?",
            "Which sex was included?",
            "Were males, females, or both included?",
            "What was the gender distribution?"
        ],
        "SPECIAL_POP": [
            "What special population was studied?",
            "Which vulnerable group was included?",
            "What specific population subgroup was investigated?",
            "Which at-risk population was addressed?"
        ],
        "RISK_GROUP": [
            "What risk group was studied?",
            "Which high-risk population was included?",
            "What vulnerable group was investigated?",
            "Which population at risk was addressed?"
        ],
        "INCLUSION": [
            "What were the inclusion criteria?",
            "Which studies were eligible for inclusion?",
            "What criteria were used to include studies?",
            "How were studies selected for this review?"
        ],
        "ANALYSIS": [
            "What analysis method was used?",
            "How was the data analyzed?",
            "What statistical analysis was performed?",
            "Which analytical approach was used?"
        ],
        "PERIOD": [
            "What time period was studied?",
            "During which years were studies conducted?",
            "What was the study period?",
            "When were the included studies performed?"
        ],
        "FOLLOWUP": [
            "What was the follow-up period?",
            "How long was the follow-up?",
            "What duration of follow-up was reported?",
            "What was the length of follow-up in the studies?"
        ],
        
        # Outcomes & Measures (12 labels)
        "SAFETY": [
            "What safety outcomes were reported?",
            "What adverse events were studied?",
            "What safety profile was assessed?",
            "Which safety measures were evaluated?"
        ],
        "ACCEPTANCE": [
            "What acceptance outcomes were reported?",
            "What vaccine acceptance was studied?",
            "What willingness to vaccinate was assessed?",
            "Which acceptance measures were evaluated?"
        ],
        "EFFICACY": [
            "What efficacy outcomes were reported?",
            "What vaccine efficacy was studied?",
            "What effectiveness was assessed?",
            "Which efficacy measures were evaluated?"
        ],
        "IMMUNOGENICITY": [
            "What immunogenicity outcomes were reported?",
            "What immune response was studied?",
            "What antibody response was assessed?",
            "Which immunogenicity measures were evaluated?"
        ],
        "COVERAGE": [
            "What coverage outcomes were reported?",
            "What vaccination coverage was studied?",
            "What uptake rate was assessed?",
            "Which coverage measures were evaluated?"
        ],
        "ECONOMIC": [
            "What economic outcomes were reported?",
            "What cost-effectiveness was studied?",
            "What economic evaluation was performed?",
            "Which economic measures were assessed?"
        ],
        "ADMINISTRATION": [
            "What administration details were reported?",
            "How was the vaccine administered?",
            "What dosing schedule was used?",
            "Which administration route was employed?"
        ],
        "ETHICAL": [
            "What ethical considerations were reported?",
            "What ethical issues were addressed?",
            "How was ethical approval obtained?",
            "Which ethical aspects were considered?"
        ],
        "LOGISTICS": [
            "What logistic aspects were reported?",
            "How was the vaccine delivered?",
            "What supply chain issues were addressed?",
            "Which logistic challenges were discussed?"
        ],
        "MODELLING": [
            "What modeling approach was used?",
            "How was mathematical modeling employed?",
            "What simulation methods were used?",
            "Which modeling techniques were applied?"
        ],
        "CLINICAL": [
            "What clinical outcomes were reported?",
            "What clinical effects were studied?",
            "How was clinical efficacy assessed?",
            "Which clinical measures were evaluated?"
        ],
        "LESION": [
            "What lesion outcomes were reported?",
            "What precancerous lesions were studied?",
            "How were cervical lesions assessed?",
            "Which lesion types were evaluated?"
        ],
        
        # Geography & Demographics (5 labels)
        "COUNTRY": [
            "Which countries were included?",
            "Where were the studies conducted?",
            "What countries were represented?",
            "Which geographical locations were studied?"
        ],
        "REGION": [
            "Which regions were included?",
            "What geographical regions were studied?",
            "Where were the studies located by region?",
            "Which areas of the world were represented?"
        ],
        "WHO_REGION": [
            "Which WHO regions were included?",
            "What World Health Organization regions were represented?",
            "Where were studies located by WHO region?",
            "Which WHO geographical areas were studied?"
        ],
        "INCOME_GROUP": [
            "What income groups were included?",
            "Which economic classifications were represented?",
            "Were studies from low-, middle-, or high-income countries?",
            "What country income levels were studied?"
        ],
        "TIMING": [
            "What timing of vaccination was studied?",
            "When was vaccination recommended?",
            "What schedule was used for vaccination?",
            "At what age or time was vaccination administered?"
        ],
        
        # Interventions & Programs (8 labels)
        "VACCINE_TYPE": [
            "What type of vaccine was studied?",
            "Which vaccine platform was used?",
            "What kind of vaccine was administered?",
            "Which vaccine technology was employed?"
        ],
        "VACCINE_BRAND": [
            "What brand of vaccine was studied?",
            "Which commercial vaccine was used?",
            "What specific vaccine product was administered?",
            "Which manufacturer's vaccine was employed?"
        ],
        "DOSE": [
            "What dose was administered?",
            "How many doses were given?",
            "What dosing regimen was used?",
            "Which dose schedule was followed?"
        ],
        "ROUTE": [
            "How was the vaccine administered?",
            "What route of administration was used?",
            "How was the vaccine delivered?",
            "Which administration method was employed?"
        ],
        "PROGRAM": [
            "What vaccination program was studied?",
            "Which immunization program was implemented?",
            "How was vaccination delivered programmatically?",
            "What type of vaccination initiative was conducted?"
        ],
        "COMPONENT": [
            "What program components were included?",
            "Which elements of the vaccination program were studied?",
            "What aspects of program implementation were addressed?",
            "Which components of the intervention were evaluated?"
        ],
        "SCREENING": [
            "What screening methods were used?",
            "How was screening performed?",
            "What type of screening was employed?",
            "Which screening tests were evaluated?"
        ],
        "COMBINATION": [
            "What combination interventions were studied?",
            "How were interventions combined?",
            "What integrated approaches were used?",
            "Which combination strategies were evaluated?"
        ],
        
        # Implementation Factors (3 labels)
        "BARRIER": [
            "What barriers were identified?",
            "What challenges to implementation were reported?",
            "What obstacles were encountered?",
            "Which barriers affected the intervention?"
        ],
        "FACILITATOR": [
            "What facilitators were identified?",
            "What factors helped implementation?",
            "What enablers were reported?",
            "Which facilitators supported the intervention?"
        ],
        "SAMPLE_SIZE": [
            "What was the sample size?",
            "How many participants were included?",
            "What was the number of study participants?",
            "How large was the study sample?"
        ],
        
        # Statistics & Economics (8 labels)
        "PERCENT": [
            "What percentages were reported?",
            "What proportion or percentage was found?",
            "What was the reported percentage?",
            "Which percentage values were significant?"
        ],
        "COST": [
            "What costs were reported?",
            "How much did the intervention cost?",
            "What was the financial cost?",
            "Which cost measures were evaluated?"
        ],
        "QALY": [
            "What QALY values were reported?",
            "How many quality-adjusted life years were gained?",
            "What was the QALY impact?",
            "Which QALY measures were calculated?"
        ],
        "ICER": [
            "What ICER values were reported?",
            "What was the incremental cost-effectiveness ratio?",
            "How cost-effective was the intervention?",
            "Which ICER measures were calculated?"
        ],
        "EFFECT_MEASURE": [
            "What effect measures were used?",
            "Which statistical measures were reported?",
            "How was effect size measured?",
            "What type of effect measure was employed?"
        ],
        "EFFECT_VALUE": [
            "What effect values were reported?",
            "What was the effect size?",
            "How large was the effect?",
            "Which numerical effect values were significant?"
        ],
        "CI": [
            "What confidence intervals were reported?",
            "What was the range of the confidence interval?",
            "How precise were the estimates?",
            "Which CI values were provided?"
        ],
        "PVALUE": [
            "What p-values were reported?",
            "What was the statistical significance?",
            "How significant were the findings?",
            "Which p-values indicated significance?"
        ],
    }
    
    @classmethod
    def get_templates(cls, label: str) -> List[str]:
        """Get question templates for a label"""
        return cls.TEMPLATES.get(label, [f"What is the {label.replace('_', ' ').lower()}?"])
    
    @classmethod
    def get_random_template(cls, label: str) -> str:
        """Get random question template for a label"""
        templates = cls.get_templates(label)
        return random.choice(templates) if templates else f"What is the {label.replace('_', ' ').lower()}?"
    
    @classmethod
    def get_all_labels(cls) -> List[str]:
        """Get all 55 entity labels"""
        return list(cls.TEMPLATES.keys())

# ============================================================
# QA GENERATOR
# ============================================================

class QAGenerator:
    """Generate QA pairs from NER extraction results"""
    
    def __init__(self, max_context_length: int = 4000, min_answers: int = 1, max_answers: int = 10):
        self.max_context_length = max_context_length
        self.min_answers = min_answers
        self.max_answers = max_answers
        self.stats = defaultdict(int)
        
    def process_document(self, ner_result: Dict[str, Any]) -> List[QAExample]:
        """Process single NER result into QA examples"""
        doc_id = ner_result['doc_id']
        full_text = ner_result['text']
        spans = ner_result['spans']
        
        # Group spans by label
        spans_by_label = defaultdict(list)
        for span in spans:
            spans_by_label[span['label']].append(span)
        
        # Generate QA pairs for each label with sufficient answers
        qa_examples = []
        
        for label, label_spans in spans_by_label.items():
            # Check if we have enough answers
            if len(label_spans) < self.min_answers:
                logger.debug(f"Skipping {label} for {doc_id}: only {len(label_spans)} answers")
                continue
            
            # Limit number of answers
            if len(label_spans) > self.max_answers:
                label_spans = self._select_best_spans(label_spans, self.max_answers)
            
            # Create entity group
            entity_group = EntityGroup(
                doc_id=doc_id,
                label=label,
                spans=label_spans,
                text=full_text
            )
            
            # Generate QA example
            qa_example = self._create_qa_example(entity_group, full_text)
            if qa_example:
                qa_examples.append(qa_example)
                self.stats['qa_pairs_generated'] += 1
                self.stats[f'qa_{label}'] += 1
        
        self.stats['documents_processed'] += 1
        self.stats['total_qa_pairs'] += len(qa_examples)
        
        return qa_examples
    
    def _select_best_spans(self, spans: List[Dict], max_count: int) -> List[Dict]:
        """Select best spans based on diversity and position"""
        if len(spans) <= max_count:
            return spans
        
        # Sort by position
        spans.sort(key=lambda x: x['start'])
        
        # Group by text (case-insensitive)
        text_groups = defaultdict(list)
        for span in spans:
            text_groups[span['text'].lower()].append(span)
        
        # Select diverse texts
        selected = []
        unique_texts = list(text_groups.keys())
        
        # Ensure we get different texts first
        for text in unique_texts:
            if len(selected) >= max_count:
                break
            # Take the first occurrence of each text
            selected.append(text_groups[text][0])
        
        # If we need more, add additional occurrences
        if len(selected) < max_count:
            for span in spans:
                if span not in selected and len(selected) < max_count:
                    selected.append(span)
        
        return selected
    
    def _create_qa_example(self, entity_group: EntityGroup, full_text: str) -> Optional[QAExample]:
        """Create a single QA example from entity group"""
        doc_id = entity_group.doc_id
        label = entity_group.label
        
        # Get question
        question = QuestionTemplates.get_random_template(label)
        
        # Get answers
        answer_texts, answer_starts = entity_group.get_answer_spans()
        
        # Validate that answers are in context
        valid_answers = []
        valid_starts = []
        
        for text, start in zip(answer_texts, answer_starts):
            if start < len(full_text) and start + len(text) <= len(full_text):
                if full_text[start:start + len(text)] == text:
                    valid_answers.append(text)
                    valid_starts.append(start)
                else:
                    # Try to find the text in context
                    found_start = full_text.find(text)
                    if found_start != -1:
                        valid_answers.append(text)
                        valid_starts.append(found_start)
                    else:
                        logger.warning(f"Could not find answer '{text}' in context for {doc_id}")
            else:
                logger.warning(f"Answer span out of bounds for {doc_id}")
        
        if not valid_answers:
            return None
        
        # Create unique ID
        qa_id = f"{doc_id}_q{self.stats['qa_pairs_generated'] + 1}_{label}"
        
        # Truncate context if needed
        context = full_text
        if len(context) > self.max_context_length:
            # Try to keep context around answers
            context = self._truncate_context_around_answers(context, valid_starts, valid_answers)
        
        # Create QA example
        qa_example = QAExample(
            id=qa_id,
            doc_id=doc_id,
            question=question,
            context=context,
            answers={
                'text': valid_answers,
                'answer_start': valid_starts
            },
            metadata={
                'entity_label': label,
                'num_answers': len(valid_answers),
                'generation_date': datetime.now().isoformat()
            }
        )
        
        return qa_example
    
    def _truncate_context_around_answers(self, context: str, answer_starts: List[int], answer_texts: List[str]) -> str:
        """Truncate context while keeping answers visible"""
        if not answer_starts:
            return context[:self.max_context_length]
        
        # Find a window that includes all answers
        min_start = min(answer_starts)
        max_end = max(start + len(text) for start, text in zip(answer_starts, answer_texts))
        
        window_size = max_end - min_start
        if window_size > self.max_context_length:
            # Can't fit all answers, return first max_context_length chars
            return context[:self.max_context_length]
        
        # Calculate padding
        padding = (self.max_context_length - window_size) // 2
        
        start_pos = max(0, min_start - padding)
        end_pos = min(len(context), max_end + padding)
        
        # Adjust if we're at the beginning or end
        if start_pos == 0:
            end_pos = min(len(context), self.max_context_length)
        elif end_pos == len(context):
            start_pos = max(0, len(context) - self.max_context_length)
        
        truncated = context[start_pos:end_pos]
        
        # Adjust answer positions
        for i in range(len(answer_starts)):
            answer_starts[i] -= start_pos
        
        return truncated
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics"""
        return dict(self.stats)

# ============================================================
# DATASET MANAGER
# ============================================================

class QADatasetManager:
    """Manage QA dataset creation and validation"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.train_dir = output_dir / "train"
        self.val_dir = output_dir / "validation"
        self.test_dir = output_dir / "test"
        
        for directory in [self.train_dir, self.val_dir, self.test_dir]:
            directory.mkdir(exist_ok=True)
    
    def load_ner_results(self, ner_file: Path) -> List[Dict[str, Any]]:
        """Load NER extraction results from JSONL file"""
        results = []
        
        try:
            with open(ner_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            result = json.loads(line)
                            results.append(result)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON line: {e}")
                            continue
        except FileNotFoundError:
            logger.error(f"NER file not found: {ner_file}")
            return []
        
        logger.info(f"Loaded {len(results)} NER results from {ner_file}")
        return results
    
    def create_dataset(self, ner_file: Path, train_split: float = 0.7, val_split: float = 0.15, 
                      test_split: float = 0.15, seed: int = 42):
        """Create train/val/test splits from NER results"""
        
        # Validate splits
        total_split = train_split + val_split + test_split
        if abs(total_split - 1.0) > 0.01:  # Allow small floating point errors
            logger.warning(f"Splits sum to {total_split}, normalizing to 1.0")
            train_split /= total_split
            val_split /= total_split
            test_split /= total_split
        
        # Load NER results
        ner_results = self.load_ner_results(ner_file)
        if not ner_results:
            logger.error("No NER results to process")
            return
        
        # Initialize QA generator
        qa_generator = QAGenerator()
        
        # Generate QA pairs for all documents
        all_qa_examples = []
        doc_qa_counts = []
        
        for ner_result in ner_results:
            doc_id = ner_result['doc_id']
            qa_examples = qa_generator.process_document(ner_result)
            
            if qa_examples:
                all_qa_examples.extend(qa_examples)
                doc_qa_counts.append((doc_id, len(qa_examples)))
                
                logger.info(f"Generated {len(qa_examples)} QA pairs for {doc_id}")
        
        if not all_qa_examples:
            logger.error("No QA pairs generated")
            return
        
        # Shuffle examples
        random.seed(seed)
        random.shuffle(all_qa_examples)
        
        # Split dataset
        n_total = len(all_qa_examples)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        train_examples = all_qa_examples[:n_train]
        val_examples = all_qa_examples[n_train:n_train + n_val]
        test_examples = all_qa_examples[n_train + n_val:]
        
        logger.info(f"Dataset split: Train={len(train_examples)}, Val={len(val_examples)}, Test={len(test_examples)}")
        
        # Write splits to files
        self._write_split("train", train_examples)
        self._write_split("validation", val_examples)
        self._write_split("test", test_examples)
        
        # Write statistics
        stats = qa_generator.get_stats()
        stats.update({
            'total_examples': n_total,
            'train_examples': len(train_examples),
            'val_examples': len(val_examples),
            'test_examples': len(test_examples),
            'documents_with_qa': len(doc_qa_counts),
            'avg_qa_per_doc': sum(count for _, count in doc_qa_counts) / len(doc_qa_counts) if doc_qa_counts else 0,
            'split_ratios': {
                'train': train_split,
                'validation': val_split,
                'test': test_split
            },
            'generation_date': datetime.now().isoformat()
        })
        
        stats_file = self.output_dir / "dataset_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset statistics saved to {stats_file}")
        
        # Show sample QA pairs
        self._show_samples(train_examples, "train")
    
    def _write_split(self, split_name: str, examples: List[QAExample]):
        """Write QA examples to JSONL file for a split"""
        if split_name == "train":
            output_file = self.train_dir / f"{split_name}.jsonl"
        elif split_name == "validation":
            output_file = self.val_dir / f"{split_name}.jsonl"
        else:  # test
            output_file = self.test_dir / f"{split_name}.jsonl"
        
        count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                json_line = json.dumps(example.to_dict(), ensure_ascii=False)
                f.write(json_line + '\n')
                count += 1
        
        logger.info(f"Wrote {count} examples to {output_file}")
        
        # Also create a version with pretty JSON for inspection
        pretty_file = output_file.with_suffix('.json')
        examples_dict = [example.to_dict() for example in examples[:100]]  # First 100 examples
        with open(pretty_file, 'w', encoding='utf-8') as f:
            json.dump(examples_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created sample file {pretty_file} with first 100 examples")
    
    def _show_samples(self, examples: List[QAExample], split_name: str):
        """Show sample QA pairs for verification"""
        if not examples:
            return
        
        logger.info(f"\n{'='*60}")
        logger.info(f"SAMPLE QA PAIRS ({split_name.upper()} SPLIT)")
        logger.info(f"{'='*60}")
        
        for i, example in enumerate(examples[:3]):  # Show first 3 examples
            logger.info(f"\nExample {i+1}:")
            logger.info(f"  ID: {example.id}")
            logger.info(f"  Document: {example.doc_id}")
            logger.info(f"  Question: {example.question}")
            logger.info(f"  Context preview: {example.context[:100]}...")
            logger.info(f"  Number of answers: {len(example.answers['text'])}")
            
            if example.answers['text']:
                logger.info("  Answers:")
                for j, (text, start) in enumerate(zip(example.answers['text'], example.answers['answer_start'])):
                    logger.info(f"    {j+1}. '{text}' (position: {start})")
            
            logger.info(f"  Metadata: {example.metadata}")
        
        # Show label distribution
        label_counts = defaultdict(int)
        for example in examples:
            label = example.metadata.get('entity_label', 'unknown')
            label_counts[label] += 1
        
        logger.info(f"\nLabel distribution in {split_name} split:")
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        for label, count in sorted_labels[:10]:  # Top 10 labels
            logger.info(f"  {label}: {count} examples")
        
        if len(sorted_labels) > 10:
            logger.info(f"  ... and {len(sorted_labels) - 10} more labels")

# ============================================================
# VALIDATION AND QUALITY CONTROL
# ============================================================

class QAValidator:
    """Validate QA dataset quality"""
    
    @staticmethod
    def validate_qa_example(qa_example: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a single QA example"""
        issues = []
        
        # Check required fields
        required_fields = ['id', 'question', 'context', 'answers']
        for field in required_fields:
            if field not in qa_example:
                issues.append(f"Missing required field: {field}")
                return False, issues
        
        # Check answers structure
        answers = qa_example['answers']
        if not isinstance(answers, dict):
            issues.append("Answers must be a dictionary")
            return False, issues
        
        if 'text' not in answers or 'answer_start' not in answers:
            issues.append("Answers must contain 'text' and 'answer_start' fields")
            return False, issues
        
        text_list = answers['text']
        start_list = answers['answer_start']
        
        if not isinstance(text_list, list) or not isinstance(start_list, list):
            issues.append("Answer text and start must be lists")
            return False, issues
        
        if len(text_list) != len(start_list):
            issues.append("Answer text and start lists must have same length")
            return False, issues
        
        if not text_list:
            issues.append("Must have at least one answer")
            return False, issues
        
        # Validate each answer
        context = qa_example['context']
        for i, (text, start) in enumerate(zip(text_list, start_list)):
            if not isinstance(text, str) or not isinstance(start, int):
                issues.append(f"Answer {i}: text must be string, start must be integer")
                continue
            
            if start < 0 or start >= len(context):
                issues.append(f"Answer {i}: start position {start} out of context bounds (context length: {len(context)})")
                continue
            
            # Check if text matches context
            if context[start:start + len(text)] != text:
                # Try to find the text
                found_start = context.find(text)
                if found_start == -1:
                    issues.append(f"Answer {i}: text '{text[:50]}...' not found in context")
                else:
                    issues.append(f"Answer {i}: text found at different position (expected {start}, found {found_start})")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_dataset(file_path: Path) -> Dict[str, Any]:
        """Validate entire dataset file"""
        stats = {
            'total_examples': 0,
            'valid_examples': 0,
            'invalid_examples': 0,
            'common_issues': defaultdict(int),
            'label_distribution': defaultdict(int)
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    stats['total_examples'] += 1
                    
                    try:
                        qa_example = json.loads(line)
                        
                        # Validate QA example
                        is_valid, issues = QAValidator.validate_qa_example(qa_example)
                        
                        if is_valid:
                            stats['valid_examples'] += 1
                            
                            # Track label distribution
                            metadata = qa_example.get('metadata', {})
                            label = metadata.get('entity_label', 'unknown')
                            stats['label_distribution'][label] += 1
                        else:
                            stats['invalid_examples'] += 1
                            for issue in issues:
                                stats['common_issues'][issue] += 1
                    
                    except json.JSONDecodeError as e:
                        stats['invalid_examples'] += 1
                        stats['common_issues'][f"JSON parse error: {str(e)}"] += 1
        
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {file_path}")
            return stats
        
        # Calculate percentages
        if stats['total_examples'] > 0:
            stats['valid_percentage'] = (stats['valid_examples'] / stats['total_examples']) * 100
            stats['invalid_percentage'] = (stats['invalid_examples'] / stats['total_examples']) * 100
        
        return stats

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main entry point for QA dataset generation"""
    
    parser = argparse.ArgumentParser(
        description="QA Dataset Generation from NER Extractions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate QA dataset from NER extractions
  python qa_generator.py --ner ner_results.jsonl --output qa_dataset
  
  # With custom split ratios
  python qa_generator.py --ner ner_results.jsonl --output qa_dataset \\
    --train-split 0.8 --val-split 0.1 --test-split 0.1
  
  # Validate existing QA dataset
  python qa_generator.py --validate qa_dataset/train.jsonl
  
  # Generate with specific random seed
  python qa_generator.py --ner ner_results.jsonl --output qa_dataset --seed 123
        """
    )
    
    parser.add_argument("--ner", help="Input NER JSONL file path")
    parser.add_argument("--output", help="Output directory for QA dataset")
    parser.add_argument("--train-split", type=float, default=0.7, help="Training split ratio (default: 0.7)")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split ratio (default: 0.15)")
    parser.add_argument("--test-split", type=float, default=0.15, help="Test split ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42)")
    parser.add_argument("--validate", help="Validate existing QA dataset file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Validate mode
    if args.validate:
        logger.info(f"Validating QA dataset: {args.validate}")
        
        stats = QAValidator.validate_dataset(Path(args.validate))
        
        logger.info(f"\n{'='*60}")
        logger.info("VALIDATION RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Total examples: {stats['total_examples']}")
        logger.info(f"Valid examples: {stats['valid_examples']} ({stats.get('valid_percentage', 0):.1f}%)")
        logger.info(f"Invalid examples: {stats['invalid_examples']} ({stats.get('invalid_percentage', 0):.1f}%)")
        
        if stats['common_issues']:
            logger.info("\nCommon issues:")
            for issue, count in sorted(stats['common_issues'].items(), key=lambda x: x[1], reverse=True)[:10]:
                logger.info(f"  {count}x: {issue}")
        
        if stats['label_distribution']:
            logger.info("\nLabel distribution (top 10):")
            sorted_labels = sorted(stats['label_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]
            for label, count in sorted_labels:
                logger.info(f"  {label}: {count}")
        
        return
    
    # Check required arguments for generation
    if not args.ner or not args.output:
        logger.error("Both --ner and --output arguments are required for dataset generation")
        parser.print_help()
        return
    
    ner_file = Path(args.ner)
    output_dir = Path(args.output)
    
    # Check if NER file exists
    if not ner_file.exists():
        logger.error(f"NER file not found: {ner_file}")
        return
    
    logger.info("=" * 60)
    logger.info("QA DATASET GENERATION")
    logger.info("=" * 60)
    logger.info(f"Input NER file: {ner_file.absolute()}")
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info(f"Train/Val/Test split: {args.train_split:.2f}/{args.val_split:.2f}/{args.test_split:.2f}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("=" * 60)
    
    # Create dataset manager
    dataset_manager = QADatasetManager(output_dir)
    
    # Generate dataset
    dataset_manager.create_dataset(
        ner_file=ner_file,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Dataset saved to: {output_dir.absolute()}")
    logger.info(f"Check the following directories:")
    logger.info(f"  - Training data: {dataset_manager.train_dir}/")
    logger.info(f"  - Validation data: {dataset_manager.val_dir}/")
    logger.info(f"  - Test data: {dataset_manager.test_dir}/")
    logger.info(f"  - Statistics: {output_dir}/dataset_statistics.json")

# ============================================================
# SAMPLE DATA GENERATOR (FOR TESTING)
# ============================================================

def create_sample_ner_results(output_file: Path):
    """Create sample NER results for testing QA generation"""
    
    sample_text = """A systematic review and meta-analysis of HPV vaccination in HIV-positive populations.

We conducted a systematic review and meta-analysis of 25 studies from PubMed, MEDLINE, and Embase.
The literature search was conducted up to December 2023. Studies were included if they reported on HPV vaccine efficacy in HIV-positive individuals.

Total sample size was 5,800 participants across 15 countries in sub-Saharan Africa, Latin America, and Asia.
Age ranged from 9 to 45 years, with median age of 28 years. 65% of participants were female.

Results: Vaccine efficacy against HPV types 16/18 was 78% (95% CI 65-87%) in HIV-positive individuals.
Seroconversion rates were 92% after 3 doses. Adverse events were mild and occurred in 15% of participants.

Cost-effectiveness analysis showed ICER of $2,500 per QALY gained in low-income countries.
Barriers included cost ($15 per dose), cold chain requirements, and vaccine hesitancy (reported by 30%).

Quality assessment using Cochrane Risk of Bias tool showed low risk for 18 studies.
This review supports HPV vaccination in HIV-positive populations as cost-effective and safe."""
    
    sample_entities = [
        {"start": 2, "end": 21, "label": "REVIEW_TYPE", "text": "systematic review"},
        {"start": 26, "end": 38, "label": "REVIEW_TYPE", "text": "meta-analysis"},
        {"start": 46, "end": 52, "label": "PATHOGEN", "text": "HPV"},
        {"start": 79, "end": 82, "label": "N_STUDIES", "text": "25"},
        {"start": 89, "end": 95, "label": "DATABASE", "text": "PubMed"},
        {"start": 97, "end": 104, "label": "DATABASE", "text": "MEDLINE"},
        {"start": 109, "end": 115, "label": "DATABASE", "text": "Embase"},
        {"start": 146, "end": 159, "label": "DATE_OF_LAST_LITERATURE_SEARCH", "text": "December 2023"},
        {"start": 205, "end": 208, "label": "PATHOGEN", "text": "HPV"},
        {"start": 232, "end": 235, "label": "EFFICACY", "text": "efficacy"},
        {"start": 259, "end": 263, "label": "SAMPLE_SIZE", "text": "5,800"},
        {"start": 291, "end": 307, "label": "N_STUDIES", "text": "15 countries"},
        {"start": 312, "end": 329, "label": "REGION", "text": "sub-Saharan Africa"},
        {"start": 331, "end": 345, "label": "REGION", "text": "Latin America"},
        {"start": 350, "end": 354, "label": "REGION", "text": "Asia"},
        {"start": 359, "end": 389, "label": "AGE_GROUP", "text": "Age ranged from 9 to 45 years"},
        {"start": 413, "end": 417, "label": "PERCENT", "text": "65%"},
        {"start": 459, "end": 462, "label": "PATHOGEN", "text": "HPV"},
        {"start": 477, "end": 483, "label": "HPV_TYPE", "text": "16/18"},
        {"start": 497, "end": 499, "label": "PERCENT", "text": "78%"},
        {"start": 516, "end": 522, "label": "CI", "text": "65-87%"},
        {"start": 549, "end": 551, "label": "PERCENT", "text": "92%"},
        {"start": 593, "end": 595, "label": "PERCENT", "text": "15%"},
        {"start": 645, "end": 649, "label": "ICER", "text": "$2,500"},
        {"start": 657, "end": 661, "label": "QALY", "text": "QALY"},
        {"start": 697, "end": 704, "label": "COST", "text": "$15"},
        {"start": 738, "end": 740, "label": "PERCENT", "text": "30%"},
        {"start": 774, "end": 798, "label": "QUALITY_TOOL", "text": "Cochrane Risk of Bias"},
        {"start": 826, "end": 829, "label": "PATHOGEN", "text": "HPV"},
        {"start": 873, "end": 885, "label": "ECONOMIC", "text": "cost-effective"},
    ]
    
    sample_result = {
        "doc_id": "sample_hpv_review",
        "text": sample_text,
        "spans": sample_entities,
        "metadata": {
            "filepath": "sample.txt",
            "file_size": len(sample_text),
            "processing_time": datetime.now().isoformat()
        },
        "stats": {
            "rule_based_entities": 25,
            "llm_entities": 5,
            "total_entities": 30
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json_line = json.dumps(sample_result, ensure_ascii=False)
        f.write(json_line + '\n')
    
    logger.info(f"Created sample NER results at {output_file}")
    return output_file

# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Add command for creating sample data
    if len(sys.argv) > 1 and sys.argv[1] == "--create-sample":
        sample_file = Path("sample_ner_results.jsonl")
        create_sample_ner_results(sample_file)
        
        # Generate QA dataset from sample
        output_dir = Path("sample_qa_dataset")
        dataset_manager = QADatasetManager(output_dir)
        dataset_manager.create_dataset(sample_file)
        sys.exit(0)
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
        


# Basic generation with default splits (70/15/15)
# python -m src.SR.data_generation.qa_generator --ner ner_results.jsonl --output qa_dataset

# # Custom split ratios
# python -m src.SR.data_generation.qa_generator --ner ner_results.jsonl --output qa_dataset \
#   --train-split 0.8 --val-split 0.1 --test-split 0.1

# # With verbose logging
# python -m src.SR.data_generation.qa_generator --ner ner_results.jsonl --output qa_dataset --verbose

# # Specific random seed for reproducibility
# python -m src.SR.data_generation.qa_generator --ner ner_results.jsonl --output qa_dataset --seed 123