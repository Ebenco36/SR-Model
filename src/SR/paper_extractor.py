#!/usr/bin/env python3

"""
BiEncoder Semantic Paper Extractor - FULLY FIXED.

Purpose: 
    Takes unstructured paper text and a definition dictionary,
    returns structured JSON using semantic similarity instead of Regex.

Features:
  Handles complex nested structures
  Semantic tagging (not regex)
  Handles paraphrasing and synonyms
  Extracts dates, countries, study counts
  Returns clean JSON structure
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Union
from collections import defaultdict, Counter

import torch
import numpy as np
from sentence_transformers import util

from src.SR.BI.BiEncoderInference import BiEncoderInference

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PaperExtractor:
    """
    Extracts structured data from papers using Semantic Tagging.
    
    Instead of regex, uses BiEncoder to match semantic meaning:
    - "immunocompromised patients" matches "immunocompromised" tag
    - "vaccine effectiveness was 95%" matches "effectiveness" tag
    - Handles paraphrasing naturally
    """

    def __init__(
        self, 
        model_path: Path, 
        search_definitions: Dict[str, Any],
        threshold: float = 0.65
    ):
        """
        Initialize extractor.

        Args:
            model_path: Path to trained BiEncoder model
            search_definitions: Nested dict of terms to extract (searchRegEx)
            threshold: Minimum similarity (0-1) to tag a sentence
        """
        logger.info(f"Loading BiEncoder from {model_path}")
        self.model = BiEncoderInference(model_dir=model_path)
        self.definitions = search_definitions
        self.threshold = threshold
        
        # Build semantic index of all keywords
        self.tag_index = []
        self.tag_embeddings = None
        self._build_tag_index()

    def _build_tag_index(self):
        """
        Flatten searchRegEx dictionary and pre-compute embeddings for all keywords.
        
        Handles flexible structure:
        - Category -> Subcategory -> Field -> [(Keyword, Code)]
        - Category -> Subcategory -> [(Keyword, Code)]
        - Category -> Field -> [(Keyword, Code)]
        
        Robust to various nesting levels.
        """
        logger.info("Building semantic tag index...")
        queries = []
        
        for category, subcats in self.definitions.items():
            if not isinstance(subcats, dict):
                logger.debug(f"Skipping non-dict at category {category}")
                continue
            
            self._process_nested_dict(subcats, category, queries)
        
        # Batch encode all definition keywords once
        if queries:
            logger.info(f"Encoding {len(queries)} keywords...")
            self.tag_embeddings = self.model.model.encode(
                queries, 
                batch_size=32,
                convert_to_tensor=True,
                show_progress_bar=True
            )
            logger.info(f"✓ Built index with {len(queries)} semantic tags")
        else:
            logger.warning("No keywords found in schema!")

    def _process_nested_dict(
        self, 
        current_dict: Dict, 
        category: str,
        queries: List[str],
        subcategory: str = "",
        field: str = ""
    ):
        """
        Recursively process nested dictionary structure.
        
        Flexible handling for different nesting levels.
        """
        for key, value in current_dict.items():
            # Case 1: Value is a list of tuples/strings (keywords)
            if isinstance(value, list):
                for item in value:
                    keyword, code = self._parse_item(item)
                    if keyword:
                        queries.append(keyword)
                        self.tag_index.append({
                            "cat": category,
                            "sub": subcategory or key,
                            "field": field or key,
                            "code": code,
                            "phrase": keyword
                        })
            
            # Case 2: Value is a dictionary (recurse deeper)
            elif isinstance(value, dict):
                self._process_nested_dict(
                    value,
                    category,
                    queries,
                    subcategory=subcategory or key,
                    field=field or key
                )
            else:
                logger.debug(f"Skipping unexpected type at {category}/{subcategory}/{key}: {type(value)}")

    def _parse_item(self, item: Union[Tuple, str]) -> Tuple[str, str]:
        """
        Parse a keyword item, handling multiple formats.
        
        Formats:
        - (keyword_str, code_str)
        - keyword_str
        - None
        """
        if item is None:
            return "", ""
        
        if isinstance(item, tuple):
            if len(item) == 2:
                keyword, code = item
                return str(keyword).strip(), str(code).strip()
            elif len(item) == 1:
                return str(item[0]).strip(), str(item[0]).strip()
            else:
                logger.warning(f"Unexpected tuple length: {item}")
                return "", ""
        
        elif isinstance(item, str):
            return item.strip(), item.strip()
        
        else:
            logger.warning(f"Unexpected item type: {type(item)}")
            return "", ""

    def _extract_dates(self, text: str) -> str:
        """
        Extract literature search date from paper text.
        
        Strategy:
          1. Split text into sentences
          2. Find sentence most similar to "literature search date"
          3. Extract date from that sentence
        """
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]', text) 
                    if len(s.strip()) > 20]
        
        if not sentences:
            return "Not specified"
        
        # Find sentence about literature search using BiEncoder
        query_phrase = "date of last literature search database inception"
        query_embedding = self.model.model.encode(query_phrase, convert_to_tensor=True)
        
        # Encode all sentences
        sent_embeddings = self.model.model.encode(
            sentences,
            batch_size=32,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        # Find most relevant sentence
        cos_scores = util.pytorch_cos_sim(query_embedding, sent_embeddings)[0]
        best_idx = torch.argmax(cos_scores).item()
        best_sentence = sentences[best_idx]
        
        logger.debug(f"Found date sentence: {best_sentence[:100]}")
        
        # Extract date using regex from that sentence
        date_patterns = [
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}',
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}',
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}',
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, best_sentence, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return "Not specified"

    def _extract_number_of_studies(self, text: str) -> str:
        """
        Extract number of included studies.
        
        Strategy:
          1. Find sentences about "number of studies"
          2. Extract first number from those sentences
        """
        query_phrase = "number of included studies"
        query_embedding = self.model.model.encode(query_phrase, convert_to_tensor=True)
        
        sentences = [s.strip() for s in re.split(r'[.!?]', text) 
                    if len(s.strip()) > 20]
        
        if not sentences:
            return "Not specified"
        
        sent_embeddings = self.model.model.encode(
            sentences,
            batch_size=32,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        cos_scores = util.pytorch_cos_sim(query_embedding, sent_embeddings)[0]
        best_idx = torch.argmax(cos_scores).item()
        best_sentence = sentences[best_idx]
        
        logger.debug(f"Found studies sentence: {best_sentence[:100]}")
        
        # Extract number
        matches = re.findall(r'\b(\d+)\b', best_sentence)
        if matches:
            return matches[0]
        
        return "Not specified"

    def _extract_countries(self, text: str) -> Dict[str, int]:
        """
        Extract country mentions using simple pattern matching.
        """
        countries_dict = {
            "Germany": r'\bGermany\b',
            "USA": r'\b(?:USA|United States|US)\b',
            "UK": r'\b(?:UK|United Kingdom|England)\b',
            "Canada": r'\bCanada\b',
            "Australia": r'\bAustralia\b',
            "France": r'\bFrance\b',
            "Italy": r'\bItaly\b',
            "Spain": r'\bSpain\b',
            "Netherlands": r'\bNetherlands\b',
            "Belgium": r'\bBelgium\b',
            "Sweden": r'\bSweden\b',
            "Norway": r'\bNorway\b',
            "Denmark": r'\bDenmark\b',
            "Switzerland": r'\bSwitzerland\b',
            "Austria": r'\bAustria\b',
            "Poland": r'\bPoland\b',
            "China": r'\bChina\b',
            "Japan": r'\bJapan\b',
            "India": r'\bIndia\b',
            "Brazil": r'\bBrazil\b',
            "Mexico": r'\bMexico\b',
            "South Africa": r'\bSouth Africa\b',
        }
        
        country_counts = {}
        for country, pattern in countries_dict.items():
            count = len(re.findall(pattern, text, re.IGNORECASE))
            if count > 0:
                country_counts[country] = count
        
        return dict(sorted(country_counts.items(), 
                          key=lambda x: x[1], reverse=True))

    def extract(self, paper_text: str) -> Dict[str, Any]:
        """
        Main extraction function.
        
        1. Segment paper into sentences
        2. Encode sentences with BiEncoder
        3. Calculate similarity to all semantic tags
        4. Aggregate results into JSON structure
        """
        logger.info("Starting extraction...")
        
        # 1. Segment text into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]', paper_text) 
                    if len(s.strip()) > 20]
        
        if not sentences:
            logger.warning("No valid sentences found in text")
            return {}
        
        logger.info(f"Found {len(sentences)} sentences")

        # 2. Encode all sentences
        logger.info("Encoding sentences...")
        sent_embeddings = self.model.model.encode(
            sentences, 
            batch_size=32, 
            convert_to_tensor=True,
            show_progress_bar=True
        )

        # 3. Calculate similarity between every sentence and every tag
        if self.tag_embeddings is None:
            logger.error("No tag embeddings! Check schema.")
            return {}
        
        logger.info("Computing semantic similarities...")
        cos_scores = util.pytorch_cos_sim(sent_embeddings, self.tag_embeddings)
        
        # For each tag, find the maximum similarity across all sentences
        max_scores_per_tag, _ = torch.max(cos_scores, dim=0)
        max_scores = max_scores_per_tag.cpu().numpy()
        
        # 4. Initialize output structure
        output = {
            "date of last literature search": self._extract_dates(paper_text),
            "number of studies": self._extract_number_of_studies(paper_text),
            "Population": defaultdict(list),
            "topic": defaultdict(list),
            "outcome": defaultdict(list),
            "intervention": defaultdict(list),
            "study_country": self._extract_countries(paper_text)
        }
        
        # 5. Fill structured data based on threshold
        logger.info(f"Matching tags (threshold={self.threshold})...")
        matched_tags = 0
        
        for tag_idx, score in enumerate(max_scores):
            if score >= self.threshold:
                matched_tags += 1
                tag_meta = self.tag_index[tag_idx]
                
                category = tag_meta['cat']
                subcategory = tag_meta['sub']
                code = tag_meta['code']
                phrase = tag_meta['phrase']
                
                logger.debug(f"  [{score:.3f}] {phrase} -> {category}/{subcategory}/{code}")
                
                # Map category to output section
                if category == "popu":
                    json_section = output["Population"]
                elif category == "topic":
                    json_section = output["topic"]
                elif category == "outcome":
                    json_section = output["outcome"]
                elif category == "intervention":
                    json_section = output["intervention"]
                else:
                    continue
                
                # Add code to list (avoid duplicates)
                if code not in json_section[subcategory]:
                    json_section[subcategory].append(code)
        
        logger.info(f"Matched {matched_tags} tags")
        
        # Convert defaultdicts to regular dicts for JSON serialization
        for key in ["Population", "topic", "outcome", "intervention"]:
            output[key] = dict(output[key])
            # Remove empty subcategories
            output[key] = {k: v for k, v in output[key].items() if v}
        
        logger.info("✓ Extraction complete")
        return output