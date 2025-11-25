"""CPC Dataset Construction Pipeline.

This module implements Priority 3, Part A: Dataset construction for training
the Causal Plausibility Classifier (CPC).

Steps:
1. Generate weak labels (unlabeled dataset)
2. Teacher LLM labeling and rationale augmentation
3. Generate adversarial (hard) negatives
4. Balance dataset (50% positive, 25% easy negative, 25% hard negative)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import google.generativeai as genai
from google.generativeai.types import GenerationConfig


@dataclass
class CPCDatasetEntry:
    """A single entry in the CPC training dataset."""
    node_i: str
    node_j: str
    context: str
    label_plausible: bool
    label_temporal: bool
    label_mechanistic: bool
    rationale: str


class CPCDatasetBuilder:
    """Builds the CPC training dataset with Teacher LLM labeling."""
    
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.2,
        api_key_env: str = "GEMINI_API_KEY",
        timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize CPC dataset builder.
        
        Args:
            api_key: Gemini API key
            model_name: Teacher LLM model name
            temperature: Temperature for labeling
            api_key_env: Environment variable for API key
            timeout_seconds: API timeout
        """
        env_name = api_key_env or "GEMINI_API_KEY"
        resolved_key = api_key or os.getenv(env_name)
        if not resolved_key:
            raise RuntimeError(
                f"Gemini API key is missing. Provide api_key or set the {env_name} environment variable."
            )
        
        genai.configure(api_key=resolved_key)
        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
    
    def label_with_teacher_llm(
        self,
        node_i: str,
        node_j: str,
        context: str,
    ) -> Optional[CPCDatasetEntry]:
        """Step 2: Label a tuple with Teacher LLM.
        
        Args:
            node_i: First node name
            node_j: Second node name
            context: Retrieved context string
        
        Returns:
            CPCDatasetEntry with labels and rationale, or None if labeling fails
        """
        prompt = self._build_labeling_prompt(node_i, node_j, context)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=self.temperature,
                    response_mime_type="application/json",
                ),
            )
            
            payload = self._extract_text(response)
            data = json.loads(payload)
            
            # Parse labels
            label_plausible = self._parse_label(data.get("label_plausible", ""))
            label_temporal = self._parse_label(data.get("label_temporal", ""))
            label_mechanistic = self._parse_label(data.get("label_mechanistic", ""))
            rationale = data.get("rationale", "").strip()
            
            if not rationale:
                rationale = "No rationale provided."
            
            return CPCDatasetEntry(
                node_i=node_i,
                node_j=node_j,
                context=context,
                label_plausible=label_plausible,
                label_temporal=label_temporal,
                label_mechanistic=label_mechanistic,
                rationale=rationale,
            )
            
        except Exception as e:
            print(
                f"[cpc-dataset] Labeling failed for ({node_i}/{node_j}): "
                f"{type(e).__name__}: {e}"
            )
            return None
    
    def build_balanced_dataset(
        self,
        positive_entries: List[CPCDatasetEntry],
        easy_negative_entries: List[CPCDatasetEntry],
        hard_negative_entries: List[CPCDatasetEntry],
    ) -> List[CPCDatasetEntry]:
        """Step 4: Balance dataset (50% positive, 25% easy negative, 25% hard negative).
        
        Args:
            positive_entries: Positive examples (Teacher says YES)
            easy_negative_entries: Easy negative examples
            hard_negative_entries: Hard negative (adversarial) examples
        
        Returns:
            Balanced dataset
        """
        # Determine target size based on positives
        n_pos = len(positive_entries)
        n_easy_neg = n_pos // 2  # 25% of total = 50% of positives
        n_hard_neg = n_pos // 2  # 25% of total = 50% of positives
        
        # Sample if we have more than needed
        import random
        random.seed(42)  # For reproducibility
        
        sampled_pos = random.sample(positive_entries, min(n_pos, len(positive_entries)))
        sampled_easy = random.sample(
            easy_negative_entries, min(n_easy_neg, len(easy_negative_entries))
        )
        sampled_hard = random.sample(
            hard_negative_entries, min(n_hard_neg, len(hard_negative_entries))
        )
        
        # Combine and shuffle
        balanced = sampled_pos + sampled_easy + sampled_hard
        random.shuffle(balanced)
        
        return balanced
    
    def save_dataset(
        self,
        entries: List[CPCDatasetEntry],
        output_path: Path,
        format: str = "jsonl",
    ) -> None:
        """Save dataset to file.
        
        Args:
            entries: Dataset entries to save
            output_path: Output file path
            format: Format to save in ("jsonl" or "json")
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "jsonl":
            with output_path.open("w", encoding="utf-8") as f:
                for entry in entries:
                    json.dump(
                        {
                            "node_i": entry.node_i,
                            "node_j": entry.node_j,
                            "context": entry.context,
                            "label_plausible": "YES" if entry.label_plausible else "NO",
                            "label_temporal": "YES" if entry.label_temporal else "NO",
                            "label_mechanistic": "YES" if entry.label_mechanistic else "NO",
                            "rationale": entry.rationale,
                        },
                        f,
                        ensure_ascii=False,
                    )
                    f.write("\n")
        else:  # JSON
            data = [
                {
                    "node_i": entry.node_i,
                    "node_j": entry.node_j,
                    "context": entry.context,
                    "label_plausible": "YES" if entry.label_plausible else "NO",
                    "label_temporal": "YES" if entry.label_temporal else "NO",
                    "label_mechanistic": "YES" if entry.label_mechanistic else "NO",
                    "rationale": entry.rationale,
                }
                for entry in entries
            ]
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"[cpc-dataset] Saved {len(entries)} entries to {output_path}")
    
    @staticmethod
    def load_dataset(input_path: Path) -> List[CPCDatasetEntry]:
        """Load dataset from file.
        
        Args:
            input_path: Input file path
        
        Returns:
            List of CPCDatasetEntry objects
        """
        entries: List[CPCDatasetEntry] = []
        
        if input_path.suffix == ".jsonl":
            with input_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    entries.append(
                        CPCDatasetEntry(
                            node_i=data["node_i"],
                            node_j=data["node_j"],
                            context=data["context"],
                            label_plausible=data["label_plausible"] == "YES",
                            label_temporal=data["label_temporal"] == "YES",
                            label_mechanistic=data["label_mechanistic"] == "YES",
                            rationale=data.get("rationale", ""),
                        )
                    )
        else:  # JSON
            with input_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    entries.append(
                        CPCDatasetEntry(
                            node_i=item["node_i"],
                            node_j=item["node_j"],
                            context=item["context"],
                            label_plausible=item["label_plausible"] == "YES",
                            label_temporal=item["label_temporal"] == "YES",
                            label_mechanistic=item["label_mechanistic"] == "YES",
                            rationale=item.get("rationale", ""),
                        )
                    )
        
        return entries
    
    @staticmethod
    def _build_labeling_prompt(node_i: str, node_j: str, context: str) -> str:
        """Build prompt for Teacher LLM labeling."""
        return (
            f"Context: {context}\n\n"
            f"Node pair: ['{node_i}', '{node_j}']\n\n"
            f"Analyze the relationship between these two nodes in the given context. "
            f"Provide labels for three dimensions:\n"
            f"1. Plausibility: Is it possible and sensible that a connection exists? "
            f"(Does this link make sense at all, or is it nonsense?)\n"
            f"2. Temporality: Is there evidence of a time-based order? "
            f"(Does the text say one thing led to or followed another?)\n"
            f"3. Mechanism: Does the text describe a process or pathway? "
            f"(Does it explain how one affects the other or through what means?)\n\n"
            f"Respond with JSON:\n"
            f'{{\n'
            f'  "rationale": "<one sentence explaining your reasoning>",\n'
            f'  "label_plausible": "YES" or "NO",\n'
            f'  "label_temporal": "YES" or "NO",\n'
            f'  "label_mechanistic": "YES" or "NO"\n'
            f'}}'
        )
    
    @staticmethod
    def _parse_label(label: str | bool) -> bool:
        """Parse label from Teacher LLM response."""
        if isinstance(label, bool):
            return label
        if isinstance(label, str):
            return label.upper() in ("YES", "TRUE", "1", "Y")
        return False
    
    @staticmethod
    def _extract_text(response) -> str:
        """Extract text from Gemini response."""
        if getattr(response, "text", None):
            return response.text
        chunks: list[str] = []
        for candidate in getattr(response, "candidates", []):
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []):
                text = getattr(part, "text", "")
                if text:
                    chunks.append(text)
        if not chunks:
            raise ValueError("Model response did not contain text content.")
        return "".join(chunks)

