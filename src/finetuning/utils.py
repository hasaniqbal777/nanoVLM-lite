"""
Shared utilities for A-OKVQA fine-tuning and evaluation.
"""

from typing import List


def build_mc_prompt(question: str, choices: List[str]) -> str:
    """
    Build a multiple-choice prompt from question and choices.
    
    This format is used for both training and evaluation to ensure consistency.
    
    Args:
        question: The question text
        choices: List of 4 answer choices
        
    Returns:
        Formatted prompt string
    """
    return (
        f"Question: {question}\n"
        f"A) {choices[0]}\n"
        f"B) {choices[1]}\n"
        f"C) {choices[2]}\n"
        f"D) {choices[3]}\n"
        "Answer:"
    )


# Mapping between letters and indices
LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}
IDX_TO_LETTER = "ABCD"
