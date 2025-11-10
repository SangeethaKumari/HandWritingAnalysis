"""
Prompt templates for Knowledge Graph extraction
Contains all LLM prompts used in the knowledge graph building process
"""


class KnowledgeGraphPrompts:
    """Class containing all prompt templates for knowledge graph extraction"""
    
    @staticmethod
    def get_triplet_extraction_prompt(chunk: str) -> str:
        """
        Generate prompt for extracting triplets from text chunk
        
        Args:
            chunk: Text chunk to extract triplets from
            
        Returns:
            Formatted prompt string
        """
        return f"""Extract knowledge graph triplets from this AI/ML text.

RULES:
1. Extract ONLY the most important relationships
2. Use clear predicates: is_type_of, uses, requires, improves, part_of, applied_in, measures, optimizes, based_on
3. Focus on AI concepts, algorithms, techniques, metrics
4. Return EXACTLY 10-15 triplets
5. Return ONLY valid JSON array, no markdown, no explanation

TEXT:
{chunk}

FORMAT (return this exact structure):
[
  ["Concept1", "relationship", "Concept2"],
  ["Concept3", "relationship", "Concept4"]
]

JSON:"""

    @staticmethod
    def get_triplet_extraction_rules() -> list:
        """
        Get the rules for triplet extraction
        
        Returns:
            List of extraction rules
        """
        return [
            "Extract ONLY the most important relationships",
            "Use clear predicates: is_type_of, uses, requires, improves, part_of, applied_in, measures, optimizes, based_on",
            "Focus on AI concepts, algorithms, techniques, metrics",
            "Return EXACTLY 10-15 triplets",
            "Return ONLY valid JSON array, no markdown, no explanation"
        ]

    @staticmethod
    def get_valid_predicates() -> list:
        """
        Get list of valid predicate types
        
        Returns:
            List of valid predicate strings
        """
        return [
            "is_type_of",
            "uses",
            "requires",
            "improves",
            "part_of",
            "applied_in",
            "measures",
            "optimizes",
            "based_on"
        ]

