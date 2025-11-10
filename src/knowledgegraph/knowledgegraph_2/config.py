"""
Configuration file for Knowledge Graph Builder
Contains all configurable parameters and constants
"""
from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path


@dataclass
class LLMConfig:
    """Configuration for LLM interactions"""
    default_model: str = "llama3.1"
    temperature: float = 0.1
    num_predict: int = 1500
    max_iter_centrality: int = 1000


@dataclass
class TextProcessingConfig:
    """Configuration for text processing"""
    default_chunk_size: int = 2000
    extraction_chunk_size: int = 3000
    max_chunks_to_process: int = 10
    page_marker_pattern: str = r'\n--- Page \d+ ---\n'


@dataclass
class VisualizationConfig:
    """Configuration for graph visualizations"""
    max_nodes: int = 50
    figure_size: tuple = (24, 18)
    dpi: int = 300
    k_spring_layout: float = 2.5
    iterations_spring_layout: int = 50
    seed: int = 42
    node_size_multiplier: float = 15000
    node_size_base: int = 500
    arrow_size: int = 25
    edge_alpha: float = 0.4
    edge_width: int = 2
    node_alpha: float = 0.8
    line_width: int = 2
    font_size: int = 10
    title_font_size: int = 26
    colorbar_shrink: float = 0.8


@dataclass
class WordCloudConfig:
    """Configuration for word cloud generation"""
    width: int = 1600
    height: int = 800
    background_color: str = 'white'
    colormap: str = 'viridis'
    relative_scaling: float = 0.5
    min_font_size: int = 12
    pagerank_multiplier: float = 1000
    title_font_size: int = 24
    dpi: int = 300


@dataclass
class SemanticReasoningConfig:
    """Configuration for semantic reasoning"""
    max_nodes_to_process: int = 100
    max_successors_level_1: int = 5
    max_successors_level_2: int = 5
    inferred_relation_type: str = "inferred_from"


@dataclass
class OutputConfig:
    """Configuration for output files"""
    graph_visualization_file: str = "knowledge_graph.png"
    wordcloud_file: str = "wordcloud.png"
    triplets_file: str = "triplets.json"
    centrality_file: str = "centrality_measures.json"


@dataclass
class CentralityConfig:
    """Configuration for centrality measures"""
    top_nodes_to_display: int = 5
    centrality_measure_names: Dict[str, str] = None

    def __post_init__(self):
        if self.centrality_measure_names is None:
            self.centrality_measure_names = {
                'degree': 'Degree Centrality (Most Connected)',
                'in_degree': 'In-Degree Centrality (Most Referenced)',
                'out_degree': 'Out-Degree Centrality (Most References Others)',
                'betweenness': 'Betweenness Centrality (Bridges Between Concepts)',
                'closeness': 'Closeness Centrality (Central to Graph)',
                'eigenvector': 'Eigenvector Centrality (Connected to Important Nodes)',
                'pagerank': 'PageRank (Overall Importance)'
            }


@dataclass
class KnowledgeGraphConfig:
    """Main configuration class combining all configs"""
    llm: LLMConfig = None
    text_processing: TextProcessingConfig = None
    visualization: VisualizationConfig = None
    wordcloud: WordCloudConfig = None
    semantic_reasoning: SemanticReasoningConfig = None
    output: OutputConfig = None
    centrality: CentralityConfig = None

    def __post_init__(self):
        if self.llm is None:
            self.llm = LLMConfig()
        if self.text_processing is None:
            self.text_processing = TextProcessingConfig()
        if self.visualization is None:
            self.visualization = VisualizationConfig()
        if self.wordcloud is None:
            self.wordcloud = WordCloudConfig()
        if self.semantic_reasoning is None:
            self.semantic_reasoning = SemanticReasoningConfig()
        if self.output is None:
            self.output = OutputConfig()
        if self.centrality is None:
            self.centrality = CentralityConfig()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'KnowledgeGraphConfig':
        """Create configuration from dictionary"""
        return cls(
            llm=LLMConfig(**config_dict.get('llm', {})),
            text_processing=TextProcessingConfig(**config_dict.get('text_processing', {})),
            visualization=VisualizationConfig(**config_dict.get('visualization', {})),
            wordcloud=WordCloudConfig(**config_dict.get('wordcloud', {})),
            semantic_reasoning=SemanticReasoningConfig(**config_dict.get('semantic_reasoning', {})),
            output=OutputConfig(**config_dict.get('output', {})),
            centrality=CentralityConfig(**config_dict.get('centrality', {}))
        )


# Default configuration instance
DEFAULT_CONFIG = KnowledgeGraphConfig()

