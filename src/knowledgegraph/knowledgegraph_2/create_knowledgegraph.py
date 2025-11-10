"""
Knowledge Graph Builder using LLM for triplet extraction
Applies OOP principles with separation of concerns
"""
import re
import json
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import ollama
import PyPDF2

# Constants
EDGE_ATTRIBUTE_RELATION = "relation"

from config import KnowledgeGraphConfig, DEFAULT_CONFIG
from triplets_prompts import KnowledgeGraphPrompts


class PDFExtractor:
    """Handles PDF text extraction"""
    
    def __init__(self, config: KnowledgeGraphConfig = None):
        """
        Initialize PDF extractor
        
        Args:
            config: Configuration object (default: DEFAULT_CONFIG)
        """
        self.config = config or DEFAULT_CONFIG
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF using PyPDF2
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: For other PDF reading errors
        """
        print(f"📄 Extracting text from PDF using PyPDF2...")
        pdf_path_obj = Path(pdf_path)
        
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                print(f"✓ Document contains {len(pdf_reader.pages)} pages")
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    text += f"\n\n--- Page {page_num} ---\n\n{page_text}"
            
            print(f"✓ Extracted {len(text)} characters")
            return text
            
        except Exception as e:
            print(f"✗ Error extracting PDF: {e}")
            raise


class TextChunker:
    """Handles text chunking for processing"""
    
    def __init__(self, config: KnowledgeGraphConfig = None):
        """
        Initialize text chunker
        
        Args:
            config: Configuration object (default: DEFAULT_CONFIG)
        """
        self.config = config or DEFAULT_CONFIG
    
    def chunk_text(self, text: str, chunk_size: Optional[int] = None) -> List[str]:
        """
        Chunk text intelligently by pages and paragraphs
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk (default: from config)
            
        Returns:
            List of text chunks
        """
        if chunk_size is None:
            chunk_size = self.config.text_processing.default_chunk_size
        
        # Split by page markers first
        pattern = self.config.text_processing.page_marker_pattern
        pages = re.split(pattern, text)
        
        chunks = []
        current_chunk = ""
        
        for page in pages:
            # Split page into paragraphs
            paragraphs = page.split('\n\n')
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                    
                if len(current_chunk) + len(para) < chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        print(f"✓ Created {len(chunks)} chunks from text")
        return chunks


class LLMTripletExtractor:
    """Handles triplet extraction using LLM"""
    
    def __init__(self, model: str = None, config: KnowledgeGraphConfig = None):
        """
        Initialize LLM triplet extractor
        
        Args:
            model: Ollama model name (default: from config)
            config: Configuration object (default: DEFAULT_CONFIG)
        """
        self.config = config or DEFAULT_CONFIG
        self.model = model or self.config.llm.default_model
        self.client = ollama.Client()
        self.prompts = KnowledgeGraphPrompts()
    
    def extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract triplets from text using LLM
        
        Args:
            text: Text to extract triplets from
            
        Returns:
            List of triplets as (subject, predicate, object) tuples
        """
        print(f"🤖 Using Ollama model: {self.model}")
        
        chunker = TextChunker(self.config)
        chunk_size = self.config.text_processing.extraction_chunk_size
        chunks = chunker.chunk_text(text, chunk_size=chunk_size)
        
        max_chunks = self.config.text_processing.max_chunks_to_process
        chunks_to_process = chunks[:max_chunks]
        all_triplets = []
        
        for idx, chunk in enumerate(chunks_to_process, 1):
            print(f"\nProcessing chunk {idx}/{len(chunks_to_process)}...")
            
            try:
                triplets = self._extract_from_chunk(chunk)
                all_triplets.extend(triplets)
                print(f"  ✓ Extracted {len(triplets)} triplets")
                
            except Exception as e:
                print(f"  ✗ Error processing chunk: {e}")
        
        print(f"\n✓ Total triplets extracted: {len(all_triplets)}")
        return all_triplets
    
    def _extract_from_chunk(self, chunk: str) -> List[Tuple[str, str, str]]:
        """
        Extract triplets from a single chunk
        
        Args:
            chunk: Text chunk to process
            
        Returns:
            List of triplets
        """
        prompt = self.prompts.get_triplet_extraction_prompt(chunk)
        
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": self.config.llm.temperature,
                "num_predict": self.config.llm.num_predict,
            }
        )
        
        content = response['message']['content']
        content = self._clean_response(content)
        
        triplets = self._parse_json_response(content)
        valid_triplets = [
            (t[0], t[1], t[2]) 
            for t in triplets 
            if len(t) == 3 and all(isinstance(item, str) for item in t)
        ]
        
        return valid_triplets
    
    @staticmethod
    def _clean_response(content: str) -> str:
        """
        Clean markdown code blocks from LLM response
        
        Args:
            content: Raw response content
            
        Returns:
            Cleaned content
        """
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        return content.strip()
    
    @staticmethod
    def _parse_json_response(content: str) -> List[List[str]]:
        """
        Parse JSON array from response
        
        Args:
            content: Response content
            
        Returns:
            Parsed triplets list
            
        Raises:
            json.JSONDecodeError: If JSON parsing fails
        """
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            raise ValueError("Could not parse JSON from response")


class GraphBuilder:
    """Handles knowledge graph construction"""
    
    def __init__(self, config: KnowledgeGraphConfig = None):
        """
        Initialize graph builder
        
        Args:
            config: Configuration object (default: DEFAULT_CONFIG)
        """
        self.config = config or DEFAULT_CONFIG
        self.graph = nx.DiGraph()
    
    def build_from_triplets(self, triplets: List[Tuple[str, str, str]]) -> nx.DiGraph:
        """
        Build graph from triplets
        
        Args:
            triplets: List of (subject, predicate, object) tuples
            
        Returns:
            Built NetworkX directed graph
        """
        for subject, predicate, obj in triplets:
            self.graph.add_edge(subject, obj, **{EDGE_ATTRIBUTE_RELATION: predicate})
        
        print(f"✓ Built graph: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        return self.graph
    
    def get_graph(self) -> nx.DiGraph:
        """Get the current graph"""
        return self.graph


class GraphAnalyzer:
    """Handles graph analysis and centrality calculations"""
    
    def __init__(self, graph: nx.DiGraph, config: KnowledgeGraphConfig = None):
        """
        Initialize graph analyzer
        
        Args:
            graph: NetworkX directed graph
            config: Configuration object (default: DEFAULT_CONFIG)
        """
        self.graph = graph
        self.config = config or DEFAULT_CONFIG
        self.centrality_measures: Dict[str, Dict[str, float]] = {}
    
    def calculate_centrality_measures(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate multiple centrality measures
        
        Returns:
            Dictionary of centrality measures
        """
        print("\n📊 Calculating centrality measures...")
        
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty!")
            return {}
        
        max_iter = self.config.llm.max_iter_centrality
        
        self.centrality_measures = {
            'degree': nx.degree_centrality(self.graph),
            'in_degree': nx.in_degree_centrality(self.graph),
            'out_degree': nx.out_degree_centrality(self.graph),
            'betweenness': nx.betweenness_centrality(self.graph),
            'closeness': nx.closeness_centrality(self.graph),
            'eigenvector': nx.eigenvector_centrality(self.graph, max_iter=max_iter),
            'pagerank': nx.pagerank(self.graph)
        }
        
        print("✓ Calculated 7 centrality measures")
        return self.centrality_measures
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive graph statistics
        
        Returns:
            Dictionary of graph statistics
        """
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty!")
            return {}
        
        degree_dict = dict(self.graph.degree())
        
        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "avg_degree": sum(degree_dict.values()) / self.graph.number_of_nodes(),
            "density": nx.density(self.graph),
            "is_connected": nx.is_weakly_connected(self.graph),
            "num_components": nx.number_weakly_connected_components(self.graph)
        }
        
        return stats
    
    def print_centrality_analysis(self) -> None:
        """Print detailed centrality analysis"""
        if not self.centrality_measures:
            self.calculate_centrality_measures()
        
        print("\n" + "="*80)
        print("CENTRALITY ANALYSIS")
        print("="*80)
        
        measure_names = self.config.centrality.centrality_measure_names
        top_n = self.config.centrality.top_nodes_to_display
        
        for measure_key, measure_name in measure_names.items():
            centrality = self.centrality_measures.get(measure_key, {})
            top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            print(f"\n{measure_name}:")
            print("-" * 80)
            for i, (node, score) in enumerate(top_nodes, 1):
                print(f"  {i}. {node:50s} {score:.4f}")
        
        print("="*80 + "\n")


class SemanticReasoner:
    """Handles semantic reasoning to infer new relationships"""
    
    def __init__(self, graph: nx.DiGraph, config: KnowledgeGraphConfig = None):
        """
        Initialize semantic reasoner
        
        Args:
            graph: NetworkX directed graph
            config: Configuration object (default: DEFAULT_CONFIG)
        """
        self.graph = graph
        self.config = config or DEFAULT_CONFIG
    
    def infer_relationships(self) -> List[Tuple[str, str, str]]:
        """
        Apply semantic reasoning to infer new relationships
        
        Returns:
            List of inferred triplets
        """
        print("\n🧠 Applying semantic reasoning...")
        new_triplets = []
        
        config = self.config.semantic_reasoning
        max_nodes = config.max_nodes_to_process
        max_succ_1 = config.max_successors_level_1
        max_succ_2 = config.max_successors_level_2
        relation_type = config.inferred_relation_type
        
        # Transitivity: if A->B and B->C, then A->C
        nodes = list(self.graph.nodes())[:max_nodes]
        
        for node in nodes:
            successors_1 = list(self.graph.successors(node))[:max_succ_1]
            for succ in successors_1:
                successors_2 = list(self.graph.successors(succ))[:max_succ_2]
                for final in successors_2:
                    if final != node and not self.graph.has_edge(node, final):
                        new_triplets.append((node, relation_type, final))
                        self.graph.add_edge(node, final, **{EDGE_ATTRIBUTE_RELATION: relation_type})
        
        print(f"✓ Inferred {len(new_triplets)} new relationships")
        return new_triplets


class GraphVisualizer:
    """Handles graph visualization"""
    
    def __init__(self, graph: nx.DiGraph, centrality_measures: Dict[str, Dict[str, float]], 
                 config: KnowledgeGraphConfig = None):
        """
        Initialize graph visualizer
        
        Args:
            graph: NetworkX directed graph
            centrality_measures: Dictionary of centrality measures
            config: Configuration object (default: DEFAULT_CONFIG)
        """
        self.graph = graph
        self.centrality_measures = centrality_measures
        self.config = config or DEFAULT_CONFIG
    
    def visualize_graph(self, output_path: Optional[str] = None, 
                       max_nodes: Optional[int] = None) -> None:
        """
        Visualize the knowledge graph with centrality-based sizing
        
        Args:
            output_path: Path to save visualization (default: from config)
            max_nodes: Maximum nodes to display (default: from config)
        """
        if not self.centrality_measures:
            raise ValueError("Centrality measures must be calculated first")
        
        output_path = output_path or self.config.output.graph_visualization_file
        max_nodes = max_nodes or self.config.visualization.max_nodes
        
        # Get top nodes by PageRank
        pagerank = self.centrality_measures['pagerank']
        top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_names = [node for node, _ in top_nodes]
        
        subgraph = self.graph.subgraph(top_node_names)
        
        viz_config = self.config.visualization
        
        plt.figure(figsize=viz_config.figure_size)
        pos = nx.spring_layout(
            subgraph, 
            k=viz_config.k_spring_layout, 
            iterations=viz_config.iterations_spring_layout, 
            seed=viz_config.seed
        )
        
        # Node sizes based on PageRank
        node_sizes = [
            pagerank[node] * viz_config.node_size_multiplier + viz_config.node_size_base 
            for node in subgraph.nodes()
        ]
        
        # Node colors based on betweenness centrality
        betweenness = self.centrality_measures['betweenness']
        node_colors = [betweenness.get(node, 0) for node in subgraph.nodes()]
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(
            subgraph, pos, 
            node_size=node_sizes,
            node_color=node_colors,
            cmap='YlOrRd',
            alpha=viz_config.node_alpha,
            edgecolors='black',
            linewidths=viz_config.line_width
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            subgraph, pos,
            edge_color='gray',
            arrows=True,
            arrowsize=viz_config.arrow_size,
            alpha=viz_config.edge_alpha,
            width=viz_config.edge_width,
            connectionstyle='arc3,rad=0.1'
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            subgraph, pos,
            font_size=viz_config.font_size,
            font_weight='bold',
            font_color='black'
        )
        
        # Add colorbar
        plt.colorbar(nodes, label='Betweenness Centrality', shrink=viz_config.colorbar_shrink)
        
        plt.title(
            f'AI Knowledge Graph - Top {max_nodes} Concepts\n'
            f'(Size: PageRank, Color: Betweenness)', 
            fontsize=viz_config.title_font_size, 
            pad=20, 
            fontweight='bold'
        )
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=viz_config.dpi, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✓ Graph visualization saved to {output_path}")
    
    def generate_word_cloud(self, output_path: Optional[str] = None) -> None:
        """
        Generate word cloud based on PageRank
        
        Args:
            output_path: Path to save word cloud (default: from config)
        """
        if not self.centrality_measures:
            raise ValueError("Centrality measures must be calculated first")
        
        pagerank = self.centrality_measures['pagerank']
        
        if not pagerank:
            print("No data for word cloud")
            return
        
        output_path = output_path or self.config.output.wordcloud_file
        wc_config = self.config.wordcloud
        
        # Scale PageRank values for word cloud
        word_freq = {
            node: score * wc_config.pagerank_multiplier 
            for node, score in pagerank.items()
        }
        
        wordcloud = WordCloud(
            width=wc_config.width,
            height=wc_config.height,
            background_color=wc_config.background_color,
            colormap=wc_config.colormap,
            relative_scaling=wc_config.relative_scaling,
            min_font_size=wc_config.min_font_size
        ).generate_from_frequencies(word_freq)
        
        plt.figure(figsize=(20, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(
            'AI Knowledge Graph - Concept Importance (PageRank)', 
            fontsize=wc_config.title_font_size, 
            pad=20, 
            fontweight='bold'
        )
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=wc_config.dpi, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Word cloud saved to {output_path}")


class DataExporter:
    """Handles data export functionality"""
    
    def __init__(self, triplets: List[Tuple[str, str, str]], 
                 centrality_measures: Dict[str, Dict[str, float]],
                 config: KnowledgeGraphConfig = None):
        """
        Initialize data exporter
        
        Args:
            triplets: List of triplets
            centrality_measures: Dictionary of centrality measures
            config: Configuration object (default: DEFAULT_CONFIG)
        """
        self.triplets = triplets
        self.centrality_measures = centrality_measures
        self.config = config or DEFAULT_CONFIG
    
    def export_all(self) -> None:
        """Export triplets and centrality measures"""
        self.export_triplets()
        if self.centrality_measures:
            self.export_centrality_measures()
    
    def export_triplets(self, output_path: Optional[str] = None) -> None:
        """
        Export triplets to JSON file
        
        Args:
            output_path: Path to save triplets (default: from config)
        """
        output_path = output_path or self.config.output.triplets_file
        
        with open(output_path, 'w') as f:
            json.dump(self.triplets, f, indent=2)
        print(f"✓ Triplets exported to {output_path}")
    
    def export_centrality_measures(self, output_path: Optional[str] = None) -> None:
        """
        Export centrality measures to JSON file
        
        Args:
            output_path: Path to save centrality measures (default: from config)
        """
        output_path = output_path or self.config.output.centrality_file
        
        centrality_export = {}
        for measure, values in self.centrality_measures.items():
            centrality_export[measure] = [
                {"concept": node, "score": score}
                for node, score in sorted(values.items(), key=lambda x: x[1], reverse=True)
            ]
        
        with open(output_path, 'w') as f:
            json.dump(centrality_export, f, indent=2)
        print(f"✓ Centrality measures exported to {output_path}")


class AIKnowledgeGraph:
    """
    Main orchestrator class for building knowledge graphs
    Uses composition to delegate responsibilities to specialized classes
    """
    
    def __init__(self, model: Optional[str] = None, config: Optional[KnowledgeGraphConfig] = None):
        """
        Initialize Knowledge Graph Builder
        
        Args:
            model: Ollama model name (default: from config)
            config: Configuration object (default: DEFAULT_CONFIG)
        """
        self.config = config or DEFAULT_CONFIG
        self.model = model or self.config.llm.default_model
        
        # Initialize components
        self.pdf_extractor = PDFExtractor(self.config)
        self.triplet_extractor = LLMTripletExtractor(self.model, self.config)
        self.graph_builder = GraphBuilder(self.config)
        self.graph_analyzer: Optional[GraphAnalyzer] = None
        self.semantic_reasoner: Optional[SemanticReasoner] = None
        self.visualizer: Optional[GraphVisualizer] = None
        self.exporter: Optional[DataExporter] = None
        
        # Data storage
        self.triplets: List[Tuple[str, str, str]] = []
        self.graph: Optional[nx.DiGraph] = None
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        return self.pdf_extractor.extract_text(pdf_path)
    
    def extract_triplets_with_llm(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract triplets from text using LLM
        
        Args:
            text: Text to extract triplets from
            
        Returns:
            List of triplets
        """
        self.triplets = self.triplet_extractor.extract_triplets(text)
        return self.triplets
    
    def build_graph(self) -> nx.DiGraph:
        """
        Build knowledge graph from triplets
        
        Returns:
            Built NetworkX graph
        """
        self.graph = self.graph_builder.build_from_triplets(self.triplets)
        return self.graph
    
    def calculate_centrality_measures(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate centrality measures
        
        Returns:
            Dictionary of centrality measures
        """
        if self.graph is None:
            raise ValueError("Graph must be built before calculating centrality")
        
        self.graph_analyzer = GraphAnalyzer(self.graph, self.config)
        return self.graph_analyzer.calculate_centrality_measures()
    
    def semantic_reasoning(self) -> List[Tuple[str, str, str]]:
        """
        Apply semantic reasoning to infer new relationships
        
        Returns:
            List of inferred triplets
        """
        if self.graph is None:
            raise ValueError("Graph must be built before semantic reasoning")
        
        self.semantic_reasoner = SemanticReasoner(self.graph, self.config)
        new_triplets = self.semantic_reasoner.infer_relationships()
        self.triplets.extend(new_triplets)
        return new_triplets
    
    def print_triplets(self) -> None:
        """Print all extracted triplets"""
        print("\n" + "="*80)
        print("EXTRACTED KNOWLEDGE TRIPLETS")
        print("="*80)
        
        if not self.triplets:
            print("No triplets found!")
            return
        
        for idx, (subject, predicate, obj) in enumerate(self.triplets, 1):
            print(f"{idx:3d}. ({subject}) --[{predicate}]--> ({obj})")
        
        print("="*80 + "\n")
    
    def print_centrality_analysis(self) -> None:
        """Print detailed centrality analysis"""
        if self.graph_analyzer is None:
            raise ValueError("Centrality measures must be calculated first")
        self.graph_analyzer.print_centrality_analysis()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive graph statistics
        
        Returns:
            Dictionary of statistics
        """
        if self.graph_analyzer is None:
            if self.graph is None:
                raise ValueError("Graph must be built first")
            self.graph_analyzer = GraphAnalyzer(self.graph, self.config)
        
        stats = self.graph_analyzer.get_statistics()
        stats["total_triplets"] = len(self.triplets)
        
        print("\n" + "="*80)
        print("KNOWLEDGE GRAPH STATISTICS")
        print("="*80)
        print(f"Total Nodes (Concepts):        {stats['total_nodes']}")
        print(f"Total Edges (Relations):       {stats['total_edges']}")
        print(f"Total Triplets:                {stats['total_triplets']}")
        print(f"Average Degree:                {stats['avg_degree']:.2f}")
        print(f"Graph Density:                 {stats['density']:.4f}")
        print(f"Weakly Connected:              {stats['is_connected']}")
        print(f"Number of Components:          {stats['num_components']}")
        print("="*80 + "\n")
        
        return stats
    
    def visualize_graph(self, output_path: Optional[str] = None, 
                       max_nodes: Optional[int] = None) -> None:
        """
        Visualize the knowledge graph
        
        Args:
            output_path: Path to save visualization
            max_nodes: Maximum nodes to display
        """
        if self.graph_analyzer is None:
            raise ValueError("Centrality measures must be calculated first")
        
        centrality_measures = self.graph_analyzer.centrality_measures
        self.visualizer = GraphVisualizer(self.graph, centrality_measures, self.config)
        self.visualizer.visualize_graph(output_path, max_nodes)
    
    def generate_word_cloud(self, output_path: Optional[str] = None) -> None:
        """
        Generate word cloud
        
        Args:
            output_path: Path to save word cloud
        """
        if self.graph_analyzer is None:
            raise ValueError("Centrality measures must be calculated first")
        
        centrality_measures = self.graph_analyzer.centrality_measures
        self.visualizer = GraphVisualizer(self.graph, centrality_measures, self.config)
        self.visualizer.generate_word_cloud(output_path)
    
    def export_data(self) -> None:
        """Export triplets and centrality measures"""
        if self.graph_analyzer is None:
            raise ValueError("Centrality measures must be calculated first")
        
        centrality_measures = self.graph_analyzer.centrality_measures
        self.exporter = DataExporter(self.triplets, centrality_measures, self.config)
        self.exporter.export_all()


def main(pdf_path: str, model: Optional[str] = None, 
         config: Optional[KnowledgeGraphConfig] = None) -> Optional[AIKnowledgeGraph]:
    """
    Main pipeline to build knowledge graph from PDF
    
    Args:
        pdf_path: Path to PDF file
        model: Ollama model name (default: from config)
        config: Configuration object (default: DEFAULT_CONFIG)
        
    Returns:
        AIKnowledgeGraph instance or None if failed
    """
    config = config or DEFAULT_CONFIG
    model = model or config.llm.default_model
    
    print("\n" + "="*80)
    print("🚀 AI KNOWLEDGE GRAPH BUILDER")
    print("="*80)
    print(f"Model: Ollama - {model}")
    print(f"PDF: {pdf_path}")
    print("="*80 + "\n")
    
    try:
        # Initialize
        kg = AIKnowledgeGraph(model=model, config=config)
        
        # Step 1: Extract text from PDF
        print("Step 1: Extracting text from PDF...")
        text = kg.extract_text_from_pdf(pdf_path)
        
        if not text:
            print("Failed to extract text from PDF!")
            return None
        
        # Step 2: Extract triplets with LLM
        print("\nStep 2: Extracting knowledge triplets with LLM...")
        kg.extract_triplets_with_llm(text)
        kg.print_triplets()
        
        # Step 3: Build graph
        print("\nStep 3: Building knowledge graph...")
        kg.build_graph()
        
        # Step 4: Calculate centrality measures
        print("\nStep 4: Calculating centrality measures...")
        kg.calculate_centrality_measures()
        
        # Step 5: Semantic reasoning
        print("\nStep 5: Applying semantic reasoning...")
        kg.semantic_reasoning()
        
        # Display results
        kg.get_statistics()
        kg.print_centrality_analysis()
        
        # Step 6: Generate visualizations
        print("\nStep 6: Generating visualizations...")
        kg.visualize_graph()
        kg.generate_word_cloud()
        
        # Export data
        print("\nStep 7: Exporting data...")
        kg.export_data()
        
        print("\n" + "="*80)
        print("✅ KNOWLEDGE GRAPH CREATION COMPLETE!")
        print("="*80 + "\n")
        
        return kg
        
    except Exception as e:
        print(f"\n✗ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build knowledge graph from PDF")
    parser.add_argument(
        "pdf_path",
        type=str,
        nargs="?",
        default="/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/AI Agents Week 8 Summary.pdf",
        help="Path to PDF file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ollama model name (default: from config)"
    )
    
    args = parser.parse_args()
    
    kg = main(args.pdf_path, model=args.model)
