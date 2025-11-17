import re
import json
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import ollama
import PyPDF2

class AIKnowledgeGraph:
    def __init__(self, model="gpt-oss:20b"):
        """
        Initialize Knowledge Graph Builder with Ollama
        
        Args:
            model: Ollama model name (default: "llama3.1")
        """
        self.graph = nx.DiGraph()
        self.triplets = []
        self.model = model
        self.client = ollama.Client()
        self.centrality_measures = {}
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using PyPDF2"""
        print(f"📄 Extracting text from PDF using PyPDF2...")
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
            return ""
    
    def chunk_markdown_normal(self, text, chunk_size=2000):
        """Chunk text intelligently by pages and paragraphs"""
        # Split by page markers first
        pages = re.split(r'\n--- Page \d+ ---\n', text)
        
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
    """
    def chunk_with_docling(self, pdf_path, chunk_size=2000):
        print("📘 Chunking PDF using Docling...")

        # Step 1: Parse PDF into structured Docling Document
        converter = DocumentConverter(pipeline=StandardPdfPipeline())
        result = converter.convert(pdf_path)

        # Step 2: Get clean markdown from Docling
        markdown = result.document.export(ExportFormat.MARKDOWN)
        markdown_text = markdown.markdown

        print(f"✓ Extracted {len(markdown_text)} characters of structured text")

        # Step 3: Actual chunking
        chunks = []
        current = ""

        for para in markdown_text.split("\n\n"):
            if len(current) + len(para) < chunk_size:
                current += para + "\n\n"
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = para + "\n\n"

        if current.strip():
            chunks.append(current.strip())

        print(f"✓ Created {len(chunks)} high-quality Docling chunks")
        return chunks
"""
    def extract_triplets_with_llm(self, text):
        """Extract triplets using Ollama LLM"""
        print(f"🤖 Using Ollama model: {self.model}")
        
        chunks = self.chunk_markdown_normal(text, chunk_size=3000)
        all_triplets = []
        
        for idx, chunk in enumerate(chunks[:10]):  # Process first 10 chunks
            print(f"\nProcessing chunk {idx+1}/{min(10, len(chunks))}...")
            
            prompt = f"""Extract knowledge graph triplets from this AI/ML text.

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

            try:
                response = self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "temperature": 0.1,
                        "num_predict": 1500,
                    }
                )
                
                content = response['message']['content']
                
                # Clean markdown code blocks
                content = re.sub(r'```json\s*', '', content)
                content = re.sub(r'```\s*', '', content)
                content = content.strip()
                
                # Extract JSON array
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    triplets = json.loads(json_match.group())
                    valid_triplets = [(t[0], t[1], t[2]) for t in triplets if len(t) == 3]
                    all_triplets.extend(valid_triplets)
                    print(f"  ✓ Extracted {len(valid_triplets)} triplets")
                else:
                    print(f"  ✗ Could not parse JSON from response")
                
            except json.JSONDecodeError as e:
                print(f"  ✗ JSON parsing error: {e}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        print(f"\n✓ Total triplets extracted: {len(all_triplets)}")
        return all_triplets
    
    def build_graph(self):
        """Build the knowledge graph from triplets"""
        for subject, predicate, obj in self.triplets:
            self.graph.add_edge(subject, obj, relation=predicate)
        
        print(f"✓ Built graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def calculate_centrality_measures(self):
        """Calculate multiple centrality measures"""
        print("\n📊 Calculating centrality measures...")
        
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty!")
            return {}
        
        # Calculate various centrality measures
        self.centrality_measures = {
            'degree': nx.degree_centrality(self.graph),
            'in_degree': nx.in_degree_centrality(self.graph),
            'out_degree': nx.out_degree_centrality(self.graph),
            'betweenness': nx.betweenness_centrality(self.graph),
            'closeness': nx.closeness_centrality(self.graph),
            'eigenvector': nx.eigenvector_centrality(self.graph, max_iter=1000),
            'pagerank': nx.pagerank(self.graph)
        }
        
        print("✓ Calculated 7 centrality measures")
        return self.centrality_measures
    
    def semantic_reasoning(self):
        """Apply semantic reasoning to infer new relationships"""
        print("\n🧠 Applying semantic reasoning...")
        new_triplets = []
        
        # Transitivity: if A->B and B->C, then A->C
        for node in list(self.graph.nodes())[:100]:
            successors_1 = list(self.graph.successors(node))
            for succ in successors_1[:5]:
                successors_2 = list(self.graph.successors(succ))
                for final in successors_2[:5]:
                    if final != node and not self.graph.has_edge(node, final):
                        new_triplets.append((node, "inferred_from", final))
                        self.graph.add_edge(node, final, relation="inferred_from")
        
        print(f"✓ Inferred {len(new_triplets)} new relationships")
        return new_triplets
    
    def print_triplets(self):
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
    
    def print_centrality_analysis(self):
        """Print detailed centrality analysis"""
        if not self.centrality_measures:
            self.calculate_centrality_measures()
        
        print("\n" + "="*80)
        print("CENTRALITY ANALYSIS")
        print("="*80)
        
        centrality_names = {
            'degree': 'Degree Centrality (Most Connected)',
            'in_degree': 'In-Degree Centrality (Most Referenced)',
            'out_degree': 'Out-Degree Centrality (Most References Others)',
            'betweenness': 'Betweenness Centrality (Bridges Between Concepts)',
            'closeness': 'Closeness Centrality (Central to Graph)',
            'eigenvector': 'Eigenvector Centrality (Connected to Important Nodes)',
            'pagerank': 'PageRank (Overall Importance)'
        }
        
        for measure_key, measure_name in centrality_names.items():
            centrality = self.centrality_measures[measure_key]
            top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            
            print(f"\n{measure_name}:")
            print("-" * 80)
            for i, (node, score) in enumerate(top_nodes, 1):
                print(f"  {i}. {node:50s} {score:.4f}")
        
        print("="*80 + "\n")
    
    def get_statistics(self):
        """Get comprehensive graph statistics"""
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty!")
            return {}
        
        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "total_triplets": len(self.triplets),
            "avg_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            "density": nx.density(self.graph),
            "is_connected": nx.is_weakly_connected(self.graph),
            "num_components": nx.number_weakly_connected_components(self.graph)
        }
        
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
    
    def visualize_graph(self, output_path="knowledge_graph._gpt-oss:20b.png", max_nodes=50):
        """Visualize the knowledge graph with centrality-based sizing"""
        if not self.centrality_measures:
            self.calculate_centrality_measures()
        
        # Get top nodes by PageRank
        pagerank = self.centrality_measures['pagerank']
        top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_names = [node for node, _ in top_nodes]
        
        subgraph = self.graph.subgraph(top_node_names)
        
        plt.figure(figsize=(24, 18))
        pos = nx.spring_layout(subgraph, k=2.5, iterations=50, seed=42)
        
        # Node sizes based on PageRank
        node_sizes = [pagerank[node] * 15000 + 500 for node in subgraph.nodes()]
        
        # Node colors based on betweenness centrality
        betweenness = self.centrality_measures['betweenness']
        node_colors = [betweenness.get(node, 0) for node in subgraph.nodes()]
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(
            subgraph, pos, 
            node_size=node_sizes,
            node_color=node_colors,
            cmap='YlOrRd',
            alpha=0.8,
            edgecolors='black',
            linewidths=2
        )
        
        # Draw edges with labels
        nx.draw_networkx_edges(
            subgraph, pos,
            edge_color='gray',
            arrows=True,
            arrowsize=25,
            alpha=0.4,
            width=2,
            connectionstyle='arc3,rad=0.1'
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            subgraph, pos,
            font_size=10,
            font_weight='bold',
            font_color='black'
        )
        
        # Add colorbar
        plt.colorbar(nodes, label='Betweenness Centrality', shrink=0.8)
        
        plt.title(f'AI Knowledge Graph - Top {max_nodes} Concepts\n(Size: PageRank, Color: Betweenness)', 
                 fontsize=26, pad=20, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✓ Graph visualization saved to {output_path}")
    
    def generate_word_cloud(self, output_path="wordcloud_gpt-oss:20b.png"):
        """Generate word cloud based on PageRank"""
        if not self.centrality_measures:
            self.calculate_centrality_measures()
        
        pagerank = self.centrality_measures['pagerank']
        
        if not pagerank:
            print("No data for word cloud")
            return
        
        # Scale PageRank values for word cloud
        word_freq = {node: score * 1000 for node, score in pagerank.items()}
        
        wordcloud = WordCloud(
            width=1600,
            height=800,
            background_color='white',
            colormap='viridis',
            relative_scaling=0.5,
            min_font_size=12
        ).generate_from_frequencies(word_freq)
        
        plt.figure(figsize=(20, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('AI Knowledge Graph - Concept Importance (PageRank)', 
                 fontsize=24, pad=20, fontweight='bold')
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Word cloud saved to {output_path}")
    
    def export_data(self):
        """Export triplets and centrality measures"""
        # Export triplets
        with open("triplets.json", 'w') as f:
            json.dump(self.triplets, f, indent=2)
        print(f"✓ Triplets exported to triplets.json")
        
        # Export centrality measures
        if self.centrality_measures:
            centrality_export = {}
            for measure, values in self.centrality_measures.items():
                centrality_export[measure] = [
                    {"concept": node, "score": score}
                    for node, score in sorted(values.items(), key=lambda x: x[1], reverse=True)
                ]
            
            with open("centrality_measures.json", 'w') as f:
                json.dump(centrality_export, f, indent=2)
            print(f"✓ Centrality measures exported to centrality_measures.json")

def main(pdf_path, model="llama3.1"):
    """
    Main pipeline to build knowledge graph from PDF
    
    Args:
        pdf_path: Path to your AI class PDF
        model: Ollama model name (default: "llama3.1")
    """
    print("\n" + "="*80)
    print("🚀 AI KNOWLEDGE GRAPH BUILDER")
    print("="*80)
    print(f"Model: Ollama - {model}")
    print(f"PDF: {pdf_path}")
    print("="*80 + "\n")
    
    # Initialize
    kg = AIKnowledgeGraph(model=model)
    
    # Step 1: Extract text using Docling
    print("Step 1: Extracting text from PDF with Docling...")
    text = kg.extract_text_from_pdf(pdf_path)
    
    if not text:
        print("Failed to extract text from PDF!")
        return None
    
    # Step 2: Extract triplets with LLM
    print("\nStep 2: Extracting knowledge triplets with LLM...")
    kg.triplets = kg.extract_triplets_with_llm(text)
    
    # Print triplets
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

# Example usage
if __name__ == "__main__":
    # Run with default llama3.1
    pdf_path = "/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/AI Agents Week 8 Summary.pdf"

    kg = main(pdf_path, model="llama3.1")
    
    # Or use different model
    # kg = main("ai_class_summary.pdf", model="llama3.2")