import re
import json
from collections import Counter, defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import PyPDF2
import spacy
from itertools import combinations

class AIKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.triplets = []
        self.entities = []
        
    def extract_text_from_pdf(self, pdf_path):
        """Step 1 & 2: Extract text from PDF"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            print(f"✓ Extracted {len(text)} characters from PDF")
            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep periods and commas
        text = re.sub(r'[^\w\s\.,\-]', '', text)
        return text.strip()
    
    def extract_entities_and_relations(self, text):
        """Step 3 & 4: Extract entities and create triplets using NLP"""
        # Load spaCy model (you'll need to install: python -m spacy download en_core_web_sm)
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            print("Please install spaCy model: python -m spacy download en_core_web_sm")
            return
        
        doc = nlp(text)
        
        # Extract named entities and noun chunks
        entities = set()
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'TECH', 'CONCEPT']:
                entities.add(ent.text)
        
        for chunk in doc.noun_chunks:
            # Filter AI-related terms
            if len(chunk.text.split()) <= 4:  # Keep phrases up to 4 words
                entities.add(chunk.text.title())
        
        self.entities = list(entities)
        print(f"✓ Extracted {len(entities)} entities")
        
        # Create triplets based on sentence structure
        for sent in doc.sents:
            sent_doc = nlp(sent.text)
            
            # Extract subject-verb-object patterns
            subject, verb, obj = None, None, None
            
            for token in sent_doc:
                if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                    subject = token.text
                    verb = token.head.text
                    
                if token.dep_ in ["dobj", "attr", "pobj"] and token.head.pos_ in ["VERB", "ADP"]:
                    obj = token.text
            
            if subject and verb and obj:
                self.triplets.append((subject.title(), verb.lower(), obj.title()))
        
        # Add co-occurrence based relationships
        sentences = [sent.text for sent in doc.sents]
        for sent in sentences[:50]:  # Limit to first 50 sentences
            sent_entities = [e for e in entities if e.lower() in sent.lower()]
            if len(sent_entities) >= 2:
                for e1, e2 in combinations(sent_entities[:5], 2):
                    self.triplets.append((e1, "related_to", e2))
        
        print(f"✓ Created {len(self.triplets)} triplets")
        return self.triplets
    
    def build_graph(self):
        """Step 4: Build the knowledge graph from triplets"""
        for subject, predicate, obj in self.triplets:
            self.graph.add_edge(subject, obj, relation=predicate)
        
        print(f"✓ Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def semantic_reasoning(self):
        """Step 5: Apply semantic reasoning to infer new relationships"""
        new_triplets = []
        
        # Transitivity: if A->B and B->C, then A->C
        for node in self.graph.nodes():
            successors_1 = list(self.graph.successors(node))
            for succ in successors_1:
                successors_2 = list(self.graph.successors(succ))
                for final in successors_2:
                    if final != node and not self.graph.has_edge(node, final):
                        new_triplets.append((node, "inferred_relation", final))
                        self.graph.add_edge(node, final, relation="inferred_relation")
        
        print(f"✓ Inferred {len(new_triplets)} new relationships")
        return new_triplets
    
    def generate_word_cloud(self, output_path="wordcloud.png"):
        """Step 6: Generate word cloud from knowledge graph"""
        # Count node frequencies (degree centrality)
        node_degrees = dict(self.graph.degree())
        
        # Create word frequency dictionary
        word_freq = {node: degree * 10 for node, degree in node_degrees.items()}
        
        if not word_freq:
            print("No data for word cloud")
            return
        
        # Generate word cloud
        wordcloud = WordCloud(width=1200, height=600, 
                             background_color='white',
                             colormap='viridis',
                             relative_scaling=0.5,
                             min_font_size=10).generate_from_frequencies(word_freq)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('AI Knowledge Graph - Concept Frequency', fontsize=20, pad=20)
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Word cloud saved to {output_path}")
    
    def visualize_graph(self, output_path="knowledge_graph.png", max_nodes=50):
        """Visualize the knowledge graph"""
        # Get top nodes by degree
        node_degrees = dict(self.graph.degree())
        top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_names = [node for node, _ in top_nodes]
        
        # Create subgraph
        subgraph = self.graph.subgraph(top_node_names)
        
        plt.figure(figsize=(20, 16))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
        
        # Draw nodes with size based on degree
        node_sizes = [subgraph.degree(node) * 100 + 300 for node in subgraph.nodes()]
        nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.7, edgecolors='black', linewidths=2)
        
        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, alpha=0.5, width=2)
        
        # Draw labels
        nx.draw_networkx_labels(subgraph, pos, font_size=9, font_weight='bold')
        
        plt.title(f'AI Knowledge Graph (Top {max_nodes} Concepts)', fontsize=24, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✓ Graph visualization saved to {output_path}")
    
    def get_statistics(self):
        """Get knowledge graph statistics"""
        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "total_triplets": len(self.triplets),
            "avg_degree": sum(dict(self.graph.degree()).values()) / max(self.graph.number_of_nodes(), 1),
            "density": nx.density(self.graph)
        }
        
        # Top concepts
        degree_centrality = nx.degree_centrality(self.graph)
        top_concepts = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print("\n" + "="*60)
        print("KNOWLEDGE GRAPH STATISTICS")
        print("="*60)
        print(f"Total Nodes (Concepts): {stats['total_nodes']}")
        print(f"Total Edges (Relations): {stats['total_edges']}")
        print(f"Total Triplets: {stats['total_triplets']}")
        print(f"Average Degree: {stats['avg_degree']:.2f}")
        print(f"Graph Density: {stats['density']:.4f}")
        print("\nTop 10 Most Connected Concepts:")
        for i, (concept, centrality) in enumerate(top_concepts, 1):
            print(f"  {i}. {concept} (centrality: {centrality:.3f})")
        print("="*60 + "\n")
        
        return stats
    
    def export_triplets(self, output_path="triplets.json"):
        """Export triplets to JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.triplets, f, indent=2)
        print(f"✓ Triplets exported to {output_path}")

# Main execution function
def main(pdf_path):
    """
    Main pipeline to build knowledge graph from PDF
    
    Args:
        pdf_path: Path to your AI class PDF
    """
    print("\n🚀 Starting AI Knowledge Graph Builder")
    print("="*60 + "\n")
    
    # Initialize
    kg = AIKnowledgeGraph()
    
    # Step 1-2: Extract and preprocess text
    print("Step 1-2: Extracting text from PDF...")
    text = kg.extract_text_from_pdf(pdf_path)
    clean_text = kg.preprocess_text(text)
    
    # Step 3-4: Extract entities and create triplets
    print("\nStep 3-4: Extracting entities and creating triplets...")
    kg.extract_entities_and_relations(clean_text)
    
    # Build graph
    print("\nBuilding knowledge graph...")
    kg.build_graph()
    
    # Step 5: Semantic reasoning
    print("\nStep 5: Applying semantic reasoning...")
    kg.semantic_reasoning()
    
    # Statistics
    kg.get_statistics()
    
    # Step 6: Generate visualizations
    print("Step 6: Generating visualizations...")
    kg.visualize_graph()
    kg.generate_word_cloud()
    
    # Export triplets
    kg.export_triplets()
    
    print("\n✅ Knowledge graph creation complete!")
    
    return kg

# Example usage:
if __name__ == "__main__":
    # Replace with your PDF path
    pdf_path = "ai_class_summary.pdf"
    kg = main(pdf_path)