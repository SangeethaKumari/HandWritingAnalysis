import re
import json
from collections import Counter, defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import PyPDF2
import spacy
from itertools import combinations
import ollama
import os

class AIKnowledgeGraph:
    def __init__(self, use_llm=True, llm_provider="ollama", api_key=None, model="llama3.1"):
        """
        Initialize Knowledge Graph Builder
        
        Args:
            use_llm: Whether to use LLM for extraction (default: True)
            llm_provider: "ollama" (local), "openai", or "anthropic"
            api_key: API key for OpenAI or Anthropic (if needed)
            model: Model name for Ollama (default: "llama3.1")
        """
        self.graph = nx.DiGraph()
        self.triplets = []
        self.entities = []
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.client = None
        
        # Initialize Ollama client if using ollama
        if use_llm and llm_provider == "ollama":
            self.client = ollama.Client()
        
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
    
    def extract_with_llm_ollama(self, text, chunk_size=2000):
        """Extract triplets using local Ollama LLM"""
        print(f"Using Ollama with model: {self.model}")
        
        # Split text into chunks
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        all_triplets = []
        
        for idx, chunk in enumerate(chunks[:5]):  # Process first 5 chunks
            print(f"Processing chunk {idx+1}/{min(5, len(chunks))}...")
            
            prompt = f"""You are an expert in AI and machine learning. Extract knowledge graph triplets from the following text about AI concepts.

Format each triplet as: (Subject, Predicate, Object)

Rules:
- Subject and Object should be specific AI concepts, algorithms, or techniques
- Predicate should be a relationship like: "is_type_of", "uses", "requires", "improves", "part_of", "applied_in", "measures", "optimizes"
- Extract 10-15 most important triplets
- Return ONLY a JSON array of triplets, no additional text

Text:
{chunk}

Return format:
[
  ["Neural Network", "is_type_of", "Machine Learning Model"],
  ["Backpropagation", "uses", "Chain Rule"],
  ...
]"""

            try:
                response = self.client.chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    options={
                        "temperature": 0,  # Deterministic for extraction
                        "num_predict": 2000,  # Max tokens
                    }
                )
                
                content = response['message']['content']
                
                # Extract JSON from response
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    triplets = json.loads(json_match.group())
                    all_triplets.extend([(t[0], t[1], t[2]) for t in triplets if len(t) == 3])
                    print(f"  ✓ Extracted {len([t for t in triplets if len(t) == 3])} triplets from chunk {idx+1}")
                
            except Exception as e:
                print(f"  ✗ Error processing chunk {idx+1}: {e}")
        
        print(f"✓ Total triplets extracted: {len(all_triplets)}")
        return all_triplets
    
    def extract_with_llm_openai(self, text, chunk_size=3000):
        """Extract triplets using OpenAI API"""
        print("Using OpenAI API for extraction...")
        
        if not self.api_key:
            print("Error: OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            return []
        
        # Split text into chunks
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        all_triplets = []
        
        for idx, chunk in enumerate(chunks[:5]):  # Process first 5 chunks
            print(f"Processing chunk {idx+1}/{min(5, len(chunks))}...")
            
            prompt = f"""Extract knowledge graph triplets from this AI/ML text.

Format: Return ONLY a JSON array of triplets [subject, predicate, object]

Rules:
- Focus on AI concepts, algorithms, techniques
- Use predicates: is_type_of, uses, requires, improves, part_of, applied_in, measures, optimizes
- Extract 10-15 key triplets
- Return pure JSON, no markdown

Text:
{chunk}

JSON:"""

            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an AI knowledge graph expert. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                
                content = response.choices[0].message.content
                
                # Clean and parse JSON
                content = content.strip().strip('```json').strip('```').strip()
                triplets = json.loads(content)
                all_triplets.extend([(t[0], t[1], t[2]) for t in triplets if len(t) == 3])
                
            except Exception as e:
                print(f"Error with OpenAI: {e}")
        
        return all_triplets
    
    def extract_with_llm_anthropic(self, text, chunk_size=3000):
        """Extract triplets using Anthropic Claude API"""
        print("Using Anthropic Claude API for extraction...")
        
        if not self.api_key:
            print("Error: Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
            return []
        
        # Split text into chunks
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        all_triplets = []
        
        for idx, chunk in enumerate(chunks[:5]):
            print(f"Processing chunk {idx+1}/{min(5, len(chunks))}...")
            
            prompt = f"""Extract knowledge graph triplets from this AI/ML text.

Format: Return ONLY a JSON array of triplets [subject, predicate, object]

Rules:
- Focus on AI concepts, algorithms, techniques
- Use predicates: is_type_of, uses, requires, improves, part_of, applied_in, measures, optimizes
- Extract 10-15 key triplets
- Return pure JSON array, no other text

Text:
{chunk}"""

            try:
                import anthropic
                client = anthropic.Anthropic(api_key=self.api_key)
                
                message = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1500,
                    temperature=0.3,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                content = message.content[0].text
                
                # Clean and parse JSON
                content = content.strip().strip('```json').strip('```').strip()
                triplets = json.loads(content)
                all_triplets.extend([(t[0], t[1], t[2]) for t in triplets if len(t) == 3])
                
            except Exception as e:
                print(f"Error with Anthropic: {e}")
        
        return all_triplets
    
    def extract_with_spacy(self, text):
        """Traditional NLP extraction with spaCy"""
        print("Using spaCy for extraction...")
        
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            print("Please install spaCy model: python -m spacy download en_core_web_sm")
            return []
        
        doc = nlp(text[:100000])  # Limit text size
        
        # Extract named entities and noun chunks
        entities = set()
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'TECH', 'CONCEPT']:
                entities.add(ent.text)
        
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 4:
                entities.add(chunk.text.title())
        
        self.entities = list(entities)
        
        # Create triplets
        triplets = []
        for sent in doc.sents:
            sent_doc = nlp(sent.text)
            subject, verb, obj = None, None, None
            
            for token in sent_doc:
                if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                    subject = token.text
                    verb = token.head.text
                    
                if token.dep_ in ["dobj", "attr", "pobj"] and token.head.pos_ in ["VERB", "ADP"]:
                    obj = token.text
            
            if subject and verb and obj:
                triplets.append((subject.title(), verb.lower(), obj.title()))
        
        # Add co-occurrence relationships
        sentences = [sent.text for sent in doc.sents]
        for sent in sentences[:50]:
            sent_entities = [e for e in entities if e.lower() in sent.lower()]
            if len(sent_entities) >= 2:
                for e1, e2 in combinations(sent_entities[:5], 2):
                    triplets.append((e1, "related_to", e2))
        
        return triplets
    
    def extract_entities_and_relations(self, text):
        """Step 3 & 4: Extract entities and create triplets"""
        
        if self.use_llm:
            if self.llm_provider == "ollama":
                self.triplets = self.extract_with_llm_ollama(text)
            elif self.llm_provider == "openai":
                self.triplets = self.extract_with_llm_openai(text)
            elif self.llm_provider == "anthropic":
                self.triplets = self.extract_with_llm_anthropic(text)
            else:
                print(f"Unknown LLM provider: {self.llm_provider}, falling back to spaCy")
                self.triplets = self.extract_with_spacy(text)
        else:
            self.triplets = self.extract_with_spacy(text)
        
        # Remove duplicates
        self.triplets = list(set(self.triplets))
        
        print(f"✓ Created {len(self.triplets)} unique triplets")
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
        for node in list(self.graph.nodes())[:100]:  # Limit for performance
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
        node_degrees = dict(self.graph.degree())
        
        if not node_degrees:
            print("No data for word cloud")
            return
        
        word_freq = {node: degree * 10 for node, degree in node_degrees.items()}
        
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
        node_degrees = dict(self.graph.degree())
        top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_names = [node for node, _ in top_nodes]
        
        subgraph = self.graph.subgraph(top_node_names)
        
        plt.figure(figsize=(20, 16))
        pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
        
        node_sizes = [subgraph.degree(node) * 100 + 300 for node in subgraph.nodes()]
        nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.7, edgecolors='black', linewidths=2)
        
        nx.draw_networkx_edges(subgraph, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, alpha=0.5, width=2)
        
        nx.draw_networkx_labels(subgraph, pos, font_size=9, font_weight='bold')
        
        plt.title(f'AI Knowledge Graph (Top {max_nodes} Concepts)', fontsize=24, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✓ Graph visualization saved to {output_path}")
    
    def get_statistics(self):
        """Get knowledge graph statistics"""
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty!")
            return {}
            
        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "total_triplets": len(self.triplets),
            "avg_degree": sum(dict(self.graph.degree()).values()) / max(self.graph.number_of_nodes(), 1),
            "density": nx.density(self.graph)
        }
        
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

def main(pdf_path, use_llm=True, llm_provider="ollama", api_key=None, model="llama3.1"):
    """
    Main pipeline to build knowledge graph from PDF
    
    Args:
        pdf_path: Path to your AI class PDF
        use_llm: Use LLM for extraction (default: True)
        llm_provider: "ollama" (local, free), "openai", or "anthropic"
        api_key: API key for OpenAI or Anthropic
        model: Model name for Ollama (default: "llama3.1")
    """
    print("\n🚀 Starting AI Knowledge Graph Builder")
    print("="*60)
    print(f"LLM Mode: {'Enabled' if use_llm else 'Disabled'}")
    if use_llm:
        print(f"LLM Provider: {llm_provider}")
        if llm_provider == "ollama":
            print(f"Model: {model}")
    print("="*60 + "\n")
    
    # Initialize
    kg = AIKnowledgeGraph(use_llm=use_llm, llm_provider=llm_provider, api_key=api_key, model=model)
    
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
    # Option 1: Use local Ollama (FREE, recommended)
    pdf_path = "/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/AI Agents Week 8 Summary.pdf"

    kg = main(pdf_path, use_llm=True, llm_provider="ollama")
    
    # Option 2: Use OpenAI
    # kg = main("ai_class_summary.pdf", use_llm=True, llm_provider="openai", api_key="your-key")
    
    # Option 3: Use Anthropic Claude
    # kg = main("ai_class_summary.pdf", use_llm=True, llm_provider="anthropic", api_key="your-key")
    
    # Option 4: Use spaCy (no LLM)
    # kg = main("ai_class_summary.pdf", use_llm=False)