Readme document for AI Knowledge Graph Builder
Automatically extract concepts, relationships, and insights from any AI/ML PDF using an LLM

This project builds a full AI Knowledge Graph from a PDF document using:

Ollama LLM (llama3.1 or any supported model)

NetworkX for building the graph

Matplotlib for visualization

PyPDF2 for text extraction

WordCloud for importance-based cloud

Semantic reasoning for inferring new relationships

The output includes:

✔ Extracted AI concepts and relationships (triplets)
✔ A directed knowledge graph
✔ Centrality scores (degree, betweenness, eigenvector, PageRank, etc.)
✔ Semantic inference
✔ Graph visualization PNG
✔ Word cloud PNG
✔ Exported JSON files

🔥 Features
1. PDF → Text Extraction

Extracts all text from a PDF using PyPDF2 and splits the content by pages.

2. LLM-Powered Triplet Extraction

Uses an Ollama model to extract knowledge graph triplets such as:

["Neural Network", "uses", "Backpropagation"]
["Gradient Descent", "optimizes", "Loss Function"]

3. Knowledge Graph Construction

Builds a directed graph (DiGraph) using NetworkX.

4. Graph Analytics

Calculates 7 centrality measures:

Degree

In-degree

Out-degree

Betweenness

Closeness

Eigenvector

PageRank

5. Semantic Reasoning

Automatically infers new relationships:

If A → B and B → C, the system infers A → C.

6. Visual Output

Generates:

knowledge_graph.png

wordcloud.png

7. JSON Export

Creates:

triplets.json

centrality_measures.json

🧩 Project Structure
AIKnowledgeGraph/
│
├── ai_knowledge_graph.py    # Main code (your file)
├── README.md                # Documentation
├── sample.pdf               # Your input PDF
│
├── triplets.json            # Exported triplets
├── centrality_measures.json # Exported centrality
├── knowledge_graph.png      # Graph visualization
└── wordcloud.png            # Word cloud

🚀 How It Works (Pipeline Overview)
Step 1 — Extract PDF Text
text = kg.extract_text_from_pdf(pdf_path)


Uses PyPDF2 to extract and mark pages.

Step 2 — Chunk Text

Large PDFs are split into manageable chunks for LLM processing:

kg.chunk_markdown(text, chunk_size=3000)

Step 3 — Extract Triplets (LLM)

Ollama model is used to extract 10–15 triplets per chunk:

response = self.client.chat(...)

Step 4 — Build the Knowledge Graph

Adds each extracted relationship as:

subject --predicate--> object

Step 5 — Compute Graph Analytics

NetworkX is used to compute centrality scores.

Step 6 — Semantic Inference

Automatic reasoning:
If A→B and B→C, infer A→C.

Step 7 — Visualize

Creates:

Graph visualization (top PageRank nodes)

Word cloud (importance-based)

Step 8 — Export

Triplets and metrics exported as JSON.

💻 Usage
Run the script
python ai_knowledge_graph.py

Or call main() manually
kg = main("lecture_notes.pdf", model="llama3.1")

🧪 Requirements

Install dependencies:

pip install networkx matplotlib wordcloud ollama PyPDF2


Make sure Ollama is installed and running:

ollama pull llama3.1

🧠 Example Output (Triplet)
1. (Neural Network) --[uses]--> (Activation Function)
2. (Transformer) --[based_on]--> (Self-Attention)
3. (Loss Function) --[optimized_by]--> (Gradient Descent)

🖼 Example Visualization

Size = PageRank importance

Color = Betweenness score

Edges show direction + relationships

📂 Exports
triplets.json

Structured list of all extracted relationships.

centrality_measures.json

Full analytics for every node.

🏁 Final Notes

This system is ideal for:

AI/ML learning material

Summaries of textbooks

Lecture notes

Research papers

Technical documentation
