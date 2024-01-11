from langchain.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship,
    GraphDocument,
)
from langchain.schema import Document
from typing import List, Dict, Any, Optional
from langchain.pydantic_v1 import Field, BaseModel
import os
from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)
from langchain_community.vectorstores import Chroma
from langchain.graphs import Neo4jGraph
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter
from tqdm import tqdm
from langchain import hub
from pyvis.network import Network
import networkx as nx
from networkx.readwrite import json_graph
import time
os.environ["OPENAI_API_KEY"] = "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"



class Property(BaseModel):
  """A single property consisting of key and value"""
  key: str = Field(..., description="key")
  value: str = Field(..., description="value")

class Node(BaseNode):
    properties: Optional[List[Property]] = Field(
        None, description="List of node properties")

class Relationship(BaseRelationship):
    properties: Optional[List[Property]] = Field(
        None, description="List of relationship properties"
    )

class KnowledgeGraph(BaseModel):
    """Generate a knowledge graph with entities and relationships."""
    nodes: List[Node] = Field(
        ..., description="List of nodes in the knowledge graph")
    rels: List[Relationship] = Field(
        default_factory=list, description="List of relationships in the knowledge graph"
    )


def format_property_key(s: str) -> str:
    words = s.split()
    if not words:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)

def props_to_dict(props) -> dict:
    """Convert properties to a dictionary."""
    properties = {}
    if not props:
      return properties
    for p in props:
        properties[format_property_key(p.key)] = p.value
    return properties

def map_to_base_node(node: Node) -> BaseNode:
    """Map the KnowledgeGraph Node to the base Node."""
    properties = props_to_dict(node.properties) if node.properties else {}
    # Add name property for better Cypher statement generation
    properties["name"] = node.id.title()
    return BaseNode(
        id=node.id.title(), type=node.type.capitalize(), properties=properties
    )


def map_to_base_relationship(rel: Relationship) -> BaseRelationship:
    """Map the KnowledgeGraph Relationship to the base Relationship."""
    source = map_to_base_node(rel.source)
    target = map_to_base_node(rel.target)
    properties = props_to_dict(rel.properties) if rel.properties else {}
    return BaseRelationship(
        source=source, target=target, type=rel.type, properties=properties
    )


llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

def get_extraction_chain(
    allowed_nodes: Optional[List[str]] = None,
    allowed_rels: Optional[List[str]] = None
    ):
    prompt = ChatPromptTemplate.from_messages(
        [(
          "system",
          f"""# Knowledge Graph Instructions for GPT-4
## 1. Overview
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
- **Nodes** represent entities and concepts. They're akin to Wikipedia nodes.
- The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible for a vast audience.
## 2. Labeling Nodes
- **Consistency**: Ensure you use basic or elementary types for node labels.
  - For example, when you identify an entity representing a topic, always label it as **"topic"**. Avoid using more specific terms like "effect" or "subject".
- **Node IDs**: Never utilize integers as node IDs. Node IDs should be names or human-readable identifiers found in the text.
{'- **Allowed Node Labels:**' + ", ".join(allowed_nodes) if allowed_nodes else ""}
{'- **Allowed Relationship Types**:' + ", ".join(allowed_rels) if allowed_rels else ""}
## 3. Handling Numerical Data and Dates
- Numerical data, like age or other related information, should be incorporated as attributes or properties of the respective nodes.
- **No Separate Nodes for Dates/Numbers**: Do not create separate nodes for dates or numerical values. Always attach them as attributes or properties of nodes.
- **Property Format**: Properties must be in a key-value format.
- **Quotation Marks**: Never use escaped single or double quotes within property values.
- **Naming Convention**: Use camelCase for property keys, e.g., `birthDate`.
## 4. Coreference Resolution
- **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency.
If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"),
always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.
Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.
## 5. Strict Compliance
Adhere to the rules strictly. Non-compliance will result in termination.
          """),
            ("human", "Use the given format to extract information from the following input: {input}"),
            ("human", "Tip: Make sure to answer in the correct format"),
        ])
    return create_structured_output_chain(KnowledgeGraph, llm, prompt, verbose=True)

rels_set = set()
G = nx.DiGraph()


def extract_and_store_graph(
    document: Document,
    nodes: Optional[List[str]] = None,
    rels: Optional[List[str]] = None) -> None:
    # Extract graph data using OpenAI functions
    extract_chain = get_extraction_chain(nodes, rels)
    data = extract_chain.run(document.page_content)
    # Construct a graph document
    graph_document = GraphDocument(
      nodes = [map_to_base_node(node) for node in data.nodes],
      relationships = [map_to_base_relationship(rel) for rel in data.rels],
      source = document
    )
    for rel in graph_document.relationships:
        G.add_edge(rel.source.id, rel.target.id)
    for n in graph_document.nodes:
        G.add_node(n.id)
    # Store information into a graph
    # graph.add_graph_documents([graph_document])

# Read the wikipedia article
# with open("Квантовая физика/Глава 11. Световые кванты./87. Фотоэффект.txt", 'r') as f1, \
#         open("Квантовая физика/Глава 11. Световые кванты./88. Теория фотоэффекта.txt", 'r') as f2, \
#         open("Квантовая физика/Глава 11. Световые кванты./90. Применение фотоэффекта.txt", 'r') as f3, \
raw_documents=[]
def ret_dirs(path):
    dirs = set()
    flag = False
    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)):
            dirs.update(ret_dirs(os.path.join(path, file)))
        else:
            flag = True
    if flag:
        dirs.update([path])
    return dirs

dirs = set()
for path in ['Оптика', 'Колебания и волны', 'Квантовая физика', 'Астрономия']:
    dirs.update(ret_dirs(path))

for dir in dirs:
    for file in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, file)):
            if file[-1] == 't':
                with open(os.path.join(dir, file), 'r') as f:
                    raw_documents.append(Document(page_content=f.read()))
    # raw_documents = [
    #     Document(page_content=f1.read()), Document(page_content=f2.read()),
    #     Document(page_content=f3.read()), Document(page_content=f4.read())
    #     ]
# Define chunking strategy
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=2048,
    chunk_overlap=200,
    add_start_index=True
)

# Only take the first the raw_documents
documents = raw_documents # text_splitter.split_documents(raw_documents)

# Specify which node labels should be extracted by the LLM
allowed_nodes = ["Term", "Definition", "Theorem"]
allowed_rels = ["Depends"]
# vectorstore = Chroma.from_documents(documents=documents[:10], embedding=OpenAIEmbeddings())

for i, d in tqdm(enumerate(documents), total=len(documents)):
    s =time.time()
    extract_and_store_graph(d)
    d = time.time() - s
    if d < 25:
        time.sleep(25 - d)

# Retrieve and generate using the relevant snippets of the blog.
# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


# Create a NetworkX graph from the extracted relation triplets
def create_graph_from_triplets():
    return G

# Convert the NetworkX graph to a PyVis network
def nx_to_pyvis(networkx_graph):
    pyvis_graph = Network(notebook=True, cdn_resources='remote')
    for node in networkx_graph.nodes():
        pyvis_graph.add_node(node)
    for edge in networkx_graph.edges(data=True):
        pyvis_graph.add_edge(edge[0], edge[1])
    return pyvis_graph

graph = create_graph_from_triplets()
with open('first_graph.json', 'w') as f:
    f.write(json_graph.dumps(graph))
pyvis_network = nx_to_pyvis(graph)

# Customize the appearance of the graph
pyvis_network.toggle_hide_edges_on_drag(True)
pyvis_network.toggle_physics(False)
pyvis_network.set_edge_smooth('discrete')

# Show the interactive knowledge graph visualization
pyvis_network.show("knowledge_graph.html")
pyvis_network.save_graph()