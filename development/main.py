import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from utils import prepare_teacher_vectors, load_teachers

# Load environment variables
load_dotenv()
api_key = os.environ.get("PINECONE_API_KEY")

# Init Pinecone
pc = Pinecone(api_key=api_key)
index_name = "ibero-agent-index"
namespace_name = "teachers"

# Load external embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Try to connect to index and Create index if it doesn't exist
try:
    index = pc.Index(index_name)
except Exception:
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec={
                "serverless": {
                    "cloud": "aws",
                    "region": "us-east-1"
                }
            }
        )
        print(f"Índice {index_name} creado.")

    index = pc.Index(index_name)

    # Load teacher records and vectorize
    teachers = load_teachers("teachers.csv")
    vectors = prepare_teacher_vectors(teachers, embed_model)

    # Upsert data
    index.upsert(vectors=vectors, namespace=namespace_name)


# Input del usuario
query = input("¿Qué deseas saber?: ")
query_vector = embed_model.encode(query).tolist()

results = index.query(
    namespace=namespace_name,
    vector=query_vector,
    top_k=5,
    include_metadata=True,
    include_values=True
)


# Prepara documentos para rerank
docs = []
for hit in results["matches"]:
    metadata = hit.get("metadata", {})
    if "info" in metadata:
        docs.append({"id": hit["id"], "text": metadata["info"]})

# Paso 2. Aplicas rerank manual
rr = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query=query,
    documents=docs,
    top_n=3,
    return_documents=True,
    parameters={"truncate": "END"}
)

# Mostrar resultados rerankeados
for result in rr.data:
    doc = result["document"]
    score = result["score"]
    print(f"- ID: {doc['id']} | Score: {round(score, 4)}")
    print(f"  Text: {doc['text']}\n")
