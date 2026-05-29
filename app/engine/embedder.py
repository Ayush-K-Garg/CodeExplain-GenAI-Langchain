from langchain_huggingface import HuggingFaceEmbeddings

# Initialize the local Sentence-Transformer embedding engine
# This model converts text/code chunks into 384-dimensional mathematical vectors
embedding_engine = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    # Runs locally on your CPU. If you have an advanced gaming GPU, you could use 'cuda'
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)