


api_key = "pcsk_4D2ykq_A7TRYdCkTgkamSc9JtuVFhM3NWERngzs6KkUq1aGwEKyVEbsxm8V91f99xeP6yr"
environment="us-east-1"
index_name = "resume-matching-index"
your_dimension = 384
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=api_key,environment=environment)

existing_indexes = pc.list_indexes()
if index_name in existing_indexes:
    pc.delete_index(index_name)

spec = ServerlessSpec(cloud="aws", region="us-east-1")  # Modify cloud/region as needed

pc.create_index(
    name=index_name,
    spec=spec,  # Required spec argument
    dimension=your_dimension,  # Example: 1536 for OpenAI embeddings
    metric="cosine"  # Options: 'cosine', 'euclidean', 'dotproduct'
)

# Connect to the new index
index = pc.Index(index_name)

print("New index created successfully!")
