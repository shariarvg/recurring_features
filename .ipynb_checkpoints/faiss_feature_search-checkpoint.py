import numpy as np
import faiss
import feature_corrs
import preprocessing_and_loading as pal

cpu_index = faiss.IndexFlatIP(1000)
#res = faiss.StandardGpuResources()  # default GPU resources

model, sae1, sae2 = pal.load_model_and_saes('cpu', 'gpt2-small-res-jb', 2, 11)

# Create example data
W_a = sae1.W_dec.detach().numpy()# np.random.randn(20000, 1000).astype('float32')
W_b = sae2.W_dec.detach().numpy()#np.random.randn(20000, 1000).astype('float32')


# Optional: normalize for cosine similarity (unit vectors)
faiss.normalize_L2(W_a)
faiss.normalize_L2(W_b)

# Step 1: Build index on W_b
index = faiss.IndexFlatIP(W_a.shape[1])  # IP = inner product
index.add(W_b)                   # index has 20000 vectors

# Step 2: Search for top-k most similar rows from W_b for each row in W_a
k = 5  # top-k neighbors
D, I = index.search(W_a, k)  # D: similarities, I: indices of matches in W_b

# Print sample of results
print("Similarity scores (D):")
print(D[:5])  # print for first 5 rows

print("\nIndices of top-k matches in W_b (I):")
print(I[:5])  # print for first 5 rows

