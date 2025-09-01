import pickle
from collections import defaultdict
from typing import Any
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class LSHIndex:
    """
    Locality Sensitive Hashing (LSH) for approximate nearest neighbor search

    This index allows for efficient retrieval of image embeddings to identify potential duplicates
    without performing exhaustive pairwise comparisons.
    Optimized for cosine similarity using Random Projections.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_hash_tables: int = 4,
        num_hyperplanes_per_table: int = 16,
        random_seed: Optional[int] = None,
        center_embedding: Optional[torch.Tensor] = None,
        l2_normalize: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initializes the LSH index

        Parameters
        ----------
        embedding_dim
            The dimensionality of the embedding vectors.
        num_hash_tables
            The number of independent hash tables to use. More tables increase recall
            but also memory usage and query time.
        num_hyperplanes_per_table
            The number of random hyperplanes used to generate a hash for each table.
            More hyperplanes per table increase precision but decrease the chance
            of two similar items hashing to the same bucket in a single table.
        random_seed
            Seed for random number generation for reproducibility.
        center_embedding
            If provided, this tensor will be subtracted from all embeddings before hashing/querying.
            Must be a 1D tensor with shape (embedding_dim,).
        l2_normalize
            If True, embeddings will be L2-normalized after centering.
        device
            The PyTorch device on which to store hyperplanes and perform computations.
            If None, defaults to CPU.
        dtype
            The PyTorch data type for internal computations and storage.
        """

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")
        if num_hash_tables <= 0:
            raise ValueError("num_hash_tables must be a positive integer")
        if num_hyperplanes_per_table <= 0:
            raise ValueError("num_hyperplanes_per_table must be a positive integer")

        self.embedding_dim = embedding_dim
        self.num_hash_tables = num_hash_tables
        self.num_hyperplanes_per_table = num_hyperplanes_per_table
        self.l2_normalize = l2_normalize
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self._g = torch.Generator(device="cpu")
        if random_seed is not None:
            self._g.manual_seed(random_seed)

        if center_embedding is not None:
            if center_embedding.shape != (embedding_dim,):
                raise ValueError("center_embedding must match embedding_dim")

            self.center_embedding = center_embedding.to(device=self.device, dtype=self.dtype).contiguous()
        else:
            self.center_embedding = None

        self._hyperplanes: list[torch.Tensor] = []
        self._hash_tables: list[dict[bytes, list[str]]] = []
        for _ in range(self.num_hash_tables):
            # Generate random hyperplanes for each hash table
            # Each hyperplane is a vector that defines a separating plane in the embedding space.
            # The sign of the dot product determines which side of the plane an embedding falls on.
            self._hyperplanes.append(
                torch.randn(
                    (num_hyperplanes_per_table, embedding_dim),
                    generator=self._g,
                    dtype=self.dtype,
                    device=self.device,
                )
            )

            # Each dict maps a hash key (tuple of signs) to a list of (embedding_id, embedding_vector)
            self._hash_tables.append(defaultdict(list))

        self._indexed_ids: set[str] = set()

    def _preprocess(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Apply centering and/or L2 normalization to an embedding
        """

        if vec.device != self.device:
            vec = vec.to(self.device)
        if self.center_embedding is not None:
            vec = vec - self.center_embedding
        if self.l2_normalize is True:
            vec = F.normalize(vec, p=2, dim=-1, eps=1e-12)

        return vec

    def _preprocess_batch(self, mat: torch.Tensor) -> torch.Tensor:
        """
        Apply centering and/or L2 normalization to a batch of embeddings
        """

        if mat.device != self.device:
            mat = mat.to(self.device)
        if self.center_embedding is not None:
            mat = mat - self.center_embedding.view(1, -1)
        if self.l2_normalize is True:
            mat = F.normalize(mat, p=2, dim=-1, eps=1e-12)

        return mat

    def _generate_hash(self, embedding: torch.Tensor, table_idx: int) -> bytes:
        """
        Generates a hash key for a given embedding and hash table

        The hash key is a compact bytes object representing the sign pattern
        of dot products with the table's hyperplanes.
        """

        hyperplanes_for_table = self._hyperplanes[table_idx]
        dot_products = torch.mv(hyperplanes_for_table, embedding)
        bits = (dot_products > 0).cpu().numpy().astype(np.uint8)
        return np.packbits(bits, bitorder="little").tobytes()

    def _generate_batch_hashes(self, embeddings: torch.Tensor, table_idx: int) -> list[bytes]:
        """
        Generates hash keys for a batch of embeddings for a given hash table
        """

        hyperplanes_for_table = self._hyperplanes[table_idx]
        dot_products = torch.mm(hyperplanes_for_table, embeddings.T)
        bits = (dot_products > 0).cpu().numpy().astype(np.uint8)

        return [np.packbits(row, bitorder="little").tobytes() for row in bits.T]

    def add_embedding(self, embedding_id: str, embedding_vector: torch.Tensor) -> None:
        """
        Adds a single embedding to the LSH index

        The embedding is hashed across all configured hash tables and its ID is stored
        in the corresponding buckets. The embedding will be automatically moved to the
        index's device and preprocessed according to the centering/normalization settings.

        Parameters
        ----------
        embedding_id
            A unique string identifier for the embedding.
        embedding_vector
            The tensor representing the embedding vector.

        Raises
        ------
        ValueError
            If an embedding with the given 'embedding_id' already exists in the index.
        """

        if embedding_id in self._indexed_ids:
            raise ValueError(f"Embedding with id '{embedding_id}' already exists in the index")

        embedding_vector = self._preprocess(embedding_vector)
        self._indexed_ids.add(embedding_id)
        for i in range(self.num_hash_tables):
            hash_key = self._generate_hash(embedding_vector, i)
            self._hash_tables[i][hash_key].append(embedding_id)

    def add_embeddings(self, embedding_ids: list[str], embedding_vectors: torch.Tensor) -> None:
        """
        Adds a batch of embeddings to the LSH index

        Embeddings are hashed across all configured hash tables and their IDs are stored
        in the corresponding buckets. The embeddings will be automatically moved to the
        index's device and preprocessed according to the centering/normalization settings.

        Parameters
        ----------
        embedding_ids
            A list of unique string identifiers for the embeddings in the batch.
        embedding_vectors
            A 2D tensor where each row represents an embedding vector
            (shape: batch_size, embedding_dim).

        Raises
        ------
        ValueError
            If the number of 'embedding_ids' does not match the number of
            'embedding_vectors' or if any 'embedding_id' already exists in the index.
        """

        batch_size = len(embedding_ids)
        if batch_size == 0:
            return

        if batch_size != embedding_vectors.shape[0]:
            raise ValueError("Number of embedding_ids must match the number of embedding_vectors (tensor rows)")
        if len(embedding_ids) != len(set(embedding_ids)):
            raise ValueError("Duplicate embedding IDs in the provided batch")

        # Check for duplicates efficiently for the entire batch
        duplicate_ids = self._indexed_ids.intersection(embedding_ids)
        if len(duplicate_ids) > 0:
            raise ValueError(f"One or more embedding IDs already exist in the index: {list(duplicate_ids)}")

        embedding_vectors = self._preprocess_batch(embedding_vectors)
        self._indexed_ids.update(embedding_ids)
        for i in range(self.num_hash_tables):
            batch_hash_keys = self._generate_batch_hashes(embedding_vectors, i)
            for j, embedding_id in enumerate(embedding_ids):
                hash_key = batch_hash_keys[j]
                self._hash_tables[i][hash_key].append(embedding_id)

    def remove_embedding(self, embedding_id: str, embedding_vector: torch.Tensor) -> bool:
        """
        Remove an embedding from the index

        Parameters
        ----------
        embedding_id
            The ID of the embedding to remove.
        embedding_vector
            The actual embedding tensor corresponding to 'embedding_id'.
            This is needed to re-compute the hash keys and find the correct buckets to remove the ID from.

        Returns
        -------
        True if the embedding was found and removed, False if it did not exist.
        """

        if embedding_id not in self._indexed_ids:
            return False

        embedding_vector = self._preprocess(embedding_vector)
        self._indexed_ids.remove(embedding_id)

        # Remove from all hash tables
        for i in range(self.num_hash_tables):
            hash_key = self._generate_hash(embedding_vector, i)
            bucket = self._hash_tables[i].get(hash_key)
            if bucket is not None:
                bucket.remove(embedding_id)
                if len(bucket) == 0:
                    # Clean up empty bucket
                    del self._hash_tables[i][hash_key]

        return True

    def query_candidates(self, query_embedding: torch.Tensor, query_id: Optional[str] = None) -> list[str]:
        """
        Queries the LSH index for candidate approximate nearest neighbors to a given query embedding

        Parameters
        ----------
        query_embedding
            The embedding tensor for which to find candidates.
        query_id
            An optional ID associated with the query_embedding. If provided, candidates
            with this same ID will be excluded from the results, as they represent the query item itself.

        Returns
        -------
        A list of 'embedding_id' representing potential nearest neighbors found in the same hash buckets as the query.

        Notes
        -----
        The query embedding itself is excluded from the results if its ID is provided.
        """

        query_embedding = self._preprocess(query_embedding)
        candidate_ids = set()
        for i in range(self.num_hash_tables):
            hash_key = self._generate_hash(query_embedding, i)
            if hash_key in self._hash_tables[i]:
                for candidate_id in self._hash_tables[i][hash_key]:
                    # Avoid adding the query embedding itself to the candidate list
                    if candidate_id not in candidate_ids and (query_id is None or candidate_id != query_id):
                        candidate_ids.add(candidate_id)

        return list(candidate_ids)

    def table(self, table_idx: int) -> dict[bytes, list[str]]:
        return self._hash_tables[table_idx]

    def num_buckets(self, table_idx: Optional[int] = None) -> int:
        """
        Number of non-empty buckets

        Parameters
        ----------
        table_idx
            If None, returns the total number of non-empty buckets across all tables.
            If int, returns the number of non-empty buckets in that specific table.
        """

        if table_idx is None:
            return sum(len(table) for table in self._hash_tables)

        return len(self._hash_tables[table_idx])

    def __len__(self) -> int:
        return len(self._indexed_ids)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()

        state["_g_state"] = self._g.get_state()
        del state["_g"]

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        g_state = state.pop("_g_state")
        self.__dict__.update(state)

        self._g = torch.Generator(device="cpu")
        self._g.set_state(g_state)

    def save(self, file_path: str) -> None:
        """
        Saves the LSHIndex instance to a file using pickle

        Parameters
        ----------
        file_path
            The path to the file where the index will be saved.
        """

        with open(file_path, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file_path: str) -> "LSHIndex":
        """
        Loads an LSHIndex instance from a file

        Parameters
        ----------
        file_path
            The path to the file from which the index will be loaded.

        Returns
        -------
        The loaded LSHIndex instance.

        Important Security Note
        -----------------------
        Deserializing data with 'pickle.load()' can execute arbitrary code.
        Only load pickle files from sources you trust completely.
        """

        with open(file_path, "rb") as handle:
            loaded_index: LSHIndex = pickle.load(handle)

        return loaded_index
