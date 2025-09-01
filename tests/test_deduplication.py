# pylint: disable=protected-access

import pickle
import unittest

import torch

from vdc.deduplication.dsu import DSU
from vdc.deduplication.lsh import LSHIndex


class TestLSHIndex(unittest.TestCase):
    def setUp(self) -> None:
        self.embedding_dim = 10
        self.num_hash_tables = 5
        self.num_hyperplanes_per_table = 8
        self.lsh = LSHIndex(
            embedding_dim=self.embedding_dim,
            num_hash_tables=self.num_hash_tables,
            num_hyperplanes_per_table=self.num_hyperplanes_per_table,
            random_seed=0,
        )

    def test_initialization(self) -> None:
        self.assertEqual(self.lsh.embedding_dim, self.embedding_dim)
        self.assertEqual(self.lsh.num_hash_tables, self.num_hash_tables)
        self.assertEqual(self.lsh.num_hyperplanes_per_table, self.num_hyperplanes_per_table)
        self.assertEqual(len(self.lsh), 0)
        self.assertEqual(len(self.lsh._hyperplanes), self.num_hash_tables)
        self.assertEqual(len(self.lsh._hash_tables), self.num_hash_tables)
        self.assertTrue(
            all(hp.shape == (self.num_hyperplanes_per_table, self.embedding_dim) for hp in self.lsh._hyperplanes)
        )

    def test_add_embedding(self) -> None:
        embedding1 = torch.rand(self.embedding_dim, dtype=torch.float32)
        self.lsh.add_embedding("id1", embedding1)

        self.assertEqual(len(self.lsh), 1)
        self.assertIn("id1", self.lsh._indexed_ids)

        for i in range(self.num_hash_tables):
            hash_key = self.lsh._generate_hash(embedding1, i)
            self.assertIn("id1", self.lsh._hash_tables[i][hash_key])

        # Adding the same ID again should raise ValueError
        with self.assertRaises(ValueError):
            self.lsh.add_embedding("id1", embedding1)

    def test_add_embeddings_batch_basic(self) -> None:
        ids = ["batch_id1", "batch_id2", "batch_id3"]
        embeddings = torch.rand(len(ids), self.embedding_dim, dtype=torch.float32)
        self.lsh.add_embeddings(ids, embeddings)

        self.assertEqual(len(self.lsh), 3)
        for _id in ids:
            self.assertIn(_id, self.lsh._indexed_ids)

        # Verify presence in hash tables for each embedding
        for idx, _id in enumerate(ids):
            embedding = embeddings[idx]
            for i in range(self.num_hash_tables):
                hash_key = self.lsh._generate_hash(embedding, i)
                self.assertIn(_id, self.lsh._hash_tables[i][hash_key])

    def test_add_embeddings_batch_duplicate_id_in_index(self) -> None:
        existing_id = "existing_id"
        existing_emb = torch.rand(self.embedding_dim, dtype=torch.float32)
        self.lsh.add_embedding(existing_id, existing_emb)

        # Attempt to add a batch with a duplicate ID
        ids = ["batch_id1", existing_id, "batch_id2"]
        embeddings = torch.rand(len(ids), self.embedding_dim, dtype=torch.float32)

        with self.assertRaises(ValueError) as cm:
            self.lsh.add_embeddings(ids, embeddings)

        self.assertIn(existing_id, str(cm.exception))
        self.assertEqual(len(self.lsh), 1)  # Size should remain 1, as batch addition should be atomic failure

    def test_add_embeddings_batch_vs_individual(self) -> None:
        num_items = 10
        ids_to_add = [f"item_{i}" for i in range(num_items)]
        embeddings_to_add = torch.rand(num_items, self.embedding_dim, dtype=torch.float32)

        # Add individually to lsh_individual
        lsh_individual = LSHIndex(
            embedding_dim=self.embedding_dim,
            num_hash_tables=self.num_hash_tables,
            num_hyperplanes_per_table=self.num_hyperplanes_per_table,
            random_seed=0,
        )
        for i in range(num_items):
            lsh_individual.add_embedding(ids_to_add[i], embeddings_to_add[i])

        # Add as batch to lsh_batch
        lsh_batch = LSHIndex(
            embedding_dim=self.embedding_dim,
            num_hash_tables=self.num_hash_tables,
            num_hyperplanes_per_table=self.num_hyperplanes_per_table,
            random_seed=0,
        )
        lsh_batch.add_embeddings(ids_to_add, embeddings_to_add)

        # Verify sizes
        self.assertEqual(len(lsh_individual), num_items)
        self.assertEqual(len(lsh_batch), num_items)
        self.assertEqual(len(lsh_individual), len(lsh_batch))

        # Verify indexed IDs are identical
        self.assertEqual(lsh_individual._indexed_ids, lsh_batch._indexed_ids)

        # Verify hash tables are identical (structure and content)
        self.assertEqual(len(lsh_individual._hash_tables), len(lsh_batch._hash_tables))
        for i in range(self.num_hash_tables):
            individual_table = lsh_individual._hash_tables[i]
            batch_table = lsh_batch._hash_tables[i]

            self.assertEqual(len(individual_table), len(batch_table))
            self.assertEqual(individual_table.keys(), batch_table.keys())

            for hash_key in individual_table:
                self.assertCountEqual(individual_table[hash_key], batch_table[hash_key])

        # Verify query results are identical for a sample query
        query_emb = embeddings_to_add[num_items // 2]
        query_id = ids_to_add[num_items // 2]

        individual_candidates = lsh_individual.query_candidates(query_emb, query_id)
        batch_candidates = lsh_batch.query_candidates(query_emb, query_id)

        self.assertCountEqual(individual_candidates, batch_candidates)

    def test_remove_embedding(self) -> None:
        embedding = torch.rand(self.embedding_dim, dtype=torch.float32)
        self.lsh.add_embedding("id_remove", embedding)

        # Ensure its in the index
        self.assertEqual(len(self.lsh), 1)
        self.assertIn("id_remove", self.lsh._indexed_ids)

        # Remove it
        removed = self.lsh.remove_embedding("id_remove", embedding)
        self.assertTrue(removed)

        # Should not be retrievable anymore
        self.assertEqual(len(self.lsh), 0)
        self.assertNotIn("id_remove", self.lsh._indexed_ids)

        # Buckets should not contain the id
        for i in range(self.num_hash_tables):
            hash_key = self.lsh._generate_hash(embedding, i)
            self.assertNotIn("id_remove", self.lsh._hash_tables[i].get(hash_key, []))

        # Removing again should return False
        removed_again = self.lsh.remove_embedding("id_remove", embedding)
        self.assertFalse(removed_again)

        # Querying with the old embedding should return no candidates
        candidates = self.lsh.query_candidates(embedding)
        self.assertEqual(len(candidates), 0)

    def test_query_candidates_no_match(self) -> None:
        lsh = LSHIndex(
            embedding_dim=self.embedding_dim,
            num_hash_tables=1,
            num_hyperplanes_per_table=48,
            random_seed=1,
        )
        embedding1 = torch.randn(self.embedding_dim, dtype=torch.float32)
        embedding2 = torch.randn(self.embedding_dim, dtype=torch.float32) ** -2
        lsh.add_embedding("id1", embedding1)

        candidates = lsh.query_candidates(embedding2)
        self.assertEqual(len(candidates), 0)

    def test_query_candidates_exact_match(self) -> None:
        embedding1 = torch.rand(self.embedding_dim, dtype=torch.float32)
        self.lsh.add_embedding("id1", embedding1)
        self.lsh.add_embedding("id2", embedding1)

        candidates = self.lsh.query_candidates(embedding1, query_id="id1")

        self.assertEqual(len(candidates), 1)  # Expect the "id2" duplicate
        self.assertTrue(any(cid == "id2" for cid in candidates))
        self.assertFalse(any(cid == "id1" for cid in candidates))

    def test_query_candidates_approximate_match_same_bucket(self) -> None:
        # Create two embeddings that are very similar
        emb_a = torch.tensor([1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=torch.float32)
        emb_b = torch.tensor([1.0, 0.11, 0.22, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.91], dtype=torch.float32)  # Slightly
        emb_c = torch.tensor([-1.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9], dtype=torch.float32)  # Very

        self.lsh.add_embedding("A", emb_a)
        self.lsh.add_embedding("B", emb_b)
        self.lsh.add_embedding("C", emb_c)

        candidates_for_a = self.lsh.query_candidates(emb_a)
        # We expect 'B' to be a candidate for 'A' due to similarity, but not 'C'
        self.assertIn("B", candidates_for_a)
        self.assertNotIn("C", candidates_for_a)
        self.assertIn("A", candidates_for_a)

        # Check for candidates of B
        candidates_for_b = self.lsh.query_candidates(emb_b, "B")
        self.assertIn("A", candidates_for_b)
        self.assertNotIn("C", candidates_for_b)
        self.assertNotIn("B", candidates_for_b)

        # Check for candidates of C
        candidates_for_c = self.lsh.query_candidates(emb_c, "C")
        self.assertNotIn("A", candidates_for_c)
        self.assertNotIn("B", candidates_for_c)
        self.assertNotIn("C", candidates_for_c)

    def test_random_seed_reproducibility(self) -> None:
        # First index
        lsh1 = LSHIndex(
            embedding_dim=self.embedding_dim,
            num_hash_tables=self.num_hash_tables,
            num_hyperplanes_per_table=self.num_hyperplanes_per_table,
            random_seed=0,
        )
        emb1 = torch.rand(self.embedding_dim, dtype=torch.float32)
        emb2 = torch.rand(self.embedding_dim, dtype=torch.float32)
        lsh1.add_embedding("e1", emb1)
        lsh1.add_embedding("e2", emb2)
        candidates1_for_e1 = lsh1.query_candidates(emb1, query_id="e1")
        candidates1_for_e1_ids = sorted(candidates1_for_e1)

        # Second index with same seed
        lsh2 = LSHIndex(
            embedding_dim=self.embedding_dim,
            num_hash_tables=self.num_hash_tables,
            num_hyperplanes_per_table=self.num_hyperplanes_per_table,
            random_seed=0,
        )
        lsh2.add_embedding("e1", emb1)
        lsh2.add_embedding("e2", emb2)
        candidates2_for_e1 = lsh2.query_candidates(emb1, query_id="e1")
        candidates2_for_e1_ids = sorted(candidates2_for_e1)

        # The hyperplanes should be identical
        for i in range(self.num_hash_tables):
            self.assertTrue(torch.equal(lsh1._hyperplanes[i], lsh2._hyperplanes[i]))

        # The hashes generated should be identical
        self.assertEqual(lsh1._generate_hash(emb1, 0), lsh2._generate_hash(emb1, 0))

        # The candidate IDs found should be identical
        self.assertEqual(candidates1_for_e1_ids, candidates2_for_e1_ids)


class TestLSHIndexSerialization(unittest.TestCase):
    def test_state_config_parameters_restored(self) -> None:
        original_index = LSHIndex(embedding_dim=128, num_hash_tables=8, num_hyperplanes_per_table=32, random_seed=0)

        pickled_data = pickle.dumps(original_index)
        loaded_index = pickle.loads(pickled_data)

        self.assertEqual(loaded_index.embedding_dim, original_index.embedding_dim)
        self.assertEqual(loaded_index.num_hash_tables, original_index.num_hash_tables)
        self.assertEqual(loaded_index.num_hyperplanes_per_table, original_index.num_hyperplanes_per_table)

    def test_state_hyperplanes_bit_perfect(self) -> None:
        original_index = LSHIndex(embedding_dim=64, num_hash_tables=2, num_hyperplanes_per_table=10, random_seed=123)

        pickled_data = pickle.dumps(original_index)
        loaded_index = pickle.loads(pickled_data)

        self.assertEqual(len(loaded_index._hyperplanes), len(original_index._hyperplanes))
        for i, original_hp in enumerate(original_index._hyperplanes):
            self.assertTrue(torch.equal(loaded_index._hyperplanes[i], original_hp))
            self.assertEqual(loaded_index._hyperplanes[i].dtype, original_hp.dtype)

    def test_state_hash_tables_fidelity(self) -> None:
        original_index = LSHIndex(embedding_dim=4, num_hash_tables=2, num_hyperplanes_per_table=4, random_seed=7)

        e1 = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
        e2 = torch.tensor([-0.1, -0.2, -0.3, -0.4], dtype=torch.float32)
        e3 = torch.tensor([0.9, 0.8, 0.7, 0.6], dtype=torch.float32)
        original_index.add_embedding("id1", e1)
        original_index.add_embedding("id2", e2)
        original_index.add_embedding("id3", e3)

        pickled_data = pickle.dumps(original_index)
        loaded_index = pickle.loads(pickled_data)

        # Ensure both hash tables are equal
        self.assertEqual(len(original_index._hash_tables), len(loaded_index._hash_tables))
        for i, orig_ht in enumerate(original_index._hash_tables):
            loaded_ht = loaded_index._hash_tables[i]
            self.assertEqual(len(orig_ht), len(loaded_ht))

            for key, orig_list in orig_ht.items():
                self.assertIn(key, loaded_ht)
                loaded_list = loaded_ht[key]
                self.assertCountEqual(orig_list, loaded_list)

    def test_state_rng_state_reproducibility(self) -> None:
        original_index = LSHIndex(embedding_dim=10, num_hash_tables=1, num_hyperplanes_per_table=5, random_seed=0)
        original_rng_state = original_index._g.get_state()

        pickled_data = pickle.dumps(original_index)
        loaded_index = pickle.loads(pickled_data)
        loaded_rng_state = loaded_index._g.get_state()

        self.assertTrue(torch.equal(original_rng_state, loaded_rng_state))

        orig_rand = torch.rand(5, generator=original_index._g)
        loaded_rand = torch.rand(5, generator=loaded_index._g)
        self.assertTrue(torch.equal(orig_rand, loaded_rand))

    def test_state_indexed_ids_restored(self) -> None:
        original_index = LSHIndex(embedding_dim=10, random_seed=0)
        emb1 = torch.rand(10, dtype=torch.float32)
        emb2 = torch.rand(10, dtype=torch.float32)
        original_index.add_embedding("test_id1", emb1)
        original_index.add_embedding("test_id2", emb2)

        pickled_data = pickle.dumps(original_index)
        loaded_index = pickle.loads(pickled_data)

        self.assertEqual(len(loaded_index), 2)
        self.assertEqual(loaded_index._indexed_ids, {"test_id1", "test_id2"})


class TestDSU(unittest.TestCase):
    def setUp(self) -> None:
        self.dsu = DSU(["a", "b", "c", "d", "e"])

    def test_initialization(self) -> None:
        # Each element should be its own parent initially
        for element in ["a", "b", "c", "d", "e"]:
            self.assertEqual(self.dsu.find(element), element)
            self.assertEqual(self.dsu.rank[element], 0)

    def test_find_nonexistent_element(self) -> None:
        with self.assertRaises(KeyError):
            self.dsu.find("x")

        with self.assertRaises(KeyError):
            self.dsu.find("")

    def test_union_different_sets(self) -> None:
        # Union should return True when connecting different sets
        self.assertTrue(self.dsu.union("a", "b"))

        # After union, both elements should have the same root
        self.assertEqual(self.dsu.find("a"), self.dsu.find("b"))

        # Should be connected
        self.assertTrue(self.dsu.connected("a", "b"))

    def test_union_same_set(self) -> None:
        self.assertTrue(self.dsu.union("a", "b"))

        # Second union of same elements should return False
        self.assertFalse(self.dsu.union("a", "b"))
        self.assertFalse(self.dsu.union("b", "a"))

    def test_connected_initially_false(self) -> None:
        self.assertFalse(self.dsu.connected("a", "b"))
        self.assertFalse(self.dsu.connected("c", "d"))
        self.assertFalse(self.dsu.connected("a", "e"))

    def test_connected_after_union(self) -> None:
        self.dsu.union("a", "b")
        self.dsu.union("c", "d")

        # Elements in same component should be connected
        self.assertTrue(self.dsu.connected("a", "b"))
        self.assertTrue(self.dsu.connected("c", "d"))

        # Elements in different components should not be connected
        self.assertFalse(self.dsu.connected("a", "c"))
        self.assertFalse(self.dsu.connected("b", "d"))

    def test_transitive_connectivity(self) -> None:
        self.dsu.union("a", "b")
        self.dsu.union("b", "c")

        # All three should be connected
        self.assertTrue(self.dsu.connected("a", "b"))
        self.assertTrue(self.dsu.connected("b", "c"))
        self.assertTrue(self.dsu.connected("a", "c"))

        # Same root for all
        root = self.dsu.find("a")
        self.assertEqual(self.dsu.find("b"), root)
        self.assertEqual(self.dsu.find("c"), root)

    def test_path_compression(self) -> None:
        # Create a chain: a -> b -> c -> d
        self.dsu.union("a", "b")
        self.dsu.union("b", "c")
        self.dsu.union("c", "d")

        # After path compression, all elements should point directly to root
        root = self.dsu.find("a")
        for element in ["a", "b", "c", "d"]:
            self.assertEqual(self.dsu.find(element), root)

    def test_get_components_initial(self) -> None:
        components = self.dsu.get_components()
        self.assertEqual(len(components), 5)

    def test_get_components_after_unions(self) -> None:
        self.dsu.union("a", "b")
        self.dsu.union("c", "d")

        components = self.dsu.get_components()

        # Should have 3 components: {a,b}, {c,d}, {e}
        self.assertEqual(len(components), 3)

        # Check component sizes
        component_sizes = sorted([len(comp) for comp in components.values()])
        self.assertEqual(component_sizes, [1, 2, 2])

    def test_union_by_rank(self) -> None:
        # Create two chains of different lengths
        # Chain 1: a-b (rank will be 1)
        # Chain 2: c (rank will be 0)
        self.dsu.union("a", "b")

        # Get the ranks before union
        root_ab = self.dsu.find("a")
        root_c = self.dsu.find("c")

        rank_ab = self.dsu.rank[root_ab]
        rank_c = self.dsu.rank[root_c]

        # Union the chains
        self.assertTrue(self.dsu.union("a", "c"))

        # The root should be from the higher rank tree
        final_root = self.dsu.find("a")
        if rank_ab > rank_c:
            self.assertEqual(final_root, root_ab)
        elif rank_c > rank_ab:
            self.assertEqual(final_root, root_c)

    def test_duplicate_elements_in_initialization(self) -> None:
        dsu = DSU(["a", "b", "a"])  # Duplicate 'a'

        # Duplicates are dropped
        self.assertEqual(len(dsu.get_components()), 2)

        # Should work normally
        self.assertEqual(dsu.find("a"), "a")
        self.assertEqual(dsu.find("b"), "b")
