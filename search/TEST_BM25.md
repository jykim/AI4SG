# BM25 Search Tests

This document describes the test functions in `test_bm25.py` which test the information retrieval functionality.

## Test Functions

### `test_tokenization`

Tests the tokenization functionality of the search system.

#### Methodology
1. Creates test documents with Korean and English content
2. Indexes the documents using BM25
3. Performs searches with Korean, English, and mixed language queries
4. Verifies search results and scores

#### Assertions
- Verifies that Korean text is properly tokenized
- Verifies that English text is properly tokenized
- Verifies that mixed language content is properly handled
- Verifies that search results are returned in the correct order
- Verifies that search scores are calculated correctly

### `test_search`

Tests the search functionality of the BM25 system.

#### Methodology
1. Creates a test corpus with various documents
2. Indexes the documents using BM25
3. Performs searches with different queries
4. Verifies search results and metrics

#### Assertions
- Verifies that search results are returned in the correct order
- Verifies that search scores are calculated correctly
- Verifies that search metrics are calculated correctly
- Verifies that search results include all required fields 