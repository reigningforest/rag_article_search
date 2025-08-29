# To Do List
High Priority:
1. Refactor Embeddings:
    1. Turn chunking off.
    2. Convert to a different embedding system. Something faster. Gemini?
    3. ensure paper's title, authors, and publication year are included in the embedding? OR make some decision on where to include this.
2. Research-Side Enhancements
    1. Add Temporal Context Node
        Purpose: Decide if query needs temporal filtering for recent papers
        Inputs: User query, classification output
        Outputs: Route to semantic search vs semantic + temporal
        Implementation: Add conditional node to existing graph

    2. Internet Search Integration
        Purpose: Supplement corpus for context/fallback
        Use Case 1: Enhanced context for HyDE rewrite generation
        Use Case 2: Direct response when classification = "no retrieval"
        Implementation: Add web search API integration

    3. Optional: Citation Metrics Node
        Purpose: Weight retrieval results by citation impact
        Research needed: Available citation data sources
        Implementation: Post-retrieval reranking logic

    4. Graph Updates
        Add routing logic for: standard semantic, temporal-enhanced, internet-augmented modes
        Keep existing workflow intact, just add decision points


Low Priority:
1. Hybrid Search: Citation based retrieval.
2. Insert footnotes for citations.
3. Implement docker container support
4. Distribute config files