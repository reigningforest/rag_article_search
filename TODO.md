# To Do List
## High Priority:
1. Immediate Bug Fixes
    1. Fix google genai package import error
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


## Low Priority:
1. Hybrid Search: Citation based retrieval.
2. Insert footnotes for citations.
3. Implement docker container support
4. Distribute config files
5. Update Tests
6. plan and add in an API option.

## Future Priority:
### Hardware & Initial Setup
1. **Buy a mini-pc**

### OS & Database Setup
2. **Install Ubuntu Server** (or preferred Linux distribution)
3. **Configure SSH access** for remote management
4. **Install PostgreSQL**: `sudo apt install postgresql postgresql-contrib`
5. **Install pgvector extension**: `sudo apt install postgresql-14-pgvector`
6. **Configure PostgreSQL** for optimal performance on mini-pc hardware
7. **Create database and enable pgvector**: `CREATE EXTENSION vector;`

### Data Migration
8. **Export your current embeddings** from Pinecone (or re-generate)
9. **Create tables** with vector columns in PostgreSQL
10. **Import embeddings** into pgvector database
11. **Set up proper indexes** for vector similarity search

### Web Server & Application
12. **Install web server**: `sudo apt install nginx` (or Apache)
13. **Install Python environment** for your RAG app
14. **Clone/deploy your RAG application** 
15. **Configure nginx** to serve your frontend and proxy API calls
16. **Set up SSL certificate** (Let's Encrypt)

### Network & Security
17. **Configure router port forwarding** (ports 80, 443, 22)
18. **Set up dynamic DNS** service (if needed)
19. **Configure firewall**: `sudo ufw enable`
20. **Set up automated backups** to external storage
