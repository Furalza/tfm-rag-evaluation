QUESTIONS = [
    # ── Original 10 questions ──────────────────────────────────────────────────
    {
        "id": "Q1_diff",
        "text": "What are the main differences between Naive RAG, GraphRAG, and TagRAG?",
        "type": "comparison",
        "difficulty": "medium"
    },
    {
        "id": "Q2_limits",
        "text": "What are the key limitations of Naive RAG systems?",
        "type": "factual",
        "difficulty": "easy"
    },
    {
        "id": "Q3_improve",
        "text": "How does GraphRAG improve upon standard RAG for complex reasoning tasks?",
        "type": "explanation",
        "difficulty": "medium"
    },
    {
        "id": "Q4_efficiency",
        "text": "Compare the efficiency of GraphRAG and TagRAG in terms of construction and retrieval time.",
        "type": "comparison",
        "difficulty": "hard"
    },
    {
        "id": "Q5_usecases",
        "text": "In which cases is standard RAG still useful compared to graph-based approaches?",
        "type": "reasoning",
        "difficulty": "hard"
    },
    {
        "id": "Q6_community",
        "text": "What is the role of community detection in GraphRAG's indexing process?",
        "type": "factual",
        "difficulty": "medium"
    },
    {
        "id": "Q7_multihop",
        "text": "How do graph-based RAG systems handle multi-hop reasoning better than vector-based RAG?",
        "type": "explanation",
        "difficulty": "hard"
    },
    {
        "id": "Q8_tagchain",
        "text": "What are tag chains in TagRAG and how do they enable hierarchical knowledge retrieval?",
        "type": "factual",
        "difficulty": "medium"
    },
    {
        "id": "Q9_cost",
        "text": "What are the computational costs of building a GraphRAG index compared to a simple vector store?",
        "type": "comparison",
        "difficulty": "medium"
    },
    {
        "id": "Q10_hybrid",
        "text": "How could combining vector databases and graph databases create a more effective hybrid RAG system?",
        "type": "reasoning",
        "difficulty": "hard"
    },

    # ── Extended question set ──────────────────────────────────────────────────

    # Factual
    {
        "id": "Q11_entities",
        "text": "What types of entities and relationships does GraphRAG extract during the indexing phase?",
        "type": "factual",
        "difficulty": "easy"
    },
    {
        "id": "Q12_embedding",
        "text": "What role do text embeddings play in vector-based RAG retrieval?",
        "type": "factual",
        "difficulty": "easy"
    },

    # Explanation
    {
        "id": "Q13_global",
        "text": "How does GraphRAG's global search differ from local search in terms of answer generation?",
        "type": "explanation",
        "difficulty": "medium"
    },
    {
        "id": "Q14_lightrag",
        "text": "How does LightRAG reduce the computational overhead of graph-based retrieval compared to GraphRAG?",
        "type": "explanation",
        "difficulty": "medium"
    },

    # Comparison
    {
        "id": "Q15_quality",
        "text": "Compare the answer quality of graph-based RAG systems versus naive RAG on complex, multi-document queries.",
        "type": "comparison",
        "difficulty": "hard"
    },
    {
        "id": "Q16_scalability",
        "text": "How do GraphRAG and TagRAG differ in their ability to scale to large document corpora?",
        "type": "comparison",
        "difficulty": "hard"
    },

    # Reasoning
    {
        "id": "Q17_hallucination",
        "text": "Why might graph-based RAG systems produce fewer hallucinations than naive RAG systems?",
        "type": "reasoning",
        "difficulty": "hard"
    },
    {
        "id": "Q18_tradeoff",
        "text": "When would a practitioner choose LightRAG over GraphRAG despite GraphRAG's higher answer quality?",
        "type": "reasoning",
        "difficulty": "hard"
    },
    {
        "id": "Q19_chunking",
        "text": "How does chunk size selection affect retrieval precision and recall in naive RAG systems?",
        "type": "reasoning",
        "difficulty": "medium"
    },
    {
        "id": "Q20_future",
        "text": "What are the most promising future research directions for improving RAG systems beyond graph-based approaches?",
        "type": "reasoning",
        "difficulty": "hard"
    },
]

if __name__ == "__main__":
    from collections import Counter
    types = Counter(q["type"] for q in QUESTIONS)
    difficulties = Counter(q["difficulty"] for q in QUESTIONS)
    print(f"Total questions : {len(QUESTIONS)}")
    print(f"Type breakdown  : {dict(types)}")
    print(f"Difficulty      : {dict(difficulties)}")