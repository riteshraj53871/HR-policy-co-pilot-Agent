from app.config import get_settings
from app.services import embeddings, vector_store, llm, web_search


def check(name, fn):
    try:
        print(f"[PASS] {name}: {fn()}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def t_config():
    s = get_settings()
    return f"index={s.pinecone_index}, model={s.embedding_model}"

def t_embed():
    q = embeddings.embed_query("annual leave days")
    d = embeddings.embed_documents(["Employees get 20 days of annual leave."])
    assert len(q) == 384 and len(d[0]) == 384
    return f"dim={len(q)}"

def t_pinecone():
    st = vector_store.stats()
    return f"dimension={st.get('dimension')}, vectors={st.get('total_vector_count')}"

def t_groq():
    s = get_settings()
    return llm.generate("Reply with the single word: ready", s.router_model)

def t_tavily():
    r = web_search.search("public holidays Bangladesh 2026", max_results=3)
    return f"{len(r)} results, first: {r[0]['url'] if r else 'none'}"


if __name__ == "__main__":
    check("config", t_config)
    check("embeddings", t_embed)
    check("pinecone", t_pinecone)
    check("groq", t_groq)
    check("tavily", t_tavily)