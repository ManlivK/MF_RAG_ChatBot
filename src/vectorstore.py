"""Stage 3-4 - Vector store (ChromaDB)."""
from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache

import chromadb
from chromadb.config import Settings

from src import config
from src.embedding import embedding_fingerprint, embed_documents


COLLECTION_NAME = getattr(config, 'CHROMA_COLLECTION', 'mf_faq_hdfc')


class IndexStaleError(Exception):
    pass


@lru_cache(maxsize=1)
def get_collection():
    chroma_dir = config.PATHS['CHROMA_DIR']
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={'hnsw:space': 'cosine'},
        embedding_function=None,
    )
    return collection


def _load_chunks():
    path = config.PATHS.get('CHUNKS_FILE') or (config.PATHS['CHUNKS_DIR'] / 'chunks.jsonl')
    chunks = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def rebuild_index(chunks: list[dict] | None = None) -> int:
    chunks = chunks or _load_chunks()
    client = chromadb.PersistentClient(path=str(config.PATHS['CHROMA_DIR']))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={'hnsw:space': 'cosine'},
        embedding_function=None,
    )
    if chunks:
        ids = [c['chunk_id'] for c in chunks]
        docs = [c['text'] for c in chunks]
        metadatas = []
        for c in chunks:
            md = {
                'chunk_id': c['chunk_id'],
                'scheme_key': c['scheme_key'],
                'scheme_name': c['scheme_name'],
                'category': c['category'],
                'section': c['section'],
                'label': c['label'],
                'source_url': c['source_url'],
                'source_title': c['source_title'],
                'fetched_at': c['fetched_at'],
                'chunk_index': int(c['chunk_index']),
                'content_hash': c['content_hash'],
            }
            metadatas.append(md)
        embeddings = embed_documents(docs)
        collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)
    model = getattr(config, 'EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    stats = {
        'collection': COLLECTION_NAME,
        'chunk_count': len(chunks),
        'built_at': datetime.utcnow().isoformat() + 'Z',
        'embedding_fingerprint': embedding_fingerprint(),
        'embedding_model': model,
        'embedding_device': config.EMBED_DEVICE,
    }
    write_index_meta(stats)
    return len(chunks)


def upsert_chunks(chunks: list[dict]) -> int:
    return rebuild_index(chunks)


def write_index_meta(stats: dict) -> None:
    meta_path = config.PATHS.get('INDEX_META_FILE') or (config.ROOT / 'data' / 'index_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)


def read_index_meta() -> dict | None:
    meta_path = config.PATHS.get('INDEX_META_FILE') or (config.ROOT / 'data' / 'index_meta.json')
    try:
        with open(meta_path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def is_index_stale() -> tuple[bool, str]:
    meta = read_index_meta()
    if not meta:
        return True, 'missing meta'
    if meta.get('embedding_fingerprint') != embedding_fingerprint():
        return True, 'fingerprint mismatch'
    return False, 'ok'


def get_stats() -> dict:
    meta = read_index_meta()
    if meta:
        return meta
    return {'chunk_count': 0}


def run_all(*args, **kwargs) -> int:
    return rebuild_index()


def main(*args, **kwargs) -> int:
    return rebuild_index()