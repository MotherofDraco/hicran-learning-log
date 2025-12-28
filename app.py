from pathlib import Path
from pyfaidx import Fasta
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import re
from typing import List, Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # development için
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).parent / "data"
FASTA_PATH = DATA_DIR / "references.fasta"

fasta_db = None


@app.on_event("startup")
def startup_event():
    _load_fasta()


def _load_fasta():
    global fasta_db
    if not FASTA_PATH.exists():
        raise RuntimeError(f"FASTA file not found at: {FASTA_PATH}")
    fasta_db = Fasta(str(FASTA_PATH), as_raw=True, sequence_always_upper=True)


class SearchRequest(BaseModel):
    sequence: str
    preview_len: int = 50
    top_k: int = 5  # <<< eklendi


def parse_header(header: str):
    # örnek: >XM_... Homo sapiens ... (SPATA31A1) ...
    record_id = header.split()[0].replace(">", "").strip() if header else None

    organism = None
    gene_name = None
    description = header.replace(">", "").strip() if header else None

    m_org = re.search(
        r"(Homo sapiens|Mus musculus|Rattus norvegicus)", header or "")
    if m_org:
        organism = m_org.group(1)

    m_gene = re.search(r"\(([^)]+)\)", header or "")
    if m_gene:
        gene_name = m_gene.group(1)

    return {
        "record_id": record_id,
        "organism": organism,
        "gene_name": gene_name,
        "description": description
    }


def best_window_match_in_record(full_seq: str, query: str):
    full_seq = (full_seq or "").upper().replace(" ", "").replace("\n", "")
    query = (query or "").upper().replace(" ", "").replace("\n", "")
    if not full_seq or not query:
        return None

    qlen = len(query)
    window_len = min(qlen, len(full_seq))

    best_score = -1
    best_i = 0

    # basit skor: aynı pozisyonda aynı harf
    for i in range(0, len(full_seq) - window_len + 1):
        window = full_seq[i: i + window_len]
        score = 0
        for j in range(window_len):
            if window[j] == query[j]:
                score += 1
        if score > best_score:
            best_score = score
            best_i = i

    match = full_seq[best_i: best_i + window_len]
    similarity = best_score / window_len if window_len else 0.0

    return {
        "start": best_i,
        "end": best_i + window_len,
        "score": best_score,
        "window_len": window_len,
        "similarity": similarity,
        "match_full": match,
    }


@app.post("/search")
def search(req: SearchRequest):
    if fasta_db is None:
        raise HTTPException(
            status_code=500, detail="FASTA database not loaded")

    query = (req.sequence or "").strip().upper()
    if not query:
        raise HTTPException(status_code=400, detail="Sequence is empty")

    hits = []

    for key in fasta_db.keys():
        rec = fasta_db[key]
        seq = str(rec)
        header = rec.long_name

        res = best_window_match_in_record(seq, query)
        if res is None:
            continue

        meta = parse_header(header)

        match_full = res["match_full"]
        preview_len = max(1, int(req.preview_len))
        match_preview = match_full[:preview_len] + \
            ("…" if len(match_full) > preview_len else "")

        hits.append({
            "matched_record_id": meta["record_id"],
            "organism": meta["organism"],
            "gene_name": meta["gene_name"],
            "description": meta["description"],
            "start": res["start"],
            "end": res["end"],
            "similarity": res["similarity"],
            "match_preview": match_preview,
            "match_full": match_full,
        })

    if not hits:
        return {"found": False, "results": []}

    hits.sort(key=lambda x: x["similarity"], reverse=True)
    top_k = max(1, min(int(req.top_k), 50))
    return {"found": True, "results": hits[:top_k]}

##########################################################
# 3) API MODELLERİ
##########################################################


class AlignRequest(BaseModel):
    seq1: str
    seq2: str

##########################################################
# 4) ENDPOINT'LER
##########################################################


@app.get("/")
def home():
    return {"message": "DNA Similarity API is running"}


@app.get("/db/status")
def db_status():
    return {
        "fasta_path": str(FASTA_PATH),
        "exists": FASTA_PATH.exists(),
        "records": 0 if fasta_db is None else len(fasta_db.keys()),
    }


@app.get("/db/records")
def db_records(limit: int = 10):
    if fasta_db is None:
        raise HTTPException(status_code=500, detail="FASTA not loaded")
    keys = list(fasta_db.keys())[:limit]
    return {"keys": keys}


# @app.get("/")
# def home():
    # ©return {"message": "FAISS DNA Search API çalışıyor!"}


# @app.post("/search")
# def search_sequence(query: Query):
    # seq = query.sequence.upper().replace("\n", "").replace(" ", "")
    # query_vec = dna_to_vector(seq).astype("float32").reshape(1, -1)

    # distances, indices = index.search(query_vec, k=10)

    # results = []
    # for rank, idx in enumerate(indices[0]:
    # results.append({
    # "rank": rank + 1,
    # "distance": float(distances[0][rank]),
    # "sequence": seq_db[int(idx)][:len(seq)]
    # })

    # return {
    # "query": seq,
    # "results": results
    # } (FAISS kullanan kod)

#####################################
# 4) Alignment API (Global & Local)
#####################################

@app.post("/align-global")
def align_global(body: AlignRequest):
    seq1 = body.seq1
    seq2 = body.seq2

    if not seq1 or not seq2:
        return {"error": "Sequences missing"}

    alignments = pairwise2.align.globalxx(seq1, seq2)

    if not alignments:
        return {"alignment": "No alignment found."}

    formatted = format_alignment(*alignments[0])
    return {"alignment": formatted}


@app.post("/align-local")
def align_local(body: AlignRequest):
    seq1 = body.seq1
    seq2 = body.seq2

    aligns = pairwise2.align.localms(seq1, seq2, 2, -1, -2, -0.5)
    if not aligns:
        return {"alignment": None}

    best = aligns[0]
    formatted = format_alignment(*best)
    return {"alignment": formatted}
