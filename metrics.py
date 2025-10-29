# metrics.py
import csv
import math
import time
from typing import List, Dict, Any, Iterable, Tuple
from collections import defaultdict
from tqdm import tqdm

# ============ Retrieval Metrics ============

def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    topk = retrieved_ids[:k]
    hits = sum(1 for _id in topk if _id in relevant_ids)
    return hits / max(1, len(topk))

def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    topk = set(retrieved_ids[:k])
    hits = sum(1 for _id in relevant_ids if _id in topk)
    return hits / len(relevant_ids)

def hit_rate_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    topk = set(retrieved_ids[:k])
    return 1.0 if any(_id in topk for _id in relevant_ids) else 0.0

def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    rel = set(relevant_ids)
    for rank, _id in enumerate(retrieved_ids, start=1):
        if _id in rel:
            return 1.0 / rank
    return 0.0

def average_precision(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    rel = set(relevant_ids)
    ap_sum, hits = 0.0, 0
    for i, _id in enumerate(retrieved_ids[:k], start=1):
        if _id in rel:
            hits += 1
            ap_sum += hits / i
    return ap_sum / max(1, len(rel))

def _dcg(relevances: List[int]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))

def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    rel_set = set(relevant_ids)
    gains = [1 if _id in rel_set else 0 for _id in retrieved_ids[:k]]
    dcg = _dcg(gains)
    ideal = sorted(gains, reverse=True)
    idcg = _dcg(ideal)
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_retrieval(
    retriever,
    dataset: List[Dict[str, Any]],
    k_values: Iterable[int] = (3, 6, 10),
    id_getter=lambda doc: doc.metadata.get("chunk_id", doc.metadata.get("source", "")),
) -> Dict[str, Dict[str, float]]:
    """
    dataset item format:
    {
      "question": str,
      "relevant_ids": List[str],  # ground-truth chunk or doc ids
    }
    """
    metrics_sum = {f"P@{k}": 0.0 for k in k_values}
    metrics_sum.update({f"R@{k}": 0.0 for k in k_values})
    metrics_sum.update({f"Hit@{k}": 0.0 for k in k_values})
    metrics_sum.update({f"NDCG@{k}": 0.0 for k in k_values})
    mrr_sum, ap_sum = 0.0, 0.0
    n = len(dataset)

    for ex in tqdm(dataset, desc="Retrieval eval"):
        q = ex["question"]
        rel_ids = ex.get("relevant_ids", [])
        docs = retriever.invoke(q)
        ret_ids = [id_getter(d) for d in docs]

        for k in k_values:
            metrics_sum[f"P@{k}"] += precision_at_k(ret_ids, rel_ids, k)
            metrics_sum[f"R@{k}"] += recall_at_k(ret_ids, rel_ids, k)
            metrics_sum[f"Hit@{k}"] += hit_rate_at_k(ret_ids, rel_ids, k)
            metrics_sum[f"NDCG@{k}"] += ndcg_at_k(ret_ids, rel_ids, k)
        mrr_sum += reciprocal_rank(ret_ids, rel_ids)
        ap_sum += average_precision(ret_ids, rel_ids, max(k_values))

    results = {name: v / max(1, n) for name, v in metrics_sum.items()}
    results["MRR"] = mrr_sum / max(1, n)
    results[f"MAP@{max(k_values)}"] = ap_sum / max(1, n)
    return results

def write_dict_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# ============ RAGAS Metrics ============

def evaluate_ragas(
    llm, retriever, dataset: List[Dict[str, Any]],
    output_csv: str = "ragas_results.csv",
    batch_sleep: float = 0.0,
):
    """
    dataset item format:
    {
      "question": str,
      "ground_truths": List[str],  # reference answers or key facts
    }
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset as HFDataset
    except Exception as e:
        print("Install ragas and datasets: pip install ragas datasets")
        raise

    records = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truths": [],
    }

    for ex in tqdm(dataset, desc="Generate answers for RAGAS"):
        q = ex["question"]
        ctx_docs = retriever.invoke(q)
        ctx_texts = [d.page_content for d in ctx_docs]
        # Your existing answering prompt can be reused; simple direct call:
        answer = llm.invoke(q).content
        records["question"].append(q)
        records["answer"].append(answer)
        records["contexts"].append(ctx_texts)
        records["ground_truths"].append(ex.get("ground_truths", []))
        if batch_sleep > 0:
            time.sleep(batch_sleep)

    hfds = HFDataset.from_dict(records)
    result = evaluate(
        hfds,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
    )
    df = result.to_pandas()
    df.to_csv(output_csv, index=False)
    return df
