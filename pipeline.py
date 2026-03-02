"""
Full processing pipeline: sync → chunk → embed → index.

Usage:
    python pipeline.py              # process all new opinions
    python pipeline.py --sync-only  # just sync, don't embed
    python pipeline.py --reindex    # re-chunk and re-embed everything
"""
import argparse
import logging
import os
from pathlib import Path

import numpy as np
from sqlalchemy import text

import config
from db import init_local_db, get_local_engine, get_session, Opinion, OpinionStatute, Statute
from sync import sync_from_source
from chunk import chunk_opinion
from embed import embed_chunks, save_checkpoint
from index import build_index, add_to_index, save_index, load_index
from label import run_labeling
from classify import train_outcome_model, train_claim_type_model, predict_outcomes, predict_claim_types, update_chunk_map_with_predictions
from topics import run_topic_modeling
from citations import run_citation_analysis
from ner import run_ner_extraction

logger = logging.getLogger(__name__)

CHECKPOINT_BATCH = 500  # opinions per checkpoint


def run_pipeline(sync_only=False, reindex=False, classify=False, predict_new=False, topics=False, citations=False, ner=False):
    # Ensure data dirs exist
    os.makedirs(os.path.dirname(config.FAISS_INDEX), exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Step 1: Sync
    engine = get_local_engine()
    init_local_db(engine)
    logger.info("Syncing from source database...")
    new_count = sync_from_source(local_engine=engine)
    logger.info(f"Sync complete: {new_count} new opinions")

    if sync_only:
        return

    # Step 2: Find opinions to process
    session = get_session(engine)
    if reindex:
        session.execute(text("UPDATE opinions SET chunked = 0"))
        session.commit()

    opinions = session.execute(text(
        "SELECT id, plain_text FROM opinions WHERE chunked = 0 AND plain_text IS NOT NULL AND plain_text != ''"
    )).fetchall()
    logger.info(f"Opinions to process: {len(opinions)}")

    if not opinions:
        logger.info("Nothing to process")
        session.close()
        return

    # Step 3: Load or create FAISS index
    if reindex:
        index = build_index()
        chunk_map = []
    else:
        index, chunk_map = load_index()
        if index is None:
            index = build_index()
            chunk_map = []

    # Build statute lookup for metadata
    statute_map = {}
    rows = session.execute(text(
        "SELECT os.opinion_id, s.key FROM opinion_statutes os JOIN statutes s ON os.statute_id = s.id"
    )).fetchall()
    for opinion_id, key in rows:
        statute_map.setdefault(opinion_id, []).append(key.upper())

    # Build opinion metadata lookup
    meta_rows = session.execute(text(
        "SELECT id, title, court_name, court_type, circuit, date_issued FROM opinions WHERE chunked = 0"
    )).fetchall()
    meta_map = {
        r[0]: {"title": r[1], "court_name": r[2], "court_type": r[3],
               "circuit": r[4] or "", "date_issued": r[5] or ""}
        for r in meta_rows
    }

    total_chunks = 0
    processed = 0

    for batch_start in range(0, len(opinions), CHECKPOINT_BATCH):
        batch = opinions[batch_start:batch_start + CHECKPOINT_BATCH]
        all_chunks = []

        for opinion_id, plain_text in batch:
            chunks = chunk_opinion(
                opinion_id=opinion_id,
                text=plain_text,
                chunk_size=config.CHUNK_SIZE,
                overlap=config.CHUNK_OVERLAP,
            )

            meta = meta_map.get(opinion_id, {})
            statutes_str = ",".join(sorted(statute_map.get(opinion_id, [])))
            for c in chunks:
                c.update({
                    "title": meta.get("title", ""),
                    "court_name": meta.get("court_name", ""),
                    "court_type": meta.get("court_type", ""),
                    "circuit": meta.get("circuit", ""),
                    "date_issued": meta.get("date_issued", ""),
                    "statutes": statutes_str,
                })
            all_chunks.extend(chunks)

        if not all_chunks:
            # Mark opinions as chunked even if they produced no chunks
            opinion_ids = [oid for oid, _ in batch]
            placeholders = ",".join(str(oid) for oid in opinion_ids)
            session.execute(text(f"UPDATE opinions SET chunked = 1 WHERE id IN ({placeholders})"))
            session.commit()
            processed += len(batch)
            continue

        # Embed in sub-batches to avoid MemoryError on large batches
        texts = [c["text"] for c in all_chunks]
        logger.info(f"Embedding {len(texts)} chunks (batch {batch_start // CHECKPOINT_BATCH + 1})...")
        EMBED_SUB_BATCH = 5000
        if len(texts) <= EMBED_SUB_BATCH:
            embeddings = embed_chunks(texts)
        else:
            sub_embeddings = []
            for sub_start in range(0, len(texts), EMBED_SUB_BATCH):
                sub_texts = texts[sub_start:sub_start + EMBED_SUB_BATCH]
                logger.info(f"  Sub-batch {sub_start // EMBED_SUB_BATCH + 1}: {len(sub_texts)} chunks")
                sub_embeddings.append(embed_chunks(sub_texts))
            embeddings = np.vstack(sub_embeddings)

        # Save checkpoint
        checkpoint_name = f"batch_{batch_start}"
        save_checkpoint(embeddings, [c["chunk_id"] for c in all_chunks], checkpoint_name)

        # Add to FAISS index
        add_to_index(index, chunk_map, all_chunks, embeddings)

        # Save index after each batch
        save_index(index, chunk_map)

        # Mark opinions as chunked
        opinion_ids = [oid for oid, _ in batch]
        placeholders = ",".join(str(oid) for oid in opinion_ids)
        session.execute(text(f"UPDATE opinions SET chunked = 1 WHERE id IN ({placeholders})"))
        session.commit()

        total_chunks += len(all_chunks)
        processed += len(batch)
        logger.info(f"Progress: {processed}/{len(opinions)} opinions, {total_chunks} total chunks")

    session.close()
    logger.info(f"Pipeline complete. {processed} opinions -> {total_chunks} chunks indexed.")
    logger.info(f"FAISS index total vectors: {index.ntotal}")

    # Step 4: Classification (if requested)
    if classify:
        logger.info("Running labeling pipeline...")
        stats = run_labeling(engine)
        logger.info(f"Labeling stats: {stats}")

        logger.info("Training outcome model...")
        outcome_result = train_outcome_model(engine)
        logger.info(f"Outcome model: {outcome_result.get('accuracy', 'N/A')} accuracy")

        logger.info("Training claim type model...")
        claim_result = train_claim_type_model(engine)
        logger.info(f"Claim type model: {claim_result.get('sections_trained', 0)} sections")

        logger.info("Predicting all opinions...")
        predict_outcomes(engine)
        predict_claim_types(engine)

        logger.info("Updating FAISS index metadata...")
        update_chunk_map_with_predictions(engine)

    if predict_new:
        logger.info("Predicting new opinions with existing models...")
        n1 = predict_outcomes(engine)
        n2 = predict_claim_types(engine)
        if n1 > 0 or n2 > 0:
            update_chunk_map_with_predictions(engine)
        logger.info(f"Predicted {n1} outcomes, {n2} claim types")

    if topics:
        logger.info("Running topic modeling...")
        run_topic_modeling(engine, refit=reindex)

    if citations:
        logger.info("Running citation analysis...")
        run_citation_analysis(engine)

    if ner:
        logger.info("Running NER extraction...")
        run_ner_extraction(engine)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(config.PROJECT_ROOT) / "data" / "pipeline.log"),
        ],
    )

    parser = argparse.ArgumentParser(description="Run the full processing pipeline")
    parser.add_argument("--sync-only", action="store_true", help="Only sync, don't embed")
    parser.add_argument("--reindex", action="store_true", help="Re-process all opinions")
    parser.add_argument("--classify", action="store_true", help="Run labeling + training + prediction")
    parser.add_argument("--predict-only", action="store_true", help="Predict new opinions with existing models")
    parser.add_argument("--topics", action="store_true", help="Run topic modeling after indexing")
    parser.add_argument("--citations", action="store_true",
                        help="Run citation network analysis")
    parser.add_argument("--ner", action="store_true",
                        help="Run named entity recognition")
    args = parser.parse_args()

    run_pipeline(
        sync_only=args.sync_only,
        reindex=args.reindex,
        classify=args.classify,
        predict_new=args.predict_only,
        topics=args.topics,
        citations=args.citations,
        ner=args.ner,
    )
