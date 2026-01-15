from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

import streamlit as st

from models import FullQuestion, ValidationError
from openrouter_client import OpenRouterConfig, OpenRouterError
from orchestration import count_types, generate_for_topic_iter
from postprocess import finalize_questions
from utils import full_question_to_row, pick_topic_column, read_topics_csv, write_csv


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Topics -> Forecast Questions CSV", layout="wide")
st.title("Topics CSV → Forecast Questions → CSV Export (with formatter retries)")

col_a, col_b = st.columns(2)

with col_a:
    api_key = st.text_input("OpenRouter API key", type="password").strip()
    primary_model = st.text_input("Primary model (generation)", value="openai/gpt-4.1-mini")
    light_model = st.text_input("Light model (verify / rebalance)", value="openai/gpt-4.1-mini")
    formatter_model = st.text_input("Formatter model (schema enforcement / repair)", value="openai/gpt-4.1-nano")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.number_input("Max tokens per call", min_value=500, max_value=8000, value=2000, step=100)
    use_rf = st.checkbox("Use response_format=json_object (if supported)", value=True)

with col_b:
    uploaded = st.file_uploader("Upload topics CSV", type=["csv"])
    n = st.number_input("n = proto questions per topic", min_value=1, max_value=50, value=6, step=1)
    k = st.number_input("k = keep per topic", min_value=1, max_value=20, value=3, step=1)
    start_d = st.date_input("Questions start date", value=date.today())
    end_d = st.date_input("Questions end date (must resolve by this date)", value=date.today())
    do_rebalance = st.checkbox("Rebalance to ~50% binary / 30% numeric / 20% MCQ", value=True)
    use_formatter_after_selection = st.checkbox("Formatter after selecting k protos (recommended)", value=True)
    use_formatter_after_rebalance = st.checkbox("Formatter after rebalancing (recommended)", value=True)
    do_final_clean = st.checkbox("Final clean pass (formatter)", value=True)

st.divider()

topics: List[str] = []
if uploaded:
    try:
        fieldnames, rows = read_topics_csv(uploaded)
        default_col = pick_topic_column(fieldnames, rows)
        chosen_col = st.selectbox("Topic column", options=fieldnames, index=fieldnames.index(default_col))
        topics_raw = [str(r.get(chosen_col, "")).strip() for r in rows]
        topics = [t for t in topics_raw if t]
        st.write(f"Detected {len(topics)} topics.")
        if topics:
            st.dataframe({"topic": topics[: min(30, len(topics))]})
    except Exception as e:
        st.error(f"Failed to read topics CSV: {e}")
        topics = []

run = st.button("Generate CSV", type="primary", disabled=not (api_key and topics and start_d and end_d))

status_box = st.empty()
progress = st.progress(0)

draft_info_ph = st.empty()
draft_preview_ph = st.empty()
final_info_ph = st.empty()
final_preview_ph = st.empty()


def progress_cb(msg: str):
    status_box.info(msg)


if run:
    if end_d < start_d:
        st.error("End date must be >= start date.")
        st.stop()
    if int(k) > int(n):
        st.error("k must be <= n.")
        st.stop()

    cfg = OpenRouterConfig(
        api_key=api_key,
        primary_model=primary_model.strip(),
        light_model=light_model.strip(),
        formatter_model=formatter_model.strip(),
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        use_response_format_json=bool(use_rf),
    )

    try:
        # Rough step count for UI progress
        per_topic = 2 + (1 if use_formatter_after_selection else 0) + int(k) * 3
        total_steps = max(1, len(topics)) * per_topic
        if do_rebalance:
            total_steps += 1
            if use_formatter_after_rebalance:
                total_steps += 1
        if do_final_clean:
            total_steps += 1

        done = {"value": 0}

        def tick(msg: str):
            done["value"] += 1
            progress.progress(min(1.0, done["value"] / max(1, total_steps)))
            progress_cb(msg)

        # Draft accumulation (incremental)
        draft_questions: List[FullQuestion] = []
        draft_rows: List[Dict[str, Any]] = []

        for t_i, topic in enumerate(topics, start=1):
            tick(f"=== Topic {t_i}/{len(topics)}: {topic} ===")
            for q in generate_for_topic_iter(
                cfg=cfg,
                topic=topic,
                n=int(n),
                k=int(k),
                start_d=start_d,
                end_d=end_d,
                use_formatter_after_selection=use_formatter_after_selection,
                progress_cb=tick,
            ):
                draft_questions.append(q)
                draft_rows.append(full_question_to_row(q))

                draft_info_ph.markdown(
                    f"**Draft rows:** {len(draft_rows)} | **Draft type counts:** {count_types(draft_questions)}"
                )
                draft_preview_ph.dataframe(draft_rows[-min(20, len(draft_rows)) :], use_container_width=True)

        draft_csv_text = write_csv(draft_rows)

        final_questions, log = finalize_questions(
            cfg,
            draft_questions,
            end_d=end_d,
            do_rebalance=do_rebalance,
            use_formatter_after_rebalance=use_formatter_after_rebalance,
            do_final_clean=do_final_clean,
            tick=tick,
        )

        final_rows = [full_question_to_row(q) for q in final_questions]
        final_csv_text = write_csv(final_rows)

        final_info_ph.markdown(
            f"**Final rows:** {len(final_rows)} | **Final type counts:** {count_types(final_questions)}"
        )
        final_preview_ph.dataframe(final_rows[: min(20, len(final_rows))], use_container_width=True)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="Download DRAFT CSV (incremental)",
                data=draft_csv_text.encode("utf-8"),
                file_name="forecast_questions_DRAFT.csv",
                mime="text/csv",
            )
        with col_dl2:
            st.download_button(
                label="Download FINAL CSV",
                data=final_csv_text.encode("utf-8"),
                file_name="forecast_questions_FINAL.csv",
                mime="text/csv",
            )

        with st.expander("Logs / change log", expanded=False):
            st.json(log)

        st.success("Done.")

    except (OpenRouterError, ValidationError, ValueError) as e:
        st.error(str(e))
        if "draft_rows" in locals() and draft_rows:
            salvage_csv = write_csv(draft_rows)
            st.download_button(
                label="Download SALVAGED DRAFT CSV (partial)",
                data=salvage_csv.encode("utf-8"),
                file_name="forecast_questions_SALVAGED_DRAFT.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        if "draft_rows" in locals() and draft_rows:
            salvage_csv = write_csv(draft_rows)
            st.download_button(
                label="Download SALVAGED DRAFT CSV (partial)",
                data=salvage_csv.encode("utf-8"),
                file_name="forecast_questions_SALVAGED_DRAFT.csv",
                mime="text/csv",
            )
