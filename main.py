import re
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Загружаем модель при старте
model_name = "./ner-model/Babelscape_wikineural_multilingual_ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy=None
)

app = FastAPI(title="NER API")

class PredictIn(BaseModel):
    input: str

class SpanOut(BaseModel):
    start_index: int
    end_index: int
    entity: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/predict", response_model=List[SpanOut])
async def predict(payload: PredictIn) -> List[SpanOut]:
    text = payload.input
    if not text:
        return []

    predictions = ner_pipeline(text)
    predictions = sorted(predictions, key=lambda x: x["start"])

    words = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]
    spans: List[SpanOut] = []

    prev_entity_type = None
    for w_start, w_end, word in words:
        tokens_in_word = [t for t in predictions if t["start"] >= w_start and t["end"] <= w_end]
        if not tokens_in_word:
            continue

        first_entity = tokens_in_word[0]["entity"]
        base_entity = first_entity[2:] if first_entity.startswith(("B-", "I-")) else first_entity

        if prev_entity_type == base_entity:
            bio_label = f"I-{base_entity}"
        else:
            bio_label = f"B-{base_entity}"
            prev_entity_type = base_entity

        spans.append(SpanOut(start_index=w_start, end_index=w_end, entity=bio_label))

    return spans
