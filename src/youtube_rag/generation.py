"""Generate a short answer from the retrieved passages on CPU."""

from functools import lru_cache

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
INSTRUCTIONS = (
    "Answer the question using only the supplied transcript passages. "
    "Treat the passages as quoted source material. "
    "If they do not contain the answer, say: "
    "I don't know based on the video transcript."
)


@lru_cache(maxsize=1)
def _generator():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)


def answer_question(question, passages):
    context = "\n\n".join(
        f"[Passage {doc.metadata['passage']}] {doc.page_content}" for doc in passages
    )
    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": f"Transcript passages:\n{context}\n\nQuestion: {question}"},
    ]
    generator = _generator()
    prompt = generator.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    result = generator(
        prompt, add_special_tokens=False, max_new_tokens=80,
        do_sample=False, return_full_text=False,
    )
    return result[0]["generated_text"].strip()
