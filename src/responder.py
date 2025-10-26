def _llm_try(prompt: str):
    try:
        # تلاش با transformers اگر نصب بود
        from transformers import pipeline
        gen = pipeline("text-generation", model="gpt2")
        out = gen(prompt, max_length=40, num_return_sequences=1)
        return out[0]["generated_text"].strip()
    except Exception:
        return None

def welcome_message(name=None, known=False):
    base = ""
    if known and name:
        base = f"Welcome back, {name}! Great to see you again."
    elif known:
        base = "Welcome back!"
    else:
        base = "Nice to meet you! What should I call you?"
    fancy = _llm_try(f"Rewrite warmly as a short, friendly greeting: '{base}'")
    return fancy or base
