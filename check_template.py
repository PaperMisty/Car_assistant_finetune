import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "model/Qwen/models/Qwen--Qwen3-8B/snapshots/master",
    trust_remote_code=True
)

with open('data/v2/sft/category1_sft_dataset.jsonl', 'r', encoding='utf-8') as f:
    sample = json.loads(f.readline())

convo = sample["messages"]
rendered = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)

print("Rendered length:", len(rendered))
print("=== RENDERED TAIL 300 CHARS ===")
print(repr(rendered[-300:]))
print()
print("Does it contain <think>\\n\\n</think>?")
print("<think>\n\n</think>" in rendered or "<think>\n</think>" in rendered or "<think>" in rendered)
