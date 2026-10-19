import json
import re
from pathlib import Path

import requests
from vllm import LLM, SamplingParams

MODEL = "./models/Qwen2.5-3B-Instruct"
URL = "http://127.0.0.1:8000/retrieve"

question = (
    "In which city was Microsoft co-founder "
    "Paul Allen born?"
)

llm = LLM(
    model=MODEL,
    dtype="float16",
    tensor_parallel_size=1,
    max_model_len=2048,
    gpu_memory_utilization=0.35,
    enforce_eager=True,
)

tokenizer = llm.get_tokenizer()
sampling = SamplingParams(
    temperature=0,
    max_tokens=192,
)

system = (
    "You are a multi-hop retrieval agent. "
    "First, issue a search with <search>query</search>. "
    "After receiving <documents>, extract important facts "
    "inside <retrieved_facts>...</retrieved_facts>, "
    "analyze sufficiency inside <analysis>...</analysis>, "
    "then either search again or provide "
    "<answer>short answer</answer>. "
    "Do not invent unsupported facts."
)

messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": "Question: " + question},
]

trace = []
search_count = 0
final_answer = None

def generate():
    global messages
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Keep the smoke test within the validated context window.
    if len(tokenizer.encode(prompt)) + 192 > 2048:
        messages = [
            messages[0],
            {"role": "user", "content":
                "Question: " + question +
                "\nLatest evidence:\n" +
                messages[-1]["content"][:2200]}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if len(tokenizer.encode(prompt)) + 192 > 2048:
            raise RuntimeError("Prompt exceeds context budget")

    return llm.generate(
        [prompt], sampling, use_tqdm=False
    )[0].outputs[0].text

for turn in range(1, 6):
    response = generate()
    print(f"\n===== Agent step {turn} =====")
    print(response)

    matches = list(re.finditer(
        r"<(search|answer)>(.*?)</\1>",
        response,
        flags=re.S | re.I,
    ))

    if not matches:
        print("No valid action tag; stopping.")
        break

    match = matches[0]
    action = match.group(1).lower()
    content = match.group(2).strip()

    event = {
        "turn": turn,
        "action": action,
        "content": content,
        "raw": response,
    }

    if action == "answer":
        final_answer = content
        trace.append(event)
        break

    if search_count >= 3:
        print("Search budget exhausted.")
        messages.append({
            "role": "assistant",
            "content": response,
        })
        messages.append({
            "role": "user",
            "content": (
                "No more searches. Output a concise "
                "<answer>...</answer> based on evidence."
            ),
        })
        trace.append(event)
        continue

    search_count += 1

    result = requests.post(
        URL,
        json={
            "queries": [content],
            "topk": 3,
            "return_scores": True,
        },
        timeout=180,
    )
    result.raise_for_status()
    docs = result.json()["result"][0]

    event["docs"] = [
        {
            "score": d["score"],
            "document": d["document"],
        }
        for d in docs
    ]
    trace.append(event)

    print(f"\nRetrieved {len(docs)} documents")

    snippets = []
    for i, item in enumerate(docs, 1):
        text = item["document"]["contents"]
        print(f"Doc {i}: {text[:120]}")
        snippets.append(f"Doc {i}: {text[:550]}")

    observation = (
        "<documents>\n" +
        "\n".join(snippets) +
        "\n</documents>\n"
        "Extract retrieved facts and analyze whether "
        "they answer the original question. "
        "Then search again or answer."
    )

    messages.append({
        "role": "assistant",
        "content": response,
    })
    messages.append({
        "role": "user",
        "content": observation,
    })

output = {
    "question": question,
    "search_count": search_count,
    "answer": final_answer,
    "trace": trace,
}

path = Path("results/smoke/agent_1.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(
    json.dumps(output, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\n===== SUMMARY =====")
print("Search calls:", search_count)
print("Final answer:", final_answer)
print("Saved:", path)
