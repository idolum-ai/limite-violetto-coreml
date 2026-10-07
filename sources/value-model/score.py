"""Score one prompt and continuation with the downloaded value-model release."""
import argparse
import json
from pathlib import Path

from value_model import load_value_model, tokenize_pair


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default=str(Path(__file__).resolve().parent))
    p.add_argument("--device", default="cuda")
    p.add_argument("--attention-backend", default="eager", choices=["eager", "flash_attention_2", "flash_attention_3"])
    p.add_argument("--prompt", default="What is 2 + 2?")
    p.add_argument("--response", default="2 + 2 = 4. The answer is \\boxed{4}.")
    p.add_argument("--tokens-json", help="JSON with prompt_ids and response_ids; use original rollout IDs when available.")
    p.add_argument("--output")
    args = p.parse_args()
    if args.tokens_json:
        tokens = json.loads(Path(args.tokens_json).read_text())
        prompt_ids, response_ids = tokens["prompt_ids"], tokens["response_ids"]
    else:
        prompt_ids, response_ids = tokenize_pair(args.model_dir, args.prompt, args.response)
    model = load_value_model(args.model_dir, device=args.device, attention_backend=args.attention_backend)
    result = model.score_tokens(prompt_ids, response_ids)
    result.update(response_token_ids=response_ids, attention_backend=args.attention_backend,
                  alignment="V(prompt, response strictly before current token)",
                  applies_sigmoid_or_clipping=False)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
