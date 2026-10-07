# Step 5: bounded practical math tasks

The prepared [CPU reference runner](../../scripts/evaluate_practical_tasks.py)
uses three fixed prompts absent from the previously audited `17 * 23`
generation fixture. It has not been run on the model yet.

| ID | Prompt | Expected boxed value |
| --- | --- | --- |
| `arithmetic` | `Compute 28 * 17.` | `476` |
| `simple_algebra` | `Solve 4x + 3 = 13. What is x?` | `5/2` |
| `short_word_problem` | `Maya has 18 marbles. She gives away one third of them, then buys 5 more. How many marbles does she have now?` | `17` |

Each task is independently tokenized with the pinned local tokenizer and its
unchanged `chat_template.jinja`. The template supplies the fixed mathematical
system prompt and assistant generation prefix. Prompt IDs must round-trip to
the formatted prompt, and each prompt length plus its 1,024-token generation
budget must fit the 2,048-token context. One verified portable CPU model is
loaded and reused, with four Torch threads. Each task samples with seed 0,
temperature 0.6, and top-p 0.95. The explicit stop set contains published
model EOS 151645 and tokenizer EOS 151643, matching the qualified local
completion policy. A fresh generator is seeded for each task.

Scoring requires a completed response (`eos` or `stop_token`) and exactly one
numeric `\boxed{...}` after the last `</think>`. The parser accepts signed
integers, terminating decimals, plain fractions, and `\frac`/`\dfrac` with
integer numerator and denominator, comparing them as exact rational numbers.
It rejects missing or malformed boxes, multiple final boxes, zero
denominators, nonnumeric text, and answers found only in reasoning. Exhausting
the token budget is incomplete even if a box appears.

The runner writes `results.json` through a flushed, fsynced temporary file
and atomic replacement after every generated token. It retains prompt and
generated IDs, formatted prompts, decoded responses, settings, source and
checkpoint identity, completion reasons, per-step and wall timing, parsed
answers, and errors. The generation helper resets its cache in `finally` on
normal completion and exceptions. Per-token persistence affects wall timing;
this run is a small descriptive usefulness check, not a speed benchmark or
general quality certification.

Parser checks only, without model inference:

```sh
.venv/bin/python scripts/evaluate_practical_tasks.py --self-check
```

After review and when no benchmark is competing for CPU, run with a fresh
attempt name and capture stdout separately:

```sh
.venv/bin/python scripts/evaluate_practical_tasks.py --attempt cpu-reference-1 > evidence/05-practical/cpu-reference-1.log 2>&1
```

The runner refuses an existing attempt directory. Its exit status is zero
only when all three tasks complete and score correctly; individual results
remain in `evidence/05-practical/<attempt>/results.json` after any failure.
