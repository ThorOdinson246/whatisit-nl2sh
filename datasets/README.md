# Chatter / robustness pairs for NL->shell models

246 hand-written `(natural language, shell command)` pairs that teach a
model the *boring reflex*: conversational, ambiguous or nonsensical input
gets a harmless command (`echo hello`, `pwd`, `echo "you're welcome!"`)
instead of garbage or something network-touching.

Motivation: task-only training data makes bare greetings leak task
behavior (e.g. `hello` producing a `curl` to a random host).

## Files

- `chatter_robustness.jsonl` -- the data. One JSON object per line:
  `{"nl": "...", "bash": "...", "source": "organic"}`
- `organic_augment.py` -- generator. Running it reproduces the JSONL exactly
  (`python organic_augment.py --output chatter_robustness.jsonl`, deterministic, seed 42)

## Composition

Greetings, small talk, thanks/bye, identity and capability questions,
time/system-state questions, keyboard-mash nonsense, ambiguous requests,
and a few fun ones. De-duplicated on normalized input; identical bash
targets capped at 120.

## License

Apache-2.0, matching the whatisit-nl2sh repository.
