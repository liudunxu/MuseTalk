# AGENTS.md

Behavioral guidelines for coding agents working in this repository.

Source: https://github.com/multica-ai/andrej-karpathy-skills/blob/main/CLAUDE.md
License: MIT, per the upstream repository README.

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

Do not assume, do not hide confusion, and surface tradeoffs.

Before implementing:

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them instead of choosing silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop, name what is confusing, and ask.

## 2. Simplicity First

Write the minimum code that solves the problem. Avoid speculative work.

- No features beyond what was asked.
- No abstractions for single-use code.
- No flexibility or configurability that was not requested.
- No error handling for impossible scenarios.
- If 200 lines could be 50, rewrite it.

Ask: would a senior engineer say this is overcomplicated? If yes, simplify.

## 3. Surgical Changes

Touch only what is necessary. Clean up only your own mess.

When editing existing code:

- Do not improve adjacent code, comments, or formatting unless required.
- Do not refactor things that are not broken.
- Match existing style, even if you would choose a different style elsewhere.
- If you notice unrelated dead code, mention it instead of deleting it.

When your changes create orphans:

- Remove imports, variables, and functions made unused by your changes.
- Do not remove pre-existing dead code unless asked.

Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

Define success criteria and loop until verified.

Transform tasks into verifiable goals:

- "Add validation" means write tests for invalid inputs, then make them pass.
- "Fix the bug" means write or identify a repro, then make it pass.
- "Refactor X" means ensure tests pass before and after.

For multi-step tasks, state a brief plan:

```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong success criteria let agents loop independently. Weak criteria like "make it work" require clarification.

These guidelines are working if diffs contain fewer unnecessary changes, implementations avoid needless rewrites, and clarifying questions happen before mistakes rather than after them.

## Repository Runtime Notes

The API may be deployed from a server where the VAE weights directory is named `models/sd-vae-ft-mse` instead of `models/sd-vae`, and where MuseTalk weights are under `models/musetalk` without `models/musetalkV15`. Do not assume local model directory names; resolve or validate actual paths before loading models. Whisper must include `config.json`, `pytorch_model.bin`, and `preprocessor_config.json` in `models/whisper`.
