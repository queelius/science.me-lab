# science.me-lab

The public side of the private worlds behind [science.me](https://github.com/queelius/science.me).

science.me scores theories against data from synthetic epidemic worlds. The generator has to stay private, or programs could be fit to it. But a private generator is a black box, and a black box asks to be trusted. This repository is how it avoids that: everything needed to check the process is public, and what has to stay private is committed to now and revealed once the worlds that depend on it close.

## What is here

- `preregistrations/tier1.json`: the tier 1 evaluation, fixed before any result: its runs, programs, thresholds and seed. It was committed privately on 2026-10-23 at 18:06 CDT and is published here before the evaluation has run. Its sha256 is `4f067d6fd3134bbc433fdbe5f4cff59c62dfdefd78843f3434ebebe825be10fc`.
- `commitments.jsonl`: an append-only log of seals. A seal is `sha256(nonce || content)`, with a random 32-byte nonce. The nonce and the content stay private until the reveal.
- `src/scienceme_lab/`: the code that writes and checks the log, and the `lab` command.

The log commits to the configurations of the three prospective worlds (`w-alpha`, `w-beta`, `w-gamma`) and to the generator package that drew and simulates them. The worlds were created on 2026-10-23 at 18:09 CDT and had already published some issues (through issue 18 for `w-alpha`, issue 4 for the other two) when these commitments were made, so a commitment says nothing about anything before it.

## When things are revealed

- A world's configuration, when the world closes (its last target resolves).
- A generator version's source, once no open world uses it.

After both, anyone can run the generator on the configuration and compare every vintage it produces with the archive in science.me.

## Checking

With [science.me](https://github.com/queelius/science.me) checked out next to this repository:

```
uv sync
uv run pytest -q
uv run lab check
```

`lab check` recomputes the seal of every reveal from its published content and nonce. By hand: a seal is the sha256 of the nonce's bytes followed by the content's bytes. For a world the content is its configuration file; for a generator it is the UTF-8 of its `package_hash` (`scienceme.worker.package_hash`), which covers every module of the package.

## Status

The pre-registration and the commitments are here. Next, the protocol moves here from the private repository: how a world's data is released (`tick`, `fulfill`, `add-channel`) and how tier 1 is judged (strategies, criteria, code stamps), with generators plugging in from outside. After that only the generators, their prior and the worlds' configurations stay private.

## License

MIT
