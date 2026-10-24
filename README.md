# science.me-lab

The public side of the private worlds behind [science.me](https://github.com/queelius/science.me).

science.me scores theories against data from synthetic epidemic worlds. The generator has to stay private, or programs could be fit to it. But a private generator is a black box, and a black box asks to be trusted. This repository is how it avoids that: everything needed to check the process is public, and what has to stay private is committed to now and revealed once the worlds that depend on it close.

## What is here

- `preregistrations/tier1.json`: the tier 1 evaluation, fixed before any result: its runs, programs, thresholds and seed. It was committed privately on 2026-10-23 at 18:06 CDT and is published here before the evaluation has run. Its sha256 is `4f067d6fd3134bbc433fdbe5f4cff59c62dfdefd78843f3434ebebe825be10fc`.
- `commitments.jsonl`: an append-only log of seals. A seal is `sha256(nonce || content)`, with a random 32-byte nonce. The nonce and the content stay private until the reveal.
- `src/scienceme_lab/`: the public side of the worlds, behind one protocol (`generator.py`: what a world must answer, and how generators are found):
  - `custodian.py`, the only bridge from a world's private truth to the public archive: it creates worlds, publishes their vintages (`tick`), runs the observations agents ask for (`fulfill`), adds channels, and seals every world and every generator version in the commitment log before it publishes anything from them;
  - `evaluate.py`, the tier 1 harness that `preregistrations/tier1.json` configures;
  - `commitments.py`, the log;
  - `toy.py`, a toy generator with invented dynamics, which the tests run on and which shows what a generator has to provide, and `contract.py`, the checks every generator must pass;
  - `cli.py`, the `lab` command.

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

The protocol, the custodian and the harness are here. The generators, their prior and the worlds' configurations stay in a private repository, whose generators plug in as entry points of the group `scienceme_lab.generators`. Anyone can run the lab on the toy generator; the private worlds run wherever that repository is installed.

## Running tier 1

The evaluation runs from a checkout of the private repository, which installs this package and the private generators side by side:

```
uv run lab evaluate --config ../science.me-lab/preregistrations/tier1.json \
  --public-root "$(realpath ../science.me)" --workers N \
  --scratch worlds/eval-scratch-tier1 --out ../science.me-lab/results/tier1.json
```

`--public-root` and `--workers` say where and how fast it runs, not what it evaluates, so the pre-registered file runs unchanged. Programs are matched by closure hash, so any checkout holding the same freezes works. Each run deletes its forecast records once its result is written (they are a cache, rebuilt by running the run again), so peak disk is what the runs in flight hold. From a first attempt on a workstation: a run takes 15 to more than 70 minutes, each worker wants about 10 GB of memory and 25 GB of disk, and at 4 workers the whole evaluation takes 12 to 15 hours. An interrupted evaluation resumes with finished runs read back; delete any run directory without a `result.json` first, since the evaluation refuses to reuse one.

## License

MIT
