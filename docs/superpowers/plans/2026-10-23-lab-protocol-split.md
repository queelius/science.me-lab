# The protocol split: custodian and harness go public Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the release protocol (the custodian) and the tier 1 evaluation harness from the private `science.me-worlds` repository into this public one, behind a small generator protocol, so that only the generators, their prior and the worlds' configurations stay private.

**Architecture:** `scienceme_lab.generator` defines what a world must answer (`World`: public facts, `vintage(spec, issue)`, `truth()`, `dump()`) and how a generator is found (entry points in the group `scienceme_lab.generators`). The custodian and the harness move here and talk to worlds only through that protocol; the custodian seals each world and each generator version in the commitment log before anything of it is published. A public toy generator drives the public tests; the private SEIR and ABM generators implement the protocol in `science.me-worlds` and register as entry points.

**Tech Stack:** Python 3.13, pydantic 2, numpy, `importlib.metadata` entry points, pytest, uv. Three sibling checkouts: `science.me` (kernel), `science.me-lab` (this), `science.me-worlds` (private).

**Spec:** `README.md` of this repository ("When things are revealed", "Status"), and the partition Alex approved on 2026-10-23: public is everything needed to check the process; private is only what decides the answers (generators, prior, lens parameters, drawn configurations, the private plan), committed now and revealed when the worlds that depend on it close.

## Global Constraints

- Nothing private enters this repository: no prior ranges, no generator equations, no lens parameters, no drawn configuration. The toy generator and tests use invented values.
- `preregistrations/tier1.json` parses unchanged as `EvalConfig` in this repository's environment, with no private generator installed.
- The prospective worlds keep working: every capture already published for `w-alpha`, `w-beta` and `w-gamma` is reproduced exactly by the private worlds behind the new protocol (Task 5 verification), and their private files (`worlds/<id>/config.json`, `worlds/<id>/cache/`, `commitments/*.nonce`) keep their paths.
- One definition per vocabulary item: stream roles, intervention variables, claim ids and `SUPPORT_MAX = 1_000_000` live in `scienceme_lab.generator`; the private package imports them.
- Code style follows the surrounding code: docstrings with two spaces between sentences, one-line domain errors on stderr, exit codes 0/1/2, only `cli.py` reads the clock.
- No em-dashes in any written file.
- Commit trailers: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_014ysBBpk9BBJvD9JzT1xKWM`.

## Review Focus

1. A generator's package is edited between two ticks: the custodian seals the new version before it publishes anything with it (Task 2 test `test_a_changed_generator_is_sealed_before_it_publishes`).
2. An entry point registered under a name other than its generator's `name`: discovery refuses it with a `GeneratorError` naming both (Task 1 test `test_an_entry_point_must_carry_its_generators_name`).
3. A world whose generator is not installed (the private package missing from the environment): the custodian refuses with a `CustodianError` that lists the installed generators, and writes nothing (Task 2 test `test_a_world_of_an_uninstalled_generator_is_refused`).
4. A held-out target resolving after the world's last week: `vintage` is defined past `weeks`, and the realized value equals the row a later vintage would show (Task 1 contract check `counts_are_defined_past_the_last_week`, Task 3 test `test_the_heldout_truth_is_what_the_resolution_vintage_shows`).
5. A create interrupted after its world was sealed but before its config was written: a second create of that id is refused with a `CustodianError` saying so, rather than a bare write-once error (Task 2 test `test_a_sealed_but_unwritten_world_is_refused_clearly`).

---

### Task 1: The generator protocol, a toy generator, and the contract

**Files:**
- Create: `src/scienceme_lab/generator.py`, `src/scienceme_lab/toy.py`, `src/scienceme_lab/contract.py`
- Modify: `pyproject.toml` (dependency `numpy>=2.1`; entry point `[project.entry-points."scienceme_lab.generators"] toy = "scienceme_lab.toy:GENERATOR"`)
- Test: `tests/test_generator.py`

**Interfaces:**
- Produces, in `generator.py`:
  - Constants: `GENERATORS_GROUP = "scienceme_lab.generators"`, `SUPPORT_MAX = 1_000_000` (with the comment now on the private lenses' constant: a fixed bound, because one derived from populations would disclose a population), `COVARIATE_X_ROLE = "covariate-x"`, `SCHOOL_TERM_ROLE = "school-term"`, `SCHOOLS_CLOSED = "schools_closed"`, `VACCINATE_FRACTION = "vaccinate_fraction"`, `X_CLAIM = "covariate-x"`, `FLAT_CLAIM = "covariate-x-flat"`, `SCHOOL_CLAIM = "school"`, `TRUTH_CLAIMS = (X_CLAIM, SCHOOL_CLAIM)`.
  - `class World(Protocol)`: read-only `id: str`, `weeks: int`, `jurisdictions: tuple[str, ...]`, `flat_jurisdiction: str`, `channels: frozenset[str]`; `vintage(self, spec: StreamSpec, issue: int) -> Vintage`; `truth(self) -> dict[str, bool]` (keys exactly `TRUTH_CLAIMS`); `dump(self) -> bytes` (a JSON object whose `"generator"` field is its generator's name).
  - `class Generator(Protocol)`: `name: str`, `interventions: frozenset[str]`; `draw(self, world_id: str, weeks: int, n_jurisdictions: int, rng: np.random.Generator, cache: Path) -> World`; `load(self, config: bytes, cache: Path) -> World`.
  - `class GeneratorError(Exception)`.
  - `generators() -> dict[str, Generator]`: every entry point of the group, loaded, by name; GeneratorError when an entry point's name is not its object's `name`.
  - `generator_of(config: bytes) -> str`: the `"generator"` field of a dumped config; GeneratorError when it is missing or the bytes are not a JSON object.
  - `generator_source(generator: Generator) -> Path`: the directory of the top-level package of `type(generator).__module__`.
  - `source_name(generator: Generator) -> str`: the distribution providing that package (`importlib.metadata.packages_distributions()`), else the package's import name.
- Produces, in `toy.py`: `GENERATOR: Generator` named `"toy"` with `interventions = {SCHOOLS_CLOSED, VACCINATE_FRACTION}` and channels `{"admissions", "ed-visits"}`; invented, public dynamics (independent Poisson weekly counts around a per-jurisdiction level, lowered by `exp(-x_effect * x)`, raised in school term, scaled down by vaccination), drawn week-major so an arm equals its baseline before the intervention starts; counts final at once (no reporting delay); x released 2 weeks ahead, zero in the flat jurisdiction (the first); the whole term calendar released at any issue; `truth` from whether the two effects are nonzero.
- Produces, in `contract.py`: `CONTRACT: tuple[Callable[[Generator, Path], None], ...]`, each a named check that raises `AssertionError` with a message: `draws_reproducibly`, `dumps_and_loads_the_same_world`, `counts_cover_the_weeks_before_the_issue` (rows for every jurisdiction and every week before `min(issue, weeks)`, integer values in `[0, spec.support_max]`), `counts_are_defined_past_the_last_week`, `arms_match_the_baseline_before_they_start` (both intervention variables, every jurisdiction, every issue), `covariates_are_released` (x rows reach at least `issue - 1` and are zero in `flat_jurisdiction`; the term vintage at issue 0 covers every week), `truth_names_the_claims`.

- [ ] **Step 1: Write the failing tests** in `tests/test_generator.py`: `test_the_toy_generator_is_installed` (`generators()["toy"].name == "toy"`); `test_the_toy_generator_keeps_the_contract` parametrized over `CONTRACT`; `test_a_broken_generator_fails_the_contract_check_it_breaks` (a toy subclass whose arms differ from the baseline from week 0 fails `arms_match_the_baseline_before_they_start` and passes `draws_reproducibly`); `test_an_entry_point_must_carry_its_generators_name` (monkeypatch `importlib.metadata.entry_points` to return an entry point `other = scienceme_lab.toy:GENERATOR`; `pytest.raises(GeneratorError, match="other.*toy")`); `test_generator_of_reads_the_generator_field` and `test_generator_of_refuses_bytes_without_one` (`b"[]"`, `b"{}"`, `b"not json"`); `test_the_toy_generators_source_is_this_package` (`generator_source(GENERATOR) == Path(scienceme_lab.__file__).parent`, `source_name(GENERATOR) == "scienceme-lab"`).
- [ ] **Step 2: Run** `uv run pytest tests/test_generator.py -q`. Expected: FAIL (modules missing).
- [ ] **Step 3: Implement** the three modules and the entry point; `uv sync` to register it.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `feat: the generator protocol, a toy generator and its contract`.

### Task 2: The custodian, generic over generators, sealing as it goes

**Files:**
- Create: `src/scienceme_lab/custodian.py` (ported from `science.me-worlds/src/scienceme_worlds/custodian.py`)
- Test: `tests/test_custodian.py` (ported from the private `tests/test_custodian.py`), `tests/helpers.py`

**Interfaces:**
- Consumes: Task 1's protocol and vocabulary; `CommitmentLog`, `Subject`, `SubjectKind` from `commitments.py`.
- Produces: `class Custodian(public: Workspace, private_root: Path, log: CommitmentLog | None, generators: Mapping[str, Generator] | None = None)` (None means `generators()`; a None log means no sealing, for simulation runs); methods `create(generator: str, world_id: str, weeks: int, n_jurisdictions: int, channel: str, now: datetime, rng: np.random.Generator | None = None, required_isolation: Isolation = Isolation.CONTAINER) -> World`, `world(world_id: str) -> World` (was `truth`), `current_issue(world_id: str) -> int`, `tick(world_id: str, to_issue: int, now: datetime) -> list[Capture]`, `fulfill(world_id: str, item_id: str, now: datetime) -> StreamSpec`, `add_channel(world_id: str, channel: str, now: datetime) -> StreamSpec`, `reveal(world_id: str, now: datetime) -> Reveal` (refused until the world has published its last issue, `weeks`; copies `config.json` to `revealed/worlds/<id>/config.json` in the log's directory and opens the world's latest commitment with its kept nonce); `CustodianError`; the mission and grid constants as they are now. Private layout: `worlds/<id>/config.json` (= `world.dump()`), `worlds/<id>/cache/` (passed to `draw`/`load`), `commitments/world-<id>.nonce`, `commitments/generator-<source_name>-<first 12 hex of package_hash>.nonce`.

The port changes only this: count and covariate vintages come from `world.vintage(spec, issue)`; the term calendar is `world.vintage(term_spec, 0)`; an intervention's variable must be in the generator's `interventions`; channels are `world.channels`; a loaded world's generator is `generator_of(config)`. With a log, `create` seals the generator version (when its nonce file is absent) and then the world, before writing the config or publishing; `tick`, `fulfill` and `add_channel` seal the generator version first when its current `package_hash` has no nonce file yet.

- [ ] **Step 1: Port the tests** that exercise the protocol (every private custodian test except those about one generator's dynamics, the latent cache, or a named generator), against the toy generator through `tests/helpers.py`. Add: `test_create_seals_the_generator_then_the_world_before_publishing` (log entries are a generator commitment then a world commitment; the world seal opens with the kept nonce over `config.json`); `test_a_generator_version_is_sealed_once`; `test_a_changed_generator_is_sealed_before_it_publishes` (a generator class in a temporary package on `sys.path`; edit a module of it between two ticks; the second tick appends one generator commitment before its captures); `test_without_a_log_nothing_is_sealed`; `test_a_world_of_an_uninstalled_generator_is_refused` (a config naming `"absent"`; `CustodianError` listing `toy`; nothing published); `test_a_sealed_but_unwritten_world_is_refused_clearly` (a nonce file without a config: `CustodianError`, match `"sealed"`); `test_a_world_is_revealed_only_once_it_has_closed` (refused at `weeks - 1`; after the last tick the reveal opens the commitment and `log.problems() == []`).
- [ ] **Step 2: Run** `uv run pytest tests/test_custodian.py -q`. Expected: FAIL.
- [ ] **Step 3: Port and adapt** `custodian.py` as above.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `feat: the custodian, generic over generators, sealing each world and generator version`.

### Task 3: The tier 1 harness

**Files:**
- Create: `src/scienceme_lab/evaluate.py` (ported from the private `evaluate.py`)
- Test: `tests/test_evaluate.py` (ported)

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: everything the private `evaluate.py` exports, with these changes: `Stamps(programs: dict[str, str], scorer: str, lab: str, generators: dict[str, str])` (`lab` is this package's `package_hash`, `generators` maps each of `cfg.generators` to the `package_hash` of its `generator_source`); `lab_hash() -> str` replaces `worlds_hash`; `heldout_realized(world: World, arm: StreamSpec, targets: Sequence[Target]) -> list[int]` reads each target's row from `world.vintage(arm, target.resolution_issue)`; `mixture_log_score(pmfs, advantages, tau, epsilon, realized, support_max: int)`; truth comes from `world.truth()`, keyed by claim id, so `matching_program(truth)` reads `truth[X_CLAIM]` and `truth[SCHOOL_CLAIM]`; `EvalConfig` no longer checks generator names (it parses anywhere), and `stamps`, `run_replicate` and `evaluate` raise `EvaluationError` for a generator that is not installed, naming the installed ones; run custodians get `log=None`.

- [ ] **Step 1: Port the tests**, the tiny-evaluation ones on `generators=("toy",)`. Add `test_the_preregistration_parses_here` (`read_model(REPO_ROOT / "preregistrations" / "tier1.json", EvalConfig)` succeeds and its `generators == ("seir", "abm")`); `test_an_uninstalled_generator_is_refused_before_any_run` (`EvaluationError` naming `toy`, scratch left empty); rename the held-out truth test to `test_the_heldout_truth_is_what_the_resolution_vintage_shows`, which ticks a toy world past its last week's resolution and compares with the archive.
- [ ] **Step 2: Run** `uv run pytest tests/test_evaluate.py -q`. Expected: FAIL.
- [ ] **Step 3: Port and adapt** `evaluate.py`.
- [ ] **Step 4: Run** `uv run pytest -q --cov=scienceme_lab`. Expected: all pass, coverage at least 99%.
- [ ] **Step 5: Commit** `feat: the tier 1 harness, judged through the generator protocol`.

### Task 4: One `lab` command

**Files:**
- Modify: `src/scienceme_lab/cli.py` (add the custodian commands and `evaluate`, ported from the private `cli.py`)
- Test: `tests/test_cli.py` (add the ported tests)

**Interfaces:**
- Consumes: Tasks 2 and 3.
- Produces: `lab [--log DIR] [--public DIR] [--private DIR] COMMAND`, with defaults `REPO_ROOT`, `REPO_ROOT.parent / "science.me"` and `REPO_ROOT.parent / "science.me-worlds"`; commands `commit`, `reveal`, `check` (as now), `create --generator NAME --world ID [--weeks 30] [--jurisdictions 4] [--channel admissions]` (`--generator` choices from `generators()`), `tick --world ID --to-issue N`, `fulfill --world ID --item ID`, `add-channel --world ID --channel ROLE`, `reveal-world --world ID`, `evaluate --config PATH --scratch DIR --out PATH [--public-root DIR] [--workers N]`. Custodian commands pass `CommitmentLog(args.log)`.

- [ ] **Step 1: Port the CLI tests** (toy generator, temporary roots) and add `test_create_seals_the_world_in_the_given_log`.
- [ ] **Step 2: Run** `uv run pytest tests/test_cli.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q --cov=scienceme_lab`. Expected: all pass.
- [ ] **Step 5: Commit** `feat: the lab command drives the custodian and the evaluation`.

### Task 5: The private generators behind the protocol (in `science.me-worlds`)

**Files (private repository):**
- Create: `src/scienceme_worlds/world.py`, `tests/test_world.py`
- Modify: `pyproject.toml` (depend on `scienceme-lab` at `../science.me-lab`, editable; entry points `seir = "scienceme_worlds.world:SEIR"`, `abm = "scienceme_worlds.world:ABM"`; drop `[project.scripts]`), `src/scienceme_worlds/lenses.py` (`reported_final` takes `support_max: int` and clips at it; its constant goes), `src/scienceme_worlds/seir.py` (the intervention variable names come from `scienceme_lab.generator`)
- Delete: `src/scienceme_worlds/custodian.py`, `evaluate.py`, `cli.py`; `tests/test_custodian.py`, `tests/test_evaluate.py`, `tests/test_cli.py` (ported in Tasks 2 to 4, apart from what moves to `tests/test_world.py`)

**Interfaces:**
- Consumes: Task 1's protocol; Task 2's `Custodian` for the integration tests.
- Produces: `class SimulatedWorld` (a `World` over a `WorldConfig`, a simulator and a cache directory, holding the latent cache the custodian used to hold, same file layout) and `class SimulatorGenerator(name: str, simulate: Simulator)` with `draw` (the prior's `draw_world`) and `load` (GeneratorError when the config's generator is another); `SEIR`, `ABM`.

- [ ] **Step 1: Write the tests** in `tests/test_world.py`: the lab's `CONTRACT` over `SEIR` and `ABM`; and, moved from the old custodian tests, the ones about one generator's dynamics, the latent cache (cached privately, simulated once per set, keyed by the set), and a world published end to end through the lab's `Custodian`.
- [ ] **Step 2: Run** `uv run pytest tests/test_world.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `world.py`, the lens and vocabulary changes, the entry points; delete the ported modules and tests; `uv sync`.
- [ ] **Step 4: Run** `uv run pytest -q --cov=scienceme_worlds`. Expected: all pass.
- [ ] **Step 5: Verify against the published record:** for each of `w-alpha`, `w-beta`, `w-gamma` and every capture of its streams in `../science.me/archive`, `SEIR`/`ABM`.load(config, cache).vintage(spec, issue) equals the published vintage, row for row. Expected: every capture reproduced; any difference stops the task.
- [ ] **Step 6: Commit** `refactor: the generators implement the lab's protocol; the custodian and harness are public now`.

### Task 6: Documents, and publish

**Files:**
- Modify: this repository's `README.md` (Status: the protocol is here; add "Running tier 1" from the private `evaluation/README.md`, with `uv run lab evaluate --config ../science.me-lab/preregistrations/tier1.json ...` run from the private checkout); the private `evaluation/README.md` (a pointer to this README); `science.me/SKILL.md` (the public `../science.me-lab` holds the release protocol, the harness, the pre-registration and the commitments, and may be read; `../science.me-worlds` stays forbidden).

- [ ] **Step 1: Edit** the three documents.
- [ ] **Step 2: Run** each repository's tests and `uv run lab check`. Expected: all pass, `ok`.
- [ ] **Step 3: Commit** in each repository and push all three; check CI on the two public ones. Expected: green.
