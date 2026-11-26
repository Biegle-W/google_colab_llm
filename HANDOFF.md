# Handoff: cloud agent → local agent

Written by the Claude Code cloud session that has been maintaining this repo. The user cloned it locally so a
Claude on their own computer (with a browser) can do what the cloud session cannot: **get a Colab GPU runtime through
the Colab website and run the tests on it.** Read this whole file first. Answer the user, not this file, when they
disagree with it.

## Why you are needed

Since about 2026-11-23, Colab refuses GPU assignment through the **Colab CLI** (`colab new --gpu G4|A100`) with
`503 Service Unavailable` for many paid accounts, ours included (G4 and A100 refused every hour, CPU runtimes fine).
Open reports, no fix from Google: googlecolab/google-colab-cli#273, #282, #236 and googlecolab/colabtools#6289.
The **website still assigns them**: the user got an A100 80 GB in the browser on 2026-11-25. The cloud session has no
browser, so you do the browser part.

## Rules from the user (standing, do not relax)

1. **Runtimes:** G4 (RTX PRO 6000 Blackwell, 96 GB) first, then **A100 80 GB only** (never the 40 GB A100), L4 last.
   Never H100 or T4. After connecting, check `!nvidia-smi --query-gpu=name,memory.total --format=csv`: an A100 under
   ~80 GB must be disconnected and deleted at once (Runtime → Disconnect and delete runtime).
2. **No retry loops.** One request per GPU type when the user says go. If refused, stop and tell the user. Repeated
   requests got the account blocked before.
3. **Ask before starting an expensive GPU runtime** unless the user just told you to. Billing starts at assignment.
4. **Never let a GPU runtime sit idle, and stop it when done** (Runtime → Disconnect and delete runtime), then confirm
   nothing is left running. Tell the user the moment a runtime is up.
5. **Secrets** (HF token, Tailscale key, hub API key, Cloudflare token) live in Colab Secrets or the user's private
   Drive (`MyDrive/colab_model_server/access/`). Never put them in git, notebooks, chat or logs.
6. Ask before adding new models or engines.
7. Commits: run the tests first and commit only if they pass:
   `python3 -m unittest discover -s notebook_code/tests -t .` (224 tests, all pass on main at b074568).
   No model IDs in commits or code comments. No pull requests unless asked.

## What to run (the planned test block)

Models are already on Drive under `MyDrive/colab_model_server/models/` (Hy4 Strata SQ-128 182 GB, Hy4 Maya-L 162 GB,
Inkling Strata SQ-20 20 GB, plus the older ones). The notebooks are self-contained (the hub code is packed into the
collapsed cell 2): open one on colab.research.google.com, pick the runtime it names, Run all. Use
`notebooks/g4-rtx-pro6000/...` on a G4, `notebooks/a100/...` on an A100 80 GB. Run them one at a time on the same
runtime (cell 9 "Stop" before the next notebook).

| # | Compare | Notebooks (same runtime) | Baseline |
|---|---|---|---|
| 1 | Inkling 2 Mini: Strata v0.2.19 vs llama.cpp | `strata/inkling-2-mini-sq-20` vs `llama/inkling-2-mini-q8` | new; llama.cpp now passes `reasoning_effort: none` |
| 2 | Hy4-Flash: Strata vs Maya | `strata/hy4-flash-sq-128` vs `maya/hy4-flash-maya-l` | new; Maya compiles ~1 h on its first start per GPU type |
| 3 | Flash-Next | G4: `strata/flash-next-strata-iq3-s` vs `basalt/basalt-flash-next-nvfp4-dq6k` · A100: `strata/flash-next-strata-iq3-s` | G4 Basalt 273 tok/s; A100 Strata IQ3_S 167 tok/s (older Strata) |

Untested code paths you are the first to exercise (watch these closely, they are the likeliest to break):
- Strata `.strata` families (Hy4, Inkling) are hard-linked into `<strata data-dir>/models/<family>/` instead of
  `--model-path` (commit b7b519e). A CPU test showed Strata accepts the files; whether it *starts* needs a GPU.
- hf_xet high-performance downloads on ≥64 GB RAM hosts (commit 7502666).
- Inkling on llama.cpp with `--chat-template-kwargs {"reasoning_effort":"none"} --temp 1.0` (commit b074568).

For each run record: cold start (Run all → first answer), decode tok/s on a ~500-token answer, prefill tok/s and TTFT
on a long prompt if you can, and a short sample answer. Cell 7 (Try chat) and cell 8 (Status and logs) print most of it.

Also worth one measurement while a GPU runtime is up (the user asked earlier): Drive vs Hugging Face download speed.
On a CPU runtime HF direct was ~200 MB/s (8 streams) vs Drive ~50-70 MB/s.

## How to hand results back

- Add one row per run to `results/results.csv` (columns: runtime,engine,model,kind,prefill_tok_s,ttft_s,
  prompt_tokens,decode_tok_s,cold_start_s,per_item_s,sample,date; runtime is `g4` / `a100` / `l4`), then
  `python3 -m notebook_code.results_table` to refresh the README tables, run the tests, commit and push to `main`.
- Append a dated entry to the **Log** section below: what you ran, what broke (exact error text from the hub log, cell
  8, or `MyDrive/colab_model_server/logs/`), and anything you changed. The cloud session pulls `main` and reads it.
- If something in `notebook_code/` breaks, either fix it here (tests must pass) or describe it in the log and the cloud
  session will fix it.

## Log

- 2026-11-25 (cloud): CLI GPU requests refused hourly 2026-11-22..25 (503). Scheduled CLI grab turned off. Handed the
  browser runtime and the test block to the local agent.
- 2026-11-25 (cloud → desktop session "Google Colab LLM", answering its question; cross-session messages from the
  cloud are refused, so the answer is here): **take all the GPU retests.** The cloud session is not editing the repo
  and stays out of `results/results.csv`, the README tables and `notebooks/` while you work; it touches Colab no more.
  Order on one runtime (G4 preferred), each pair back to back: (1) Inkling: `strata/inkling-2-mini-sq-20` (now
  Strata v0.2.20) vs `llama/inkling-2-mini-q8`; (2) Hy4: `strata/hy4-flash-sq-128` vs `maya/hy4-flash-maya-l`;
  (3) Flash-Next: `strata/flash-next-strata-iq3-s` vs `basalt/basalt-flash-next-nvfp4-dq6k` on the G4 (A100: Strata
  IQ3_S alone vs 167 tok/s); (4) then the 15db690 engine retests (llama b12648, comfyui v0.46.2, tensorfold v1.3.4,
  ninfer 520d9ce6, basalt v0.5.5, strata-nvfp4 nvfp4.7). Log each run here and push; for a break in `notebook_code/`,
  fix it with the tests passing or log the exact error here for the cloud session to fix.
- 2026-11-26 (cloud, browser): one G4 request sat at "Allocating runtime" 15 min and was cancelled (nothing billed, no
  retry). One A100 80 GB connected 03:43 UTC, deleted 06:51 UTC (Manage sessions empty). Results in
  `results/results.csv` (A100, 2026-11-26; one ~3.3K-token streamed prompt, 512 tokens, temperature 0; cold start =
  first request incl. Drive→local copy and load):
  - Inkling: Strata v0.2.20 3,094 tok/s prefill / 77 decode vs llama.cpp b12648 1,028 / 40. llama.cpp still emits
    reasoning with `reasoning_effort: none` (the chat template maps none → effort 0.0). llama.cpp sm_80 build took
    807 s and is now cached on Drive.
  - TensorFold v1.3.4: decode 71–81 tok/s vs 144 on v1.2.2. GPU 95% busy; its speculative drafting is on by default
    and accepts only 17% of drafted tokens. Likely regression or changed default: check before trusting v1.3.4.
  - Hy4-Flash: Maya v1.1.74 refused `--drafting auto` (choices measure/on/off/mtp/dfly2); catalog now asks for `on`
    (f432aea). On the A100 both engines offload heavily: Maya 51–77 prefill / 16–25 decode, Strata SQ-128 94–96 /
    2 decode. Not a fair comparison off the G4. When Maya's load failed, the hub restarted the 162 GB import on
    every 503-retried request; the load should fail fast instead. Maya also copies files already staged in
    local-scratch a second time into its own models dir (double copy).
  - Flash-Next Strata IQ3_S: 2,611 prefill / 156 decode (second run 117), vs 167 decode on 2026-11-10.
  - Fixed (8566ae7): Colab mounts Drive noexec, so Tailscale/cloudflared now run from a local copy.
  - Not run (G4 only): Basalt v0.5.5, NInfer 520d9ce6, Strata NVFP4 nvfp4.7. ComfyUI v0.46.2 not run.
  - Lesson: the A100 sat idle during copies, compiles and the failed Maya load. Next time prestage files and build
    engines before connecting a GPU.
