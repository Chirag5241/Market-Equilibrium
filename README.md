# Decentralized Equilibrium Discovery After Shocks

CS 580 (Algorithmic Game Theory), Fall 2026 · Chirag Rastogi and Sejun Park

An over-the-counter corporate-bond market calibrated to March 2025 Treasury yields and corporate spreads. A shock changes investors' preferences, and the market re-trades one bilateral Nash bargain at a time. The site compares how quickly different negotiators get back to the terminal allocation: an exact full-information solver, estimated-utility solvers with private information, and LLM agents.

**Live site:** https://chirag5241.github.io/Market-Equilibrium/ (after GitHub Pages is enabled, see below)

## What's here

| Path | Contents |
|---|---|
| `website/` | The static site: `index.html`, the market engine `engine.js`, and recorded LLM results in `data/`. Everything except the recorded LLM runs is computed in the browser. |
| `website/archive/duration-market/` | The earlier version of the site, built on the no-cash barter market. |
| `agents/` | LLM experiment harnesses (Azure gpt-6-luna) and their recorded results in `agents/results/`. |
| `report/` | Report drafts (LaTeX section on the term structure, with a "To review" part) and the model handoff note. |

## View locally

No build step. Serve the `website/` folder with any static server:

```bash
python3 -m http.server 8000 --directory website
```

Then open http://localhost:8000. (Opening `index.html` directly from disk also works, except the recorded-LLM panels, which load JSON with `fetch`.)

## Publish with GitHub Pages

The workflow in `.github/workflows/pages.yml` deploys `website/` on every push to `main`. One-time setup: repository **Settings → Pages → Build and deployment → Source: GitHub Actions**.

## Re-running the LLM experiments

The harnesses need Azure OpenAI credentials in `agents/.env` (`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, `DEPLOYMENT_NAME_6_LUNA`, `DEPLOYMENT_NAME_6_LUNA_2`). That file is git-ignored; never commit it.

```bash
node agents/llm_full_info.mjs --case both     # full-information LLMs, minimal vs. objective prompt
node agents/export_for_site.mjs
node agents/llm_estimated.mjs                 # estimated-utility solvers: particle beliefs and LLM estimates
node agents/export_estimated.mjs
```

Each run logs every API call's tokens and cost to `agents/results/` and stops at a hard spending cap (`--cap`, default $10). Add `--resume` to continue an interrupted run. Recorded runs: full information $5.73 in total; estimated-utility (LLM estimates) $2.29; the particle solver makes no API calls.

## Notes

- The negotiation lab in section 08 calls Claude through the claude.ai artifact runtime, so it only works when the page is opened as a published claude.ai artifact. On GitHub Pages it shows a notice instead.
- The engine reproduces the interim deck's seed-0 reference values exactly when given the same investor draws and matchings. Its own seeds use a different random generator from numpy, so its numbers vary around the deck's.
