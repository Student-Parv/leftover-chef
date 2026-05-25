# Leftover Chef 👨‍🍳

Turn leftover ingredients into quick recipe ideas with an AI-powered chef assistant.

## Overview

**Leftover Chef** is a lightweight web app for students, home cooks, and anyone trying to reduce food waste.  
You enter ingredients you already have, and the app generates a short recipe with a creative name using an LLM via OpenRouter.

## Features

- Simple single-page UI for entering ingredients
- FastAPI backend with a `/api/recipe` endpoint
- AI-generated recipe responses constrained to provided ingredients
- Deploy-ready structure for Vercel (static frontend + Python API route)

## Tech Stack

- **Frontend:** HTML, Tailwind CSS, vanilla JavaScript
- **Backend:** Python 3.11, FastAPI, Pydantic
- **AI Integration:** `pydantic-ai` + OpenRouter-compatible OpenAI client
- **Deployment:** Vercel (`vercel.json` rewrite for API routing)

## Project Structure

```text
leftover-chef/
├── api/
│   └── index.py        # FastAPI app + AI agent setup
├── index.html          # Frontend UI
├── requirements.txt    # Python dependencies
├── runtime.txt         # Python runtime (3.11)
└── vercel.json         # API rewrite config for deployment
```

## Setup

### Prerequisites

- Python **3.11**
- An OpenRouter API key

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Configure environment variables

Set the OpenRouter key used by the backend:

```bash
export OPENROUTER_API_KEY="your_key_here"
```

On Windows (PowerShell):

```powershell
$env:OPENROUTER_API_KEY="your_key_here"
```

## Running Locally

### Backend only (FastAPI)

```bash
uvicorn api.index:app --reload
```

Backend will run at `http://127.0.0.1:8000`.

You can test the endpoint directly:

```bash
curl -X POST http://127.0.0.1:8000/api/recipe \
  -H "Content-Type: application/json" \
  -d '{"ingredients":["eggs","bread","milk"]}'
```

### Full app (frontend + API) with Vercel CLI

If you want the static page and API served together with the same routing behavior as deployment:

```bash
vercel dev
```

Then open the local URL shown by Vercel CLI.

## Usage

1. Open the app in your browser.
2. Enter ingredients separated by commas (for example: `eggs, stale bread, milk`).
3. Click **Cook Something**.
4. Read the generated recipe in the result panel.

## Configuration

Model/provider settings are in `api/index.py`:

- Base URL: `https://openrouter.ai/api/v1`
- Model: `google/gemini-2.0-flash-exp:free`

You can change these values to use a different OpenRouter-supported model.

## Screenshots

No screenshots are committed yet. Add images here after UI updates, for example:

- `docs/images/home.png` – ingredient input screen
- `docs/images/result.png` – generated recipe example

## Troubleshooting

- **`500` errors from `/api/recipe`:** Verify `OPENROUTER_API_KEY` is set correctly.
- **Auth/provider errors:** Confirm your OpenRouter key is active and has access to the configured model.
- **Frontend works but no recipe appears:** Check browser dev tools and backend logs for API/network errors.

## Contributing

Contributions are welcome. A typical flow:

1. Fork the repository
2. Create a feature branch
3. Make and test your changes
4. Open a pull request with a clear description

## License

This repository currently does not include a license file.  
Add a `LICENSE` file if you want to define reuse terms explicitly.