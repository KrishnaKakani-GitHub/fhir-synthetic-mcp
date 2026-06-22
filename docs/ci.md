# CI / GitHub Actions Setup

The `.github/workflows/ci.yml` file requires the `workflows` scope on your
GitHub token to be pushed via API. Push it manually:

```bash
git clone https://github.com/KrishnaKakani-GitHub/fhir-synthetic-mcp
cd fhir-synthetic-mcp
mkdir -p .github/workflows
```

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    name: Tests (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install
        run: |
          pip install fastmcp pydantic rank-bm25 pytest pytest-asyncio httpx
          pip install -e . --no-deps
        env:
          FHIR_MCP_RAG_DISABLE_CHROMA: "1"
      - name: Seed DB
        run: python scripts/seed_db.py
        env:
          FHIR_MCP_DB: data/fhir.db
      - name: pytest
        run: pytest -q --tb=short
        env:
          FHIR_MCP_DB: data/fhir.db
          FHIR_MCP_RAG_DISABLE_CHROMA: "1"

  eval:
    name: Eval regression gate
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install
        run: |
          pip install fastmcp pydantic rank-bm25 pytest pytest-asyncio
          pip install -e . --no-deps
        env:
          FHIR_MCP_RAG_DISABLE_CHROMA: "1"
      - name: Seed DB
        run: python scripts/seed_db.py
        env:
          FHIR_MCP_DB: data/fhir.db
      - name: Eval smoke suite
        run: python scripts/run_evals.py --suite smoke --output eval_report.json
        env:
          FHIR_MCP_DB: data/fhir.db
          FHIR_MCP_RAG_DISABLE_CHROMA: "1"
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: eval-report
          path: eval_report.json
```

Then commit and push:

```bash
git add .github/workflows/ci.yml
git commit -m "ci: GitHub Actions pytest + eval regression gate"
git push
```

GitHub Actions will activate immediately on the next push.
