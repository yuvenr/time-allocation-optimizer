# Time Allocation Optimizer

A smart tool for planning students' study schedules and optimizing study load. It combines **constraint satisfaction**, **association rule mining (ARM / FP-Growth)**, and **large language models (LLMs)** to estimate cognitive load.

The system recommends a semester-by-semester study plan from the current semester to graduation. It considers:

1. **Curriculum and academic rules**: course prerequisites, odd/even semester offerings, credit limits based on the previous GPA, first-semester course packages, PLP and Capstone requirements, topic choices, and maximum study duration.
2. **Association Rule Mining (FP-Growth)**: patterns from past course registrations are used to rank elective recommendations based on student interests and historical trends.
3. **Cognitive load and time allocation**: an LLM estimates the revised Bloom's taxonomy level for each course learning outcome. In `hybrid` mode, this sets a per-semester cognitive-load limit (θ).

`arm` and `hybrid` use the same planning algorithm. They differ only in the cognitive-load limit (θ).

## Features

| Role | Features |
|---|---|
| Student | Grade and profile history, topic-based course choices, semester plans with explanations, cognitive-load charts, validation of 16 academic rules, and interactive drag-and-drop course changes |
| Academic advisor (password protected) | Review study plans, compare them with system recommendations, approve plans, and add advising notes |
| Administrator (password protected) | Data readiness checklist, data templates, academic data import, preprocessing, ARM and sensitivity analysis, LLM-based Bloom assessment, LLM-vs-expert evaluation (ICC), comparison simulations, CSV export for JASP, and test-data packages |

Guides: [User Guide](docs/PANDUAN_PENGGUNA.md) · [Administrator Guide](docs/PANDUAN_ADMINISTRATOR.md) · [Deployment Guide](docs/DEPLOY_STREAMLIT_NEON.md)

## Run locally

Requires Python 3.11 or later.

```bash
python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                    # set role password hashes: python -m src.auth
streamlit run app/streamlit_app.py
```

Without `DATABASE_URL`, the app uses SQLite at `data/crs.db` and creates it automatically. The system starts empty. Log in as an Administrator and open **Start here** to load test data or add data step by step.

A study plan can still be generated from the curriculum structure and rules without historical data.

## Testing

```bash
python -m pytest
```

The tests use deterministic fictional data. Set `CRS_TEST_DATABASE_URL=postgresql://...` to run the full test suite with PostgreSQL. GitHub Actions runs both configurations on every push.

## Project structure

```text
app/                  Streamlit interface (streamlit_app.py, tampilan.py)
src/                  rules, planner, ARM, LLM/Bloom scoring, preprocessing, simulation,
                      metrics, statistics, export, database, adjustment, readiness,
                      templates, test packages, file handling, and authentication
data/kurikulum_2024/  Sample 2024 curriculum dataset
tests/                Pytest tests
docs/                 User, administrator, and deployment guides
config.yaml           All thresholds and parameters
```

## Data and privacy

- The repository contains no student data, student IDs, databases, or API keys. Administrators import academic data through the app. Student IDs are salted and hashed before storage.
- Only curriculum text (course descriptions, CPMK, and Sub-CPMK) is sent to the LLM.
- `data/kurikulum_2024/` comes from the 2024–2029 curriculum documents of the Information Technology Education program at UB.

## License

- **Source code**: [MIT License](LICENSE), © 2026 Yuven Reksa Pratama.
- **2024 Curriculum Dataset** (`data/kurikulum_2024/`) is **not** covered by the MIT License. It was extracted from curriculum documents of the Bachelor of Information Technology Education program at FILKOM UB and remains the property of the program. It is included as a reference sample so the app can run. See [data/kurikulum_2024/README.md](data/kurikulum_2024/README.md).
