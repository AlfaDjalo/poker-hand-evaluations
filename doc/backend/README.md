# Backend documentation

This folder contains developer-facing documentation for the backend API and
the model-running / evaluation components.

Files
- `api_flowcharts.md` — flowcharts and sequence diagrams for each public API
- `model_running.md` — in-depth flow and sequence diagrams showing how the
  model/evaluator is invoked and how data flows between layers
- `components.md` — descriptions of backend components and responsibilities

If you want docstrings for additional modules (bindings, `db_plo`, etc.), I
can add them — currently the primary server endpoints are documented in
`backend/server.py`.

Tips
- Use VS Code with the Mermaid preview extension to render `.md` diagrams
- Run backend locally and call `/evaluate` and `/showdown` to validate flows

--
Generated on: documentation scaffolding
