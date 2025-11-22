# API Flowcharts

This document contains flowcharts and sequence diagrams for the backend
public APIs. Use a Markdown viewer that supports Mermaid to render the
diagrams.

## Evaluate API (POST /evaluate)

```mermaid
flowchart TD
  A[Frontend] -->|POST /evaluate| B[backend/server.py: evaluate]
  B --> C[prepare_player_hands]
  C --> D[normalize_card]
  B -->|call| E[bindings.equity_wrapper.compute_equity]
  E --> F[OMPEval / native evaluator]
  F --> E
  E --> B
  B -->|JSON: {equities}| A
```

### Sequence (simplified)

```mermaid
sequenceDiagram
  participant FE as Frontend
  participant API as backend/server
  participant BIND as equity_wrapper
  participant NATIVE as OMPEval

  FE->>API: POST /evaluate (playerHands, board)
  API->>API: prepare_player_hands()
  API->>BIND: compute_equity(player_hands, board)
  BIND->>NATIVE: compute/evaluate (fast C++)
  NATIVE-->>BIND: result dict (equities, wins, ties, total_hands)
  BIND-->>API: result
  API-->>FE: {equities: [...]}
```

## Showdown API (POST /showdown)

The `/showdown` flow is identical to `/evaluate` in the current
implementation — the difference is semantic (showdown implies exhaustive
evaluation). See the above diagrams.
