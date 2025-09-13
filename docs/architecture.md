# System Architecture

The project consists of:
- **Backend** (Python) -> handles API, data processing, ML models
- **Frontend** (React, Vite, Tailwind) -> handles UI, charting, feature selection
- **Deployment** static frontend hosted on GitHub pages, backend hosted separately

```mermaid
flowchart LR
    subgraph Frontend [React Frontend]
        A[PokerTable.jsx]
    end

    subgraph Backend [Python]
        B[Calculate Equity]
        C@{ shape: lin-cyl, label: "Evaluations DB" }
    end

    A -->|Sends board, handList| B
    B -->|Returns player equities| A
    B -->|Sends board| C
    C -->|Returns hand evaluations| B

