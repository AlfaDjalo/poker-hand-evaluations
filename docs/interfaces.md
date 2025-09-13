# Interfaces Between Components

## Backend ↔ Frontend API

| Endpoint          | Method | Input (JSON)                           | Output (JSON)                   |
|-------------------|--------|----------------------------------------|---------------------------------|
| `/api/calculate_equity` | POST   | `{ board, handList, variant }`   | `{ equityList }`      |


## Internal Component Interfaces

```mermaid
classDiagram
    class EquityCalculator {
      +calculate_equity(board, handList, variant) equityList
    }

    class API {
      +POST /api/calculate_equity()
    }


    API --> EquityCalculator