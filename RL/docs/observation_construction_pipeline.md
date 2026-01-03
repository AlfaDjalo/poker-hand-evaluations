```mermaid
flowchart LR
    Cards[Private Cards] --> Embedding[32D Hand Embedding]
    Embedding --> Concat

    Stack[Stack Size BB] --> Concat
    Pot[Pot Size] --> Concat
    Position[SB / BB] --> Concat
    Blinds[Blind Size] --> Concat

    Concat --> Observation[Observation Vector]
