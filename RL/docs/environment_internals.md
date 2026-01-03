```mermaid
stateDiagram-v2
    [*] --> PostBlinds
    PostBlinds --> DealCards
    DealCards --> P1Decision

    P1Decision --> Terminal_FoldP1: P1 folds
    P1Decision --> P2Decision: P1 pushes

    P2Decision --> Terminal_FoldP2: P2 folds
    P2Decision --> Showdown: P2 calls

    Showdown --> Terminal_Showdown

    Terminal_FoldP1 --> [*]
    Terminal_FoldP2 --> [*]
    Terminal_Showdown --> [*]
