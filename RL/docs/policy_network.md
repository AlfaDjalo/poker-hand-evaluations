```mermaid
flowchart TB
    Obs[Observation Vector] --> FC1
    FC1 --> FC2
    FC2 --> PolicyHead

    PolicyHead --> ActionProbs[Action Probabilities]

    subgraph Optional
        FC2 --> ValueHead
    end
