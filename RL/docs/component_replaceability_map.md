```mermaid
flowchart LR
    Env --> ObsBuilder
    ObsBuilder --> Embedding
    ObsBuilder --> Policy

    Policy --> Trainer
    Trainer --> Optimizer
    Trainer --> LossFn
