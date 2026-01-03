```mermaid
flowchart LR
    Trainer -->|episodes| Env
    Env -->|observations| Trainer
    Trainer -->|actions| Env
    Env -->|rewards, done| Trainer

    Trainer --> Policy
    Policy --> Trainer

    Policy --> Embedding
