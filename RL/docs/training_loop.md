```mermaid
sequenceDiagram
    participant Trainer
    participant Env
    participant Policy

    Trainer->>Env: reset()
    Env-->>Trainer: observation (P1)

    Trainer->>Policy: act(obs)
    Policy-->>Trainer: action, logprob

    Trainer->>Env: step(action)
    Env-->>Trainer: observation (P2)

    Trainer->>Policy: act(obs)
    Policy-->>Trainer: action, logprob

    Trainer->>Env: step(action)
    Env-->>Trainer: reward, done

    Trainer->>Trainer: compute returns
    Trainer->>Policy: update parameters
