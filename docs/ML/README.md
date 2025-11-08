# Poker Embeddings ML Module

This module trains and serves neural networks that learn **embeddings** and **strength estimations** for poker hands.  
It is designed to integrate with the main poker simulator backend and ultimately provide real-time evaluations to the frontend.

## 🧩 System Overview

```mermaid
graph TD
  A["Frontend (React)"] -->|HTTP / WebSocket| B["Backend (Python)"]
  B -->|Evaluation Requests| C[C++ Evaluator]
  B -->|Model Inference| D[Poker ML Module]
  D --> E[Model Output / Embeddings]

