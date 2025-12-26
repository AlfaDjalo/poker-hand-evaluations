"""Backend API server for poker hand evaluation.

This module exposes HTTP endpoints used by the frontend. It wraps the
underlying C++/pybind11 evaluation bindings and provides input validation
and normalization for requests coming from the UI.

Key endpoints
- POST /evaluate  — run equity calculations for a set of player hands and an optional board
- POST /showdown  — run an exhaustive showdown/evaluation and return equities

Helper functions in this module normalize incoming card strings and prepare
player hands into the form expected by the native bindings.

The code intentionally performs light-weight validation and logs input and
result summaries to stdout for quick debugging during development.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
from bindings.equity_wrapper import compute_equity as compute_calc
# from bindings.compute_equity import compute_calc
from routes import setup_api_routes
# from flask_bootstrap import Bootstrap
from fastapi.middleware.cors import CORSMiddleware

from python_hand_evaluator import calculate_equity_for_multiple_hands_exhaustive
from db_plo import DB_PLO, open_db


app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React dev server
    # allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_api_routes(app)
