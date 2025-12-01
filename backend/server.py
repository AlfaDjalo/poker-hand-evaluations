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

# app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # Add a secret key for session management
# # CORS(app)
# # Allow all origins (for local dev)
# CORS(app, resources={r"/api/*": {"origins": "*"}})
# Bootstrap(app)

setup_api_routes(app)
# Define what data the frontend will send
# class HandRequest(BaseModel):
#     players: Dict[str, List[str]]  # e.g. { "1": ["As", "Kd"], "2": ["Jh", "Jc"] }
#     board: List[str]               # e.g. ["2d", "7s", "9c"]

# Request schema
# class HandRequest(BaseModel):
#     """Request body for evaluation endpoints.

#     Attributes:
#         playerHands: list of player hand arrays. Each player hand is a list of
#             card strings (e.g. ["As", "Kd"]) or None values for empty slots.
#             Example: [["As","Kd"],["Qh","Qc"],[None,None]]
#         board: list of 0..5 board card strings (e.g. ["2c","7d","9h"]).
#             Elements may be None or empty strings and will be filtered.
#     """
#     playerHands: List[List[Optional[str]]]
#     board: List[Optional[str]]

# @app.post("/evaluate")
# async def evaluate(req: HandRequest):
#     """Run equity calculation for given player hands and an optional board.

#     This endpoint expects a JSON body conforming to :class:`HandRequest`.

#     The function performs the following steps:
#     1. Normalizes and filters incoming card strings.
#     2. Prepares the player hands into the compact format required by the
#        native evaluation bindings.
#     3. Calls the compiled equity routine and returns the equities array to
#        the client. For debugging the full result (wins/ties/total_hands) is
#        logged to stdout.

#     Returns:
#         JSON object with key `equities` mapping to a list of floats (one per
#         player) or an `error` string if the call failed.
#     """
#     print("In evaluate API")
#     try:
#         print(req.playerHands)
#         print(req.board)

#         player_hands = prepare_player_hands(req.playerHands)
#         board = [normalize_card(c) for c in (req.board or []) if c and c.strip()]

#         print("Filtered player hands:", player_hands)
#         print("Filtered board:", board)

#         # compute_equity now returns a dict with keys: 'equities', 'wins', 'ties', 'total_hands', 'exact'
#         result = compute_calc(player_hands, board, debug=True)
#         # result = equity_wrapper.compute_equity(player_hands, board, debug=True)
        
#         # For debugging, you can log the full result
#         print("Full result:", result)
#         print(f"Total hands evaluated: {result.get('total_hands')}")
#         print(f"Wins: {result.get('wins')}")
#         print(f"Ties: {result.get('ties')}")

#         # Return only equities to the API client
#         return {"equities": result['equities']}

#     except Exception as e:
#         print(f"Error: {e}")
#         return {"error": str(e)}
    
# @app.post("/showdown")
# async def showdown(req: HandRequest):
#     """Run an exhaustive showdown/evaluation and return equities.

#     This endpoint is similar to `/evaluate` but intended for cases where a
#     full exhaustive showdown is required (for example to inspect exact
#     distributions). The returned object currently contains the same
#     `equities` array as `/evaluate`. The endpoint logs inputs and the
#     full native result for debugging.

#     Returns:
#         JSON object with key `equities` or `error` on failure.
#     """
#     print("In showdown API")
#     try:
#         print(req.playerHands)
#         print(req.board)

#         player_hands = prepare_player_hands(req.playerHands)
#         board = [normalize_card(c) for c in (req.board or []) if c and c.strip()]

#         print("Filtered player hands:", player_hands)
#         print("Filtered board:", board)

#         result = compute_calc(player_hands, board, debug=True)
#         # result = compute_showdown(player_hands, board, debug=True)

#         return {"equities": result['equities']}

#     except Exception as e:
#         print(f"Error: {e}")
#         return {"error": str(e)}
    
# def normalize_card(card: str) -> str:
#     """Normalize a single card string.

#     The frontend may send cards in a variety of formats (e.g. `as`, `AS`,
#     `10d`, `td`). This helper returns a normalized two-character (or
#     three-character for 10) string where the rank portion is upper-case and
#     the suit character is lower-case. Examples:

#     - "as" -> "As"
#     - "TH" -> "Th"
#     - "10d" -> "10d"

#     If the input is invalid or too short the original value is returned so
#     the caller can decide how to handle it.

#     Args:
#         card: string representing a card.

#     Returns:
#         Normalized card string.
#     """
#     if not card or len(card) < 2:
#         return card  # skip invalid entries
#     rank = card[:-1].upper()
#     suit = card[-1].lower()
#     return rank + suit

# def prepare_player_hands(raw_hands):
#     """Prepare and validate player hands for the evaluator.

#     This helper accepts the raw list-of-lists as provided by the frontend,
#     normalizes each card, filters out empty slots, and returns a list of
#     players where each player is a compact list of card strings. Empty
#     player entries (no cards) are omitted.

#     Args:
#         raw_hands: List of player hand lists as received from the frontend.

#     Returns:
#         List[List[str]]: cleaned, normalized player hands ready to be passed
#         to the native evaluation function.
#     """
#     valid_hands = []
#     for hand in raw_hands:
#         # Filter out None or empty strings
#         clean_hand = [normalize_card(c) for c in hand if c and c.strip()]
#         if clean_hand:  # Only include non-empty hands
#             valid_hands.append(clean_hand)
#     return valid_hands