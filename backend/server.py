import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
from bindings.equity_wrapper import compute_equity as compute_calc
# from bindings.compute_equity import compute_calc

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

# Define what data the frontend will send
# class HandRequest(BaseModel):
#     players: Dict[str, List[str]]  # e.g. { "1": ["As", "Kd"], "2": ["Jh", "Jc"] }
#     board: List[str]               # e.g. ["2d", "7s", "9c"]

# Request schema
class HandRequest(BaseModel):
    playerHands: List[List[Optional[str]]]
    board: List[Optional[str]]

@app.post("/evaluate")
async def evaluate(req: HandRequest):
    """
    hands_list: list of list of strings, e.g. [["As", "Kd"], ["Qh", "Qc"]]
    board_cards: list of strings, e.g. ["2c", "7d", "9h"]
    """
    print("In evaluate API")
    try:
        print(req.playerHands)
        print(req.board)

        player_hands = prepare_player_hands(req.playerHands)
        board = [normalize_card(c) for c in (req.board or []) if c and c.strip()]

        print("Filtered player hands:", player_hands)
        print("Filtered board:", board)

        # compute_equity now returns a dict with keys: 'equities', 'wins', 'ties', 'total_hands', 'exact'
        result = compute_calc(player_hands, board, debug=True)
        # result = equity_wrapper.compute_equity(player_hands, board, debug=True)
        
        # For debugging, you can log the full result
        print("Full result:", result)
        print(f"Total hands evaluated: {result['total_hands']}")
        print(f"Wins: {result['wins']}")
        print(f"Ties: {result['ties']}")

        # Return only equities to the API client
        return {"equities": result['equities']}

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}
    
# @app.post("/evaluate")
# async def evaluate(req: HandRequest):
#     """
#     hands_list: list of list of strings, e.g. [["As", "Kd"], ["Qh", "Qc"]]
#     board_cards: list of strings, e.g. ["2c", "7d", "9h"]
#     """
#     print("In evaluate API")
#     try:
#         print(req.playerHands)
#         print(req.board)

#         # player_hands = [[normalize_card(c) for c in hand if c] for hand in req.playerHands]
#         # board = [normalize_card(c) for c in (req.board or []) if c]

#         # print(player_hands)
#         # print(board)

#         player_hands = prepare_player_hands(req.playerHands)
#         board = [normalize_card(c) for c in (req.board or []) if c and c.strip()]

#         print("Filtered player hands:", player_hands)
#         print("Filtered board:", board)

#         # equities = equity_wrapper.compute_equity(player_hands, board)

#         equities = compute_calc(player_hands, board, debug=True)

#         print(equities)

#         return {"equities": equities}

#     except Exception as e:
#         print(f"Error: {e}")
#         return {"error": str(e)}


async def evaluate_old(req: HandRequest):
    print("In evaluate API")
    try:
        db = open_db()

        print(req.playerHands)
        print(req.board)

        player_hands = [[normalize_card(c) for c in hand if c] for hand in req.playerHands]
        board = [normalize_card(c) for c in (req.board or []) if c]

        print(player_hands)
        print(board)

        equities = calculate_equity_for_multiple_hands_exhaustive(
            db=db,
            player_hands=player_hands,
            board=board,
            # player_hands=req.playerHands,
            # board=req.board,
            debug=True  # optional
        )

        print(f"equities: {equities}")

        return {"equities": equities}

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

def normalize_card(card: str) -> str:
    """Convert rank to uppercase, suit to lowercase (e.g., 'as' -> 'As')."""
    if not card or len(card) < 2:
        return card  # skip invalid entries
    rank = card[:-1].upper()
    suit = card[-1].lower()
    return rank + suit

def prepare_player_hands(raw_hands):
    """Filter and normalize player hands before sending to C++."""
    valid_hands = []
    for hand in raw_hands:
        # Filter out None or empty strings
        clean_hand = [normalize_card(c) for c in hand if c and c.strip()]
        if clean_hand:  # Only include non-empty hands
            valid_hands.append(clean_hand)
    return valid_hands