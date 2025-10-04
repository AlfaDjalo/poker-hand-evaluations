from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional

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