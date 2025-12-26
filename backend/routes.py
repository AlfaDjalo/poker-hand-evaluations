import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ML")))

from flask import request, jsonify
from ML.config import get_config
from ML.training.trainer import get_custom_objects

from typing import List, Dict, Optional
from bindings.equity_wrapper import compute_equity as compute_calc
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import os

# Request schema
class HandRequest(BaseModel):
    """Request body for evaluation endpoints.

    Attributes:
        playerHands: list of player hand arrays. Each player hand is a list of
            card strings (e.g. ["As", "Kd"]) or None values for empty slots.
            Example: [["As","Kd"],["Qh","Qc"],[None,None]]
        board: list of 0..5 board card strings (e.g. ["2c","7d","9h"]).
            Elements may be None or empty strings and will be filtered.
    """
    playerHands: List[List[Optional[str]]]
    board: List[Optional[str]]

class EmbeddingRequest(BaseModel):
    """Request body for embedding endpoints.

    Attributes:
        embeddingHands: list of player hand arrays. Each player hand is a list of
            card strings (e.g. ["As", "Kd"]) or None values for empty slots.
            Example: [["As","Kd"],["Qh","Qc"],[None,None]]
    """
    embeddingHands: List[List[Optional[str]]]
    mode: str

# API Routes for Front-End
def setup_api_routes(app):
    """
    Set up all routes for the Flask application.

    Args:
        app (Flask): The Flask application instance.
    """
    
    @app.post("/evaluate")
    async def evaluate(req: HandRequest):
        """Run equity calculation for given player hands and an optional board.

        This endpoint expects a JSON body conforming to :class:`HandRequest`.

        The function performs the following steps:
        1. Normalizes and filters incoming card strings.
        2. Prepares the player hands into the compact format required by the
        native evaluation bindings.
        3. Calls the compiled equity routine and returns the equities array to
        the client. For debugging the full result (wins/ties/total_hands) is
        logged to stdout.

        Returns:
            JSON object with key `equities` mapping to a list of floats (one per
            player) or an `error` string if the call failed.
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
            
            # For debugging, you can log the full result
            print("Full result:", result)
            print(f"Total hands evaluated: {result.get('total_hands')}")
            print(f"Wins: {result.get('wins')}")
            print(f"Ties: {result.get('ties')}")

            # Return only equities to the API client
            return {"equities": result['equities']}

        except Exception as e:
            print(f"Error: {e}")
            return {"error": str(e)}
        
    @app.post("/showdown")
    async def showdown(req: HandRequest):
        """Run an exhaustive showdown/evaluation and return equities.

        This endpoint is similar to `/evaluate` but intended for cases where a
        full exhaustive showdown is required (for example to inspect exact
        distributions). The returned object currently contains the same
        `equities` array as `/evaluate`. The endpoint logs inputs and the
        full native result for debugging.

        Returns:
            JSON object with key `equities` or `error` on failure.
        """
        print("In showdown API")
        try:
            print(req.playerHands)
            print(req.board)

            player_hands = prepare_player_hands(req.playerHands)
            board = [normalize_card(c) for c in (req.board or []) if c and c.strip()]

            print("Filtered player hands:", player_hands)
            print("Filtered board:", board)

            result = compute_calc(player_hands, board, debug=True)

            return {"equities": result['equities']}

        except Exception as e:
            print(f"Error: {e}")
            return {"error": str(e)}
            

    @app.post("/embeddings")
    async def embeddings(req: EmbeddingRequest):
        """
        Loads the encoder, computes embeddings, and returns them.
        """

        try:
            print("\n=== Embeddings Request ===")
            print("Requested mode:", req.mode)
            print("Hands:", req.embeddingHands)

            # ------------------------
            # 1. Validate mode
            # ------------------------
            mode = str(req.mode).lower().strip()
            if mode not in ("2", "3", "5", "combined", "hand", "board"):
                return {"error": f"Invalid mode: {req.mode}"}

            # Normalise mode names
            if mode == "2" or mode == "hand":
                mode = "hand"
            elif mode == "3" or mode == "board":
                mode = "board"
            else:
                mode = "combined"

            # ------------------------
            # 2. Load config paths
            # ------------------------
            cfg = get_config()
            encoder_path = os.path.join(cfg["save_directory"], cfg["encoder_filename"])
            # model_paths = {
            #     "hand": os.path.join(cfg["save_directory"], cfg["hand_encoder_filename"]),
            #     "board": os.path.join(cfg["save_directory"], cfg["board_encoder_filename"]),
            #     "combined": os.path.join(cfg["save_directory"], cfg["combined_encoder_filename"]),
            #     "shared": os.path.join(cfg["save_directory"], cfg["shared_encoder_filename"]),
            # }

            # if cfg["use_shared_encoder"]:
            #     encoder_path = model_paths["shared"]
            # else:
            #     encoder_path = model_paths[mode]
            
            if not os.path.exists(encoder_path):
                return {"error": f"Encoder file not found: {encoder_path}"}

            print(f"Loading {mode} encoder from: {encoder_path}")

            # ------------------------
            # 3. Load only this encoder
            # ------------------------
            encoder = tf.keras.models.load_model(
                encoder_path,
                custom_objects=get_custom_objects(),
                compile=False
            )

            # ------------------------
            # 4. Convert hands to tensors
            # ------------------------
            embedding_hands = prepare_player_hands(req.embeddingHands)

            tensors = []
            for h in embedding_hands:
                t = cards_to_tensor(h, mode)   # mode controls hand/board/combined input structure
                tensors.append(t)

            if not tensors:
                return {"error": "No valid hands provided"}

            batch = tf.stack(tensors)
            print("Batch:", batch.shape)

            # ------------------------
            # 5. Run encoder
            # ------------------------
            try:
                if hasattr(encoder, "predict"):
                    out = encoder.predict(batch)
                else:
                    out = encoder(batch, training=False)
            except Exception as e:
                print("Encoder error:", e)
                return {"error": f"Encoder run failed: {e}"}

            # ------------------------
            # 6. Normalize outputs
            # ------------------------
            try:
                embeddings = out[0].numpy().tolist()
            except Exception:
                embeddings = out[0].tolist()

            print(f"Generated {len(embeddings)} {mode} embeddings")

            return {"embeddings": embeddings}

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}


    def normalize_card(card: str) -> str:
        """Normalize a single card string.

        The frontend may send cards in a variety of formats (e.g. `as`, `AS`,
        `10d`, `td`). This helper returns a normalized two-character (or
        three-character for 10) string where the rank portion is upper-case and
        the suit character is lower-case. Examples:

        - "as" -> "As"
        - "TH" -> "Th"
        - "10d" -> "10d"

        If the input is invalid or too short the original value is returned so
        the caller can decide how to handle it.

        Args:
            card: string representing a card.

        Returns:
            Normalized card string.
        """
        if not card or len(card) < 2:
            return card  # skip invalid entries
        rank = card[:-1].upper()
        suit = card[-1].lower()
        return rank + suit

    def prepare_player_hands(raw_hands):
        """Prepare and validate player hands for the evaluator.

        This helper accepts the raw list-of-lists as provided by the frontend,
        normalizes each card, filters out empty slots, and returns a list of
        players where each player is a compact list of card strings. Empty
        player entries (no cards) are omitted.

        Args:
            raw_hands: List of player hand lists as received from the frontend.

        Returns:
            List[List[str]]: cleaned, normalized player hands ready to be passed
            to the native evaluation function.
        """
        valid_hands = []
        for hand in raw_hands:
            # Filter out None or empty strings
            clean_hand = [normalize_card(c) for c in hand if c and c.strip()]
            if clean_hand:  # Only include non-empty hands
                valid_hands.append(clean_hand)
        return valid_hands

    def cards_to_tensor(cards: List[str], mode: str) -> tf.Tensor:
        """
        Convert cards to a (13, 4, 2) tensor.

        Channels:
            0 = hand cards
            1 = board cards

        Modes:
            - "hand":     all cards go to hand channel
            - "board":    all cards go to board channel
            - "combined": cards[0:2] -> hand channel, cards[2:5] -> board channel

        Args:
            cards: List[str]   e.g. ["As", "Kd"] or ["As", "Kd", "7h", "2d", "Tc"]
            mode: str          "hand", "board", "combined"

        Returns:
            tf.Tensor of shape (13, 4, 2)
        """

        # ----- Rank & suit lookup -----
        rank_to_idx = {
            'A': 0, 'K': 1, 'Q': 2, 'J': 3, 'T': 4,
            '9': 5, '8': 6, '7': 7, '6': 8, '5': 9,
            '4': 10, '3': 11, '2': 12
        }
        suit_to_idx = {'s': 0, 'h': 1, 'd': 2, 'c': 3}

        # ----- Base empty grid -----
        grid = np.zeros((14, 4, 1), dtype=np.float32)

        # ----- Split cards based on mode -----
        if mode == "hand":
            hand_cards = cards
            board_cards = []

        elif mode == "board":
            hand_cards = []
            board_cards = cards

        elif mode == "combined":
            hand_cards = cards[:2]
            board_cards = cards[2:5]

        else:
            raise ValueError(f"Unknown mode '{mode}'. Expected: hand, board, combined.")

        # ----- Fill hand channel (0) -----
        for card in hand_cards:
            if len(card) < 2:
                continue
            rank = card[0].upper()
            suit = card[-1].lower()
            if rank in rank_to_idx and suit in suit_to_idx:
                grid[rank_to_idx[rank], suit_to_idx[suit], 0] = 1.0

        # ----- Fill board channel (1) -----
        for card in board_cards:
            if len(card) < 2:
                continue
            rank = card[0].upper()
            suit = card[-1].lower()
            if rank in rank_to_idx and suit in suit_to_idx:
                grid[rank_to_idx[rank], suit_to_idx[suit], 0] = 1.0

        return tf.constant(grid, dtype=tf.float32)


def debug_print_tensor(tensor, label=""):
    """Pretty-print a 13x4x2 poker tensor."""
    print(f"\n===== {label} =====")

    rank_labels = ["A","K","Q","J","T","9","8","7","6","5","4","3","2"]
    suit_labels = ["♠","♥","♦","♣"]

    for ch in range(tensor.shape[-1]):    # channel 0/1
        print(f"\n--- Channel {ch} ---")
        for r in range(14):               # each rank row
        # for r in range(13):               # each rank row
            row_vals = []
            for s in range(4):            # each suit column
                v = int(tensor[r, s, ch])
                row_vals.append(str(v))
            print(f"{rank_labels[r]} {row_vals}")

    print("\n====================\n")