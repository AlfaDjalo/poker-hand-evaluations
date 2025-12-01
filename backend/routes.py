import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ML")))
# import pandas as pd
# import numpy as np
# import io
# import os
from flask import request, jsonify
from ML.config import get_config
from ML.training.trainer import get_custom_objects
# import json
# import requests
# from io import StringIO
# from flask_cors import CORS
# import traceback

from typing import List, Dict, Optional
from bindings.equity_wrapper import compute_equity as compute_calc
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import os

# from stock_data import StockData

# DEBUG = True
# BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# # DATA_PATH = os.path.join(BASE_DIR, "data", "feature_sets.json")
# # FEATURE_SETS_FILE = os.path.join(os.path.dirname(__file__), 'data\\feature_sets.json')
# FEATURE_SETS_FILE = os.path.join(BASE_DIR, "back-end", "data", "feature_sets.json")

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
    
    # Load encoder model once at startup
    encoder_model = None
    encoder_path = None
    
    def load_encoder():
        nonlocal encoder_model, encoder_path
        try:
            # Import config to get encoder path
            config = get_config()
            # encoder_path = os.path.join(config["save_directory"], config["embedding_filename"])
            encoder_path = os.path.join(config["save_directory"], config["encoder_filename"])
            # encoder_path = os.path.join(config["embeddings_directory"], config["encoder_filename"])
            
            if os.path.exists(encoder_path):
                # use trainer's helper for custom objects (was ML.models.implementation.get_custom_objects which doesn't exist)
                encoder_model = tf.keras.models.load_model(
                    encoder_path,
                    custom_objects=get_custom_objects(),
                    compile=False
                )
                print(f"✅ Encoder loaded from {encoder_path}; type={type(encoder_model)}")
                # attempt to print summary if it's a keras Model
                try:
                    if hasattr(encoder_model, "summary"):
                        print("Encoder model summary:")
                        encoder_model.summary()
                except Exception as e:
                    print("Unable to print model summary:", e)
            else:
                print(f"⚠️ Encoder file not found at {encoder_path}")
        except Exception as e:
            print(f"⚠️ Failed to load encoder: {e}")

    # Load encoder on first request
    load_encoder()

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
            # result = equity_wrapper.compute_equity(player_hands, board, debug=True)
            
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
            # result = compute_showdown(player_hands, board, debug=True)

            return {"equities": result['equities']}

        except Exception as e:
            print(f"Error: {e}")
            return {"error": str(e)}
        
    @app.post("/embeddings")
    async def embeddings(req: EmbeddingRequest):
        """Return the embeddings for the requested hands.

        Loads hands from the API request, converts them to tensor format,
        passes through the saved encoder model, and returns embedding vectors.

        Returns:
            JSON object with key `embeddings` (list of embedding vectors) 
            or `error` on failure.
        """
        print("In embeddings API")
        try:
            if encoder_model is None:
                return {"error": "Encoder model not loaded"}

            print(req.embeddingHands)
            print(req.mode)

            embedding_hands = prepare_player_hands(req.embeddingHands)
            mode = req.mode

            print("Filtered embedding hands:", embedding_hands)

            # Convert hands to tensor format (13, 4, 2)
            hand_tensors = []
            for hand in embedding_hands:
                tensor = cards_to_tensor(hand, mode)
                # debug_print_tensor(tensor, label=f"Input for {hand}")
                hand_tensors.append(tensor)
            
            if not hand_tensors:
                return {"error": "No valid hands provided"}
            
            # print("Tensor for each hand:")

            # for i, t in enumerate(hand_tensors):
            #     print(i, tf.reduce_sum(t).numpy(), t.numpy().nonzero())
                # print(i, t)

            # Stack all hands into batch
            batch = tf.stack(hand_tensors)
            print(f"Batch shape: {batch.shape}")

            # Call encoder: prefer .predict if present (covers Keras models loaded), otherwise try calling
            try:
                if hasattr(encoder_model, "predict"):
                    out = encoder_model.predict(batch)
                elif callable(encoder_model):
                    out = encoder_model(batch, training=False)
                else:
                    return {"error": "Loaded encoder is not callable or does not expose predict()"}
            except Exception as e:
                print("Error while running encoder:", e)
                return {"error": f"Encoder run failed: {e}"}

            print("out:", out)

            # Normalize output: if encoder returned multiple tensors (hand, board, combined),
            # pick the combined embedding (last element). Otherwise use the single output.
            if isinstance(out, (list, tuple)):
                try:
                    hand_embeddings = out[0].numpy().tolist()
                    board_embeddings = out[1].numpy().tolist()
                    combined_embeddings = out[2].numpy().tolist()

                    print("hand_embeddings:", hand_embeddings)
                    print("board_embeddings:", board_embeddings)
                    print("combined_embeddings:", combined_embeddings)

                except Exception:
                    # Fallback if already numpy arrays
                    hand_embeddings = out[0].tolist()
                    board_embeddings = out[1].tolist()
                    combined_embeddings = out[2].tolist()
            else:
                # Single output fallback: treat as combined embeddings
                try:
                    combined_embeddings = out.numpy().tolist()
                except Exception:
                    combined_embeddings = out.tolist()

            # Return embeddings according to requested mode
            mode = req.mode.lower() if hasattr(req, "mode") else "combined"

            if mode == "2": #"hand":
                print("Embeddings for hand.")
                embeddings_list = hand_embeddings
            elif mode == "3": #"board":
                print("Embeddings for board.")
                embeddings_list = board_embeddings
            else:  # Default or "combined"
                print("Embeddings for combined.")
                embeddings_list = combined_embeddings

            print(f"Generated {len(embeddings_list)} embeddings for mode '{mode}'")

            return {"embeddings": embeddings_list}

        except Exception as e:
            print(f"Error: {e}")
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
        grid = np.zeros((13, 4, 2), dtype=np.float32)

        # ----- Split cards based on mode -----
        if mode == "2": #"hand":
            hand_cards = cards
            board_cards = []

        elif mode == "3": #"board":
            hand_cards = []
            board_cards = cards

        elif mode == "5": #"combined":
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
                grid[rank_to_idx[rank], suit_to_idx[suit], 1] = 1.0

        return tf.constant(grid, dtype=tf.float32)


    # def cards_to_tensor(mode: str, cards: List[str]) -> tf.Tensor:
    #     """Convert a list of card strings to a (13, 4, 2) tensor.
        
    #     The tensor has two channels:
    #     - Channel 0: hand cards
    #     - Channel 1: board cards
    #     Mode is hand / board / combined and determines which channel
    #     to send the cards to.

    #     Args:
    #         cards: List of card strings like ["As", "Kd", "Qh"]
        
    #     Returns:
    #         tf.Tensor of shape (13, 4, 2) with float32 dtype
    #     """
    #     # Map ranks to indices (0-12)
    #     rank_to_idx = {
    #         'A': 0, 'K': 1, 'Q': 2, 'J': 3, 'T': 4,
    #         '9': 5, '8': 6, '7': 7, '6': 8, '5': 9,
    #         '4': 10, '3': 11, '2': 12
    #     }
        
    #     # Map suits to indices (0-3)
    #     suit_to_idx = {'s': 0, 'h': 1, 'd': 2, 'c': 3}
        
    #     # Initialize 13x4x2 grid (ranks x suits x channels)
    #     grid = tf.zeros((13, 4, 2), dtype=tf.float32)
    #     grid_np = grid.numpy()
        
    #     # Fill in hand cards (channel 0)
    #     for card in cards:
    #         if len(card) < 2:
    #             continue
    #         rank_char = card[0].upper()
    #         suit_char = card[-1].lower()
            


    #         if rank_char in rank_to_idx and suit_char in suit_to_idx:
    #             rank_idx = rank_to_idx[rank_char]
    #             suit_idx = suit_to_idx[suit_char]
    #             grid_np[rank_idx, suit_idx, 0] = 1.0
        
    #     return tf.constant(grid_np, dtype=tf.float32)

def debug_print_tensor(tensor, label=""):
    """Pretty-print a 13x4x2 poker tensor."""
    print(f"\n===== {label} =====")

    rank_labels = ["A","K","Q","J","T","9","8","7","6","5","4","3","2"]
    suit_labels = ["♠","♥","♦","♣"]

    for ch in range(tensor.shape[-1]):    # channel 0/1
        print(f"\n--- Channel {ch} ---")
        for r in range(13):               # each rank row
            row_vals = []
            for s in range(4):            # each suit column
                v = int(tensor[r, s, ch])
                row_vals.append(str(v))
            print(f"{rank_labels[r]} {row_vals}")

    print("\n====================\n")