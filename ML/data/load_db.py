import os
import json
import random

import psycopg2

# from db_plo import DB_PLO

CONFIG_FILE = "db_config.json"

def open_db():
    """
    Opens database object.

    Returns:
        Database
    """
    config = load_db_config(CONFIG_FILE)
    db = DB_PLO(
        dbname=config["dbname"],
        user=config["user"],
        password=config["password"],
        host=config.get("host", "localhost"),
        port=config.get("port", 5432)
    )
    return db

def load_db_config(config_path="config.json"):
    """
    Loads DB config from a JSON file.

    Arguments:
        config_path: location of config file.

    Returns:
        Dictionary of config elements loaded from json file.
    """
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Database config file '{config_path}' not found.")

    with open(config_path, "r") as f:
        return json.load(f)
    
class DB_PLO:
    def __init__(self, dbname, user, password, host="localhost", port=5432):
        try:
            # Connection management
            self.conn = psycopg2.connect(
                dbname = dbname,
                user = user,
                password = password,
                host = host,
                port = port
            )
            self.conn.autocommit = True
            self.cursor = self.conn.cursor()
            print("✅ Connected to PostgreSQL")
        except Exception as e:
            print("❌ Failed to connect to PostgreSQL:", e)
            raise
        return

    def get_sample_evaluations(self, batch_size: int):
        self.cursor.execute(f"SELECT * FROM plo_evaluations_bm ORDER BY RANDOM() LIMIT {batch_size};")
        rows = self.cursor.fetchall()

        print(f"✅ Returned {len(rows)} evaluations")
        return rows

    def get_sample_board_evaluations(self, batch_size: int):
        query = f"SELECT e1.board_mask, e1.hand_mask AS hand_a, e2.hand_mask AS hand_b, e1.high_value AS value_a, e2.high_value AS value_b \
            FROM plo_evaluations_bm e1 JOIN plo_evaluations_bm e2 ON e1.board_mask = e2.board_mask AND e1.hand_mask <> e2.hand_mask \
            ORDER BY RANDOM() LIMIT {batch_size};"

        self.cursor.execute(query)
        # self.cursor.execute(f"SELECT * FROM plo_evaluations_bm ORDER BY RANDOM() LIMIT {batch_size};")
        rows = self.cursor.fetchall()

        print(f"✅ Returned {len(rows)} evaluations")
        return rows

    def get_comparison_pairs(self, mode, batch_size=10000):
        """
        Efficient version of comparison-pair generation:
        1. Randomly sample a small set of base rows using TABLESAMPLE.
        2. Join only within this small set.
        """

        # Heuristic: sample 5× as many as we need
        sample_size = batch_size * 5

        if mode == "board":
            # Same board_mask, different hands.
            query = f"""
                WITH base AS (
                    SELECT *
                    FROM plo_evaluations_bm TABLESAMPLE SYSTEM (1)
                    LIMIT {sample_size}
                )
                SELECT 
                    b1.hand_mask AS hand_a,
                    b2.hand_mask AS hand_b,
                    b1.board_mask AS board_a,
                    b2.board_mask AS board_b,
                    b1.high_value AS value_a,
                    b2.high_value AS value_b
                FROM base b1
                JOIN base b2
                ON b1.board_mask = b2.board_mask
                AND b1.hand_mask <> b2.hand_mask
                LIMIT {batch_size};
            """

        elif mode == "hand":
            # Same hand_mask, different boards.
            query = f"""
                WITH base AS (
                    SELECT *
                    FROM plo_evaluations_bm TABLESAMPLE SYSTEM (1)
                    LIMIT {sample_size}
                )
                SELECT 
                    b1.hand_mask AS hand_a,
                    b2.hand_mask AS hand_b,
                    b1.board_mask AS board_a,
                    b2.board_mask AS board_b,
                    b1.high_value AS value_a,
                    b2.high_value AS value_b
                FROM base b1
                JOIN base b2
                ON b1.hand_mask = b2.hand_mask
                AND b1.board_mask <> b2.board_mask
                LIMIT {batch_size};
            """

        elif mode == "mix":
            # Completely different hand and board masks.
            query = f"""
                WITH base AS (
                    SELECT *
                    FROM plo_evaluations_bm TABLESAMPLE SYSTEM (1)
                    LIMIT {sample_size}
                )
                SELECT 
                    b1.hand_mask AS hand_a,
                    b2.hand_mask AS hand_b,
                    b1.board_mask AS board_a,
                    b2.board_mask AS board_b,
                    b1.high_value AS value_a,
                    b2.high_value AS value_b
                FROM base b1
                JOIN base b2
                ON b1.hand_mask <> b2.hand_mask
                AND b1.board_mask <> b2.board_mask
                LIMIT {batch_size};
            """

        else:
            raise ValueError(f"Invalid mode: {mode}")

        self.cursor.execute(query)
        rows = self.cursor.fetchall()

        for i in range(len(rows)):
            if random.random() < 0.5:
                hand_a, hand_b = rows[i][0], rows[i][1]
                board_a, board_b = rows[i][2], rows[i][3]
                value_a, value_b = rows[i][4], rows[i][5]
                rows[i] = (hand_b, hand_a, board_b, board_a, value_b, value_a)
       
        # Shuffle rows to add randomness
        # random.shuffle(rows)

        # Then return only up to batch_size, just in case
        return rows[:batch_size]



        # self.cursor.execute(query)
        # return self.cursor.fetchall()
