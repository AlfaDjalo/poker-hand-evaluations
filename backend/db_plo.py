import json
import os
import io
import csv
import random

import psycopg2
from psycopg2.extras import execute_values

CONFIG_FILE = "config.json"

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
    
    # Methods to implement schema
    def init_schema(self):
        # Schema Initialization
        self.cursor.execute("""
                    CREATE TABLE IF NOT EXISTS plo_boards (
                    board_id SERIAL PRIMARY KEY,
                    card1_str TEXT NOT NULL,
                    card2_str TEXT NOT NULL,
                    card3_str TEXT NOT NULL,
                    suit_pattern INT,
                    UNIQUE(card1_str, card2_str, card3_str)
                    );
                    """)

        self.cursor.execute("""
                    CREATE TABLE IF NOT EXISTS hands (
                    hand_id SERIAL PRIMARY KEY,
                    hand_str TEXT UNIQUE NOT NULL,
                    suitedness TEXT NOT NULL        
                    );
                    """)

        self.cursor.execute("""
                    CREATE TABLE IF NOT EXISTS plo_evaluations (
                    board_id INT REFERENCES boards(board_id),
                    hand_id INT REFERENCES hands(hand_id),
                    hand_value INT NOT NULL,
                    rank_dense FLOAT,
                    PRIMARY KEY (board_id, hand_id)
                    );
                    """)

        self.cursor.execute("""
                    CREATE TABLE IF NOT EXISTS plo_evaluations_bm (
                        hand_mask BIGINT,
                        board_mask BIGINT,
                        high_value INT NOT NULL,
                        low_value INT NOT NULL,
                        PRIMARY KEY (hand_mask, board_mask)
                    );
                    """)
        
        print("🛠️ Created PLO tables")

        return

    def remove_indices_from_evaluations(self):
        """
        Function to remove indices from plo_evaluations table.
        Required to be able to add rows to the table.
        """
        self.cursor.execute("ALTER TABLE plo_evaluations DROP CONSTRAINT evaluations_pkey;")
        self.conn.commit()
        print("✅ Primary key removed from PLO evaluations table.")

    def recreate_indices_on_evaluations(self):
        """
        Function to recreate indices for plo_evaluations table.
        Required once rows have been added to the table.
        """
        self.cursor.execute("ALTER TABLE plo_evaluations ADD PRIMARY KEY (board_id, hand_id);")
        self.conn.commit()
        print("✅ Primary key added to PLO evaluations table.")

    def add_indices_on_evaluations(self):
        """
        Function to recreate indices for plo_evaluations table.
        Required once rows have been added to the table.
        """
        self.cursor.execute("CREATE INDEX idx_board_mask ON plo_evaluations_bm(board_mask);")
        self.cursor.execute("CREATE INDEX idx_hand_mask  ON plo_evaluations_bm(hand_mask);")
        self.conn.commit()
        print("✅ Indices added to PLO evaluations table.")


    # Reset tables
    def clear_table(self, table_name):
        if table_name not in {"plo_boards", "plo_hands", "plo_evaluations", "plo_evaluations_bm"}:
            raise ValueError("Invalid table name")

        self.cursor.execute(f"DELETE FROM {table_name};")
        print(f"🧹 Cleared table: {table_name}")

    def truncate_table(self, table_name):
        if table_name not in {"plo_boards", "plo_hands", "plo_evaluations", "plo_evaluations_bm"}:
            raise ValueError("Invalid table name")

        self.cursor.execute(f"TRUNCATE TABLE {table_name} RESTART IDENTITY CASCADE;")
        print(f"🧹 Truncated table: {table_name}")

    def drop_table(self, table_name):
        """
        Drops table if it exists.
        """
        if table_name not in {"plo_boards", "plo_hands", "plo_evaluations", "plo_evaluations_bm"}:
            raise ValueError("Invalid table name")

        self.cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
        print(f"🗑️ Dropped table: {table_name}")

    # Methods to populate tables
    def insert_board(self, board_str):
        self.cursor.execute("INSERT INTO plo_boards (board_str) VALUES (%s);", (board_str,))
        print("📥 Inserted board data")
        return

    def insert_hand(self, hand_str):
        self.cursor.execute("INSERT INTO plo_hands (hand_str) VALUES (%s);", (hand_str,))
        print("📥 Inserted hand data")
        return

    def insert_evaluation(self, board_id, hand_id, high_value, low_value):
        self.cursor.execute("""
            INSERT INTO plo_evaluations (board_id, hand_id, high_value, low_value)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (board_id, hand_id) DO NOTHING;
        """, (board_id, hand_id, high_value, low_value))
        print("📥 Inserted evaluation data")


    def bulk_insert_boards(self, data, suit_pattern):
        # data: list of tuples [(board_str,), ...]
        # board_pattern: string or None
        query = """
            INSERT INTO plo_boards (board_str, suit_pattern)
            VALUES %s
            ON CONFLICT (board_str) DO NOTHING
            RETURNING board_id, board_str;
        """
        print(type(data), type(data[0]), data[0])
        # Add board_pattern (should be suit_pattern) to each tuple in data
        data_with_pattern = [(board, suit_pattern) for board in data]

        execute_values(self.cursor, query, data_with_pattern)
    
        all_board_strs = data
        # all_board_strs = [item[0] for item in data]
        self.cursor.execute(
            "SELECT board_id, board_str FROM plo_boards WHERE board_str = ANY(%s);",
            (all_board_strs,)
        )

        rows = self.cursor.fetchall()

        board_id_map = {board_str: board_id for board_id, board_str in rows}

        print(f"✅ Inserted {len(board_id_map)} new boards")
        return board_id_map    


    def bulk_insert_flops(self, flop_data):
        """
        Bulk insert flop data into the plo_boards table.
        flop_data: list of dictionaries with keys: card1_str, card2_str, card3_str, suit_pattern
        """
        query = """
            INSERT INTO plo_boards (card1_str, card2_str, card3_str, suit_pattern)
            VALUES %s
            ON CONFLICT (card1_str, card2_str, card3_str) DO NOTHING
            RETURNING board_id, card1_str, card2_str, card3_str
        """

        values = [
            (f["card1_str"], f["card2_str"], f["card3_str"], f.get("suit_pattern"))
            for f in flop_data
        ]

        rows = execute_values(self.cursor, query, values, fetch=True)
        # print(len(values))

        # rows = self.cursor.fetchall()
        self.conn.commit()
        # print(len(rows))

        # build a flop string to use as dict key
        # board_id_map = {
        #     f"{c1}{c2}{c3}": board_id
        #     for board_id, c1, c2, c3 in rows
        # }
        board_id_map = {
                f"{r[1]}{r[2]}{r[3]}": r[0] for r in rows
            }

        print(f"✅ Inserted {len(board_id_map)} new boards")
        return board_id_map


    def bulk_insert_evaluations(self, data, chunk_size=5000000):
        # data: list of tuples [(board_id, hand_id, hand_value, rank_min, rank_max, rank_avg, rank_dense), ...]

        if not data:
            return

        total = len(data)
        for i in range(0, total, chunk_size):
            chunk = data[i:i+chunk_size]
            processed_data = [
                ["" if v is None else v for v in row]
                for row in chunk
            ]
            buffer = io.StringIO()
            writer = csv.writer(buffer, delimiter='\t', lineterminator='\n')
            writer.writerows(processed_data)
            buffer.seek(0)

            self.cursor.copy_from(
                buffer,
                'plo_evaluations_bm',
                columns=('board_mask', 'hand_mask', 'high_value', 'low_value'),
                sep='\t'
            )
            print(f"✅ COPY inserted {len(chunk)} plo_evaluations_bm (rows {i+1}-{min(i+chunk_size, total)})")
        return


    # Query / Utility Methods
    def select_hands(self):
        self.cursor.execute("SELECT * FROM hands LIMIT 10;")
        rows = self.cursor.fetchall()
        print("📤 Retrieved data:")
        for row in rows:
            print(row)


    def select_boards(self):
        self.cursor.execute("SELECT * FROM plo_boards LIMIT 10;")
        rows = self.cursor.fetchall()
        print("📤 Retrieved data:")
        for row in rows:
            print(row)


    def get_hand_ids(self):    
        # After insertion, fetch all hand_strs to get their IDs
        self.cursor.execute(
            "SELECT hand_id, hand_str FROM hands;",
        )
    
        rows = self.cursor.fetchall()
    
        # Create mapping from hand string to hand_id
        hand_id_map = {hand_str: hand_id for hand_id, hand_str in rows}

        print(f"✅ Returned {len(hand_id_map)} hands")
        return hand_id_map
    

    def get_board_ids(self, suit_pattern=None):    
        # After insertion, fetch all board_strs to get their IDs
        # print(suit_pattern)
        # print(str(suit_pattern))
        
        if suit_pattern is not None:
            # print("Suit pattern")
            self.cursor.execute(
                "SELECT board_id, card1_str, card2_str, card3_str FROM plo_boards WHERE suit_pattern = %s;",
                (suit_pattern,)
            )
        else:
            # print("No suit pattern")
            self.cursor.execute(
                "SELECT board_id, card1_str, card2_str, card3_str FROM plo_boards;"
            )
    
        rows = self.cursor.fetchall()
        print(len(rows))
        # print(rows)
        # Create mapping from board string to board_id
        # board_id_map = {board_str: board_id for board_id in rows}

        board_id_map = {
                f"{r[1]}{r[2]}{r[3]}": r[0] for r in rows
            }

        print(f"✅ Returned {len(board_id_map)} boards")
        return board_id_map

    def get_evaluations(self, hand_id: int, board_ids: list[int]):
        """
        Return all evaluations for a given hand_id restricted to a list of board_ids.
        """
        if not board_ids:
            return []

        placeholders = ",".join(["%s"] * len(board_ids))
        query = f"""
            SELECT board_id, hand_id, hand_value, rank_dense
            FROM plo_evaluations
            WHERE hand_id = %s AND board_id IN ({placeholders});
        """
        self.cursor.execute(query, [hand_id] + board_ids)
        return self.cursor.fetchall()

    def get_evaluations_for_hands_and_boards_bm(self, hand_masks, board_masks):
        """
        Fetch evaluations for a given list of hand_ids and board_ids.
        
        Returns:
            dict mapping (hand_id, board_id) -> (high_hand_value, rank_dense)
        """
        # Query using hand_mask and board_mask directly
        # Returns dict: {(hand_mask, board_mask): (high_value, low_value)}
        if not hand_masks or not board_masks:
            return {}

        # Convert lists to tuples for SQL IN clause
        hand_masks_tuple = tuple(hand_masks)
        board_masks_tuple = tuple(board_masks)

        # Construct SQL
        query = f"""
            SELECT hand_mask, board_mask, high_value, low_value
            FROM plo_evaluations_bm
            WHERE hand_mask IN %s AND board_mask IN %s;
        """

        self.cursor.execute(query, (hand_masks_tuple, board_masks_tuple))
        rows = self.cursor.fetchall()

        eval_dict = {(h_id, b_id): (value, rank) for h_id, b_id, value, rank in rows}

        print(f"✅ Loaded {len(eval_dict)} evaluations for {len(hand_masks)} hands x {len(board_masks)} boards")
        return eval_dict


    def get_evaluations_for_hands_and_boards(self, hand_ids, board_ids):
        """
        Fetch evaluations for a given list of hand_ids and board_ids.
        
        Returns:
            dict mapping (hand_id, board_id) -> (high_hand_value, rank_dense)
        """
        if not hand_ids or not board_ids:
            return {}

        # Convert lists to tuples for SQL IN clause
        hand_ids_tuple = tuple(hand_ids)
        board_ids_tuple = tuple(board_ids)

        # Construct SQL
        query = f"""
            SELECT hand_id, board_id, hand_value, rank_dense
            FROM plo_evaluations
            WHERE hand_id IN %s AND board_id IN %s;
        """

        self.cursor.execute(query, (hand_ids_tuple, board_ids_tuple))
        rows = self.cursor.fetchall()

        eval_dict = {(h_id, b_id): (value, rank) for h_id, b_id, value, rank in rows}

        print(f"✅ Loaded {len(eval_dict)} evaluations for {len(hand_ids)} hands x {len(board_ids)} boards")
        return eval_dict

    # def get_evaluations(self):
    #     self.cursor.execute(
    #         "SELECT * FROM plo_evaluations;"
    #     )
    
    #     rows = self.cursor.fetchall()
    
    #     # # Create mapping from board string to board_id
    #     # board_id_map = {board_str: board_id for board_id, board_str in rows}

    #     print(f"✅ Returned {len(rows)} evaluations")
    #     return rows

    def get_evaluations_for_hand(self, hand_id):
        self.cursor.execute(
            "SELECT * FROM plo_evaluations WHERE hand_id = %s;",
            (hand_id,)
        )
    
        rows = self.cursor.fetchall()
    
        # # Create mapping from board string to board_id
        # board_id_map = {board_str: board_id for board_id, board_str in rows}

        print(f"✅ Returned {len(rows)} evaluations")
        return rows


    def get_evaluations_count_for_hand(self, hand_id):
        # self.cursor.execute(
        #     "SELECT * FROM evaluations WHERE ;"
        # )
        self.cursor.execute("SELECT COUNT(*) FROM plo_evaluations WHERE hand_id = %s;",
                            (hand_id,))
    
        (count,) = self.cursor.fetchone()
        # rows = self.cursor.fetchall()
    
        # # Create mapping from board string to board_id
        # board_id_map = {board_str: board_id for board_id, board_str in rows}

        print(f"✅ Table has {count:,} rows for hand_id={hand_id}.")
        return count

    def get_evaluations_for_suitedness(self, hand_str):
        query = """
            SELECT e.*, b.suit_pattern
            FROM plo_evaluations e
            JOIN hands h1 ON e.hand_id = h1.hand_id
            JOIN plo_boards b ON e.board_id = b.board_id
            WHERE h1.suitedness = %s;
        """        

        self.cursor.execute(query, (hand_str,))
        rows = self.cursor.fetchall()

        print(f"✅ Returned {len(rows)} evaluations for suitedness of '{hand_str}'")
        return rows


    def get_evaluations_count(self):
        self.cursor.execute("SELECT COUNT(*) FROM plo_evaluations;")
        (count,) = self.cursor.fetchone()
        print(f"✅ Table has {count:,} rows")
        return count

    def get_sample_evaluations(self, batch_size: int):
        self.cursor.execute(f"SELECT * FROM plo_evaluations_bm ORDER BY RANDOM() LIMIT {batch_size};")
        rows = self.cursor.fetchall()

        print(f"✅ Returned {len(rows)} evaluations")
        return rows


    def get_boards_with_card(self, card_str, suit_pattern=None):
        if suit_pattern is not None:
            self.cursor.execute(
                """
                SELECT board_id, card1_str, card2_str, card3_str 
                FROM plo_boards
                WHERE suit_pattern = %s
                AND (%s IN (card1_str, card2_str, card3_str));
                """,
                (suit_pattern, card_str)
            )
        else:
            self.cursor.execute(
                """
                SELECT board_id, card1_str, card2_str, card3_str 
                FROM plo_boards
                WHERE %s IN (card1_str, card2_str, card3_str);
                """,
                (card_str,)
            )

        rows = self.cursor.fetchall()
        print(f"✅ Found {len(rows)} boards containing {card_str}")
        return rows
    
    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("🔌 Connection closed")


# Helper functions. (Not in class)
def get_suitedness(hand_str:str) -> str:
    """
    Helper function to convert a two-card hand string to a simple suitedness classification.

    Args:
        hand_str: A string representing the two-card hand (e.g., 'AhKh', '7h7d').
                  Assumes single-character ranks (2-9, T, J, Q, K, A).

    Returns:
        A string representing the classification (e.g., 'AKs', 'AKo', '77').
    """
    # print(hand_str, type(hand_str))

    hand_str_suitedness = hand_str[0] + hand_str[2]
    if (hand_str[0]!=hand_str[2]):
        if (hand_str[1]==hand_str[3]):
            hand_str_suitedness = hand_str_suitedness + "s"
        else:
            hand_str_suitedness = hand_str_suitedness + "o"

    return hand_str_suitedness

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