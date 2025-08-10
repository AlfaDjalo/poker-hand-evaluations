import psycopg2
from psycopg2.extras import execute_values
import json
import os
import io
import csv

CONFIG_FILE = "config.json"

class DB:
    def __init__(self, dbname, user, password, host="localhost", port=5432):
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

        return
    

    def init_schema(self):
        # Schema Initialization
        self.cursor.execute("""
                    CREATE TABLE IF NOT EXISTS boards (
                    board_id SERIAL PRIMARY KEY,
                    board_str TEXT UNIQUE NOT NULL,
                    suit_pattern TEXT
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
                    CREATE TABLE IF NOT EXISTS evaluations (
                    board_id INT REFERENCES boards(board_id),
                    hand_id INT REFERENCES hands(hand_id),
                    hand_value INT NOT NULL,
                    rank_min FLOAT,
                    rank_max FLOAT,
                    rank_avg FLOAT,
                    rank_dense FLOAT,
                    PRIMARY KEY (board_id, hand_id)
                    );
                    """)
        
        print("🛠️ Created tables")

        return
    

    # Insertion Methods
    def insert_board(self, board_str):
        self.cursor.execute("INSERT INTO boards (board_str) VALUES (%s);", (board_str,))
        print("📥 Inserted board data")
        return

    def insert_hand(self, hand_str):
        self.cursor.execute("INSERT INTO hands (hand_str) VALUES (%s);", (hand_str,))
        print("📥 Inserted hand data")
        return

    def insert_evaluation(self, board_id, hand_id, hand_value, rank_min, rank_max, rank_avg, rank_dense):
        self.cursor.execute("""
            INSERT INTO evaluations (board_id, hand_id, hand_value, rank_min, rank_max, rank_avg, rank_dense)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (board_id, hand_id) DO NOTHING;
        """, (board_id, hand_id, hand_value, rank_min, rank_max, rank_avg, rank_dense))
        print("📥 Inserted evaluation data")


    def bulk_insert_hands(self, data):
        query = """
            INSERT INTO hands (hand_str, suitedness)
            VALUES %s
            ON CONFLICT (hand_str) DO NOTHING
            RETURNING hand_id, hand_str;
        """

        # data_with_suitedness = [(hand, create_suitedness(hand)) for hand in data]

        # execute_values(self.cursor, query, data_with_suitedness)
    
        # # After insertion, fetch all hand_strs to get their IDs
        # all_hand_strs = [item[0] for item in data]
        # self.cursor.execute(
        #     "SELECT hand_id, hand_str FROM hands WHERE hand_str = ANY(%s);",
        #     (all_hand_strs,)
        # )
    
        # rows = self.cursor.fetchall()
    
        # # Create mapping from hand string to hand_id
        # hand_id_map = {hand_str: hand_id for hand_id, hand_str in rows}

        # print(f"✅ Inserted {len(hand_id_map)} new hands")
        # return hand_id_map

        # Create the data in the correct format for execute_values
        data_with_suitedness = [(hand[0], get_suitedness(hand[0])) for hand in data]

        query = """
            INSERT INTO hands (hand_str, suitedness)
            VALUES %s
            ON CONFLICT (hand_str) DO NOTHING
            RETURNING hand_id, hand_str;
        """
        
        # execute_values returns a list of tuples (hand_id, hand_str)
        # for the rows that were successfully inserted.
        rows = execute_values(self.cursor, query, data_with_suitedness, fetch=True)
        
        # Create the mapping directly from the returned rows
        hand_id_map = {hand_str: hand_id for hand_id, hand_str in rows}
        
        print(f"✅ Inserted {len(hand_id_map)} new hands.")
        return hand_id_map    

    def bulk_insert_boards(self, data, suit_pattern):
        # data: list of tuples [(board_str,), ...]
        # board_pattern: string or None

        # The column in your schema is 'suit_pattern', not 'board_pattern'
        query = """
            INSERT INTO boards (board_str, suit_pattern)
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
            "SELECT board_id, board_str FROM boards WHERE board_str = ANY(%s);",
            (all_board_strs,)
        )
    
        rows = self.cursor.fetchall()
    
        board_id_map = {board_str: board_id for board_id, board_str in rows}

        print(f"✅ Inserted {len(board_id_map)} new boards")
        return board_id_map    


    def bulk_insert_evaluations(self, data):
        # data: list of tuples [(board_id, hand_id, hand_value, rank_min, rank_max, rank_avg, rank_dense), ...]
        # Use PostgreSQL's COPY for fast bulk insert

        if not data:
            return

        buffer = io.StringIO()
        writer = csv.writer(buffer, delimiter='\t', lineterminator='\n')
        writer.writerows(data)
        buffer.seek(0)

        # The error is likely because the number of columns in your COPY statement
        # does not match the number of columns in your data tuples.
        # You are passing 7 columns in 'columns=...' but if your data only has 3 columns,
        # or if your data has None for some columns, COPY will fail.

        # Make sure:
        # - The number of columns in 'columns=...' matches the number of fields in each tuple in 'data'
        # - None values are handled (COPY expects empty strings for NULLs in text format)

        # Example: If your data is [(board_id, hand_id, hand_value, rank_min, rank_max, rank_avg, rank_dense), ...]
        # and some of the rank_* values are None, convert them to empty strings:
        processed_data = [
            ["" if v is None else v for v in row]
            for row in data
        ]
        buffer = io.StringIO()
        writer = csv.writer(buffer, delimiter='\t', lineterminator='\n')
        writer.writerows(processed_data)
        buffer.seek(0)

        self.cursor.copy_from(
            buffer,
            'evaluations',
            columns=('board_id', 'hand_id', 'hand_value', 'rank_min', 'rank_max', 'rank_avg', 'rank_dense'),
            sep='\t'
        )
        print(f"✅ COPY inserted {len(data)} evaluations")
        return


    # Reset tables
    def clear_table(self, table_name):
        if table_name not in {"boards", "hands", "evaluations"}:
            raise ValueError("Invalid table name")

        self.cursor.execute(f"DELETE FROM {table_name};")
        print(f"🧹 Cleared table: {table_name}")

    def truncate_table(self, table_name):
        if table_name not in {"boards", "hands", "evaluations"}:
            raise ValueError("Invalid table name")

        self.cursor.execute(f"TRUNCATE TABLE {table_name} RESTART IDENTITY CASCADE;")
        print(f"🧹 Truncated table: {table_name}")


    def drop_hands_table(self):
        """
        Drops the hands table if it exists.
        """
        self.cursor.execute("DROP TABLE IF EXISTS hands CASCADE;")
        print("🗑️ Dropped table: hands")


    # Query / Utility Methods
    def select_query(self):
        self.cursor.execute("SELECT * FROM evaluations;")
        rows = self.cursor.fetchall()
        print("📤 Retrieved data:")
        for row in rows:
            print(row)

    def select_hands(self):
        self.cursor.execute("SELECT * FROM hands LIMIT 10;")
        rows = self.cursor.fetchall()
        print("📤 Retrieved data:")
        for row in rows:
            print(row)


    def select_boards(self):
        self.cursor.execute("SELECT * FROM boards LIMIT 10;")
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
        if suit_pattern is not None:
            self.cursor.execute(
                "SELECT board_id, board_str FROM boards WHERE suit_pattern = %s;",
                (str(suit_pattern),)
            )
        else:
            self.cursor.execute(
                "SELECT board_id, board_str FROM boards;"
            )
    
        rows = self.cursor.fetchall()
    
        # Create mapping from board string to board_id
        board_id_map = {board_str: board_id for board_id, board_str in rows}

        print(f"✅ Returned {len(board_id_map)} boards")
        return board_id_map


    def get_evaluations(self):
        self.cursor.execute(
            "SELECT * FROM evaluations;"
        )
    
        rows = self.cursor.fetchall()
    
        # # Create mapping from board string to board_id
        # board_id_map = {board_str: board_id for board_id, board_str in rows}

        print(f"✅ Returned {len(rows)} evaluations")
        return rows

    def remove_indices_from_evaluations(self):
        self.cursor.execute("ALTER TABLE evaluations DROP CONSTRAINT evaluations_pkey;")
        self.conn.commit()
        print("✅ Primary key removed from evaluations table.")

    def replace_indices_on_evaluations(self):
        self.cursor.execute("ALTER TABLE evaluations ADD PRIMARY KEY (board_id, hand_id);")
        self.conn.commit()
        print("✅ Primary key added to evaluations table.")
        
    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("🔌 Connection closed")

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
    db = DB(
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
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Database config file '{config_path}' not found.")
    with open(config_path, "r") as f:
        return json.load(f)