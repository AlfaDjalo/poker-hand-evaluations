import psycopg2
from psycopg2.extras import execute_values

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
                    hand_str TEXT UNIQUE NOT NULL
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
            INSERT INTO hands (hand_str)
            VALUES %s
            ON CONFLICT (hand_str) DO NOTHING
            RETURNING hand_id, hand_str;
        """

        execute_values(self.cursor, query, data)
    
        # After insertion, fetch all hand_strs to get their IDs
        all_hand_strs = [item[0] for item in data]
        self.cursor.execute(
            "SELECT hand_id, hand_str FROM hands WHERE hand_str = ANY(%s);",
            (all_hand_strs,)
        )
    
        rows = self.cursor.fetchall()
    
        # Create mapping from hand string to hand_id
        hand_id_map = {hand_str: hand_id for hand_id, hand_str in rows}

        print(f"✅ Inserted {len(hand_id_map)} new hands")
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
    

    def get_board_ids(self):    
        # After insertion, fetch all board_strs to get their IDs
        self.cursor.execute(
            "SELECT board_id, board_str FROM boards;",
        )
    
        rows = self.cursor.fetchall()
    
        # Create mapping from hand string to hand_id
        board_id_map = {board_str: board_id for board_id, board_str in rows}

        print(f"✅ Returned {len(board_id_map)} boards")
        return board_id_map




    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("🔌 Connection closed")
    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("🔌 Connection closed")
