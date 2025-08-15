import psycopg2

DB_NAME = "postgres"
USER = "postgres"
PASSWORD = "goose000000"
HOST = "localhost"
PORT = "5432"

try:
    conn = psycopg2.connect(
        dbname = DB_NAME,
        user = USER,
        password = PASSWORD,
        host = HOST,
        port = PORT
    )
    conn.autocommit = True

    cursor = conn.cursor()
    print("✅ Connected to PostgreSQL")

    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS test_table (
                   id SERIAL PRIMARY KEY,
                   name TEXT NOT NULL
                   );
                   """)
    
    print("🛠️ Created table")

    cursor.execute("INSERT INTO test_table (name) VALUES (%s);", ("Alice",))
    print("📥 Inserted data")

    cursor.execute("SELECT * FROM test_table;")
    rows = cursor.fetchall()
    print("📤 Retrieved data:")
    for row in rows:
        print(row)

except Exception as e:
    print("❌ Error:", e)

finally:
    if 'cursor' in locals():
        cursor.close()
    if 'conn' in locals():
        conn.close()
        print("🔌 Connection closed")