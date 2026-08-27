"""
db_connector.py

Lets a user pull data directly from an external database instead of
uploading a file -- the way real analytics platforms usually source data
(a live connection + query, rather than a manual CSV export).

Scope/safety notes:
- Only SELECT statements are allowed. This is a read-only data-import
  feature, not a general SQL console, so INSERT/UPDATE/DELETE/DROP/ALTER
  etc. are rejected before ever reaching the database.
- Credentials are used only for the single connection attempt and are
  never written to disk or to the application's own database -- only the
  data that comes back from the query is persisted (as a regular dataset,
  identical to an uploaded file).
- A row cap is enforced so an accidental "SELECT * FROM huge_table" can't
  exhaust memory or disk.
"""

import re
import mysql.connector
import pandas as pd

MAX_IMPORT_ROWS = 100_000
QUERY_TIMEOUT_SECONDS = 15

# Only allow a single, read-only SELECT/SHOW/DESCRIBE statement.
_DISALLOWED_KEYWORDS = re.compile(
    r'\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|GRANT|REVOKE|EXEC|CALL)\b',
    re.IGNORECASE,
)


class DBConnectionError(Exception):
    """Raised for connection failures or invalid/unsafe queries."""
    pass


def _validate_query(sql_query: str) -> str:
    query = sql_query.strip().rstrip(';')

    if not query:
        raise DBConnectionError("Query cannot be empty.")

    if ';' in query:
        raise DBConnectionError("Only a single statement is allowed (remove any ';' inside the query).")

    if _DISALLOWED_KEYWORDS.search(query):
        raise DBConnectionError(
            "Only read-only queries are allowed here (SELECT/SHOW/DESCRIBE). "
            "This import tool is for pulling data, not modifying it."
        )

    if not re.match(r'^\s*(SELECT|SHOW|DESCRIBE|DESC)\b', query, re.IGNORECASE):
        raise DBConnectionError("Query must start with SELECT, SHOW, or DESCRIBE.")

    return query


def _connect(host: str, port: int, username: str, password: str, database: str):
    try:
        return mysql.connector.connect(
            host=host,
            port=int(port),
            user=username,
            password=password,
            database=database,
            connection_timeout=QUERY_TIMEOUT_SECONDS,
        )
    except mysql.connector.Error as e:
        raise DBConnectionError(f"Could not connect to database: {e}")


def test_connection(host: str, port: int, username: str, password: str, database: str) -> bool:
    """Verifies the credentials work. Raises DBConnectionError on failure."""
    conn = _connect(host, port, username, password, database)
    conn.close()
    return True


def list_tables(host: str, port: int, username: str, password: str, database: str) -> list:
    """Returns the list of table names visible to this user in the target database."""
    conn = _connect(host, port, username, password, database)
    try:
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return tables
    finally:
        conn.close()


def run_query(host: str, port: int, username: str, password: str, database: str,
              sql_query: str) -> pd.DataFrame:
    """
    Runs a validated, read-only query against the external database and
    returns the result as a pandas DataFrame -- from this point on, the
    data is indistinguishable from an uploaded CSV to the rest of the app.
    """
    query = _validate_query(sql_query)
    conn = _connect(host, port, username, password, database)
    try:
        df = pd.read_sql(query, conn)
    except Exception as e:
        raise DBConnectionError(f"Query failed: {e}")
    finally:
        conn.close()

    if len(df) > MAX_IMPORT_ROWS:
        df = df.head(MAX_IMPORT_ROWS)

    return df
