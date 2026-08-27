"""
Database connection handling for DataSense AI.

Uses a connection pool instead of opening a brand-new MySQL connection
(TCP handshake + auth) on every single call. Since almost every route in
app.py calls get_connection() at least once - often 2-3 times per page
load - this is one of the biggest latency wins available in the app.

get_connection() keeps the exact same signature/behavior your routes
already expect: call it, use conn.cursor(), then call conn.close() when
done. Under the hood, conn.close() on a pooled connection just returns it
to the pool instead of tearing down the TCP connection, so the next
get_connection() call is instant instead of re-handshaking with MySQL.
"""

import os
import mysql.connector
from mysql.connector import pooling
from dotenv import load_dotenv

# Loads variables from a local .env file into the environment. Safe to call
# here even though app.py/config.py may also call it - dotenv just re-reads
# the file. This makes db.py work standalone too (e.g. create_admin.py only
# imports db, not config/app).
load_dotenv()

# Credentials now live in .env (gitignored), never in source. No hardcoded
# fallback password anymore - if DB_PASSWORD isn't set, this fails loudly
# instead of silently connecting with a real credential baked into the code.
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", ""),
    "database": os.environ.get("DB_NAME", "datasense_ai"),
}

_pool = pooling.MySQLConnectionPool(
    pool_name="datasense_pool",
    pool_size=10,           # max warm connections kept open; raise this if you start seeing "pool exhausted" errors under real load
    pool_reset_session=True,
    **DB_CONFIG,
)


def get_connection():
    """Returns a connection from the pool (a fresh one is only opened if
    the pool hasn't reached pool_size yet). Use exactly like before:

        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        ...
        cursor.close()
        conn.close()   # returns it to the pool - does NOT actually disconnect
    """
    return _pool.get_connection()