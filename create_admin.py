"""
create_admin.py
Run this ONCE to create the admin account for DataSense AI.
This is not a public signup route — it's a setup script since this
is an internal, admin-only analytics tool.

Usage:
    python create_admin.py
"""

import bcrypt
from db import get_connection

# ---- Set your admin credentials here ----
USERNAME = "admin"
PASSWORD = "admin123"   # change this to something only you know
ROLE = "admin"
# ------------------------------------------


def create_admin():
    conn = get_connection()
    cursor = conn.cursor()

    # Prevent creating a duplicate admin if one already exists
    cursor.execute("SELECT user_id FROM users WHERE username = %s", (USERNAME,))
    existing = cursor.fetchone()

    if existing:
        print(f"User '{USERNAME}' already exists. No changes made.")
        cursor.close()
        conn.close()
        return

    hashed = bcrypt.hashpw(PASSWORD.encode('utf-8'), bcrypt.gensalt())

    cursor.execute(
        "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)",
        (USERNAME, hashed.decode('utf-8'), ROLE)
    )
    conn.commit()
    cursor.close()
    conn.close()

    print("Admin account created successfully.")
    print(f"Username: {USERNAME}")
    print(f"Password: {PASSWORD}")
    print("You can now log in at /login")


if __name__ == '__main__':
    create_admin()