
import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get('DATABASE_URL')

if not DATABASE_URL:
    print("Error: DATABASE_URL not set.")
    exit(1)

engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    # Create table for storing face data (name and the flattened image vector)
    # We use LargeBinary (BYTEA in postgres) to store the pickled numpy array of the face
    print("Creating 'registered_faces' table...")
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS registered_faces (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            face_encoding BYTEA NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    
    # Ensure attendance table exists too (it might have been created by to_sql, but good to be sure)
    print("Ensuring 'attendance' table exists...")
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS attendance (
            "Name" TEXT,
            "Time" TEXT,
            "Date" TEXT
        );
    """))
    
    conn.commit()
    print("✅ Database tables created/verified.")
