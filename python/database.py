import os
import psycopg2
from pandas import DataFrame


def drop_table():
    query = """
        DROP TABLE predictions
        """
    cursor.execute(query)
    conn.commit()


def create_table():
    query = """
        CREATE TABLE predictions (
          Id SERIAL PRIMARY KEY,
          Label SMALLINT,
          IsCorrect BOOLEAN,
          Image BYTEA
        );
        """
    cursor.execute(query)
    conn.commit()


def insert_table(_Label: int, _IsCorrect: bool, _Image: bytes):
    query = """INSERT INTO predictions (Label, IsCorrect, Image) VALUES (%s, %s, %s)"""
    cursor.execute(query, (_Label, _IsCorrect, _Image))
    conn.commit()


def get_data():
    labels, correct, imgs = [], [], []
    query = """SELECT * FROM predictions ORDER BY random() limit 100"""
    cursor.execute(query)
    rows = cursor.fetchall()
    for row in rows:
        print(row)
        labels.append(row[1])
        correct.append(row[2])
        imgs.append(row[3])
    return DataFrame({"labels": labels, "correct": correct, "imgs": imgs})


DATABASE_URL = os.environ['DATABASE_URL']

conn = psycopg2.connect(DATABASE_URL, sslmode='require')
cursor = conn.cursor()

# drop_table()
# create_table()
