import sqlite3


DB_NAME = "neurovision.db"


def create_database():

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions(

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            patient_id TEXT,

            patient_name TEXT,

            age INTEGER,

            gender TEXT,

            prediction TEXT,

            confidence REAL,

            scan_date TEXT

        )
    """)

    conn.commit()

    conn.close()


def save_prediction(
    patient_id,
    patient_name,
    age,
    gender,
    prediction,
    confidence,
    scan_date
):

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""

        INSERT INTO predictions(

            patient_id,
            patient_name,
            age,
            gender,
            prediction,
            confidence,
            scan_date

        )

        VALUES(?,?,?,?,?,?,?)

    """, (

        patient_id,
        patient_name,
        age,
        gender,
        prediction,
        confidence,
        scan_date

    ))

    conn.commit()

    conn.close()



def get_predictions():

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            patient_id,
            patient_name,
            age,
            gender,
            prediction,
            confidence,
            scan_date
        FROM predictions
        ORDER BY id DESC
    """)

    data = cursor.fetchall()

    conn.close()

    return data

def delete_prediction(patient_id):
    conn = sqlite3.connect("neurovision.db")
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM predictions WHERE patient_id=?",
        (patient_id,)
    )

    conn.commit()
    conn.close()
    
    