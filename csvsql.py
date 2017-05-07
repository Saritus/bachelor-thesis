def convert_csv_to_sql(csvfile, sqlfile):
    # Open sql connection
    import sqlite3
    conn = sqlite3.connect(sqlfile)
    c = conn.cursor()

    return


def main():
    convert_csv_to_sql("nwt-data/Gebaeude_Dresden.csv", "nwt-data/Gebaeude_Dresden.db")
    return


if __name__ == "__main__":
    main()
