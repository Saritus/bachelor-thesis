def convert_csv_to_sql(csvfile, database):
    # Delete sql file if it already exists
    import os
    if os.path.exists(database):
        os.remove(database)

    # Open sql connection
    import sqlite3
    conn = sqlite3.connect(database)
    c = conn.cursor()

    # Open csv file
    import csv
    csvfile = open(csvfile)
    reader = csv.reader(csvfile, delimiter='\t')

    # Create table
    headers = reader.next()
    c.execute("CREATE TABLE sqltable(" + str(headers)[1:-1] + ")")

    # Create basic insert-query
    query = 'INSERT INTO sqltable({0}) VALUES ({1})'
    query = query.format(','.join(headers), ','.join('?' * len(headers)))

    # For every row in csv
    for row in reader:
        # Insert a row of data
        c.execute(query, row)

    # Save (commit) the changes
    conn.commit()

    # Close the connection
    conn.close()

    return


def convert_sql_to_csv(database, csvfile):
    # Delete csv file if it already exists
    import os
    if os.path.exists(csvfile):
        os.remove(csvfile)

    # Open sql connection
    import sqlite3
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Get column names
    c = conn.execute("SELECT * FROM sqltable")
    fieldnames = c.fetchone().keys()


def get_column(database, column):
    # Open sql connection
    import sqlite3
    conn = sqlite3.connect(database)
    c = conn.cursor()

    # Get column
    c.execute("SELECT {} FROM sqltable".format(column))  # SQL Injection
    result = c.fetchall()

    # Close the connection
    conn.close()

    return result


def main():
    DATABASE = "nwt-data/Gebaeude_Dresden.db"
    convert_csv_to_sql("nwt-data/Gebaeude_Dresden_shuffle.csv", DATABASE)
    columns = get_column(DATABASE, "Street, ZipCode")
    for street, zipcode in columns:
        print street, zipcode
    return


if __name__ == "__main__":
    main()
