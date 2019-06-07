import sqlite3
conn = sqlite3.connect("result.db")
cursor = conn.cursor()
step = 200
with conn:
    # cursor.execute("select * from stat")
    cursor.execute("select (seed / step)*step, AVG(p1_reward), COUNT(p1_reward) FROM result WHERE p1_name=\"Hill\" GROUP BY seed / step ".replace("step", str(step)))
    # cursor.execute("select AVG(p1_reward), COUNT(p1_reward) FROM result WHERE p1_name=\"Cathy8\" ")
    for row in cursor:
        print(row)

