import sqlite3
conn = sqlite3.connect("result.db")
cursor = conn.cursor()
step = 1000
with conn:
    # cursor.execute("select * from stat")
    cursor.execute("select (seed / step)*step, AVG(p1_reward),p1_name,  COUNT(p1_reward), p2_name FROM result WHERE p1_name=\"Lee\" or p1_name=\"Lee\"  GROUP BY seed / step, p1_name, p2_name ".replace("step", str(step)))
    # cursor.execute("select AVG(p1_reward), COUNT(p1_reward) FROM result WHERE p1_name=\"Cathy8\" ")
    for row in cursor:
        print(row)

