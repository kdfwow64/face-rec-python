import logging
import mysql.connector
import csv
import datetime

# install mysql first
# sudo apt install mysql-server
# ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'detector';
# FLUSH PRIVILEGES;

class DB:
    def __init__(self, dbname, save_dir):
        self.save_dir = save_dir
        self.db = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="detector")
        self.control = self.db.cursor()
        self.dbname = dbname
        self.header = ["session", "name", "first_seen", "get_in_latitude", "get_in_longitude", "last_seen", "get_off_latitude", "get_off_longitude"]
        

    def check_persons_database(self):
        self.control.execute("SHOW DATABASES")
        for db in self.control:
            if db[0] == self.dbname:
                return True
        return False

    def check_table(self):
        self.control.execute("SHOW TABLES")
        for tbl in self.control:
            if tbl[0] == self.dbname:
                return True
        return False

    def connect(self):
        if not self.check_persons_database():
            self.control.execute("CREATE DATABASE " + self.dbname)
        self.persons = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="detector",
            database=self.dbname) 
        self.control = self.persons.cursor()
        if not self.check_table():
            self.control.execute("CREATE TABLE " + self.dbname + " (id INT AUTO_INCREMENT PRIMARY KEY, session INT, name VARCHAR(255), first_seen DATETIME, get_in_latitude DECIMAL, get_in_longitude DECIMAL, last_seen DATETIME, get_off_latitude DECIMAL, get_off_longitude DECIMAL)")

    def insert_person(self, person): 
        #person["embeddings"] = []
        # logging.info(person)
        sql = "INSERT INTO " + self.dbname + " (session, name, first_seen, get_in_latitude, get_in_longitude, last_seen, get_off_latitude, get_off_longitude) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        val = (*[person[key] for key in self.header], )
        self.control.execute(sql, val)
        self.persons.commit()

    def update_person(self, person):
        sql = "UPDATE " + self.dbname + " SET get_off_latitude = %s, SET get_off_longitude = %s, SET last_seen = %s WHERE name = %s"
        val = (person["get_off_latitude"], person["get_off_longitude"], person["last_seen"], person["name"])
        self.control.execute(sql, val)
        self.persons.commit()

    def get_session(self):        
        sql = "SELECT session FROM " + self.dbname + " ORDER BY ID DESC LIMIT 1"
        self.control.execute(sql)
        session = self.control.fetchone()          
        self.session = session[0] + 1 if session is not None else 0   
        return self.session 

    def write_csv(self, dicts):
        dicts = [ [ d[k] for k in self.header ] for (_,d) in dicts.items() if not d["detecting"] ]
        with open(self.save_dir + '/persons-session-%s-date-%s.csv' % (self.session, datetime.date.today()), 'w') as output_file:
            writter = csv.writer(output_file, self.header)
            writter.writerow(self.header)
            writter.writerows(dicts)            

    def close(self):
        if (self.persons.is_connected()):
            self.control.close()
            self.persons.close()
            logging.info("MySQL connection is closed")
    


# db = DB("personal")
# db.check_persons_database()
# db.control.execute("CREATE DATABASE TEST")
# db.control.execute("SHOW DATABASES")
# db.connect()
# db.get_session()
# for x in db.control:
#     print(x)

