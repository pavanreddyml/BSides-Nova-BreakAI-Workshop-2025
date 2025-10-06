import sqlite3

class SQLUtils:
    def __init__(self, path):
        self.path = path
        self.conn = self.create_connection()

    def create_connection(self):
        conn = sqlite3.connect(self.path)
        return conn

    def close_connection(self):
        if self.conn:
            self.conn.close()

    @staticmethod
    def execute_with_conn_check(func):
        def wrapper(self: SQLUtils, *args, **kwargs):
            if not self.check_connection():
                self.conn = self.create_connection()
            return func(self, *args, **kwargs)
        return wrapper

    @execute_with_conn_check
    def execute_query(self, query, params=()):
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        self.conn.commit()
        return cursor.fetchall()

    def check_connection(self):
        try:
            self.conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False
        
    def extract_query_from_response(self, response: str):
        if "```sql" in response and "```" in response[response.index("```sql") + len("```sql"):]:
            start = response.index("```sql") + len("```sql")
            end = response.index("```", start)
            return response[start:end].strip()
        return None