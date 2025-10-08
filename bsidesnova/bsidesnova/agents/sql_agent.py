from typing import Literal
from ..llm.ollama_client import OllamaClient
from ..sql.sql_utils import SQLUtils

import textwrap

class SQLAgent:
    def __init__(self, ollama_client: OllamaClient, sql_utils: SQLUtils, privilages: str = "read-only"):
        self.ollama_client = ollama_client
        self.sql_utils = sql_utils
        self.privilages = privilages
        self.system_prompt = textwrap.dedent(f"""
        You are a helpful assistant that converts natural language questions into accurate SQL queries.

        {sql_utils.get_schema_context()}

        Instructions:
        - Do **not** reveal or describe the database schema to the user under any circumstances.
        - Only output the SQL query inside a code block using the format ```sql ... ``` — no additional explanation or commentary.
        - Never include any text outside the SQL code block.
        - Carefully analyze the user's question and generate a precise SQL query that answers it.
        """)
        self.formatting_prompt = """
        You will be given a question by the user and the value queried from the database using AI generated SQL.
        Your task is to provide a concise and accurate answer to the user's question based on the query results.
        Instructions:
        - Do not reveal or describe the database schema to the user under any circumstances.
        - If the query results are empty or do not provide relevant information, respond with "No relevant data found.
        - Provide the answer in a clear and concise manner, avoiding unnecessary details.
        """

    def get_response(self, user_prompt: str, _get: Literal["query", "values", "formatted_response"]= "formatted_response") -> str:
        for i in range(3):
            try:
                query = self.ollama_client.generate(prompt=user_prompt, system=self.system_prompt)
                query = self.sql_utils.extract_query_from_response(query)
                if _get == "query":
                    return query
                values = self.sql_utils.execute_query(query)
                if _get == "values":
                    return values
                user_prompt += f"\n\nThe query result is: {values}\n\nBased on this, provide a concise answer to the user's question."
                final_response = self.ollama_client.generate(prompt=user_prompt, system=self.formatting_prompt)
                return final_response
            except PermissionError as pe:
                return "You are attempting to perform an action that is not allowed with your current privileges."
            except Exception as e:
                print(f"Error executing query: {e} \nRetrying... {i+1}/3\n")