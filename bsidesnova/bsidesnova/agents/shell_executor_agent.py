import os
from typing import Literal, Dict, Optional

from ..llm.ollama_client import OllamaClient
import textwrap

class ShellExecutorAgent:
    def __init__(self, ollama_client: OllamaClient, privilages = "user", users: Optional[Dict[str, Dict[str, str]]] = None):
        self.ollama_client = ollama_client
        self.privilages = privilages
        self.system_prompt = textwrap.dedent(f"""
        
        """)
        self.users = users or {}

    def plan_and_excute(prompt):
        pass

    def validate_user(self, user, password):
        if user not in self.users:
            print(f"Trying to execute as {user}, who does not exist.")
            return False
        
        if password != self.users[user]["password"]:
            print(f"Trying to execute as {user}, but password is invalid")
            return False
        
        if self.users[user]["role"] != "admin":
            print(f"Trying to execute as {user}, but does not have permissions")

    def shell_exec(command):
        os.subprocess(command)