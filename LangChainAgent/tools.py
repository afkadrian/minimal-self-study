from langchain.tools import BaseTool

import requests
import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


from typing import Dict, List
import base64
from simple_colors import get_color_code
from prompts import IMAGE_ANALYSIS_PROMPT, NAO_MOTION_PROMPT, MIRROR_TEST_PROMPT
from llm import gpt4

from datetime import datetime


# Server URL for the robot's API
server_url = "http://localhost:5000"

class Proprioception(BaseTool):
    name: str = "proprioception"
    description: str = "Returns the current position of the robot's axis joints."
    server_url: str = None

    def __init__(self, server_url: str):
        super().__init__()
        self.server_url = server_url

    def _run(self, _="Requesting the current state of the robots axis joints."):
        try:
            response = requests.get(self.server_url)
            response.raise_for_status()
            return response.json()["joint_positions"]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error capturing image: {e}")
        
    async def _arun(self, input):
        raise NotImplementedError("This tool does not support async execution.")
