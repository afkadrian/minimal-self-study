import time
from langchain.tools import BaseTool

import requests
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from datetime import datetime
from typing import Dict, List
import base64
from simple_colors import get_color_code

# Package
from prompts import IMAGE_ANALYSIS_PROMPT, NAO_MOTION_PROMPT, MIRROR_TEST_PROMPT
from llm import gpt4
from state import GlobalState

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Server URL for the robot's API
server_url = "http://localhost:5000"
image_dir = "./Images/"

class Proprioception(BaseTool):
    name: str = "proprioception"
    description: str = "Returns the current position in degree of the robot's axis joints."
    server_url: str = None
    global_state: GlobalState = None

    def __init__(self, server_url: str, global_state: GlobalState):
        super().__init__()
        self.server_url = server_url
        self.global_state = global_state

    def _run(self, _="Requesting the current state of the robots axis joints."):
        try:
            response = requests.get(self.server_url)
            response.raise_for_status()
            return response.json()["joint_positions"]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error capturing image: {e}")
        
    async def _arun(self, input):
        raise NotImplementedError("This tool does not support async execution.")
    
# Tool: CaptureImage
class CaptureImage(BaseTool):
    name: str = "capture_image"
    description: str = (
        "Captures images using the robot's top and bottom cameras."
    )
    server_url: str = None
    global_state: GlobalState = None

    def __init__(self, server_url: str, global_state: GlobalState):
        super().__init__()
        self.server_url = server_url
        self.global_state = global_state

    def _run(self, _="Capturing images with top and bottom cameras.") -> str:
        try:
            payload = {"id": self.global_state.image_id}
            response = requests.post(self.server_url, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return "Images captured."
            if response.json()["status"] == "success":
                return response.json()["image_id"]
            else:
                return response.json()["message"]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error capturing image: {e}")

    async def _arun(self, input):
        raise NotImplementedError("This tool does not support async execution.")

# Tool: Image2Text
class Image2Text(BaseTool):
    name: str = "image_to_text"
    description: str = ("Analyzes the images captured by the top and the bottom cameras of NAO and generates a textual description.")
    llm: ChatOpenAI = None
    prompt: str = None
    global_state: GlobalState = None

    def __init__(self, llm: ChatOpenAI, prompt: str, global_state: GlobalState):
        super().__init__()
        self.llm = llm
        self.prompt = prompt
        self.global_state = global_state

    def load_image(self, image_path):
        
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
            return image_base64

    def _run(self, _="Analyze the captured images.") -> str:
        try:
            id = self.global_state.image_id
            cam_1 = os.path.join(f"{image_dir}run_{self.global_state.run_id}/", f"top_cam_{id}.jpeg")
            cam_2 = os.path.join(f"{image_dir}run_{self.global_state.run_id}/", f"bot_cam_{id}.jpeg")
            if not os.path.exists(cam_1) or not os.path.exists(cam_2):
                return f"Error: Image file {id} not found at {image_dir}run_{self.global_state.run_id}/"
            #image_id += 1
            # Proceed with image processing
            encoded_img_top = self.load_image(cam_1)
            encoded_img_bot = self.load_image(cam_2)

            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": self.prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img_top}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img_bot}"}}
                ]
            )

            #formatted_prompt = self.prompt_template.format(image_url=f"data:image/jpeg;base64,{encoded_img}")
            description = self.llm.invoke([message])
            self.global_state.image_id += 1
            #print(f"\n{description.content}\n")
            if (self.global_state.t > 0):
                self.global_state.latency.append(time.time() - t)

            return description.content
        except Exception as e:
            return f"Error analyzing image: {e}"

    async def _arun(self, input = "Analyze the captured images.") -> str:
        raise NotImplementedError("This tool does not support async execution.")

# Tool: GenerateMotion
class GenerateMotion(BaseTool):
    name: str = "generate_motion"
    description: str = (
        "Creates a motor command to control the NAO robot based on a textual description of a simple motion."
    )
    llm: ChatOpenAI = None
    prompt: str = None
    prompt_template: List[Dict] = None
    server_url: str = None
    global_state: GlobalState = None


    def __init__(self, server_url: str, llm: ChatOpenAI, global_state: GlobalState):
        super().__init__()
        self.server_url = server_url
        self.llm = llm
        self.prompt = NAO_MOTION_PROMPT
        self.prompt_template = self._load_prompt_template()
        self.global_state = global_state

    def _load_prompt_template(self) -> List[Dict]:
        try:
            return [
                {"role": "system", "content": "You are a robot motion interpreter for the NAO humanoid robot."},
                {"role": "user", "content": self.prompt},
            ]
        except Exception as e:
            raise Exception(f"Error loading prompt template: {e}")


    def _send_to_server(self, code: str) -> dict:
        try:
            payload = {"code": f"{code}"}
            response = requests.post(self.server_url, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error sending data to server: {e}")


    def _run(self, input_text: str) -> str:
        global t
        t = time.time()
        try:
            # Generate the prompt
            messages = self.prompt_template.copy()
            messages.append({"role": "user", "content": input_text})

            # Generate motor commands using the prompt template
            #formatted_prompt = self.prompt_template.format(input_text=input_text)
            response = self.llm.invoke(messages)
            code = response.content
            server_response = self._send_to_server(code=code)
            #time.sleep(2)
            #return server_response['positions']
            return f"Generated code\n{code}"
        except Exception as e:
            return f"\nError generating motor commands: {e}\n"

    async def _arun(self, input_text: str):
        raise NotImplementedError("This tool does not support async execution.")
