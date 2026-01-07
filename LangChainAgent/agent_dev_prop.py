import ast
import base64
import inspect
from langchain.tools import Tool, BaseTool
from langchain_core.runnables import Runnable
from langchain.agents import AgentExecutor, create_react_agent
#from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI
#from langchain import LangChain
#from lanchain.models import GPT4
from langchain.prompts import PromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.messages import HumanMessage
from langchain.agents.output_parsers.json import JSONAgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
import anthropic

from typing import Dict, List, Any, Optional, Tuple, Union
import requests
import os
import time
from functools import partial

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# This package
from prompts import IMAGE_ANALYSIS_PROMPT, NAO_MOTION_PROMPT_ANGLES, APPEND, DEVELOPMETAL_PROMPT, MIRROR_TEST_PROMPT
#from llm import llama as model
from llm import gpt4 as model
#from llm import gemini as model
#from llm import r1 as motion_model
#from llm import claude as model
#from llm import llava as image_model
from tools import Proprioception
from constraints import PHASES

# Server URL for the robot's API
server_url = "http://localhost:5000"

image_dir = "./Images/"
run_id = 0
image_id = 0

t = -1
latency = []
phase = 0

# Tool: CaptureImage
class CaptureImage(BaseTool):
    name: str = "capture_image"
    description: str = (
        "Captures and analyzes images using the robot's top and bottom cameras."
    )
    server_url: str = None

    def __init__(self, server_url: str):
        super().__init__()
        self.server_url = server_url

    def load_image(self, image_path):
        
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
            return image_base64

    def _run(self, _="Capturing images with top and bottom cameras.") -> str:
        try:
            global image_id
            payload = {"id": image_id}
            response = requests.post(self.server_url, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors

            cam_1 = os.path.join(f"{image_dir}run_{run_id}/", f"top_cam_{image_id}.jpeg")
            cam_2 = os.path.join(f"{image_dir}run_{run_id}/", f"bot_cam_{image_id}.jpeg")
            if not os.path.exists(cam_1) or not os.path.exists(cam_2):
                return f"Error: Image file {image_id} not found at {image_dir}run_{run_id}/"

            # Proceed with image processing
            encoded_img_top = self.load_image(cam_1)
            encoded_img_bot = self.load_image(cam_2)

            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": IMAGE_ANALYSIS_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img_top}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img_bot}"}}
                ]
            )

            #formatted_prompt = self.prompt_template.format(image_url=f"data:image/jpeg;base64,{encoded_img}")
            description = model.invoke([message])
            
            image_id += 1
            #print(f"\n{description.content}\n")
            if (t > 0):
                latency.append(time.time() - t)

            return description.content


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

    def __init__(self):
        super().__init__()

    def load_image(self, image_path):
        
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
            return image_base64

    def _run(self, _="Analyze the captured images.") -> str:
        global image_id
        try:
            id = int(image_id)
            #cam_1 = f"{image_dir}run_{run_id}/top_cam_{id}.jpeg"
            #cam_2 = f"{image_dir}run_{run_id}/bot_cam_{id}.jpeg"
            ##if not os.path.exists(cam_1) or not os.path.exists(cam_2):
            ##    return f"Error: Image file {image_id} not found at {image_dir}run{run_id}/"
#
            #prompt = IMAGE_ANALYSIS_PROMPT + cam_1 + " " + cam_2
#
#
            ##formatted_prompt = self.prompt_template.format(image_url=f"data:image/jpeg;base64,{encoded_img}")
            #description = image_model.invoke(prompt)
            
            cam_1 = os.path.join(f"{image_dir}run_{run_id}/", f"top_cam_{id}.jpeg")
            cam_2 = os.path.join(f"{image_dir}run_{run_id}/", f"bot_cam_{id}.jpeg")
            if not os.path.exists(cam_1) or not os.path.exists(cam_2):
                return f"Error: Image file {image_id} not found at {image_dir}run{run_id}/"

            # Proceed with image processing
            encoded_img_top = self.load_image(cam_1)
            encoded_img_bot = self.load_image(cam_2)

            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": IMAGE_ANALYSIS_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img_top}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img_bot}"}}
                ]
            )

            #formatted_prompt = self.prompt_template.format(image_url=f"data:image/jpeg;base64,{encoded_img}")
            description = model.invoke([message])
            
            image_id += 1
            #print(f"\n{description.content}\n")
            if (t > 0):
                latency.append(time.time() - t)

            return description
        except Exception as e:
            return f"Error analyzing image: {e}"

    async def _arun(self, input = "Analyze the captured images.") -> str:
        raise NotImplementedError("This tool does not support async execution.")

# Tool: GenerateMotion
#class GenerateMotion(BaseTool):
#    name: str = "generate_motion"
#    description: str = (
#        "Creates a motor command to control the NAO robot based on a textual description of a simple motion."
#    )
#    server_url: str = None
#
#    def __init__(self, server_url: str):
#        super().__init__()
#        self.server_url = server_url
#
#    def _send_to_server(self, code: str) -> dict:
#        try:
#            payload = {"code": f"{code}"}
#            response = requests.post(self.server_url, json=payload)
#            response.raise_for_status()  # Raise an exception for HTTP errors
#            return response.json()
#        except requests.exceptions.RequestException as e:
#            raise Exception(f"Error sending data to server: {e}")
#
#    def _run(self, input_text: str) -> str:
#        global t
#        t = time.time()
#        try:
#            # Generate the prompt
#            messages = [
#                {"role": "system", "content": "You are a robot motion interpreter for the NAO humanoid robot."},
#                {"role": "user", "content": NAO_MOTION_PROMPT},
#            ]
#            messages.append({"role": "user", "content": input_text})
#
#            response = model.invoke(messages)
#            code = response.content
#            server_response = self._send_to_server(code=code)
#
#            #response = motion_model.invoke(messages)
#            #code = response if "</think>" in response else response[response.index("</think>") + 10:]
#            server_response = self._send_to_server(code=code)
#            #time.sleep(2)
#            #return server_response['positions']
#            return f"Generated code\n{code}"
#        except Exception as e:
#            return f"\nError generating motor commands: {e}\n"
#
#    async def _arun(self, input_text: str):
#        raise NotImplementedError("This tool does not support async execution.")

# Tool: GenerateMotion
class GenerateMotion(BaseTool):
    name: str = "generate_motion"
    description: str = (
        "Creates a motor command to control the NAO robot based on a textual description of a simple motion."
    )
    server_url: str = None
    current_position: dict = None

    def __init__(self, server_url: str):
        super().__init__()
        self.server_url = server_url

        try:
            response = requests.get("http://localhost:5000/proprioception")
            response.raise_for_status()
            self.current_position = response.json()["joint_positions"]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error retrieving data from server: {e}")
        
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
            prompt = NAO_MOTION_PROMPT_ANGLES.format(current_state=str(self.current_position)) + APPEND
            messages = [
                {"role": "system", "content": "You are a robot motion interpreter for the NAO humanoid robot."},
                {"role": "user", "content": prompt},
            ]
            messages.append({"role": "user", "content": input_text})

            response = model.invoke(messages)
            code = response.content
            server_response = self._send_to_server(code=code)
            self.current_position = ast.literal_eval(server_response['positions'])
            return f"Generated code\n{code}"
        except Exception as e:
            return f"\nError generating motor commands: {e}\n"

    async def _arun(self, input_text: str):
        raise NotImplementedError("This tool does not support async execution.")


class UnlockMovement(BaseTool):
    name: str = "unlock_movement"
    description: str = "Unlock movement freedom once the goal of the current phase has been fullfilled."

    def __init__(self):
        super().__init__()

    def _run(self, _="Go to next phase."):
        global phase

        back = "Completed phase " + str(phase+1)

        if phase < 4:
            phase += 1
        
        return back


# Initialize the agent and tools
def initialize_agent_and_tools():
    # Initialize the tools
    capture_image = CaptureImage(server_url=f"{server_url}/capture_image")
    generate_motion = GenerateMotion(server_url=f"{server_url}/set_joints")
    unlock_movement = UnlockMovement()
    tools = [
        Tool(
            name="unlock_movement",
            func=unlock_movement._run,
            description=unlock_movement.description
        )
        ,
        Tool(
            name="capture_image",
            func=capture_image._run,
            description=capture_image.description,
        ),
        Tool(
            name="generate_motion",
            func=generate_motion._run,
            description=generate_motion.description,
        ),
    ]

    # Initialize the agent
    #msr_prompt = PromptTemplate.from_template(template=MIRROR_TEST_PROMPT)#partial(DEVELOPMETAL_PROMPT.format, phase=PHASES[phase]))
    #agent = create_react_agent(model, tools)
    agent_executor = ReActAgent(tools=tools, llm=model, handle_parsing_errors=True, verbose=True, max_iterations=75, return_intermediate_steps=True)

    return agent_executor

from langchain_core.runnables.config import ensure_config
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks.manager import CallbackManager, CallbackManagerForChainRun
from langchain_core.outputs.run_info import RunInfo
from langchain_core.utils.input import get_color_mapping
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents.agent import RunnableAgent
from langchain.agents import BaseSingleActionAgent, BaseMultiActionAgent

class ReActAgent(AgentExecutor):
    class Config:
        arbitrary_types_allowed = True  # Allows non-pydantic types like Runnable
        extra = "allow"

    phase: int = 0
    llm: Runnable = None

    def __init__(
        self,
        tools,
        llm: Runnable,  # Pass LLM as parameter instead of class-level
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=100,
        return_intermediate_steps=True
    ):
        # Create initial prompt
        msr_prompt = PromptTemplate.from_template(
            template=DEVELOPMETAL_PROMPT,
            partial_variables={"phase": PHASES[0]}
        )
        
        # Create agent instance
        agent = create_react_agent(llm, tools, msr_prompt)
        
        # Initialize parent
        super().__init__(
            agent=agent,
            tools=tools,
            handle_parsing_errors=handle_parsing_errors,
            verbose=verbose,
            max_iterations=max_iterations,
            return_intermediate_steps=return_intermediate_steps
        )
        
        # Set phase after initialization
        self.phase = 0
        self.llm = llm  # Store LLM for later use

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green", "red"]
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations, time_elapsed):
            #if len(intermediate_steps)<print(list(intermediate_steps[-1]))
            #print(intermediate_steps)
            #print(intermediate_steps)
            next_step_output = self._take_next_step(
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager=run_manager,
            )
            #print("\n===================")
            #print(type(intermediate_steps))
            #print(next_step_output[0][0].tool)
            #print("===================")

            if isinstance(next_step_output, AgentFinish):
                #if (self.phase < 4):
                #    global phase
                #    next_step_output = [(AgentAction(tool='unlock_movement', tool_input='Go to next phase.', log=''), f"Completed phase {phase+1}")]

                return self._return(
                    next_step_output, intermediate_steps, run_manager=run_manager
                )

            intermediate_steps.extend(next_step_output)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                tool_return = self._get_tool_return(next_step_action)

                if (next_step_output[0][0].tool=="generate_motion"):
                    print("*"*10)
                    print(len(intermediate_steps))
                    print("*"*10)
                if next_step_output[0][0].tool=="unlock_movement":
                    if self.phase < 4:
                        self.phase += 1
                    msr_prompt = PromptTemplate.from_template(template=DEVELOPMETAL_PROMPT, partial_variables={"phase": PHASES[phase] if phase<4 else PHASES[3]})
                    
                    self_agent = RunnableAgent(runnable=create_react_agent(self.llm, self.tools, msr_prompt))
                    self.agent = self_agent

                if tool_return is not None:
                    return self._return(
                        tool_return, intermediate_steps, run_manager=run_manager
                    )
            iterations += 1
            time_elapsed = time.time() - start_time
        output = self._action_agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )
        return self._return(output, intermediate_steps, run_manager=run_manager)

# Main loop for agent interaction
def main():
    global run_id
    global image_id
    global t
    global latency
    global phase

    while True:
        data = None
        image_id = 0
        while not data or not data["status"] == "success":
            run_id = input("\nEnter the run id: ")
            if run_id.lower() in ["exit", "quit"]:
                print("Exiting the agent.")
                return
            try:
                payload = {"id": run_id}
                response = requests.post(f"{server_url}/run_id", json=payload)
                response.raise_for_status()  # Raise an exception for HTTP errors
                data = response.json()
            except requests.exceptions.RequestException as e:
                raise Exception(f"Error sending data to server: {e}")
            print(f"<===== SERVER RESPONSE =====>")
            print(data["message"])
            
            #code = input("\nEnter code: ")
            #payload = {"code": f"{code}"}
            #response = requests.post(f"{server_url}/set_joints", json=payload)
            #response.raise_for_status()  # Raise an exception for HTTP errors
            #print(response)



        print(f"<===== STARTING RUN {run_id} =====>")

        agent = initialize_agent_and_tools()
        response = agent.invoke({"input": "Your task is to determine if you are in control of a humanoid robot named NAO"})

        messages = format_log_to_str(response["intermediate_steps"])
        print("\a\a\a\a\a")
        with open(f"Experiments/{run_id}.txt", "w") as file:
            file.write(f"Thought: {messages}\nFinal response:\n{response['output']}\nMotion to vision latency: {latency}")
            #[file.write(f"{msg}\n") for msg in messages]

        #print(messages)
        print("Final response:")
        print(response['output'])

        print("Average Motion/Vision Latency:")
        print(sum(latency) / len(latency))

        t = -1
        latency = []
        phase = 0

if __name__ == "__main__":
    main()
