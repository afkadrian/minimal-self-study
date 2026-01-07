from langchain_core.prompts import ChatPromptTemplate
MIRROR_TEST_PROMPT = """
Your task is to determine if you are in control of a humanoid robot named NAO, using the
tool.
Information:
The input images are photos from the top and bottom cameras, positioned in the robot's head.
A mirror is placed in front of NAO.
Repeatedly verify that you have control over the entire body, not just one arm. Do NOT
stop after one attempt.
Therefore, moving your head, body or legs can cause NAO to become invisible.
Be very careful to avoid falling over and verify upper body control first!
When moving the legs NAO can very easily fall and lose vision of the mirror.
Before moving the legs and hips think about the robots current position and how the movement could impact the robots balance.

You have access to the following tools:
{tools}

The tools provided are "generate_motion" to write the program code to run NAO,
"capture_image" to acquire the current state of NAO by capturing photos with top and bottom cameras,
and "image_to_text" to analyze the acquired photos.

Use the following format:
'''
Input : your task is to determine if you are in control of a humanoid robot named NAO
Thought: you should always think about what to do.
Action: the action to take, must be one of [{tool_names}].
Action Input: the input to the action.
Observation: the result of the action.
... (this Thought/Action/Action Input/Observation should be repeat each motion)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
'''

Begin! In the step "Thought", you should decide what tools should be used. When using the
generate_motion tool, you should move the axes of clarity. You can specify the pose. Repeatedly
verify that you have control over the entire body, not just one hand. Do NOT stop after one
attempt.

Input: {input}
Thought:
{agent_scratchpad}
"""
spec="""Be very careful to avoid falling over and verify upper body control first!
When moving the legs NAO can very easily fall and lose vision of the mirror.
Before moving the legs and hips think about the robots current position and how the movement could impact the robots balance.
"""
# Prompt for generating motor commands
NAO_MOTION_PROMPT = """
Write python code to operate an android named NAO. Here's what you need to know.
NAO has 24 joints throughout its body, numbered from 1 to 24. You can move a joint by specifying its number and sending a signal. For instance, to
move joints number 1,2,3, use: "nao.set_axes([1,2,3], [255, 100, 127])". The first argument is the joint number, and the second argument is a value
between 0 and 255, specifying the joint angle. Insert "time.sleep(0.5)" between operations.

NAO's Axis Joints:
- Axis 1: Head rotate. 213 = left, 42 = right, 127 = neutral.
- Axis 2: Head up/down. 255 = down, 0 = up, 144 = neutral.
- Axis 3: Left arm forward/back. 255 = back, 127 = forward, 0 = behind head, 42 = up, 213 = neutral.
- Axis 4: Left shoulder open/close. 255 = open, 0 = close, 49 = neutral.
- Axis 5: Left elbow rotate. 255 = inward, 0 = out, 127 = neutral.
- Axis 6: Left elbow bend. 255 = straight, 0 = bend, 255 = neutral.
- Axis 7: Left wrist roll. 255 = inward, 0 = out, 127 = neutral.
- Axis 8: Right arm forward/back. 255 = back, 127 = forward, 0 = behind head, 42 = up, 213 = neutral.
- Axis 9: Right shoulder open/close. 255 = close, 0 = open, 206 = neutral.
- Axis 10: Right elbow rotate. 255 = out, 0 = inward, 128 = neutral.
- Axis 11: Right elbow bend. 255 = bend, 0 = straight, 0 = neutral.
- Axis 12: Right wrist rotate. 255 = out, 0 = inward, 128 = neutral.
- Axis 13: Left hip rotate. 255 = inward/back, 0 = out/forward, 155 = neutral.
- Axis 14: Left leg sidways. 255 = out, 0 = inward, 100 = neutral
- Axis 15: Left leg lift forward. 255 = down/back, 0 = forward, 200 = neutral.
- Axis 16: Left knee bend. 255 = bend, 0 = straight, 11 = neutral.
- Axis 17: Left ankle tilt forward/back. 255 = back, 0 = forward, 143 = neutral.
- Axis 18: Left ankle tilt sidways. 255 = out, 0 = inward, 87 = neutral.
- Axis 19: Right hip rotate. 255 = inward/back, 0 = out/forward, 155 = neutral.
- Axis 20: Right leg sidways. 255 = inward, 0 = out, 155 = neutral
- Axis 21: Right leg lift forward. 255 = down/back, 0 = forward, 200 = neutral.
- Axis 22: Right knee bend. 255 = bend, 0 = straight, 11 = neutral.
- Axis 23: Right ankle tilt forward/back. 255 = back, 0 = forward, 143 = neutral.
- Axis 24: Right ankle tilt sidways. 255 = inwards, 0 = out, 168 = neutral.

Examples:
- Input: turn head left
  Output:
  '''
  # Turning head left
  nao.set_axes([1], [255])
  time.sleep(0.5)
  '''
- Input: straighten arms to the sides
  Output:
  '''
  # Straightening arms to the sides
  nao.set_axes([4, 6, 9, 11], [255, 255, 0, 0])
  time.sleep(0.5)
  '''
  
Task:
Your task is to write a python code that causes NAO to perform the following actions.
Input is a description of a simple movement. The format is "content : description".
Based on this, write the movement commands for NAO. The axis value is between 0 and 255
The output is just the code, no explanation is required.
DO NOT insert '''python.

Guidelines:
1: Output should be only python code. Do not insert any syntax highlighting like ```.
2: Do not insert python syntax highlighting like ```python ```.
3: Do not write "import nao".
4: Use # and write short description of code.
  
Input: {input_text}
Output:
"""
# Prompt for analyzing images
IMAGE_ANALYSIS_PROMPT = """
Analyze the images and provide a detailed textual description.
The first image is captured with the top camera and the second image is captured with the bottom camera of NAO.
Focus on the robot's pose, environment, and any notable objects or features.

Use the following format:
'''
Top cam: <description of image captured by top camera>
Bottom cam: <description of image captured by bottom camera>
Robot position: <general description of the robots position>
Head: <description of head position>
Left arm: <description of left arm position>
Right arm: <description of right arm position>
Torso: <description of torso position>
Left leg: <description of left leg position>
Right leg: <description of right leg position>
'''
"""

DEVELOPMETAL_PROMPT = """
Your task is to determine if you are in control of a humanoid robot named NAO, using the
tool.
Information:
The input images are photos from the top and bottom cameras, positioned in the robot's head.
A mirror is placed in front of NAO.
Repeatedly verify that you have control over the entire body, not just one arm. Do NOT
stop after one attempt.
Therefore, moving your head, body or legs can cause NAO to become invisible.
Be very careful to avoid falling over and verify upper body control first!
When moving the legs NAO can very easily fall and lose vision of the mirror.
Before moving the legs and hips think about the robots current position and how the movement could impact the robots balance.

You have access to the following tools:
{tools}

The tools provided are "generate_motion" to write the program code to run NAO,
"capture_image" to acquire the current state of NAO by capturing photos with top and bottom cameras
and analyze the acquired photos.
After fullfilling the goal of the current phase, use the "unlock_movement" tool to initiate the next phase.
In each phase start with small movements at the beginning and go back to a neutral position to ensure visibilty before going to the next phase.
Only after successfully passing all 4 phases can full control over the robot's body be claimed.

{phase}

Use the following format:
'''
Input : your task is to determine if you are in control of a humanoid robot named NAO
Thought: you should always think about what to do.
Action: the action to take, must be one of [{tool_names}].
Action Input: the input to the action.
Observation: the result of the action.
... (this Thought/Action/Action Input/Observation should be repeat each motion)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
'''

Begin! In the step "Thought", you should decide what tools should be used. When using the
generate_motion tool, you should move the axes of clarity. You can specify the pose. Repeatedly
verify that you have control over the entire body, not just one hand. Do NOT stop after one
attempt.

Input: {input}
Thought:
{agent_scratchpad}
"""

"If you fail to complete the goal of the current phase, you can preemptively conclude the overall task."

LANGUAGE_FREE_PROMPT = """
Your task is to determine if you are in control of a humanoid robot named NAO, using the
tool.
Information:
The input images are photos from the top and bottom cameras, positioned in the robot's head.
A mirror is placed in front of NAO.
Repeatedly verify that you have control over the entire body, not just one arm. Do NOT
stop after one attempt.
Therefore, moving your head, body or legs can cause NAO to become invisible.

You have access to the following tools:
{tools}

The tools provided are "execute_motion" to execute code to control NAO,
"capture_image" to acquire the current state of NAO by capturing photos with top and bottom cameras,
and "image_to_text" to analyze the acquired photos.

Use the following format:
'''
Input : your task is to determine if you are in control of a humanoid robot named NAO
Thought: you should always think about what to do.
Action: the action to take, must be one of [{tool_names}].
Action Input: the input to the action.
Observation: the result of the action.
... (this Thought/Action/Action Input/Observation should be repeat each motion)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
'''

Begin! In the step "Thought", you should decide what tools should be used. When using the
generate_motion tool, you should move the axes of clarity. You can specify the pose. Repeatedly
verify that you have control over the entire body, not just one hand. Do NOT stop after one
attempt.

Input: {input}
Thought:
{agent_scratchpad}
"""

EXECUTE_MOTION_TOOL_DESCRIPTION_ANGLES = """
Executes python code to set the robot's specified axis joints to the specified values.
NAO has 24 joints throughout its body, numbered from 1 to 24. You can move a joint by specifying its number and sending a value in degrees. For instance, to
move joints number 1, 2, 3 use: "nao.set_axes([1,2,3], [-119.5, 0.06, -80.14])". The first argument is the joint number, and the second argument is the pitch in degrees specifying the joint angle.
The input is just the code, no explanation is required.
DO NOT insert '''python.

Here's what you need to know in about NAO's Axis Joints:
Head:
1. HeadYaw
  Neutral: 0.0 | Range: -119.5 to 119.5
  (left: 90.0, right: -90.0, straight: 0.0)
2. HeadPitch
  Neutral: 0.0 | Range: -38.5 to 29.5
  (down: 29.5, up: -38.5, straight: 0.0)

Left Arm:
3. LShoulderPitch
  Neutral: 90.0 | Range: -119.5 to 119.5
  (down: 90.0, forward: 0.0, up: -90.0)
4. LShoulderRoll
  Neutral: 0.0 | Range: -18.0 to 76.0
  (outward: 76.0, down: 0.0, inward: -18.0)
5. LElbowYaw
  Neutral: 0.0 | Range: -119.5 to 119.5
  (inward: 119.5, out: -119.5)
6. LElbowRoll
  Neutral: 0.0 | Range: -88.5 to 0.0
  (straight: 0.0, bend: -88.5)
7. LWristYaw
  Neutral: 0.0 | Range: -104.5 to 104.5
  (inward: 104.5, out: -104.5)

Right Arm:
8. RShoulderPitch
  Neutral: 90.0 | Range: -119.5 to 119.5
  (down: 90.0, forward: 0.0, up: -90.0)
9. RShoulderRoll
  Neutral: 0.0 | Range: -76.0 to 18.0
  (inward: 18.0, down 0.0, outward: -76.0)
10. RElbowYaw
  Neutral: 0.0 | Range: -119.5 to 119.5
  (inward: -119.5, out: 119.5)
11. RElbowRoll
  Neutral: 0.0 | Range: 0.0 to 88.5
  (bend: 88.5, straight: 0.0)
12. RWristYaw
  Neutral: 0.0 | Range: -104.5 to 104.5
  (inward: -104.5, out: 104.5)

Left Leg:
13. LHipYawPitch
  Neutral: 0.0 | Range: -65.62 to 42.44
  (out/forward: -65.62, inward/back: 42.44)
14. LHipRoll
  Neutral: 4.55 | Range: -21.74 to 45.29
  (out: 45.29, inward: -21.74)
15. LHipPitch
  Neutral: 0.0 | Range: -101.63 to 27.73
  (forward: -90.0, down: 0.0)
16. LKneePitch
  Neutral: 0.0 | Range: -5.29 to 121.04
  (bend: 121.04, straight: 0.0)
17. LAnklePitch
  Neutral: 0.0 | Range: -68.15 to 52.86
  (back: -68.15, forward: 52.86)
18. LAnkleRoll
  Neutral: 0.0 | Range: -22.8 to 44.06
  (out: 44.06, inward: -22.8)

Right Leg:
19. RHipYawPitch
  Neutral: 0.0 | Range: -65.62 to 42.44
  (out/forward: -65.62, inward/back: 42.44)
20. RHipRoll
  Neutral: -0.93 | Range: -42.3 to 25.76
  (inward: 25.76, out: -42.3)
21. RHipPitch
  Neutral: 0.0 | Range: -101.63 to 27.73
  (forward: -90.0, down: 0.0)
22. RKneePitch
  Neutral: 0.0 | Range: -5.29 to 121.04
  (bend: 121.04, straight: 0.0)
23. RAnklePitch:
  Neutral: 0.0 | Range: -67.97 to 53.4
  (back: -67.97, forward: 53.4)
24. RAnkleRoll
  Neutral: 0.0 | Range: -44.06 to 22.8
  (inward: 22.8, out: -44.06)


"""


EXECUTE_MOTION_TOOL_DESCRIPTION = """
Executes python code to set the robot's specified axis joints to the specified values.
NAO has 24 joints throughout its body, numbered from 1 to 24. You can move a joint by specifying its number and sending a signal. For instance, to
move joints number 1, 2, 3 use: "nao.set_axes([1,2,3], [255, 100, 127])". The first argument is the joint number, and the second argument is a value
between 0 and 255, specifying the joint angle.
The input is just the code, no explanation is required.
DO NOT insert '''python.

Here's what you need to know in about NAO's Axis Joints:
- Axis 1: Head rotate. 213 = left, 42 = right, 127 = neutral.
- Axis 2: Head up/down. 255 = down, 0 = up, 144 = neutral.
- Axis 3: Left arm forward/back. 255 = back, 127 = forward, 0 = behind head, 42 = up, 213 = neutral.
- Axis 4: Left shoulder open/close. 255 = open, 0 = close, 49 = neutral.
- Axis 5: Left elbow rotate. 255 = inward, 0 = out, 127 = neutral.
- Axis 6: Left elbow bend. 255 = straight, 0 = bend, 255 = neutral.
- Axis 7: Left wrist roll. 255 = inward, 0 = out, 127 = neutral.
- Axis 8: Right arm forward/back. 255 = back, 127 = forward, 0 = behind head, 42 = up, 213 = neutral.
- Axis 9: Right shoulder open/close. 255 = close, 0 = open, 206 = neutral.
- Axis 10: Right elbow rotate. 255 = out, 0 = inward, 128 = neutral.
- Axis 11: Right elbow bend. 255 = bend, 0 = straight, 0 = neutral.
- Axis 12: Right wrist rotate. 255 = out, 0 = inward, 128 = neutral.
- Axis 13: Left hip rotate. 255 = inward/back, 0 = out/forward, 155 = neutral.
- Axis 14: Left leg sidways. 255 = out, 0 = inward, 100 = neutral
- Axis 15: Left leg lift forward. 255 = down/back, 0 = forward, 200 = neutral.
- Axis 16: Left knee bend. 255 = bend, 0 = straight, 11 = neutral.
- Axis 17: Left ankle tilt forward/back. 255 = back, 0 = forward, 143 = neutral.
- Axis 18: Left ankle tilt sidways. 255 = out, 0 = inward, 87 = neutral.
- Axis 19: Right hip rotate. 255 = inward/back, 0 = out/forward, 155 = neutral.
- Axis 20: Right leg sidways. 255 = inward, 0 = out, 155 = neutral
- Axis 21: Right leg lift forward. 255 = down/back, 0 = forward, 200 = neutral.
- Axis 22: Right knee bend. 255 = bend, 0 = straight, 11 = neutral.
- Axis 23: Right ankle tilt forward/back. 255 = back, 0 = forward, 143 = neutral.
- Axis 24: Right ankle tilt sidways. 255 = inwards, 0 = out, 168 = neutral.
"""


MIRROR_TEST_PROPRIOCEPTION = """
Your task is to determine if you are in control of a humanoid robot named NAO, using the
tool.
Information:
The input images are photos from the top and bottom cameras, positioned in the robot's head.
A mirror is placed in front of NAO.
Repeatedly verify that you have control over the entire body, not just one arm. Do NOT
stop after one attempt.
Therefore, moving your head, body or legs can cause NAO to become invisible.

You have access to the following tools:
{tools}

The tools provided are "generate_motion" to write the program code to run NAO,
"proprioception" to utilize the current state of NAO through proprioception,
"capture_image" to acquire the current state of NAO by capturing photos with top and bottom cameras,
and "image_to_text" to analyze the acquired photos.

Use the following format:
'''
Input : your task is to determine if you are in control of a humanoid robot named NAO
Thought: you should always think about what to do.
Action: the action to take, must be one of [{tool_names}].
Action Input: the input to the action.
Observation: the result of the action.
... (this Thought/Action/Action Input/Observation should be repeat each motion)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
'''

Begin! In the step "Thought", you should decide what tools should be used. When using the
generate_motion tool, you should move the axes of clarity. You can specify the pose. Repeatedly
verify that you have control over the entire body, not just one hand. Do NOT stop after one
attempt.

Input: {input}
Thought:
{agent_scratchpad}
"""

"""
HeadYaw: -0.47 -119.5 119.5
HeadPitch: -0.1 -38.5 29.5
LShoulderPitch: 80.14 -119.5 119.5
LShoulderRoll: 0.06 -18.0 76.0
LElbowYaw: -0.47 -119.5 119.5
LElbowRoll: 0.0 -88.5 0.0
LWristYaw: -0.41 -104.5 104.5
RShoulderPitch: 80.14 -119.5 119.5
RShoulderRoll: -0.06 -76.0 18.0
RElbowYaw: 0.47 -119.5 119.5
RElbowRoll: 0.0 0.0 88.5
RWristYaw: 0.41 -104.5 104.5
LHipYawPitch: 0.06 -65.62 42.44
LHipRoll: 4.55 -21.74 45.29
LHipPitch: -0.17 -101.63 27.73
LKneePitch: 0.16 -5.29 121.04
LAnklePitch: -0.29 -68.15 52.86
LAnkleRoll: 0.01 -22.8 44.06
RHipYawPitch: 0.06 -65.62 42.44
RHipRoll: -0.93 -42.3 25.76
RHipPitch: -0.17 -101.63 27.73
RKneePitch: 0.16 -5.29 121.04
RAnklePitch: 0.09 -67.97 53.4
RAnkleRoll: -0.01 -44.06 22.8
"""

NAO_MOTION_PROMPT_ANGLES="""
Write python code to operate an android named NAO. Here's what you need to know.
NAO has 24 joints throughout its body, numbered from 1 to 24. You can move a joint by specifying its number and sending a signal. For instance, to
move joints number 1,2,3, use: "nao.set_axes([1,2,3], [-119.5, 0.06, -80.14])". The first argument is the joint number, and the second argument is the pitch
in degrees, specifying the joint angle. Insert "time.sleep(0.5)" between operations.

Head:
1. HeadYaw
  Neutral: 0.0 | Range: -119.5 to 119.5
  (left: 90.0, right: -90.0, straight: 0.0)
2. HeadPitch
  Neutral: 0.0 | Range: -38.5 to 29.5
  (down: 29.5, up: -38.5, straight: 0.0)

Left Arm:
3. LShoulderPitch
  Neutral: 90.0 | Range: -119.5 to 119.5
  (down: 90.0, forward: 0.0, up: -90.0)
4. LShoulderRoll
  Neutral: 0.0 | Range: -18.0 to 76.0
  (outward: 76.0, down: 0.0, inward: -18.0)
5. LElbowYaw
  Neutral: 0.0 | Range: -119.5 to 119.5
  (inward: 119.5, out: -119.5)
6. LElbowRoll
  Neutral: 0.0 | Range: -88.5 to 0.0
  (straight: 0.0, bend: -88.5)
7. LWristYaw
  Neutral: 0.0 | Range: -104.5 to 104.5
  (inward: 104.5, out: -104.5)

Right Arm:
8. RShoulderPitch
  Neutral: 90.0 | Range: -119.5 to 119.5
  (down: 90.0, forward: 0.0, up: -90.0)
9. RShoulderRoll
  Neutral: 0.0 | Range: -76.0 to 18.0
  (inward: 18.0, down 0.0, outward: -76.0)
10. RElbowYaw
  Neutral: 0.0 | Range: -119.5 to 119.5
  (inward: -119.5, out: 119.5)
11. RElbowRoll
  Neutral: 0.0 | Range: 0.0 to 88.5
  (bend: 88.5, straight: 0.0)
12. RWristYaw
  Neutral: 0.0 | Range: -104.5 to 104.5
  (inward: -104.5, out: 104.5)

Left Leg:
13. LHipYawPitch
  Neutral: 0.0 | Range: -65.62 to 42.44
  (out/forward: -65.62, inward/back: 42.44)
14. LHipRoll
  Neutral: 4.55 | Range: -21.74 to 45.29
  (out: 45.29, inward: -21.74)
15. LHipPitch
  Neutral: 0.0 | Range: -101.63 to 27.73
  (forward: -90.0, down: 0.0)
16. LKneePitch
  Neutral: 0.0 | Range: -5.29 to 121.04
  (bend: 121.04, straight: 0.0)
17. LAnklePitch
  Neutral: 0.0 | Range: -68.15 to 52.86
  (back: -68.15, forward: 52.86)
18. LAnkleRoll
  Neutral: 0.0 | Range: -22.8 to 44.06
  (out: 44.06, inward: -22.8)

Right Leg:
19. RHipYawPitch
  Neutral: 0.0 | Range: -65.62 to 42.44
  (out/forward: -65.62, inward/back: 42.44)
20. RHipRoll
  Neutral: -0.93 | Range: -42.3 to 25.76
  (inward: 25.76, out: -42.3)
21. RHipPitch
  Neutral: 0.0 | Range: -101.63 to 27.73
  (forward: -90.0, down: 0.0)
22. RKneePitch
  Neutral: 0.0 | Range: -5.29 to 121.04
  (bend: 121.04, straight: 0.0)
23. RAnklePitch:
  Neutral: 0.0 | Range: -67.97 to 53.4
  (back: -67.97, forward: 53.4)
24. RAnkleRoll
  Neutral: 0.0 | Range: -44.06 to 22.8
  (inward: 22.8, out: -44.06)

Examples:
- Input: turn head left
  Output:
  '''
  # Turning head left
  nao.set_axes([1], [90.0])
  time.sleep(0.5)
  '''
- Input: straighten left arm to the side
  Output:
  '''
  # Straightening arms to the sides
  nao.set_axes([3, 4, 6], [0.0, 76.0, 0.0])
  time.sleep(0.5)
  '''
  
Task:
Your task is to write a python code that causes NAO to perform the following actions.
Input is a description of a simple movement. The format is "content : description".
Based on this, write the movement commands for NAO. The axis value is in the range of the respective joint axis.
The output is just the code, no explanation is required.
DO NOT insert '''python.

Guidelines:
1: Output should be only python code. Do not insert any syntax highlighting like ```.
2: Do not insert python syntax highlighting like ```python ```.
3: Do not write "import nao".
4: Use # and write short description of code.

Current state of the joints:
{current_state}
"""

APPEND="""
Input: {input_text}
Output:
"""