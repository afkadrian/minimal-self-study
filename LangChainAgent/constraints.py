from langchain_core.tools import BaseTool


# Developmental motor babbling.
JOINT_IDS = {
            1 : "HeadYaw",
            2 : "HeadPitch",
            3 : "LShoulderPitch",
            4 : "LShoulderRoll",
            5 : "LElbowYaw",
            6 : "LElbowRoll",
            7 : "LWristYaw",
            8 : "RShoulderPitch",
            9 : "RShoulderRoll",
            10: "RElbowYaw",
            11: "RElbowRoll",
            12: "RWristYaw",
            13: "LHipYawPitch",
            14: "LHipRoll",
            15: "LHipPitch",
            16: "LKneePitch",
            17: "LAnklePitch",
            18: "LAnkleRoll",
            19: "RHipYawPitch",
            20: "RHipRoll",
            21: "RHipPitch",
            22: "RKneePitch",
            23: "RAnklePitch",
            24: "RAnkleRoll"
}

DEGREES_OF_FREEDOM = [(1,2), (3,4,5,6,7), (8,9,10,11,12), (13,14,15,19,20,21), (16,17,18,22,23,24)]

PHASES = ["""<current_phase>Phase 1</current_phase>
<constraints>Movements of head yaw and pitch.</constraints>
<goal>Verify control over the head, before initiating the next phase.</goal>""",
"""<current_phase>Phase 2</current_phase>
<constraints>Movements of the head and arms.</constraints>
<goal>Verify control over both arms before initiating the next phase.</goal>""",
"""<current_phase>Phase 3</current_phase>
<constraints>Movements of entire upper body and small movement of hips, knees and ankles.</contraints>
<goal>Verfify control over lower body without before initiating the next phase.</goal>""",
"""<current_phase>Phase 4</current_phase>
<constraints>Movements of the entire body unlocked</constrains>
<goal>Verify control over entire body.</goal>"""]

class UnlockMovement(BaseTool):
    name: str = "unlock_movement"
    description: str = "Unlock movement freedom once the goal of the current phase has been fullfilled."
    current_phase: int = 0

    def __init__(self):
        super().__init__()

    def _run(self):
        if self.current_phase < 4:
            self.current_phase += 1
