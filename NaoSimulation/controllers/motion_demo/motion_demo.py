import math
from controller import Robot, Keyboard, Motion
from PIL import Image
import numpy as np
import io
import os
from flask import Flask, request, jsonify
import threading
import ast
import time # needed for waiting after setting axes
import random
from image_dir import IMAGE_DIR

# Flask server for communication
app = Flask(__name__)
#timestemp = datetime.datetime.now()
image_dir = IMAGE_DIR
run_id = 0

IGNORE = False
DEGREES = False

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

class Nao(Robot):
    def __init__(self):
        Robot.__init__(self)
        self.currentlyPlaying = False
        self.command = None
        self.motors = {}  # Dictionary to store joint devices
        if not DEGREES:
            self.joint_positions = {1: 127, 2: 144, 3: 213, 4: 49,
                                    5: 127, 6: 255, 7: 127, 8: 213,
                                    9: 206, 10: 128, 11: 0, 12: 128, 
                                    13: 155, 14: 100, 15: 200, 16: 11,
                                    17: 143, 18: 87, 19: 155, 20: 155,
                                    21: 200, 22: 11, 23: 143, 24: 168
                                }  # Dictionary to store target joint positions
        else:
            self.joint_positions = {1: 0.0, 2: 0.0, 3: 90.0, 4: 0.0,
                                    5: 0.0, 6: 0.0, 7: 0.0, 8: 90.0,
                                    9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 
                                    13: 0.0, 14: 4.55, 15: 0.0, 16: 0.0,
                                    17: 0.0, 18: 0.0, 19: 0.0, 20: -0.93,
                                    21: 0.0, 22: 0.0, 23: 0.0, 24: 0.0
                                }  # Dictionary to store target joint positions

        self.joint_translate = {}
        # Initialize stuff
        self.findAndEnableDevices()
        dictionary = self.get_joint_positions()
        #for joint in dictionary.keys():
            #print(joint+": "+str(dictionary[joint]) + " " + str(round(self.motors[joint].getMinPosition()* 180/math.pi,2)) + " " + str(round(self.motors[joint].getMaxPosition()* 180/math.pi,2)))

            
    def captureImage(self, camera, save_path):
        width = camera.getWidth()
        height = camera.getHeight()
        print(save_path)
        image_bytes = camera.getImage()

        image_array = np.frombuffer(image_bytes, dtype=np.uint8).reshape((height, width, 4))
        image_array = image_array[:, :, [2, 1, 0, 3]]
        image_rgb = image_array[:, :, :3]
        img = Image.fromarray(image_rgb, 'RGB')  # RGB-Modus für Pillow
        img.save(os.path.expanduser(save_path))
        print(f"Image saved as {save_path}")
        return True

    def findAndEnableDevices(self):
        # get the time step of the current world.
        self.timeStep = int(self.getBasicTimeStep())
        #n = self.getNumberOfDevices()
        #for i in range(n):
        #    device = self.getDeviceByIndex(i)
        #    print(device.getName())
        # Initialize all joint devices
        self.motors = {
            "HeadYaw": self.getDevice("HeadYaw"),
            "HeadPitch": self.getDevice("HeadPitch"),
            "LShoulderPitch": self.getDevice("LShoulderPitch"),
            "LShoulderRoll": self.getDevice("LShoulderRoll"),
            "LElbowYaw": self.getDevice("LElbowYaw"),
            "LElbowRoll": self.getDevice("LElbowRoll"),
            "LWristYaw": self.getDevice("LWristYaw"),
            "RShoulderPitch": self.getDevice("RShoulderPitch"),
            "RShoulderRoll": self.getDevice("RShoulderRoll"),
            "RElbowYaw": self.getDevice("RElbowYaw"),
            "RElbowRoll": self.getDevice("RElbowRoll"),
            "RWristYaw": self.getDevice("RWristYaw"),
            "LHipYawPitch": self.getDevice("LHipYawPitch"),
            "LHipRoll": self.getDevice("LHipRoll"),
            "LHipPitch": self.getDevice("LHipPitch"),
            "LKneePitch": self.getDevice("LKneePitch"),
            "LAnklePitch": self.getDevice("LAnklePitch"),
            "LAnkleRoll": self.getDevice("LAnkleRoll"),
            "RHipYawPitch": self.getDevice("RHipYawPitch"),
            "RHipRoll": self.getDevice("RHipRoll"),
            "RHipPitch": self.getDevice("RHipPitch"),
            "RKneePitch": self.getDevice("RKneePitch"),
            "RAnklePitch": self.getDevice("RAnklePitch"),
            "RAnkleRoll": self.getDevice("RAnkleRoll")
        }
        #values
        for i in range(1, 25):
            self.joint_translate[i] = (((self.motors[JOINT_IDS[i]].getMaxPosition()-self.motors[JOINT_IDS[i]].getMinPosition())/255), self.motors[JOINT_IDS[i]].getMinPosition())

        #for i in range(1, 25):
        #    print("*"*100)
        #    print(JOINT_IDS[i])
        #    print((self.motors[JOINT_IDS[i]].getTargetPosition()-self.motors[JOINT_IDS[i]].getMinPosition())*(255/(self.motors[JOINT_IDS[i]].getMaxPosition()-self.motors[JOINT_IDS[i]].getMinPosition())))
        #    print(f"- Range: {self.motors[JOINT_IDS[i]].getMinPosition()} to {self.motors[JOINT_IDS[i]].getMaxPosition()}\n\t- Default: {self.motors[JOINT_IDS[i]].getTargetPosition()}")

        # Initialize other devices (cameras, sensors, etc.)
        self.cameraTop = self.getDevice("CameraTop")
        self.cameraBottom = self.getDevice("CameraBottom")
        self.cameraTop.enable(4 * self.timeStep)
        self.cameraBottom.enable(4 * self.timeStep)

        self.accelerometer = self.getDevice('accelerometer')
        self.accelerometer.enable(4 * self.timeStep)

        self.gyro = self.getDevice('gyro')
        self.gyro.enable(4 * self.timeStep)

        self.gps = self.getDevice('gps')
        self.gps.enable(4 * self.timeStep)

        self.inertialUnit = self.getDevice('inertial unit')
        self.inertialUnit.enable(self.timeStep)

        self.us = []
        usNames = ['Sonar/Left', 'Sonar/Right']
        for i in range(0, len(usNames)):
            self.us.append(self.getDevice(usNames[i]))
            self.us[i].enable(self.timeStep)

        self.fsr = []
        fsrNames = ['LFsr', 'RFsr']
        for i in range(0, len(fsrNames)):
            self.fsr.append(self.getDevice(fsrNames[i]))
            self.fsr[i].enable(self.timeStep)

        self.lfootlbumper = self.getDevice('LFoot/Bumper/Left')
        self.lfootrbumper = self.getDevice('LFoot/Bumper/Right')
        self.rfootlbumper = self.getDevice('RFoot/Bumper/Left')
        self.rfootrbumper = self.getDevice('RFoot/Bumper/Right')
        self.lfootlbumper.enable(self.timeStep)
        self.lfootrbumper.enable(self.timeStep)
        self.rfootlbumper.enable(self.timeStep)
        self.rfootrbumper.enable(self.timeStep)

        self.leds = []
        self.leds.append(self.getDevice('ChestBoard/Led'))
        self.leds.append(self.getDevice('RFoot/Led'))
        self.leds.append(self.getDevice('LFoot/Led'))
        self.leds.append(self.getDevice('Face/Led/Right'))
        self.leds.append(self.getDevice('Face/Led/Left'))
        self.leds.append(self.getDevice('Ears/Led/Right'))
        self.leds.append(self.getDevice('Ears/Led/Left'))

        self.keyboard = self.getKeyboard()
        self.keyboard.enable(10 * self.timeStep)

    def get_rad(self, joint, angle):
        if angle == 255:
            rad = self.motors[JOINT_IDS[joint]].getMaxPosition()
        elif angle == 0:
            rad = self.motors[JOINT_IDS[joint]].getMinPosition()
        else:
            rad = angle*self.joint_translate[joint][0] + self.joint_translate[joint][1]
        return rad


    def set_joint(self, joint, angle):
        rad = self.get_rad(joint, angle)
        self.motors[JOINT_IDS[joint]].setPosition(self.get_rad(joint, angle))
        if not DEGREES:
            self.joint_positions[joint] = angle
        else:
            self.joint_positions[joint] = round(rad*(180/math.pi), 2)
        return angle
    
    def set_axes(self, axes, angles):
        if IGNORE:
            for ax, angle in zip(axes, angles):
                self.set_joint(ax, random.randint(0, 255))
        else:
            if not DEGREES:
                for ax, angle in zip(axes, angles):
                    self.set_joint(ax, angle)
            else:
                for ax, angle in zip(axes, angles):
                    self.set_joint_angle(ax, angle)
    #angles
    def set_joint_angle(self, joint, angle):
        rad = angle * (math.pi/180)
        if rad < self.motors[JOINT_IDS[joint]].getMinPosition():
            rad = self.motors[JOINT_IDS[joint]].getMinPosition()
        elif rad > self.motors[JOINT_IDS[joint]].getMaxPosition():
            rad = self.motors[JOINT_IDS[joint]].getMaxPosition()
        
        self.motors[JOINT_IDS[joint]].setPosition(rad)
        self.joint_positions[joint] = round(rad * (180/math.pi), 2)

    def get_joint_positions(self):
        positions = {}
        if not DEGREES:
            for joint in range(1, 25):
                positions[JOINT_IDS[joint]] = round(self.get_rad(joint, self.joint_positions[joint]) * 180/math.pi, 2)
                #print("*"*100)
                #print(self.motors[JOINT_IDS[joint]].getValue())
        else:
            for joint in range(1, 25):
                positions[JOINT_IDS[joint]] = round(self.joint_positions[joint], 2)
            
        return positions

    def run(self):
        """
        Main loop of the controller.
        """
        print("Starting NAO Motion Control Demo.")
        while self.step(self.timeStep) != -1:
            # Apply the target joint positions
            if not DEGREES:
                for joint_id, angle in self.joint_positions.items():
                    self.set_joint(joint_id, angle)
            
            else:
                for joint, angle in self.joint_positions.items():
                    self.set_joint_angle(joint, angle)
                    #print("*"*10)

@app.route('/run_id', methods=['POST'])
def set_run_id():
    data = request.json
    global run_id
    run_id = data.get("id")
    try: os.mkdir(f"{image_dir}run_{run_id}/")
    except Exception as e:
        return jsonify({"status": "failure", "message": f"{e}\nRun ID {run_id} has been used. Please enter another ID"})
    return jsonify({"status": "success", "message": f"Run ID {run_id} has been set"})


@app.route('/set_joints', methods=['POST'])
def set_joints():
    data = request.json
    print("*"*10)
    print(data)
    print("*"*10)
    #code = ast.literal_eval(str(data.get("code")))
    code = data.get("code")
    print(f"Method call: {code}")
    exec(f"{code}")
    return jsonify({"status": "success", "message": "Joint positions updated", "positions": str(nao.get_joint_positions()), "code": code})


#@app.route('/record_set_joints', methods=['POST'])
#def record_set_joints():
#    data = request.json
#    print("*"*10)
#    print(data)
#    print("*"*10)
#    #code = ast.literal_eval(str(data.get("code")))
#    code = str(data.get("code"))
#    id = data.get("id")
#    insert = code.find("])\ntime")
#    run_dir = f"{image_dir}run_{run_id}/"
#    i=0
#    while insert != -1:
#        code = code[:insert+2] + f"\nnao.captureImage(camera=nao.cameraTop, save_path = \"{run_dir}\"+\"top_cam_{id}_{i}.jpeg\")\nnao.captureImage(camera=nao.cameraBottom, save_path = \"{run_dir}\"+\"bot_cam_{id}_{id}.jpeg\")" + code[insert+2:]
#        insert = code.find("])\ntime")
#        i += 1
#    print(f"Method call: {code}")
#    exec(f"{code}")
#    return jsonify({"status": "success", "message": "Joint positions updated", "positions": f"{nao.joint_positions}", "code": code})


@app.route('/capture_image', methods=['POST'])
def capture_image():
    data = request.json
    id = data.get("id")
    head_img = f"top_cam_{id}.jpeg"
    body_img = f"bot_cam_{id}.jpeg"
    run_dir = f"{image_dir}run_{run_id}/"
    nao.captureImage(camera=nao.cameraTop, save_path = run_dir+head_img)
    nao.captureImage(camera=nao.cameraBottom, save_path = run_dir+body_img)
    return jsonify({"status": "success", "message": "Image saved.",
                    "image_id": id})

@app.route('/proprioception', methods=['GET'])
def proprioception():
    joint_positions = nao.get_joint_positions()
    return jsonify({"status": "succes", "message": "Joint positions transmitted.",
                    "joint_positions": joint_positions})

def run_flask_server():
    """Run the Flask server in a separate thread."""
    app.run(host='0.0.0.0', port=5000)

# Start the Flask server in a separate thread
flask_thread = threading.Thread(target=run_flask_server)
#flask_thread.daemon = True  # Daemonize thread to exit when the main program exits
flask_thread.start()

# Create the Robot instance and run main loop
nao = Nao()
nao.run()