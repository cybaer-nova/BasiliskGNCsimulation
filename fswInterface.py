import sys
# Import mandatory Basilisk architecture modules
from Basilisk.architecture import sysModel
from Basilisk.architecture import messaging

class CustomPythonFSW(sysModel.SysModel):
    """
    A Basilisk Python module wrapper enabling pure Python controllers to run within the execution loop.
    """
    def __init__(self, custom_controller_file, fsw_inputs, thrusterCase_val):
        super(CustomPythonFSW, self).__init__()
        self.custom_controller_file = custom_controller_file
        self.fsw_inputs = fsw_inputs
        self.tCase = thrusterCase_val # Store the static thruster configuration case (0, 1, or 2)
        
        # Dynamically import the pure Python file based on the JSON string name
        try:
            __import__(custom_controller_file)
            self.external_script = sys.modules[custom_controller_file]
        except ImportError:
            raise ImportError(f"Could not find or import the custom controller file '{custom_controller_file}.py'.")
        
        # Create standalone Basilisk output command messages
        # These will be populated with data computed by the external script.
        self.torque_out_msg = messaging.CmdTorqueBodyMsg()
        self.ontime_out_msg = messaging.THRArrayOnTimeCmdMsg()

        # Initialize number of physical thrusters to 0. It will be set during the first UpdateState call.
        self.num_physical_thrusters = 0

    def Reset(self, currentTime):
        # Mandatory Basilisk Reset function; remains empty for this pure math bridge module.
        return

    def UpdateState(self, currentTime):
        """
        Executes at every Flight Software time step (0.1s) in C++ time.
        Reads sensor data, calls the custom Python algorithm, and writes actuator commands.
        """
        # --- 1. Read Current Simulation States (Inputs) ---
        # Read the current attitude/angular velocity and translation (position/velocity) navigation messages
        nav_payload = self.fsw_inputs["nav_state"].read()     # Attitudes & Angular Velocity
        trans_payload = self.fsw_inputs["trans_state"].read() # Positions & Velocity (Inertial Frame)
        
        # Convert memory-managed structs (Swig objects) into pure Python lists for the external script.
        # This prevents potential memory leaks across the language boundary and ensures clean data handling.
        current_sigma = list(nav_payload.sigma_BN)
        current_omega = list(nav_payload.omega_BN_B)
        current_r = list(trans_payload.r_BN_N)                # Inertial position vector [x, y, z] in meters
        current_v = list(trans_payload.v_BN_N)                # Inertial velocity vector [vx, vy, vz] in m/s
        
        # --- 2. Call the Pure Python External Controller ---
        # Pass position, velocity, and static thrusterCase (0, 1, or 2) integers as pure Python data types to the user's logic.
        # Pass self.tCase so the user knows which array indices are available.
        raw_torque, raw_ontime = self.external_script.evaluate_controller(
            current_sigma, current_omega, current_r, current_v, self.tCase
        )
        
        # --- 3. Pack Output Binaries (CmdTorqueBodyMsg) ---
        # Populated for Reaction Wheels (used in thrusterCase 1). The message accepts a size-3 list.
        torque_payload = messaging.CmdTorqueBodyMsgPayload()
        torque_payload.torqueRequestBody = raw_torque
        self.torque_out_msg.write(torque_payload, currentTime, self.moduleID)
        
        # --- 4. Pack Output Valve Firing Times (THRArrayOnTimeCmdMsg Payload) ---
        # Handles dynamic remapping of directional thrusters vs. main translation engines.
        # Implements memory safety guardrails.
        ontime_payload = messaging.THRArrayOnTimeCmdMsgPayload()
        
        # Determine number of physical thrusters on the first run.
        # This is the correct way to get the count of thrusters configured by Main.py[cite: 3].
        if self.num_physical_thrusters == 0:
            config_payload = self.fsw_inputs["thruster_config"].read()
            self.num_physical_thrusters = config_payload.numThrusters
            # Fallback to standard Basilisk payload limits if query fails[cite: 3].
            if self.num_physical_thrusters == 0:
                self.num_physical_thrusters = messaging.MAX_EFF_CNT

        # Iterate over the raw firing times list received from the custom Python script.
        # The script returns a dynamically sized list.
        received_list_len = len(raw_ontime)
        for i in range(received_list_len):
            # Check against allocated C++ memory buffer to prevent corruption.
            if i < self.num_physical_thrusters:
                ontime_payload.OnTimeRequest[i] = raw_ontime[i]
            
        # Write the populated valve duration payload back to the dynamic Basilisk message bus.
        self.ontime_out_msg.write(ontime_payload, currentTime, self.moduleID)
        
        # Explicit return marks end of successful update step.
        return 

def setup_fsw_controller(scSim, simTaskName, nav_msg, trans_msg, configDataMsg, thrusterCase_val, controller_type="MRP", custom_module_name=None, thrusterConfigMsg=None):
    """
    Universal Flight Software router. 
    Connects Basilisk navigation sensor outputs to designated controller type.
    Handles setup for default Basilisk controllers or pure Python custom algorithms.
    """
    fsw_inputs = {
        "nav_state": nav_msg,
        "trans_state": trans_msg,       # Storeographical dynamic inertial position message reference.
        "vehicle_config": configDataMsg # Holds physical properties (Inertia, CoM).
    }
    fsw_outputs = {}

    # Controller Route 0 (Default): Basilisk C++ MRP feedback controller.
    if controller_type == "MRP":
        from Basilisk.fswAlgorithms import mrpFeedback
        # Standard mrpFeedback setup logic here...
        pass

    # Controller Route 1: Custom external Python script (e.g., custom_controller.py).
    elif controller_type == "CUSTOM":
        if not custom_module_name:
            raise ValueError("Controller type is CUSTOM, but no filename was provided in the JSON configuration.")
            
        print(f"Instantiating Custom Python Controller from file: {custom_module_name}.py")
        
        # Add the thruster configuration message to inputs if provided.
        if thrusterConfigMsg:
            fsw_inputs["thruster_config"] = thrusterConfigMsg
        else:
            raise ValueError("Custom controller requires thrusterConfigMsg to determine the number of physical thrusters.")

        # Instantiate custom bridge module, passing dynamic inputs AND static thrusterCase integer.
        custom_fsw_module = CustomPythonFSW(custom_module_name, fsw_inputs, thrusterCase_val)
        custom_fsw_module.ModelTag = "CustomExternalController"
        
        # Add Python FSW module to designated simulation task loop.
        scSim.AddModelToTask(simTaskName, custom_fsw_module)
        
        # Route computed command messages back to universal bridge output dictionary.
        # Allows Main.py script to subscribe physical actuators to these commands.
        fsw_outputs["torque_cmd"] = custom_fsw_module.torque_out_msg
        fsw_outputs["ontime_cmd"] = custom_fsw_module.ontime_out_msg
            
    # Return command message dictionary to calling script.
    return fsw_outputs