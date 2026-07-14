import math
import os
import json

# Import the default controllers and the FSW interface
import AttitudeController
import RendezvousController
import RelativeOrbitController
from fswInterface import setup_fsw_controller

import matplotlib.pyplot as plt
import numpy as np

from Basilisk.architecture import messaging
from Basilisk.architecture import astroConstants
from Basilisk.architecture import sysModel

# Import simulation related support
from Basilisk.simulation import spacecraft 
from Basilisk.simulation import dragDynamicEffector
from Basilisk.simulation import radiationPressure
from Basilisk.simulation import gravityEffector
from Basilisk.simulation import simpleNav

# Import hardware effectors
from Basilisk.simulation import thrusterDynamicEffector
from Basilisk.simulation import reactionWheelStateEffector
from Basilisk.utilities import simIncludeRW
from Basilisk.utilities import simIncludeThruster

# Import general simulation support files
from Basilisk.utilities import (SimulationBaseClass, macros, simIncludeGravBody, unitTestSupport, vizSupport)
from Basilisk.utilities.supportDataTools.dataFetcher import get_path, DataFile

fileName = os.path.basename(os.path.splitext(__file__)[0])


def flatten_vector(vec):
    # Ensures that column-format vectors [[x],[y],[z]] are flattened to [x,y,z] for C++
    if isinstance(vec[0], list):
        return [vec[0][0], vec[1][0], vec[2][0]]
    return [vec[0], vec[1], vec[2]]


class NavToGroundBridge(sysModel.SysModel):
    """
    A custom Python bridge module to dynamically convert a NavTransMsg (from the target spacecraft)
    to a GroundStateMsg (required by the pointing FSW module) at every simulation step.
    """
    def __init__(self, chiefNavMsg):
        super(NavToGroundBridge, self).__init__()
        self.chiefNavMsg = chiefNavMsg
        self.groundStateMsg = messaging.GroundStateMsg()
        
    def UpdateState(self, CurrentSimNanos):
        # Read the current dynamic position of the target (Chief)
        chiefData = self.chiefNavMsg.read()
        
        # Write its position into the GroundStateMsg payload using r_LN_N
        # Note: GroundStateMsg does not support velocity vectors (no v_LN_N attribute)
        payload = messaging.GroundStateMsgPayload()
        payload.r_LN_N = chiefData.r_BN_N
        
        # Write the payload back to the dynamic message output
        self.groundStateMsg.write(payload, CurrentSimNanos)


def run(perturbCase, thrusterCase, controlCase):
    """
    The simulation has the following use cases:

        perturbCase (int):
            ======  ============================
            0       No perturbations, only ideal spherical gravity.
            1       Earth J2 perturbation and spherical gravity of other celestial bodies.
            2       All perturbations at the same time: Earth J2, solar radiation pressure, and drag force.
            ======  ============================

        thrusterCase (int):
            ======  ============================
            0       2 thrusters with opposite directions per axis.
            1       Reaction wheels for attitude and a higher capacity thruster for translation.
            2       Same as configuration 1 but reaction wheels are replaced by chemical thrusters.
            ======  ============================

        controlCase (int):
            ======  ============================
            0       Attitude-only controller.
            1       Rendezvous controller.
            2       Maintain a relative orbit for 4 satellites.
            ======  ============================
    """

    # Open the configuration file in read mode
    with open("Config.json", "r", encoding="utf-8") as file:
        Config = json.load(file)

    # Extract the FSW settings safely using the .get() method
    fsw_type = Config["fsw_settings"].get("controller_type", "MRP")
    fsw_file_name = Config["fsw_settings"].get("custom_controller_file", None)


    # Extract the dynamic number of satellites based on the control case
    if controlCase in [0, 1]:
        numSatellites = 1  # Attitude and Rendezvous only permit 1 satellite
    else:
        numSatellites = max(1, Config["use_cases"].get("number_of_satellites", 4))  # Formation uses N satellites

    # Create the variable names for tasks and processes
    simTaskName = "simTask"
    simProcessName = "simProcess"

    # Create the sim module as a free container
    scSim = SimulationBaseClass.SimBaseClass()

    # Set the simulation time variable used later on
    simulationTime = macros.min2nano(Config["fsw_settings"]["sim_time"])

    # Create the simulation process
    dynProcess = scSim.CreateNewProcess(simProcessName)

    # Create the dynamics task and choose the simulation time step
    simulationTimeStep = macros.sec2nano(0.1)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    # Create the Flight Software (FSW) task with a 0.1s time step
    fswTaskName = "fswTask"
    fswTimeStep = macros.sec2nano(0.1)
    dynProcess.addTask(scSim.CreateNewTask(fswTaskName, fswTimeStep))

    # Add the celestial body factory and the celestial bodies
    gravFactory = simIncludeGravBody.gravBodyFactory()
    gravBodies = gravFactory.createBodies("earth")
    gravBodies["earth"].isCentralBody = True 
    mu = gravBodies["earth"].mu

    # This loads the astronomical ephemeris data for the selected bodies based on a specific date
    timeInitString = "2021 May 04 07:47:48.965 (UTC)"
    gravFactory.createSpiceInterface(time=timeInitString)
    
    # Add the SPICE module to the simulation dynamic task
    scSim.AddModelToTask(simTaskName, gravFactory.spiceObject)

    # =========================================================================
    # MULTI-SPACECRAFT CONFIGURATION (CHASERS)
    # =========================================================================
    satellites = []
    sNavObjects = []

    pos_base = flatten_vector(Config["sat_params"]["initial_position"])

    for idx in range(numSatellites):
        # Define the spacecraft module for each Chaser
        sat = spacecraft.Spacecraft()
        sat.ModelTag = f"{Config['sat_params']['ModelTag']}_{idx}"  # Set unique model name
        sat.hub.mHub = Config["sat_params"]["mass"]                 # Set the satellite mass 
        sat.hub.r_BcB_B = Config["sat_params"]["center_of_mass"]     # Set the satellite center of mass

        I_flat = [float(val) for row in Config["sat_params"]["inertia"] for val in row]
        sat.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I_flat) # Set the inertia matrix

        # Apply a small along-track separation offset to prevent physical collision at initialization
        offset_y = pos_base[1] + (idx * 50.0)
        
        sat.hub.r_CN_NInit = [[pos_base[0]], [offset_y], [pos_base[2]]] # Set initial position with offset
        sat.hub.v_CN_NInit = Config["sat_params"]["initial_velocity"]   # Set the initial velocity
        sat.hub.sigma_BNInit = Config["sat_params"]["MRPs"]             # Set the MRPs
        sat.hub.omega_BN_BInit = Config["sat_params"]["angular_velocity"] # Set the initial angular velocity

        # Connect the satellite to the simulation
        scSim.AddModelToTask(simTaskName, sat)
        satellites.append(sat)

        # Create the Simple Navigation module (so the FSW knows its current state)
        sNav = simpleNav.SimpleNav()
        sNav.ModelTag = f"SimpleNavigation_{idx}"
        sNav.scStateInMsg.subscribeTo(sat.scStateOutMsg)
        scSim.AddModelToTask(simTaskName, sNav)
        sNavObjects.append(sNav)


    # =========================================================================
    # TARGET SPACECRAFT (CHIEF) FOR RENDEZVOUS / FORMATION SCENARIO 
    # =========================================================================

    # Create the spacecraft module for the Target
    satTarget = spacecraft.Spacecraft()
    satTarget.ModelTag = "ChiefTarget"
    
    # Fetch flat representations of velocity of the chaser 
    vel_base = flatten_vector(Config["sat_params"]["initial_velocity"])
    
    # Orbit radius (R): r=sqrt(x^2+y^2+z^2) and velocity magnitude (V) calculations
    r_mag = math.sqrt(pos_base[0]**2 + pos_base[1]**2 + pos_base[2]**2)
    v_mag = math.sqrt(vel_base[0]**2 + vel_base[1]**2 + vel_base[2]**2)
    
    # Calculate angular displacement (theta) representing a 150m orbital lead along the flight path
    # Angle in radians = arc length / radius
    theta = 150.0 / r_mag
    
    # Perform a rotation matrix transformation to place the target exactly on the same orbit plane
    # but advanced by theta radians along the velocity direction
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    # Position vector transformation (Rotating Chaser's position in the orbital plane)
    r_target_x = pos_base[0] * cos_t + (vel_base[0]/v_mag) * r_mag * sin_t
    r_target_y = pos_base[1] * cos_t + (vel_base[1]/v_mag) * r_mag * sin_t
    r_target_z = pos_base[2] * cos_t + (vel_base[2]/v_mag) * r_mag * sin_t
    
    # Velocity vector transformation (Rotating Chaser's velocity vector to match the new orbital position)
    v_target_x = vel_base[0] * cos_t - (pos_base[0]/r_mag) * v_mag * sin_t
    v_target_y = vel_base[1] * cos_t - (pos_base[1]/r_mag) * v_mag * sin_t
    v_target_z = vel_base[2] * cos_t - (pos_base[2]/r_mag) * v_mag * sin_t
    
    # Set the computed precise states back to the Chief spacecraft initialization fields
    satTarget.hub.r_CN_NInit = [[r_target_x], [r_target_y], [r_target_z]]
    satTarget.hub.v_CN_NInit = [[v_target_x], [v_target_y], [v_target_z]]
    
    # Add the Chief spacecraft to the dynamic task
    scSim.AddModelToTask(simTaskName, satTarget)

    # Create the Simple Navigation module for the Chief Target spacecraft
    chiefNavObject = simpleNav.SimpleNav()
    chiefNavObject.ModelTag = "ChiefNavigation"
    chiefNavObject.scStateInMsg.subscribeTo(satTarget.scStateOutMsg)
    scSim.AddModelToTask(simTaskName, chiefNavObject)


    # =========================================================================
    # PERTURBATION USE CASES (Applied to all satellites)
    # =========================================================================

    # Level 0: no perturbations, only ideal spherical gravity of celestial bodies
    if perturbCase == 0:
        # Add the gravity of the defined bodies to the spacecrafts
        for sat in satellites: gravFactory.addBodiesTo(sat)
        gravFactory.addBodiesTo(satTarget)
        print("Perturbations Level 0.")

    # Level 1: Earth J2 perturbation and spherical gravity of other celestial bodies
    elif perturbCase == 1:
        # Get the path to the standard Earth gravity file (GGM03S)
        ggm03sPath = get_path(DataFile.LocalGravData.GGM03S)
        # Enable spherical harmonics only for Earth, degree 2 (J2)
        gravBodies["earth"].useSphericalHarmonicsGravityModel(str(ggm03sPath), 2)
        for sat in satellites: gravFactory.addBodiesTo(sat)
        gravFactory.addBodiesTo(satTarget)
        print("Perturbations Level 1.")

    # Level 2: all perturbations at the same time: Earth J2, solar radiation pressure, and drag force
    elif perturbCase == 2:
        # Enable J2 just like in level 1
        ggm03sPath = get_path(DataFile.LocalGravData.GGM03S)  
        gravBodies["earth"].useSphericalHarmonicsGravityModel(str(ggm03sPath), 2)      
        for sat in satellites: gravFactory.addBodiesTo(sat)
        gravFactory.addBodiesTo(satTarget)

        for idx, sat in enumerate(satellites):
            # Create and attach the Atmospheric Drag effector to each satellite
            drag = dragDynamicEffector.DragDynamicEffector()
            drag.ModelTag = f"Drag_{idx}"
            drag.coreParams.projectedArea = Config["sat_params"]["drag_projectedArea"] # Area where drag acts in m^2
            drag.coreParams.dragCoeff = Config["sat_params"]["drag_dragCoeff"]         # Drag coefficient
            sat.addDynamicEffector(drag)

            # Create and attach the Solar Radiation Pressure effector to each satellite
            srp = radiationPressure.RadiationPressure()
            srp.ModelTag = f"SolRadPress_{idx}" 
            srp.area = Config["sat_params"]["srp_area"]                                   # Area where SRP acts in m^2
            srp.coefficientReflection = Config["sat_params"]["srp_coefficientReflection"] # Reflection coefficient
            sat.addDynamicEffector(srp)
        print("Perturbations Level 2.")


    # =========================================================================
    # ACTUATOR PARAMETERS SELECTION (Applied dynamically to all satellites)
    # =========================================================================

    d = Config["actuator_params"]["d"]
        
    # Force Basilisk to use empty baseline templates ("Blank_Thruster" and "custom") 
    # to allow direct injection of user-defined parameters
    thrusterType = "Blank_Thruster"
    transThrusterType = "Blank_Thruster"
    rwType = "custom"
        
    # Load custom physical property blocks from JSON directly as Python dictionaries
    orient_kwargs = Config["actuator_params"]["thruster_orient"]
    main_kwargs = Config["actuator_params"]["thruster_main"]
    rw_kwargs = Config["actuator_params"]["reaction_wheel"]

    print("Configuring actuators based on parameters from Config.json.")
    
    # Store effectors and configs for all satellites in lists
    thrusterEffectors = []
    rwStateEffectors = []
    ConfigMessage = None
    rwConfigMessage = None


    # THRUSTER USE CASES #

    for idx, sat in enumerate(satellites):
        # Configuration 0: 2 thrusters with opposite directions per axis
        if thrusterCase == 0:
            # Creating the thruster factory
            thrusterFactory = simIncludeThruster.thrusterFactory()
            # Positioned on the X-axis, thrusting in the Y direction, generates torque on the Z-axis
            thrusterFactory.create(thrusterType, [d, 0.0, 0.0], [0.0, 1.0, 0.0], **orient_kwargs)
            thrusterFactory.create(thrusterType, [-d, 0.0, 0.0], [0.0, 1.0, 0.0], **orient_kwargs)
            # Positioned on the Y-axis, thrusting in the Z direction, generates torque on the X-axis
            thrusterFactory.create(thrusterType, [0.0, d, 0.0], [0.0, 0.0, 1.0], **orient_kwargs)
            thrusterFactory.create(thrusterType, [0.0, -d, 0.0], [0.0, 0.0, 1.0], **orient_kwargs)
            # Positioned on the Z-axis, thrusting in the X direction, generates torque on the Y-axis
            thrusterFactory.create(thrusterType, [0.0, 0.0, d], [1.0, 0.0, 0.0], **orient_kwargs)
            thrusterFactory.create(thrusterType, [0.0, 0.0, -d], [1.0, 0.0, 0.0], **orient_kwargs)

            # Creating the thruster effector and adding it to the satellite
            thrEff = thrusterDynamicEffector.ThrusterDynamicEffector()
            thrusterFactory.addToSpacecraft(f"ThrusterSetup0_{idx}", thrEff, sat)

            # Add the effector to the dynamic task
            scSim.AddModelToTask(simTaskName, thrEff)
            thrusterEffectors.append(thrEff)
            
            # Extract configuration message (identical across identical spacecraft hulls)
            if idx == 0: ConfigMessage = thrusterFactory.getConfigMessage()

        # Configuration 1: reaction wheels for attitude and a higher capacity thruster for translation
        elif thrusterCase == 1:
            thrusterFactory = simIncludeThruster.thrusterFactory()
            
            # Unpack the dictionary to pass custom translation engine properties
            thrusterFactory.create(transThrusterType, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], **main_kwargs)
            thrEff = thrusterDynamicEffector.ThrusterDynamicEffector()
            thrusterFactory.addToSpacecraft(f"ThrusterSetup1_{idx}", thrEff, sat)

            rwFactory = simIncludeRW.rwFactory()
            
            # Unpack the dictionary to pass custom reaction wheel physical properties
            rwFactory.create(rwType, [1.0, 0.0, 0.0], **rw_kwargs) 
            rwFactory.create(rwType, [0.0, 1.0, 0.0], **rw_kwargs) 
            rwFactory.create(rwType, [0.0, 0.0, 1.0], **rw_kwargs) 

            # Creating the reaction wheel effector and adding it to the satellite
            rwEff = reactionWheelStateEffector.ReactionWheelStateEffector()
            rwFactory.addToSpacecraft(f"ReactionWheels_{idx}", rwEff, sat)
            
            # Add the reaction wheel and thruster effectors to the dynamic task
            scSim.AddModelToTask(simTaskName, rwEff)
            scSim.AddModelToTask(simTaskName, thrEff)
            
            thrusterEffectors.append(thrEff)
            rwStateEffectors.append(rwEff)
            
            if idx == 0: 
                ConfigMessage = thrusterFactory.getConfigMessage()
                rwConfigMessage = rwFactory.getConfigMessage()

        # Configuration 2: same as configuration 1 but reaction wheels are replaced by chemical thrusters
        elif thrusterCase == 2:
            thrusterFactory = simIncludeThruster.thrusterFactory()

            # Unpack the dictionary to pass custom main engine properties as explicit kwargs
            thrusterFactory.create(transThrusterType, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], **main_kwargs)
            
            # Unpack the dictionary to pass custom attitude control thruster properties as explicit kwargs
            thrusterFactory.create(thrusterType, [d, 0.0, 0.0], [0.0, 1.0, 0.0], **orient_kwargs)
            thrusterFactory.create(thrusterType, [-d, 0.0, 0.0], [0.0, 1.0, 0.0], **orient_kwargs)
            thrusterFactory.create(thrusterType, [0.0, d, 0.0], [0.0, 0.0, 1.0], **orient_kwargs)
            thrusterFactory.create(thrusterType, [0.0, -d, 0.0], [0.0, 0.0, 1.0], **orient_kwargs)
            thrusterFactory.create(thrusterType, [0.0, 0.0, d], [1.0, 0.0, 0.0], **orient_kwargs)
            thrusterFactory.create(thrusterType, [0.0, 0.0, -d], [1.0, 0.0, 0.0], **orient_kwargs)

            thrEff = thrusterDynamicEffector.ThrusterDynamicEffector()
            thrusterFactory.addToSpacecraft(f"ThrusterSetup2_{idx}", thrEff, sat)
            
            # Add the thruster effector to the dynamic task
            scSim.AddModelToTask(simTaskName, thrEff)
            thrusterEffectors.append(thrEff)
            if idx == 0: ConfigMessage = thrusterFactory.getConfigMessage()


    # =========================================================================
    # CONTROLLER USE CASES
    # =========================================================================

    # Create the standalone Vehicle Configuration Message required by the FSW
    vcMsgData = messaging.VehicleConfigMsgPayload()
    
    # Extract and flatten center of mass from Config.json
    jsonCoM = Config["sat_params"]["center_of_mass"]
    vcMsgData.CoM_B = flatten_vector(jsonCoM)
 
    # Extract the inertia matrix
    vcMsgData.ISCPntB_B = I_flat
    
    # Write the payload data into the message object
    fswVehConfigMsg = messaging.VehicleConfigMsg().write(vcMsgData)

    # Create the dynamic Python bridge to transform the target's dynamic NavTransMsg to a GroundStateMsg
    bridge = NavToGroundBridge(chiefNavObject.transOutMsg)
    bridge.ModelTag = "NavToGroundBridge"
    scSim.AddModelToTask(simTaskName, bridge)
    
    attErrorLogs = []

   # Controller 0: attitude only
    if controlCase == 0:
        if thrusterCase == 0:
            # Execute for the single available satellite (index 0)
            onTimeCmdMsg, attErrorModule = AttitudeController.attitude_controller(
                scSim = scSim, 
                fswTaskName = fswTaskName, 
                thrusterConfigMsg = ConfigMessage,
                navAttMsg = sNavObjects[0].attOutMsg,
                navTransMsg = sNavObjects[0].transOutMsg, 
                vehConfigMsg = fswVehConfigMsg
            )

            if fsw_type != "CUSTOM":
                log = attErrorModule.attGuidOutMsg.recorder()
                scSim.AddModelToTask(fswTaskName, log)
                attErrorLogs.append(log)

            # Connect the timing commands directly to the single physical thruster
            thrusterEffectors[0].cmdsInMsg.subscribeTo(onTimeCmdMsg)
            print("Thruster Configuration 0 with the Attitude Controller active.")
        
    # Controller 1: rendezvous 
    elif controlCase == 1:
        if thrusterCase == 1:
            # Execute the rendezvous guidance algorithm for the single chaser (index 0)
            onTimeCmdMsg, rwCmdMsg = RendezvousController.rendezvous_controller(
                scSim = scSim, 
                fswTaskName = fswTaskName, 
                thrusterConfigMsg = ConfigMessage,
                rwConfigMsg = rwConfigMessage,
                chaserTransMsg = sNavObjects[0].transOutMsg,
                chaserAttMsg = sNavObjects[0].attOutMsg,
                chiefTransMsg = bridge.groundStateMsg,
                vehConfigMsg = fswVehConfigMsg,
                chiefNavMsg = chiefNavObject.transOutMsg
            )

            # Link main thruster and reaction wheel fire commands to the single satellite
            thrusterEffectors[0].cmdsInMsg.subscribeTo(onTimeCmdMsg)
            rwStateEffectors[0].rwMotorCmdInMsg.subscribeTo(rwCmdMsg)
            print("Thruster Configuration 1 with Rendezvous Controller active.")

# Controller 2: Formation Flying (Relative Orbit)
    elif controlCase == 2:
        if thrusterCase == 2:
            print(f"Configuring Formation Flying for {numSatellites} satellites using Thruster Configuration 2.")
            for idx in range(numSatellites):
                # Pass all tracking data and the specific satellite loop index
                onTimeCmdMsg = RelativeOrbitController.formation_controller(
                    scSim = scSim,
                    fswTaskName = fswTaskName,
                    thrusterConfigMsg = ConfigMessage,
                    chaserNavTransMsg = sNavObjects[idx].transOutMsg,
                    chaserNavAttMsg = sNavObjects[idx].attOutMsg,
                    chiefNavTransMsg = chiefNavObject.transOutMsg,
                    satellite_index = idx  # Passes 0, 1, 2, 3 dynamically
                )
                
                # Connect the timing outputs to the physical thruster block of each specific satellite
                thrusterEffectors[idx].cmdsInMsg.subscribeTo(onTimeCmdMsg)
        else:
            print(f"Warning: Formation Flying (controlCase 2) requires Thruster Configuration 2 (thrusterCase 2) to handle translational maneuvers. Current thrusterCase is {thrusterCase}.")


    # Call the language bridge using the variables from the Config file (Applied template to first satellite)
    if controlCase == 0 and fsw_type == "CUSTOM":
        generated_commands = setup_fsw_controller(
            scSim = scSim, 
            simTaskName = simTaskName, 
            nav_msg = sNavObjects[0].attOutMsg, 
            trans_msg = sNavObjects[0].transOutMsg, 
            configDataMsg = fswVehConfigMsg, 
            thrusterCase_val = thrusterCase,         
            controller_type = fsw_type, 
            custom_module_name = fsw_file_name,
            thrusterConfigMsg = ConfigMessage       
        )

        # Route the generated output commands to the active hardware actuators
        if "torque_cmd" in generated_commands:
            if thrusterCase == 1:
                # Connect the torque commands directly to the reaction wheels motor input
                rwStateEffectors[0].rwMotorCmdInMsg.subscribeTo(generated_commands["torque_cmd"])
                print("Successfully linked custom torque commands to the Reaction Wheels.")
            elif thrusterCase in [0, 2]:
                # Notice for thruster configurations that expect on-time command structures instead of pure torque
                print("Note: Reaction Wheels are not active in this thrusterCase; ignoring raw torque commands.")

        # Subscribe the physical thrusters to the custom firing time instructions
        if "ontime_cmd" in generated_commands:
            thrusterEffectors[0].cmdsInMsg.subscribeTo(generated_commands["ontime_cmd"])
            print("Successfully linked custom OnTime firing commands to the Thrusters.")


    # =========================================================================
    # DATA LOGGING AND SIMULATION EXECUTION
    # =========================================================================
    navAttLogs = []
    navTransLogs = []
    thrCmdLogs = []
    rwCmdLogs = []

    for idx in range(numSatellites):
        # Record Spacecraft Actual State (Always available from SimpleNav)
        attLog = sNavObjects[idx].attOutMsg.recorder()
        scSim.AddModelToTask(simTaskName, attLog)
        navAttLogs.append(attLog)

        # Record Spacecraft Actual Translation/Position State
        transLog = sNavObjects[idx].transOutMsg.recorder()
        scSim.AddModelToTask(simTaskName, transLog)
        navTransLogs.append(transLog)

        # Record Actuator Commands Dynamically (Checks what is being used)
        if thrusterCase in [0, 2]:
            # If thrusters are active, record the commands going into them
            tLog = thrusterEffectors[idx].cmdsInMsg.recorder()
            scSim.AddModelToTask(simTaskName, tLog)
            thrCmdLogs.append(tLog)
            
        elif thrusterCase == 1:
            # If reaction wheels are active, record their motor commands
            rLog = rwStateEffectors[idx].rwMotorCmdInMsg.recorder()
            scSim.AddModelToTask(simTaskName, rLog)
            rwCmdLogs.append(rLog)

    # Configure Vizard visualization interface before initialization
    vizActive = True
    if vizActive:
        all_sats = satellites + [satTarget]
        # Build effector array matches for the dynamic multi-spacecraft models
        thr_list = [[te] for te in thrusterEffectors] + [[]]
        
        if thrusterCase == 0:
            vizSupport.enableUnityVisualization(scSim, simTaskName, all_sats, thrEffectorList=thr_list, saveFile=fileName)
        elif thrusterCase == 1:
            rw_list = rwStateEffectors + [None]
            vizSupport.enableUnityVisualization(scSim, simTaskName, all_sats, rwEffectorList=rw_list, thrEffectorList=thr_list, saveFile=fileName)
        elif thrusterCase == 2:
            vizSupport.enableUnityVisualization(scSim, simTaskName, all_sats, thrEffectorList=thr_list, saveFile=fileName)


    # Initialize Simulation
    scSim.InitializeSimulation()
    # Configure a simulation stop time and execute the simulation run
    scSim.ConfigureStopTime(simulationTime)
    scSim.ExecuteSimulation()


    # =========================================================================
    # DATA EXTRACTION AND EXPORTION
    # =========================================================================

    # Simulation times are stored in nanoseconds; convert them to seconds
    timeSec = navAttLogs[0].times() * macros.NANO2SEC
    
    # Convert the full NumPy arrays to standard Python lists for JSON serialization
    # By mapping them to explicit keys, the JSON remains perfectly structured
    simulationData = {
        "status": "Simulation successfully completed",
        "time_history_s": timeSec.tolist(),
        "satellites": []
    }

    # Extract dynamic states iteratively across the fleet
    for idx in range(numSatellites):
        satData = {
            "satellite_id": idx,
            "attitude_mrp_history": navAttLogs[idx].sigma_BN.tolist(),
            "angular_velocity_history": navAttLogs[idx].omega_BN_B.tolist(),
            "spacecraft_position_m": navTransLogs[idx].r_BN_N.tolist()
        }
        
        # Dynamically append Attitude Error history if the module was created and recorded
        if len(attErrorLogs) > idx:
            try:
                # Extract the MRP error (sigma_BR) and angular velocity error (omega_BR_B)
                satData["attitude_error_mrp"] = attErrorLogs[idx].sigma_BR.tolist()
                satData["angular_velocity_error"] = attErrorLogs[idx].omega_BR_B.tolist()
            except AttributeError:
                print(f"Warning: It wasn't possible to extract the errors for satellite {idx}.")
                pass
                
        # Dynamically append Thruster history if it was recorded
        if len(thrCmdLogs) > idx:
            try:
                # OnTimeRequest is an array of firing times per thruster
                satData["thruster_onTime_history"] = thrCmdLogs[idx].OnTimeRequest.tolist()
            except AttributeError:
                pass
                
        # Dynamically append Reaction Wheel history if it was recorded
        if len(rwCmdLogs) > idx:
            try:
                # torqueRequestBody holds the commanded internal torques for each wheel
                satData["rw_torque_history"] = rwCmdLogs[idx].torqueRequestBody.tolist()
            except AttributeError:
                pass
                
        simulationData["satellites"].append(satData)

    # Write the entire structured data history into a JSON file
    outputFileName = "SimulationData.json"
    with open(outputFileName, "w", encoding="utf-8") as jsonFile:
        jsonFile.write(json.dumps(simulationData, indent=4))

    print(f"Simulation finished! Full data history saved to '{outputFileName}'.")


if __name__ == "__main__":
    # Open the configuration file in read mode
    with open("Config.json", "r", encoding="utf-8") as file:
        Config = json.load(file)
    run(
        perturbCase = Config["use_cases"]["perturbCase"],   # perturbCase: 0 to 2
        thrusterCase = Config["use_cases"]["thrusterCase"], # thrusterCase: 0 to 2
        controlCase = Config["use_cases"]["controlCase"]    # controlCase: 0 to 2
    )