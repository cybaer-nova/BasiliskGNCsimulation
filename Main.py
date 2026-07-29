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
        payload = messaging.GroundStateMsgPayload()
        payload.r_LN_N = chiefData.r_BN_N
        
        # Write the payload back to the dynamic message output
        self.groundStateMsg.write(payload, CurrentSimNanos)


def run(perturbCase, thrusterCase, controlCase):
    """
    Simulation Executive Function handling configuration, orchestration, and execution.

    Parameters
    ----------
    perturbCase : int (0 to 2)
        Defines the level of environmental perturbations applied to the spacecraft:
        - 0: Ideal spherical gravity of central celestial bodies (Earth, Sun, Moon, Jupiter, Mars).
        - 1: Earth J2 geopotential harmonic perturbation + spherical gravity.
        - 2: Full environmental perturbations (Earth J2 + Solar Radiation Pressure + Atmospheric Drag).

    thrusterCase : int (0 to 2)
        Defines the active hardware and actuator configuration:
        - 0: 6 attitude control thrusters (2 opposing thrusters per body axis).
        - 1: 3 Reaction Wheels for attitude control + 1 high-thrust main translational thruster.
        - 2: 6 attitude control thrusters + 1 high-thrust main translational thruster.

    controlCase : int (0 to 2)
        Defines the active GNC control strategy:
        - 0: Attitude Controller (Manages exclusively spatial reorientation using MRPs).
        - 1: Rendezvous Controller (Manages proximity operations and docking relative to a chief satellite).
        - 2: Relative Orbit & Formation Flying Controller (Maintains relative orbit configuration for multi-satellite fleets).
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
        sat = spacecraft.Spacecraft()
        sat.ModelTag = f"{Config['sat_params']['ModelTag']}_{idx}"  
        sat.hub.mHub = Config["sat_params"]["mass"]                 
        sat.hub.r_BcB_B = Config["sat_params"]["center_of_mass"]     

        I_flat = [float(val) for row in Config["sat_params"]["inertia"] for val in row]
        sat.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I_flat) 

        offset_y = pos_base[1] + (idx * 50.0)
        
        sat.hub.r_CN_NInit = [[pos_base[0]], [offset_y], [pos_base[2]]] 
        sat.hub.v_CN_NInit = Config["sat_params"]["initial_velocity"]   
        sat.hub.sigma_BNInit = Config["sat_params"]["MRPs"]             
        sat.hub.omega_BN_BInit = Config["sat_params"]["angular_velocity"] 

        scSim.AddModelToTask(simTaskName, sat)
        satellites.append(sat)

        sNav = simpleNav.SimpleNav()
        sNav.ModelTag = f"SimpleNavigation_{idx}"
        sNav.scStateInMsg.subscribeTo(sat.scStateOutMsg)
        scSim.AddModelToTask(simTaskName, sNav)
        sNavObjects.append(sNav)


    # =========================================================================
    # TARGET SPACECRAFT (CHIEF) FOR RENDEZVOUS / FORMATION SCENARIO 
    # =========================================================================
    satTarget = spacecraft.Spacecraft()
    satTarget.ModelTag = "ChiefTarget"
    
    vel_base = flatten_vector(Config["sat_params"]["initial_velocity"])
    r_mag = math.sqrt(pos_base[0]**2 + pos_base[1]**2 + pos_base[2]**2)
    v_mag = math.sqrt(vel_base[0]**2 + vel_base[1]**2 + vel_base[2]**2)
    
    theta = 150.0 / r_mag
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    r_target_x = pos_base[0] * cos_t + (vel_base[0]/v_mag) * r_mag * sin_t
    r_target_y = pos_base[1] * cos_t + (vel_base[1]/v_mag) * r_mag * sin_t
    r_target_z = pos_base[2] * cos_t + (vel_base[2]/v_mag) * r_mag * sin_t
    
    v_target_x = vel_base[0] * cos_t - (pos_base[0]/r_mag) * v_mag * sin_t
    v_target_y = vel_base[1] * cos_t - (pos_base[1]/r_mag) * v_mag * sin_t
    v_target_z = vel_base[2] * cos_t - (pos_base[2]/r_mag) * v_mag * sin_t
    
    satTarget.hub.r_CN_NInit = [[r_target_x], [r_target_y], [r_target_z]]
    satTarget.hub.v_CN_NInit = [[v_target_x], [v_target_y], [v_target_z]]
    
    scSim.AddModelToTask(simTaskName, satTarget)

    chiefNavObject = simpleNav.SimpleNav()
    chiefNavObject.ModelTag = "ChiefNavigation"
    chiefNavObject.scStateInMsg.subscribeTo(satTarget.scStateOutMsg)
    scSim.AddModelToTask(simTaskName, chiefNavObject)


    # =========================================================================
    # PERTURBATION USE CASES
    # =========================================================================
    if perturbCase == 0:
        for sat in satellites: gravFactory.addBodiesTo(sat)
        gravFactory.addBodiesTo(satTarget)
        print("Perturbations Level 0.")

    elif perturbCase == 1:
        ggm03sPath = get_path(DataFile.LocalGravData.GGM03S)
        gravBodies["earth"].useSphericalHarmonicsGravityModel(str(ggm03sPath), 2)
        for sat in satellites: gravFactory.addBodiesTo(sat)
        gravFactory.addBodiesTo(satTarget)
        print("Perturbations Level 1.")

    elif perturbCase == 2:
        ggm03sPath = get_path(DataFile.LocalGravData.GGM03S)  
        gravBodies["earth"].useSphericalHarmonicsGravityModel(str(ggm03sPath), 2)      
        for sat in satellites: gravFactory.addBodiesTo(sat)
        gravFactory.addBodiesTo(satTarget)

        for idx, sat in enumerate(satellites):
            drag = dragDynamicEffector.DragDynamicEffector()
            drag.ModelTag = f"Drag_{idx}"
            drag.coreParams.projectedArea = Config["sat_params"]["drag_projectedArea"] 
            drag.coreParams.dragCoeff = Config["sat_params"]["drag_dragCoeff"]         
            sat.addDynamicEffector(drag)

            srp = radiationPressure.RadiationPressure()
            srp.ModelTag = f"SolRadPress_{idx}" 
            srp.area = Config["sat_params"]["srp_area"]                                   
            srp.coefficientReflection = Config["sat_params"]["srp_coefficientReflection"] 
            sat.addDynamicEffector(srp)
        print("Perturbations Level 2.")


    # =========================================================================
    # ACTUATOR PARAMETERS SELECTION
    # =========================================================================
    d = Config["actuator_params"]["d"]
    thrusterType = "Blank_Thruster"
    transThrusterType = "Blank_Thruster"
    rwType = "custom"
        
    orient_kwargs = Config["actuator_params"]["thruster_orient"]
    main_kwargs = Config["actuator_params"]["thruster_main"]
    rw_kwargs = Config["actuator_params"]["reaction_wheel"]

    print("Configuring actuators based on parameters from Config.json.")
    
    thrusterEffectors = []
    rwStateEffectors = []
    ConfigMessage = None
    rwConfigMessage = None

    for idx, sat in enumerate(satellites):
        if thrusterCase == 0:
            thrusterFactory = simIncludeThruster.thrusterFactory()
            thrusterFactory.create(thrusterType, [d, 0.0, 0.0], [0.0, 1.0, 0.0], **orient_kwargs)
            thrusterFactory.create(thrusterType, [-d, 0.0, 0.0], [0.0, 1.0, 0.0], **orient_kwargs)
            thrusterFactory.create(thrusterType, [0.0, d, 0.0], [0.0, 0.0, 1.0], **orient_kwargs)
            thrusterFactory.create(thrusterType, [0.0, -d, 0.0], [0.0, 0.0, 1.0], **orient_kwargs)
            thrusterFactory.create(thrusterType, [0.0, 0.0, d], [1.0, 0.0, 0.0], **orient_kwargs)
            thrusterFactory.create(thrusterType, [0.0, 0.0, -d], [1.0, 0.0, 0.0], **orient_kwargs)

            thrEff = thrusterDynamicEffector.ThrusterDynamicEffector()
            thrusterFactory.addToSpacecraft(f"ThrusterSetup0_{idx}", thrEff, sat)

            scSim.AddModelToTask(simTaskName, thrEff)
            thrusterEffectors.append(thrEff)
            if idx == 0: ConfigMessage = thrusterFactory.getConfigMessage()

        elif thrusterCase == 1:
            thrusterFactory = simIncludeThruster.thrusterFactory()
            thrusterFactory.create(transThrusterType, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], **main_kwargs)
            thrEff = thrusterDynamicEffector.ThrusterDynamicEffector()
            thrusterFactory.addToSpacecraft(f"ThrusterSetup1_{idx}", thrEff, sat)

            rwFactory = simIncludeRW.rwFactory()
            rwFactory.create(rwType, [1.0, 0.0, 0.0], **rw_kwargs) 
            rwFactory.create(rwType, [0.0, 1.0, 0.0], **rw_kwargs) 
            rwFactory.create(rwType, [0.0, 0.0, 1.0], **rw_kwargs) 

            rwEff = reactionWheelStateEffector.ReactionWheelStateEffector()
            rwFactory.addToSpacecraft(f"ReactionWheels_{idx}", rwEff, sat)
            
            scSim.AddModelToTask(simTaskName, rwEff)
            scSim.AddModelToTask(simTaskName, thrEff)
            
            thrusterEffectors.append(thrEff)
            rwStateEffectors.append(rwEff)
            
            if idx == 0: 
                ConfigMessage = thrusterFactory.getConfigMessage()
                rwConfigMessage = rwFactory.getConfigMessage()

        elif thrusterCase == 2:
            thrusterFactory = simIncludeThruster.thrusterFactory()
            thrusterFactory.create(transThrusterType, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], **main_kwargs)
            thrusterFactory.create(thrusterType, [d, 0.0, 0.0], [0.0, 1.0, 0.0], **orient_kwargs)
            thrusterFactory.create(thrusterType, [-d, 0.0, 0.0], [0.0, 1.0, 0.0], **orient_kwargs)
            thrusterFactory.create(thrusterType, [0.0, d, 0.0], [0.0, 0.0, 1.0], **orient_kwargs)
            thrusterFactory.create(thrusterType, [0.0, -d, 0.0], [0.0, 0.0, 1.0], **orient_kwargs)
            thrusterFactory.create(thrusterType, [0.0, 0.0, d], [1.0, 0.0, 0.0], **orient_kwargs)
            thrusterFactory.create(thrusterType, [0.0, 0.0, -d], [1.0, 0.0, 0.0], **orient_kwargs)

            thrEff = thrusterDynamicEffector.ThrusterDynamicEffector()
            thrusterFactory.addToSpacecraft(f"ThrusterSetup2_{idx}", thrEff, sat)
            
            scSim.AddModelToTask(simTaskName, thrEff)
            thrusterEffectors.append(thrEff)
            if idx == 0: ConfigMessage = thrusterFactory.getConfigMessage()


    # =========================================================================
    # CONTROLLER USE CASES
    # =========================================================================
    vcMsgData = messaging.VehicleConfigMsgPayload()
    jsonCoM = Config["sat_params"]["center_of_mass"]
    vcMsgData.CoM_B = flatten_vector(jsonCoM)
    vcMsgData.ISCPntB_B = I_flat
    fswVehConfigMsg = messaging.VehicleConfigMsg().write(vcMsgData)

    bridge = NavToGroundBridge(chiefNavObject.transOutMsg)
    bridge.ModelTag = "NavToGroundBridge"
    scSim.AddModelToTask(simTaskName, bridge)
    
    attErrorLogs = []

    # Controller 0: attitude only
    if controlCase == 0:
        if thrusterCase == 0:
            onTimeCmdMsg, attErrorModule = AttitudeController.attitude_controller(
                scSim = scSim, fswTaskName = fswTaskName, thrusterConfigMsg = ConfigMessage,
                navAttMsg = sNavObjects[0].attOutMsg, navTransMsg = sNavObjects[0].transOutMsg, vehConfigMsg = fswVehConfigMsg
            )
            if fsw_type != "CUSTOM":
                log = attErrorModule.attGuidOutMsg.recorder()
                scSim.AddModelToTask(fswTaskName, log)
                attErrorLogs.append(log)

            thrusterEffectors[0].cmdsInMsg.subscribeTo(onTimeCmdMsg)
            print("Thruster Configuration 0 with the Attitude Controller active.")
        
    # Controller 1: rendezvous 
    elif controlCase == 1:
        if thrusterCase == 1:
            rv_outputs = RendezvousController.rendezvous_controller(
                scSim = scSim, fswTaskName = fswTaskName, thrusterConfigMsg = ConfigMessage, rwConfigMsg = rwConfigMessage,
                chaserTransMsg = sNavObjects[0].transOutMsg, chaserAttMsg = sNavObjects[0].attOutMsg,
                chiefTransMsg = bridge.groundStateMsg, vehConfigMsg = fswVehConfigMsg, chiefNavMsg = chiefNavObject.transOutMsg
            )
            
            # Trata dinamicamente se o RendezvousController retorna 2 ou 3 argumentos (com erro de atitude)
            if isinstance(rv_outputs, tuple) and len(rv_outputs) == 3:
                onTimeCmdMsg, rwCmdMsg, attErrorModule = rv_outputs
                log = attErrorModule.attGuidOutMsg.recorder()
                scSim.AddModelToTask(fswTaskName, log)
                attErrorLogs.append(log)
            else:
                onTimeCmdMsg, rwCmdMsg = rv_outputs

            thrusterEffectors[0].cmdsInMsg.subscribeTo(onTimeCmdMsg)
            rwStateEffectors[0].rwMotorCmdInMsg.subscribeTo(rwCmdMsg)
            print("Thruster Configuration 1 with Rendezvous Controller active.")

    # Controller 2: Formation Flying (Relative Orbit)
    elif controlCase == 2:
        if thrusterCase == 2:
            print(f"Configuring Formation Flying for {numSatellites} satellites using Thruster Configuration 2.")
            for idx in range(numSatellites):
                onTimeCmdMsg = RelativeOrbitController.formation_controller(
                    scSim = scSim, fswTaskName = fswTaskName, thrusterConfigMsg = ConfigMessage,
                    chaserNavTransMsg = sNavObjects[idx].transOutMsg, chaserNavAttMsg = sNavObjects[idx].attOutMsg,
                    chiefNavTransMsg = chiefNavObject.transOutMsg, satellite_index = idx  
                )
                thrusterEffectors[idx].cmdsInMsg.subscribeTo(onTimeCmdMsg)
        else:
            print(f"Warning: Formation Flying requires Thruster Configuration 2. Current thrusterCase is {thrusterCase}.")

    # External / Custom Controller Bridge Hook
    if controlCase == 0 and fsw_type == "CUSTOM":
        generated_commands = setup_fsw_controller(
            scSim = scSim, simTaskName = simTaskName, nav_msg = sNavObjects[0].attOutMsg, 
            trans_msg = sNavObjects[0].transOutMsg, configDataMsg = fswVehConfigMsg, 
            thrusterCase_val = thrusterCase, controller_type = fsw_type, 
            custom_module_name = fsw_file_name, thrusterConfigMsg = ConfigMessage       
        )

        if "torque_cmd" in generated_commands:
            if thrusterCase == 1:
                rwStateEffectors[0].rwMotorCmdInMsg.subscribeTo(generated_commands["torque_cmd"])
                print("Successfully linked custom torque commands to the Reaction Wheels.")

        if "ontime_cmd" in generated_commands:
            thrusterEffectors[0].cmdsInMsg.subscribeTo(generated_commands["ontime_cmd"])
            print("Successfully linked custom OnTime firing commands to the Thrusters.")


    # =========================================================================
    # UNIVERSAL DATA LOGGING Setup (Independent of Case Structure)
    # =========================================================================
    navAttLogs = []
    navTransLogs = []
    thrCmdLogs = [None] * numSatellites
    rwCmdLogs = [None] * numSatellites

    # Always log the Chief/Target space state for cross-simulation data structural symmetry
    chiefTransLog = chiefNavObject.transOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, chiefTransLog)

    for idx in range(numSatellites):
        attLog = sNavObjects[idx].attOutMsg.recorder()
        scSim.AddModelToTask(simTaskName, attLog)
        navAttLogs.append(attLog)

        transLog = sNavObjects[idx].transOutMsg.recorder()
        scSim.AddModelToTask(simTaskName, transLog)
        navTransLogs.append(transLog)

        # Record Thruster commands dynamically if hardware exists
        if idx < len(thrusterEffectors) and thrusterEffectors[idx] is not None:
            thrCmdLogs[idx] = thrusterEffectors[idx].cmdsInMsg.recorder()
            scSim.AddModelToTask(simTaskName, thrCmdLogs[idx])
            
        # Record Reaction Wheel commands dynamically if hardware exists
        if idx < len(rwStateEffectors) and rwStateEffectors[idx] is not None:
            rwCmdLogs[idx] = rwStateEffectors[idx].rwMotorCmdInMsg.recorder()
            scSim.AddModelToTask(simTaskName, rwCmdLogs[idx])

    # Configure Vizard visualization interface before initialization
    vizActive = True
    if vizActive:
        all_sats = satellites + [satTarget]
        thr_list = [[te] if te is not None else [] for te in thrusterEffectors] + [[]]
        if thrusterCase == 1:
            rw_list = rwStateEffectors + [None]
            vizSupport.enableUnityVisualization(scSim, simTaskName, all_sats, rwEffectorList=rw_list, thrEffectorList=thr_list, saveFile=fileName)
        else:
            vizSupport.enableUnityVisualization(scSim, simTaskName, all_sats, thrEffectorList=thr_list, saveFile=fileName)

    # Initialize and Execute Simulation
    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(simulationTime)
    scSim.ExecuteSimulation()


    # =========================================================================
    # UNIVERSAL DATA EXTRACTION AND EXPORTATION
    # =========================================================================
    timeSec = navAttLogs[0].times() * macros.NANO2SEC
    num_points = len(timeSec)
    
    simulationData = {
        "status": "Simulation successfully completed",
        "time_history_s": timeSec.tolist(),
        "satellites": [],
        "chief": {
            "spacecraft_position_m": chiefTransLog.r_BN_N.tolist(),
            "spacecraft_velocity_m_s": chiefTransLog.v_BN_N.tolist()
        }
    }

    for idx in range(numSatellites):
        # Cria a estrutura pré-povoada com arrays de zeros correspondentes ao tamanho do tempo orbital
        satData = {
            "satellite_id": idx,
            "attitude_mrp_history": navAttLogs[idx].sigma_BN.tolist(),
            "angular_velocity_history": navAttLogs[idx].omega_BN_B.tolist(),
            "spacecraft_position_m": navTransLogs[idx].r_BN_N.tolist(),
            "spacecraft_velocity_m_s": navTransLogs[idx].v_BN_N.tolist(),
            "attitude_error_mrp": [[0.0, 0.0, 0.0] for _ in range(num_points)],
            "angular_velocity_error": [[0.0, 0.0, 0.0] for _ in range(num_points)],
            "thruster_onTime_history": [[0.0] * 6 for _ in range(num_points)],
            "rw_torque_history": [[0.0, 0.0, 0.0] for _ in range(num_points)]
        }
        
        # Se um registador de erro nativo Basilisk estiver ativo, substitui os zeros
        if idx < len(attErrorLogs) and attErrorLogs[idx] is not None:
            try:
                satData["attitude_error_mrp"] = attErrorLogs[idx].sigma_BR.tolist()
                satData["angular_velocity_error"] = attErrorLogs[idx].omega_BR_B.tolist()
            except AttributeError:
                pass
                
        # Se os propulsores registaram comandos de disparo reais, extrai e substitui
        if idx < len(thrCmdLogs) and thrCmdLogs[idx] is not None:
            try:
                satData["thruster_onTime_history"] = thrCmdLogs[idx].OnTimeRequest.tolist()
            except AttributeError:
                pass
                
        # Se as rodas de reação registaram torques reais, extrai e substitui
        if idx < len(rwCmdLogs) and rwCmdLogs[idx] is not None:
            try:
                satData["rw_torque_history"] = rwCmdLogs[idx].motorTorque.tolist()
            except AttributeError:
                pass
                
        simulationData["satellites"].append(satData)

    # Write the entire structured data history into a JSON file
    outputFileName = "SimulationData.json"
    with open(outputFileName, "w", encoding="utf-8") as jsonFile:
        jsonFile.write(json.dumps(simulationData, indent=4))

    print(f"Simulation finished! Full data history saved to '{outputFileName}'.")


if __name__ == "__main__":
    with open("Config.json", "r", encoding="utf-8") as file:
        Config = json.load(file)
    run(
        perturbCase = Config["use_cases"]["perturbCase"],   
        thrusterCase = Config["use_cases"]["thrusterCase"], 
        controlCase = Config["use_cases"]["controlCase"]    
    )
