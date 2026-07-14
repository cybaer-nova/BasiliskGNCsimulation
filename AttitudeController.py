#==============================================================================#
# File Name: AttitudeController.py
# FSW Attitude Controller (Attitude Pointing).
#==============================================================================#

 
from Basilisk.fswAlgorithms import mrpFeedback, locationPointing, attTrackingError, thrForceMapping, thrFiringSchmitt, hillPoint
from Basilisk.architecture import messaging

# Global list to store messages and prevent Python from deleting them from memory
_msg_cache = []

def attitude_controller(scSim, fswTaskName, thrusterConfigMsg, navAttMsg, navTransMsg, vehConfigMsg):

    """
    Configures the attitude control system and adds FSW modules to the execution task.

    Parameters
    ----------
    scSim : SimulationBaseClass
        The primary Basilisk simulation base instance.
    fswTaskName : str
        The name of the simulation task where the FSW modules will execute.
    thrusterConfigMsg : ThrusterArrayConfigMsgReader
        The configuration message containing physical thruster parameters/geometry.
    navAttMsg : NavAttMsgReader
        The navigation output message containing the current attitude/state estimate.
    navTransMsg : NavTransMsgReader
        The navigation output message containing the current position/velocity estimate.
    vehConfigMsg : VehicleConfigMsgReader
        The vehicle configuration message containing inertia and center of mass.

    Returns
    -------
    onTimeCmdMsg : Message
        The final commanded thruster on-time message to be linked to the physical thruster effector.
    """ 

    # 1. REFERENCE ATTITUDE (hillPoint)
    # hillPoint automatically calculates the Nadir pointing orientation 
    # AND the required orbital angular velocity.
    
    attRef = hillPoint.hillPoint()
    attRef.ModelTag = "hillPoint"
    
    # Correct message name is transNavInMsg, not transInMsg
    attRef.transNavInMsg.subscribeTo(navTransMsg)
    
    # Define the body axis that should point towards the Earth center (Nadir)
    # This aligns the +Y body axis with the nadir vector
    # attRef.pHat_B = [0.0, 1.0, 0.0] 
    
    scSim.AddModelToTask(fswTaskName, attRef)

    # 2. TRACKING ERROR (attTrackingError)
    # Now it receives both orientation AND angular velocity
    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attError"
    attError.attNavInMsg.subscribeTo(navAttMsg)
    attError.attRefInMsg.subscribeTo(attRef.attRefOutMsg) # Now contains omega_RN_N
    scSim.AddModelToTask(fswTaskName, attError)

    # 3. CONTROL LAW (mrpFeedback)
    # Calculate the required 3D control torque vector to correct the tracking error
    mrpControl = mrpFeedback.mrpFeedback()
    mrpControl.ModelTag = "mrpFeedbackControl"
    mrpControl.K = 25.0
    mrpControl.Ki = 0
    mrpControl.P = 100.0
    mrpControl.integralLimit = 2.0 / abs(mrpControl.Ki) * 0.1 if mrpControl.Ki != 0 else 0.0
    mrpControl.guidInMsg.subscribeTo(attError.attGuidOutMsg)
    # Connect vehicle configuration to the control law module as well
    mrpControl.vehConfigInMsg.subscribeTo(vehConfigMsg)
    scSim.AddModelToTask(fswTaskName, mrpControl)

    # 4. THRUSTER FORCE MAPPING (thrForceMapping)
    # Translate the requested 3D control torque into force commands distributed among the thrusters
    thrMapping = thrForceMapping.thrForceMapping()
    thrMapping.ModelTag = "thrusterMapping"
    thrMapping.controlAxes_B = [1, 0, 0,   0, 1, 0,   0, 0, 1] # Full 3-axis control execution
    thrMapping.thrForceSign = 1 # Specify the sign of the output thruster force command (1 for standard positive forces)
    thrMapping.cmdTorqueInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)
    thrMapping.thrConfigInMsg.subscribeTo(thrusterConfigMsg)
    thrMapping.vehConfigInMsg.subscribeTo(vehConfigMsg)
    scSim.AddModelToTask(fswTaskName, thrMapping)

    # 5. SCHMITT TRIGGER (thrusterFiringSchmitt)
    # Convert continuous thrust force requests into discrete on/off valve time commands (OnTime)
    schmitt = thrFiringSchmitt.thrFiringSchmitt()
    schmitt.ModelTag = "schmittTrigger"
    schmitt.thrMinFireTime = 0.015 # Must match the MinOnTime parameter from the JSON config
    schmitt.level_on = 0.005         # Turn thruster ON if requested force exceeds 30% of max thrust
    schmitt.level_off = 0.002      # Turn thruster OFF if requested force drops below 10% of max thrust
    schmitt.thrForceInMsg.subscribeTo(thrMapping.thrForceCmdOutMsg)
    schmitt.thrConfInMsg.subscribeTo(thrusterConfigMsg)
    scSim.AddModelToTask(fswTaskName, schmitt)

    # Return the firing command output message to link up with the simulation physics effector and the attError module
    return schmitt.onTimeOutMsg, attError