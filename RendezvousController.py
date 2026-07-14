#==============================================================================#
# File Name: RendezvousController.py
# FSW Rendezvous Controller (Attitude Pointing + Main Engine Translation)
#==============================================================================#

from Basilisk.fswAlgorithms import mrpFeedback, locationPointing, attTrackingError, rwMotorTorque, thrFiringSchmitt
from Basilisk.architecture import messaging
from Basilisk.architecture import sysModel
from Basilisk.utilities import macros
import math

class ConditionalThrusterController(sysModel.SysModel):
    """
    Custom Python Flight Software (FSW) module running inside the simulation task loop.
    Implements a safe 'Cruise Control' for orbital rendezvous by evaluating real-time
    relative distance and closing velocity to command precise main engine pulses.
    """
    def __init__(self, guidMsg, chaserTransMsg, chiefNavMsg, alignment_threshold):
        super(ConditionalThrusterController, self).__init__()
        self.guidMsg = guidMsg
        self.chaserTransMsg = chaserTransMsg
        self.chiefNavMsg = chiefNavMsg  # Direct NavTransMsg containing dynamic state and velocity
        self.alignment_threshold = alignment_threshold
        # Output message to send command force values to the Schmitt trigger modulator
        self.forceMsgOut = messaging.THRArrayCmdForceMsg()
        
    def UpdateState(self, CurrentSimNanos):
        # Read the current navigation and guidance states
        guidData = self.guidMsg.read()
        chaserData = self.chaserTransMsg.read()
        chiefData = self.chiefNavMsg.read()
        
        # 1. Compute the magnitude of the attitude pointing error vector (MRP norm)
        mrp_error_norm = math.sqrt(guidData.sigma_BR[0]**2 + guidData.sigma_BR[1]**2 + guidData.sigma_BR[2]**2)
        
        # 2. Compute the Relative Position Vector (Target - Chaser) in Inertial Frame
        rx = chiefData.r_BN_N[0] - chaserData.r_BN_N[0]
        ry = chiefData.r_BN_N[1] - chaserData.r_BN_N[1]
        rz = chiefData.r_BN_N[2] - chaserData.r_BN_N[2]
        
        distance = math.sqrt(rx**2 + ry**2 + rz**2)
        if distance < 0.0001: 
            distance = 0.0001  # Guard against division by zero
            
        # 3. Compute the Relative Velocity Vector (Chaser - Target) in Inertial Frame
        vx_c = chaserData.v_BN_N[0] - chiefData.v_BN_N[0]
        vy_c = chaserData.v_BN_N[1] - chiefData.v_BN_N[1]
        vz_c = chaserData.v_BN_N[2] - chiefData.v_BN_N[2]
        
        # 4. Compute Closing Speed by projecting relative velocity onto the Line of Sight (LOS)
        # Positive values indicate that the Chaser is actively closing the gap to the Target
        closing_speed = (vx_c * rx + vy_c * ry + vz_c * rz) / distance
        
        payload = messaging.THRArrayCmdForceMsgPayload()
        
        # Pre-allocate a 36-element force array matching Basilisk's internal C++ vector capacity
        thruster_forces = [0.0] * 36
        
        # 5. Cruise Control Decision Logic:
        # - Spaceship must be aligned within the threshold
        # - Target must be further than a safe 10-meter boundary
        if mrp_error_norm < self.alignment_threshold and distance > 10.0:
            if closing_speed < 1.5:
                # Command full engine force to overcome the Schmitt trigger level_on (30% of 500N)
                thruster_forces[0] = 500.0
                
        # Write the force command list back to the C++ payload structure
        payload.thrForce = thruster_forces
        self.forceMsgOut.write(payload, CurrentSimNanos)


def rendezvous_controller(scSim, fswTaskName, thrusterConfigMsg, rwConfigMsg, chaserTransMsg, chaserAttMsg, chiefTransMsg, vehConfigMsg, chiefNavMsg=None):
    """
    FSW rendezvous controller: manages pointing attitude and controls closing velocity.
    """
    
    # 1. POINTING REFERENCE (locationPointing)
    # Align the spacecraft +Z axis with the dynamic location of the Target (Chief)
    locPoint = locationPointing.locationPointing()
    locPoint.ModelTag = "locPoint"
    locPoint.scTransInMsg.subscribeTo(chaserTransMsg)
    locPoint.scAttInMsg.subscribeTo(chaserAttMsg)
    locPoint.locationInMsg.subscribeTo(chiefTransMsg) 
    
    # Set pointing axis to +Z in body frame to align the main engine thrust direction
    locPoint.pHat_B = [0.0, 0.0, 1.0] 
    scSim.AddModelToTask(fswTaskName, locPoint)

    # 2. ATTITUDE TRACKING ERROR (attTrackingError)
    # Evaluates the instantaneous angular deviation from the reference target pointing direction
    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attError"
    attError.attNavInMsg.subscribeTo(chaserAttMsg)
    attError.attRefInMsg.subscribeTo(locPoint.attRefOutMsg)
    scSim.AddModelToTask(fswTaskName, attError)

    # 3. ATTITUDE CONTROL LAW (mrpFeedback)
    # Computes the 3D control torque required to damp attitude and tracking errors
    mrpControl = mrpFeedback.mrpFeedback()
    mrpControl.ModelTag = "mrpFeedbackControl"
    mrpControl.K = 5.5
    
    # Critically damped P gain to eliminate pointing oscillations
    mrpControl.P = 150.0
    
    mrpControl.Ki = 0.0  # Integral gain disabled to prevent reaction wheel wind-up
    mrpControl.integralLimit = 0.0
    mrpControl.guidInMsg.subscribeTo(attError.attGuidOutMsg)
    mrpControl.vehConfigInMsg.subscribeTo(vehConfigMsg)
    scSim.AddModelToTask(fswTaskName, mrpControl)

    # 4. REACTION WHEEL TORQUE MAPPING (rwMotorTorque)
    # Maps the commanded 3D attitude control torque onto individual reaction wheel spin axes
    rwTorque = rwMotorTorque.rwMotorTorque()
    rwTorque.ModelTag = "rwTorque"
    rwTorque.rwParamsInMsg.subscribeTo(rwConfigMsg)
    rwTorque.vehControlInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)
    rwTorque.controlAxes_B = [1,0,0, 0,1,0, 0,0,1]
    scSim.AddModelToTask(fswTaskName, rwTorque)

    # 5. TRANSLATION CONTROL (Main Engine Conditional Firing)
    # We command thrust only when pointing error is within a safe 3.0-degree margin
    alignment_threshold = 2.0 * math.tan(macros.D2R * 3.0 / 4.0) 
    
    # Pass target state navigation to the Python FSW cruise control module
    target_nav_msg = chiefNavMsg if chiefNavMsg is not None else chiefTransMsg

    # Instantiate the custom conditional firing module to handle translation thrust safety
    condFiring = ConditionalThrusterController(attError.attGuidOutMsg, chaserTransMsg, target_nav_msg, alignment_threshold)
    condFiring.ModelTag = "conditionalFiring"
    scSim.AddModelToTask(fswTaskName, condFiring)

    # Convert continuous thrust command signals into discrete valve timing (OnTime)
    schmitt = thrFiringSchmitt.thrFiringSchmitt()
    schmitt.ModelTag = "mainThrusterSchmitt"
    schmitt.thrMinFireTime = 0.025
    schmitt.level_on = 0.3
    schmitt.level_off = 0.1
    
    # Wire the modulator input to the output of our custom conditional thruster controller
    schmitt.thrForceInMsg.subscribeTo(condFiring.forceMsgOut)
    schmitt.thrConfInMsg.subscribeTo(thrusterConfigMsg)
    
    # Prevent Python's garbage collector from destroying the module during simulation steps
    scSim.customCondFiringModule = condFiring 
    
    scSim.AddModelToTask(fswTaskName, schmitt)

    # Return commanded thruster valve timing and reaction wheel motor commands
    return schmitt.onTimeOutMsg, rwTorque.rwMotorTorqueOutMsg