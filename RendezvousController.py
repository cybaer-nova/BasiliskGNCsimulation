#==============================================================================#
# File Name: RendezvousController.py
# FSW Rendezvous Controller (Velocity Glide-Slope Architecture)
#==============================================================================#

from Basilisk.fswAlgorithms import mrpFeedback, attTrackingError, rwMotorTorque, thrFiringSchmitt
from Basilisk.architecture import messaging
from Basilisk.architecture import sysModel
from Basilisk.utilities import macros
from Basilisk.utilities import RigidBodyKinematics as rbk
import math
import numpy as np

class RendezvousGuidanceController(sysModel.SysModel):
    """
    Flight Software module implementing orbital rendezvous guidance using 
    Clohessy-Wiltshire relative dynamics and a velocity glide-slope architecture.
    """
    def __init__(self, chaserTransMsg, chaserAttMsg, chiefNavMsg):
        super(RendezvousGuidanceController, self).__init__()
        
        # Store input message handles for translation and attitude states
        self.chaserTransMsg = chaserTransMsg
        self.chaserAttMsg = chaserAttMsg
        self.chiefNavMsg = chiefNavMsg
        
        # Define output messages for attitude reference and thruster forces
        self.attRefMsgOut = messaging.AttRefMsg()
        self.forceMsgOut = messaging.THRArrayCmdForceMsg()

        # Spacecraft mass used for force-to-acceleration conversion
        self.mass = 300.0  
        
        # Glide-slope and cruise control parameters
        self.v_cruise_max = 1.0       # Maximum safe approach cruise velocity
        self.braking_dist = 85.0      # Distance threshold where braking begins
        self.Kv = 0.04                # Proportional velocity tracking gain
        self.a_max = 0.035            # Maximum acceleration limit (~10.5 N) for clean pulses      
        
        self.b_z_prev = None
        
    def UpdateState(self, CurrentSimNanos):
        """
        Periodic execution loop updating relative navigation, guidance, and control commands.
        """
        # Read current navigation and attitude data from input messages
        chaserData = self.chaserTransMsg.read()
        chaserAttData = self.chaserAttMsg.read()
        chiefData = self.chiefNavMsg.read()
        
        # Extract inertial position and velocity vectors for chief and chaser
        r_c = np.array(chiefData.r_BN_N)
        v_c = np.array(chiefData.v_BN_N)
        r_s = np.array(chaserData.r_BN_N)
        v_s = np.array(chaserData.v_BN_N)
        
        # Compute relative position and velocity vectors in the inertial frame
        r_rel = r_s - r_c
        v_rel = v_s - v_c
        
        # Guard against zero-division for chief radius magnitude
        r_mag = np.linalg.norm(r_c)
        if r_mag < 1.0: return
        
        # Compute orbital angular momentum vector and magnitude
        h_vec = np.cross(r_c, v_c)
        h_mag = np.linalg.norm(h_vec)
        if h_mag < 1.0: return
        
        # Construct the Hill (Orbital) reference frame rotation matrix (R_HN)
        i_r = r_c / r_mag                      
        i_h = h_vec / h_mag                      
        i_theta = np.cross(i_h, i_r)             
        R_HN = np.vstack([i_r, i_theta, i_h])
        
        # Transform relative position into the Hill frame
        r_H = R_HN @ r_rel
        x, y, z = r_H[0], r_H[1], r_H[2]
        
        # Compute mean motion (n) and relative velocity in the Hill frame
        n = h_mag / (r_mag**2) 
        v_H = R_HN @ v_rel - np.array([-n * y, n * x, 0.0])
        x_dot, y_dot, z_dot = v_H[0], v_H[1], v_H[2]
        
        dist_inertial = np.linalg.norm(r_rel)
        
        # 1. CRUISE CONTROL AND STATION-KEEPING LOGIC
        target_dist = 10.0 
        dist_to_go = dist_inertial - target_dist
        
        # Determine required velocity magnitude based on distance profile (Glide-slope)
        if dist_to_go > self.braking_dist:
            v_req_mag = self.v_cruise_max
        elif dist_to_go > 0:
            v_req_mag = (self.v_cruise_max / self.braking_dist) * dist_to_go
        else:
            v_req_mag = 0.05 * dist_to_go 
            
        # Compute the required velocity vector pointing toward the target
        if dist_inertial > 0.1:
            v_req_H = -(r_H / dist_inertial) * v_req_mag
        else:
            v_req_H = np.array([0.0, 0.0, 0.0])
            
        # Compute velocity error vector in the Hill frame
        v_err_H = v_req_H - np.array([x_dot, y_dot, z_dot])
        
        # 2. SELF-LIMITED ACCELERATION CONTROL
        a_ctrl_H = self.Kv * v_err_H
        a_ctrl_mag = np.linalg.norm(a_ctrl_H)
        
        # Saturate control acceleration to maximum allowed limit
        if a_ctrl_mag > self.a_max:
            a_ctrl_H = a_ctrl_H * (self.a_max / a_ctrl_mag)
            
        # 3. GRAVITY COMPENSATION (Clohessy-Wiltshire Equations)
        a_cw_x = 2.0 * n * y_dot + 3.0 * (n**2) * x
        a_cw_y = -2.0 * n * x_dot
        a_cw_z = -(n**2) * z
        a_cw_H = np.array([a_cw_x, a_cw_y, a_cw_z])
        
        # Combine control acceleration and gravity compensation, then convert to inertial force
        a_total_H = a_ctrl_H - a_cw_H
        F_N_raw = self.mass * (R_HN.T @ a_total_H)
        F_mag = np.linalg.norm(F_N_raw)
        
        # 4. ATTITUDE MANAGEMENT WITH FREEZING AND RATE LIMITING
        if F_mag > 0.01:
            # Align pointing Z-axis with the required force vector when thrusters are active
            b_z_target = F_N_raw / F_mag
        else:
            # Freeze attitude reference to the last valid position when engines are idle
            b_z_target = self.b_z_prev if self.b_z_prev is not None else -r_rel / dist_inertial

        # Apply a rate limiter to prevent abrupt attitude jumps
        if self.b_z_prev is None:
            b_z_req = b_z_target
        else:
            diff = b_z_target - self.b_z_prev
            dist = np.linalg.norm(diff)
            
            max_step = 0.04  
            if dist > max_step:
                b_z_req = self.b_z_prev + (diff / dist) * max_step
            else:
                b_z_req = b_z_target

        b_z_req = b_z_req / np.linalg.norm(b_z_req)
        self.b_z_prev = b_z_req
        
        # 5. ORTHOGONAL REFERENCE FRAME CONSTRUCTION (R_RN)
        ref_stable = i_h
        b_y_req = np.cross(b_z_req, ref_stable)
        if np.linalg.norm(b_y_req) < 1e-3:
            ref_stable = i_r
            b_y_req = np.cross(b_z_req, ref_stable)
            
        b_y_req = b_y_req / np.linalg.norm(b_y_req)
        b_x_req = np.cross(b_y_req, b_z_req)
        b_x_req = b_x_req / np.linalg.norm(b_x_req)
        
        R_RN = np.vstack([b_x_req, b_y_req, b_z_req])
        
        # Convert rotation matrix to Modified Rodrigues Parameters (MRPs)
        try:
            sigma_RN_array = rbk.C2MRP(R_RN)
            sigma_RN = [0.0, 0.0, 0.0] if np.isnan(sigma_RN_array).any() else list(sigma_RN_array)
        except:
            sigma_RN = [0.0, 0.0, 0.0]
        
        # Populate and write the attitude reference message payload
        refPayload = messaging.AttRefMsgPayload()
        refPayload.sigma_RN = sigma_RN
        omega_orb = n * i_h
        refPayload.omega_RN_N = omega_orb.tolist()
        refPayload.domega_RN_N = [0.0, 0.0, 0.0]
        self.attRefMsgOut.write(refPayload, CurrentSimNanos)
        
        # 6. MAIN ENGINE TRANSLATION ACTUATION
        sigma_BN = np.array(chaserAttData.sigma_BN)
        sigma_BR = rbk.subMRP(sigma_BN, np.array(sigma_RN))
        ang_error_deg = 4.0 * math.atan(np.linalg.norm(sigma_BR)) * (180.0 / math.pi)
        
        thruster_forces = [0.0] * 36
        
        # Fire main engine only if attitude pointing error is within the alignment threshold
        if ang_error_deg < 3.0 and not np.isnan(ang_error_deg):
            thruster_forces[0] = min(F_mag, 15.0)
            
        # Write force command payload to the output message bus
        forcePayload = messaging.THRArrayCmdForceMsgPayload()
        forcePayload.thrForce = thruster_forces
        self.forceMsgOut.write(forcePayload, CurrentSimNanos)

def rendezvous_controller(scSim, fswTaskName, thrusterConfigMsg, rwConfigMsg, chaserTransMsg, chaserAttMsg, chiefTransMsg, vehConfigMsg, chiefNavMsg=None):
    """
    Configures and integrates the complete FSW guidance, attitude pointing, and translation loop.
    """
    target_nav_msg = chiefNavMsg if chiefNavMsg is not None else chiefTransMsg

    # Instantiate and add the custom rendezvous guidance controller module
    cwControl = RendezvousGuidanceController(chaserTransMsg, chaserAttMsg, target_nav_msg)
    cwControl.ModelTag = "cwOrbitalMechanicsControl"
    scSim.AddModelToTask(fswTaskName, cwControl)

    # Configure attitude tracking error evaluation module
    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attError"
    attError.attNavInMsg.subscribeTo(chaserAttMsg)
    attError.attRefInMsg.subscribeTo(cwControl.attRefMsgOut)
    scSim.AddModelToTask(fswTaskName, attError)

    # Configure MRP feedback attitude control loop
    mrpControl = mrpFeedback.mrpFeedback()
    mrpControl.ModelTag = "mrpFeedbackControl"
    mrpControl.K = 12.0      
    mrpControl.P = 30.0      
    mrpControl.Ki = 0.001     
    mrpControl.integralLimit = 0.5
    mrpControl.guidInMsg.subscribeTo(attError.attGuidOutMsg)
    mrpControl.vehConfigInMsg.subscribeTo(vehConfigMsg)
    scSim.AddModelToTask(fswTaskName, mrpControl)

    # Configure reaction wheel torque mapping module
    rwTorque = rwMotorTorque.rwMotorTorque()
    rwTorque.ModelTag = "rwTorque"
    rwTorque.rwParamsInMsg.subscribeTo(rwConfigMsg)
    rwTorque.vehControlInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)
    rwTorque.controlAxes_B = [1,0,0, 0,1,0, 0,0,1]
    scSim.AddModelToTask(fswTaskName, rwTorque)

    # Configure Schmitt trigger modulator for thruster pulse generation
    schmitt = thrFiringSchmitt.thrFiringSchmitt()
    schmitt.ModelTag = "mainThrusterSchmitt"
    schmitt.thrMinFireTime = 0.1
    schmitt.level_on = 0.05     
    schmitt.level_off = 0.02
    schmitt.thrForceInMsg.subscribeTo(cwControl.forceMsgOut)
    schmitt.thrConfInMsg.subscribeTo(thrusterConfigMsg)
    scSim.AddModelToTask(fswTaskName, schmitt)
    
    # Store reference to prevent garbage collection
    scSim.customCwControlModule = cwControl 
    return schmitt.onTimeOutMsg, rwTorque.rwMotorTorqueOutMsg, attError
