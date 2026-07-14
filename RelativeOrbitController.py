# =============================================================================
# File Name: FormationController.py
# FSW Formation Flying Controller (Relative Orbit Control for N Satellites).
# =============================================================================

from Basilisk.architecture import sysModel
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import thrFiringSchmitt
import math

class FormationRelativeController(sysModel.SysModel):
    """
    Custom Python Flight Software (FSW) module running inside the simulation task loop.
    Implements a relative orbit PD control law in the Hill frame for multi-satellite 
    formations, transforming the required forces into the Chaser's Body Frame.
    """
    def __init__(self, chaserTransMsg, chaserAttMsg, chiefTransMsg, satellite_index):
        super(FormationRelativeController, self).__init__()
        self.chaserTransMsg = chaserTransMsg
        self.chaserAttMsg = chaserAttMsg
        self.chiefTransMsg = chiefTransMsg
        self.satellite_index = satellite_index
        
        # Output message to send command force values to the Schmitt trigger modulator
        self.forceMsgOut = messaging.THRArrayCmdForceMsg()
        
        # Define the target relative slot in the Hill Frame (X: Radial, Y: Along-track, Z: Cross-track)
        # Each satellite takes a unique position trailing the Chief (e.g., -100m, -200m, -300m...)
        self.target_r_Hill = [0.0, (satellite_index + 1) * -100.0, 0.0]
        self.target_v_Hill = [0.0, 0.0, 0.0] # Target relative velocity is zero to hold station
        
        # Control Gains (Tuned for relative translation maneuvers)
        self.Kp = 0.15  # Proportional gain (reacts to relative position error)
        self.Kd = 2.5   # Derivative gain (reacts to relative velocity error)

    def UpdateState(self, CurrentSimNanos):
        # Read current navigation messages
        chaserData = self.chaserTransMsg.read()
        chaserAtt = self.chaserAttMsg.read()
        chiefData = self.chiefTransMsg.read()
        
        # 1. Compute Chief's Orbit/Hill Frame Basis Vectors in the Inertial Frame (N)
        r_chief_N = chiefData.r_BN_N
        v_chief_N = chiefData.v_BN_N
        
        r_chief_mag = math.sqrt(sum(k**2 for k in r_chief_N))
        if r_chief_mag < 0.001: return
        
        # h_r (Radial): unit vector along Chief position
        h_r = [k / r_chief_mag for k in r_chief_N]
        
        # h_z (Cross-track/Orbit Normal): unit vector along angular momentum (r x v)
        hx = r_chief_N[1]*v_chief_N[2] - r_chief_N[2]*v_chief_N[1]
        hy = r_chief_N[2]*v_chief_N[0] - r_chief_N[0]*v_chief_N[2]
        hz = r_chief_N[0]*v_chief_N[1] - r_chief_N[1]*v_chief_N[0]
        h_mag = math.sqrt(hx**2 + hy**2 + hz**2)
        h_z = [hx/h_mag, hy/h_mag, hz/h_mag]
        
        # h_y (Along-track): completes the right-handed triad (h_z x h_r)
        h_y = [
            h_z[1]*h_r[2] - h_z[2]*h_r[1],
            h_z[2]*h_r[0] - h_z[0]*h_r[2],
            h_z[0]*h_r[1] - h_z[1]*h_r[0]
        ]
        
        # Direction Cosine Matrix from Inertial to Hill frame [HN]
        # Rows are the Hill frame unit vectors expressed in Inertial coordinates
        HN = [h_r, h_y, h_z]

        # 2. Compute Relative Position and Velocity in the Inertial Frame
        r_rel_N = [chaserData.r_BN_N[i] - chiefData.r_BN_N[i] for i in range(3)]
        v_rel_N = [chaserData.v_BN_N[i] - chiefData.v_BN_N[i] for i in range(3)]
        
        # 3. Transform Relative States from Inertial (N) to Hill Frame (H)
        r_rel_H = [sum(HN[i][j] * r_rel_N[j] for j in range(3)) for i in range(3)]
        v_rel_H = [sum(HN[i][j] * v_rel_N[j] for j in range(3)) for i in range(3)]
        
        # 4. Compute Control Errors in Hill Frame
        error_r = [self.target_r_Hill[i] - r_rel_H[i] for i in range(3)]
        error_v = [self.target_v_Hill[i] - v_rel_H[i] for i in range(3)]
        
        # 5. Apply PD Control Law to compute required ideal 3D force in Hill Frame
        F_Hill = [self.Kp * error_r[i] + self.Kd * error_v[i] for i in range(3)]
        
        # 6. Transform Required Force from Hill Frame (H) back to Inertial Frame (N)
        # F_N = [HN]^T * F_Hill
        F_Inertial = [sum(HN[j][i] * F_Hill[j] for j in range(3)) for i in range(3)]
        
        # 7. Transform Required Force from Inertial Frame (N) to Chaser's Body Frame (B)
        # Convert Chaser's current MRPs (sigma_BN) to a Rotation Matrix [BN]
        sigma = chaserAtt.sigma_BN
        s2 = sigma[0]**2 + sigma[1]**2 + sigma[2]**2
        
        # Skew-symmetric matrix components for the cross product
        tilde = [
            [0.0, -sigma[2], sigma[1]],
            [sigma[2], 0.0, -sigma[0]],
            [-sigma[1], sigma[0], 0.0]
        ]
        
        # Standard identity matrix
        I3 = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
        
        # Build [BN] Matrix using the mathematical formulation for MRPs
        BN = [[0.0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                tilde_sq = sum(tilde[i][k]*tilde[k][j] for k in range(3))
                BN[i][j] = I3[i][j] + (8.0 * tilde_sq - 4.0 * (1.0 - s2) * tilde[i][j]) / ((1.0 + s2)**2)
                
        # Compute final required 3D control force vector in Body Frame
        F_Body = [sum(BN[i][j] * F_Inertial[j] for j in range(3)) for i in range(3)]

        # 8. Actuator Mapping (Map 3D Body Forces onto the Thruster Channels)
        payload = messaging.THRArrayCmdForceMsgPayload()
        thruster_forces = [0.0] * 36
        
        # Mapping rules based on the thruster layout configurations in Config.json:
        # Index 0 is the high capacity Main Translation thruster pointing along +Z
        if F_Body[2] > 0.0:
            thruster_forces[0] = F_Body[2]
            
        # Symmetrically distribute lateral forces to the attitude ACS thruster pairs
        if F_Body[0] > 0.0: # +X Force requests
            thruster_forces[5] = F_Body[0] / 2.0
            thruster_forces[6] = F_Body[0] / 2.0
        if F_Body[1] > 0.0: # +Y Force requests
            thruster_forces[1] = F_Body[1] / 2.0
            thruster_forces[2] = F_Body[1] / 2.0
            
        payload.thrForce = thruster_forces
        self.forceMsgOut.write(payload, CurrentSimNanos)


def formation_controller(scSim, fswTaskName, thrusterConfigMsg, chaserNavTransMsg, chaserNavAttMsg, chiefNavTransMsg, satellite_index=0):
    """
    FSW formation flying architecture router: handles multi-satellite relative orbit control loop setup.
    """
    # Instantiate the custom relative orbit tracking controller module
    relativeControl = FormationRelativeController(chaserNavTransMsg, chaserNavAttMsg, chiefNavTransMsg, satellite_index)
    relativeControl.ModelTag = f"formationRelativeControl_{satellite_index}"
    scSim.AddModelToTask(fswTaskName, relativeControl)
    
    # Modulate continuous required forces into discrete valve timing (OnTime)
    schmitt = thrFiringSchmitt.thrFiringSchmitt()
    schmitt.ModelTag = f"formationSchmitt_{satellite_index}"
    schmitt.thrMinFireTime = 0.025
    schmitt.level_on = 0.3
    schmitt.level_off = 0.1
    
    # Connect the Schmitt Trigger input to our custom relative orbit force outputs
    schmitt.thrForceInMsg.subscribeTo(relativeControl.forceMsgOut)
    schmitt.thrConfInMsg.subscribeTo(thrusterConfigMsg)
    scSim.AddModelToTask(fswTaskName, schmitt)
    
    # Store reference inside the simulation base class to protect against Python memory deallocation
    if not hasattr(scSim, "customFormationModules"):
        scSim.customFormationModules = []
    scSim.customFormationModules.append(relativeControl)
    
    # Return the commanded thruster valve timings back to the script
    return schmitt.onTimeOutMsg