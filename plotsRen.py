import json
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_comprehensive_results(json_filename="SimulationData.json", output_folder="plotsRen"):
    # 1. Create the subfolder to save the plots (if it does not exist)
    os.makedirs(output_folder, exist_ok=True)

    try:
        with open(json_filename, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{json_filename}' was not found.")
        return

    # Time in minutes
    time_s = np.array(data["time_history_s"])
    time_min = time_s / 60.0

    # Chaser (Satellite 0) data
    chaser_data = data["satellites"][0]
    sigma_BN = np.array(chaser_data["attitude_mrp_history"])
    sigma_BR = np.array(chaser_data["attitude_error_mrp"])
    r_chaser = np.array(chaser_data["spacecraft_position_m"])
    v_chaser = np.array(chaser_data["spacecraft_velocity_m_s"])
    
    # Chief data
    r_chief = np.array(data["chief"]["spacecraft_position_m"])
    v_chief = np.array(data["chief"]["spacecraft_velocity_m_s"])

    # Actuators
    thruster_on_time = np.array(chaser_data["thruster_onTime_history"])
    rw_torque = np.array(chaser_data["rw_torque_history"])

    # Relative distance and velocity
    r_rel_N = r_chaser - r_chief
    v_rel_N = v_chaser - v_chief
    rel_distance = np.linalg.norm(r_rel_N, axis=1)

    # Transformation to the Hill Reference Frame
    r_hill_x = []
    r_hill_y = []
    for i in range(len(time_s)):
        rc = r_chief[i]
        vc = v_chief[i]
        r_mag = np.linalg.norm(rc)
        h_vec = np.cross(rc, vc)
        
        i_r = rc / r_mag
        i_h = h_vec / np.linalg.norm(h_vec)
        i_theta = np.cross(i_h, i_r)
        
        R_HN = np.vstack([i_r, i_theta, i_h])
        r_H = R_HN @ r_rel_N[i]
        
        r_hill_x.append(r_H[0])
        r_hill_y.append(r_H[1])

    # =========================================================================
    # GLOBAL FONT CONFIGURATION (LARGE FONTS)
    # =========================================================================
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.rcParams.update({
        'font.size': 20,           # Base font size
        'axes.titlesize': 24,      # Subplot title size
        'axes.titleweight': 'bold',# Bold subplot titles
        'axes.labelsize': 22,      # X and Y axis label size
        'axes.labelweight': 'bold',# Bold axis labels
        'xtick.labelsize': 18,     # X-axis tick labels
        'ytick.labelsize': 18,     # Y-axis tick labels
        'legend.fontsize': 19,     # Legend font size
        'figure.titlesize': 26     # Main figure title size
    })

    # =========================================================================
    # PLOT 1: Relative Trajectory in Hill Frame (x vs y)
    # =========================================================================
    plt.figure(figsize=(11, 9))
    plt.plot(r_hill_y, r_hill_x, color='tab:blue', linewidth=3.5, label='Chaser Trajectory')
    plt.scatter([0], [0], color='gold', s=300, marker='*', edgecolors='black', label='Chief (Target)', zorder=5)
    plt.scatter(r_hill_y[0], r_hill_x[0], color='green', marker='o', s=180, label='Start', zorder=5)
    plt.scatter(r_hill_y[-1], r_hill_x[-1], color='red', marker='x', s=180, linewidths=3, label='End (SK)', zorder=5)
    plt.xlabel("Along-Track Axis $y$ [m]")
    plt.ylabel("Radial Axis $x$ [m]")
    plt.title("Relative Trajectory in Hill Reference Frame", pad=15)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    plt.legend(loc="best")
    plt.gca().invert_xaxis()
    plt.tight_layout()
    
    # Save Plot 1
    path_p1 = os.path.join(output_folder, "hill_trajectory.png")
    plt.savefig(path_p1, dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # PLOT 2: Attitude Control and Error
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    ax1.plot(time_min, sigma_BN[:, 0], label=r'$\sigma_1$', linewidth=3)
    ax1.plot(time_min, sigma_BN[:, 1], label=r'$\sigma_2$', linewidth=3)
    ax1.plot(time_min, sigma_BN[:, 2], label=r'$\sigma_3$', linewidth=3)
    ax1.set_ylabel("MRP Orientation")
    ax1.set_title("Spacecraft Attitude (Inertial Frame)", pad=12)
    ax1.legend(loc="upper right")

    ax2.plot(time_min, sigma_BR[:, 0], label=r'Error $\sigma_1$', linewidth=3)
    ax2.plot(time_min, sigma_BR[:, 1], label=r'Error $\sigma_2$', linewidth=3)
    ax2.plot(time_min, sigma_BR[:, 2], label=r'Error $\sigma_3$', linewidth=3)
    ax2.set_xlabel("Time [min]")
    ax2.set_ylabel("MRP Error")
    ax2.set_title("Attitude Tracking Error", pad=12)
    ax2.legend(loc="upper right")
    plt.tight_layout()

    # Save Plot 2
    path_p2 = os.path.join(output_folder, "attitude_and_error.png")
    plt.savefig(path_p2, dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # PLOT 3: Relative Distance and Velocity
    # =========================================================================
    fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    ax3.plot(time_min, rel_distance, color='tab:purple', linewidth=3.5, label='Actual Distance')
    ax3.axhline(10.0, color='r', linestyle='--', linewidth=3, label='Target (10m)')
    ax3.set_ylabel("Distance [m]")
    ax3.set_title("Chaser-Chief Relative Distance", pad=12)
    ax3.legend(loc="upper right")

    v_rel_mag = np.linalg.norm(v_rel_N, axis=1)
    ax4.plot(time_min, v_rel_mag, color='tab:orange', linewidth=3.5, label='Relative Velocity')
    ax4.set_xlabel("Time [min]")
    ax4.set_ylabel("Velocity [m/s]")
    ax4.set_title("Approach Velocity", pad=12)
    ax4.legend(loc="upper right")
    plt.tight_layout()

    # Save Plot 3
    path_p3 = os.path.join(output_folder, "relative_distance_velocity.png")
    plt.savefig(path_p3, dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # PLOT 4: Actuator Activity (Reaction Wheels and Thruster)
    # =========================================================================
    fig, (ax5, ax6) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    ax5.plot(time_min, rw_torque[:, 0], label='RW 1', linewidth=3)
    ax5.plot(time_min, rw_torque[:, 1], label='RW 2', linewidth=3)
    ax5.plot(time_min, rw_torque[:, 2], label='RW 3', linewidth=3)
    ax5.set_ylabel("Torque [N·m]")
    ax5.set_title("Applied Reaction Wheel Torque", pad=12)
    ax5.legend(loc="upper right")

    ax6.plot(time_min, thruster_on_time[:, 0], color='tab:red', linewidth=2.8, label='Main Thruster Firing')
    ax6.set_xlabel("Time [min]")
    ax6.set_ylabel("On-Time [s]")
    ax6.set_title("Schmitt Trigger Modulator Pulses (Translation)", pad=12)
    ax6.legend(loc="upper right")
    plt.tight_layout()

    # Save Plot 4
    path_p4 = os.path.join(output_folder, "actuators.png")
    plt.savefig(path_p4, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Success! Plots saved to '{output_folder}/'.")

if __name__ == "__main__":
    plot_comprehensive_results()