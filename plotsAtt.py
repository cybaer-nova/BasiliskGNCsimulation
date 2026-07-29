import json
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_attitude_and_rate_performance(json_filename="SimulationData.json", output_folder="plotsAtt"):
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

    # Chaser (Satellite 0) attitude and angular rate data
    sat_data = data["satellites"][0]
    
    # Attitude MRPs and Tracking Error
    sigma_BN = np.array(sat_data["attitude_mrp_history"])
    sigma_BR = np.array(sat_data["attitude_error_mrp"])
    
    # Angular Velocity and Tracking Error (rad/s)
    omega_BN = np.array(sat_data["angular_velocity_history"])
    omega_BR = np.array(sat_data["angular_velocity_error"])

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
    # PLOT 1: Spacecraft Attitude and Attitude Tracking Error (MRPs)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    
    # Upper Subplot: Inertial Attitude (sigma_BN)
    ax1.plot(time_min, sigma_BN[:, 0], label=r'$\sigma_1$', linewidth=3)
    ax1.plot(time_min, sigma_BN[:, 1], label=r'$\sigma_2$', linewidth=3)
    ax1.plot(time_min, sigma_BN[:, 2], label=r'$\sigma_3$', linewidth=3)
    ax1.set_ylabel("MRP Orientation")
    ax1.set_title("Spacecraft Attitude (Inertial Frame)", pad=12)
    ax1.legend(loc="upper right")

    # Lower Subplot: Attitude Tracking Error (sigma_BR)
    ax2.plot(time_min, sigma_BR[:, 0], label=r'Error $\sigma_1$', linewidth=3)
    ax2.plot(time_min, sigma_BR[:, 1], label=r'Error $\sigma_2$', linewidth=3)
    ax2.plot(time_min, sigma_BR[:, 2], label=r'Error $\sigma_3$', linewidth=3)
    ax2.set_xlabel("Time [min]")
    ax2.set_ylabel("MRP Error")
    ax2.set_title("Attitude Tracking Error", pad=12)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    path_p1 = os.path.join(output_folder, "attitude_and_error.png")
    plt.savefig(path_p1, dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # PLOT 2: Angular Velocity and Angular Velocity Tracking Error (rad/s)
    # =========================================================================
    fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    
    # Upper Subplot: Inertial Angular Velocity (omega_BN_B)
    ax3.plot(time_min, omega_BN[:, 0], label=r'$\omega_1$', linewidth=3)
    ax3.plot(time_min, omega_BN[:, 1], label=r'$\omega_2$', linewidth=3)
    ax3.plot(time_min, omega_BN[:, 2], label=r'$\omega_3$', linewidth=3)
    ax3.set_ylabel("Angular Rate [rad/s]")
    ax3.set_title("Body Angular Velocity (Inertial Frame)", pad=12)
    ax3.legend(loc="upper right")

    # Lower Subplot: Angular Velocity Tracking Error (omega_BR_B)
    ax4.plot(time_min, omega_BR[:, 0], label=r'Error $\omega_1$', linewidth=3)
    ax4.plot(time_min, omega_BR[:, 1], label=r'Error $\omega_2$', linewidth=3)
    ax4.plot(time_min, omega_BR[:, 2], label=r'Error $\omega_3$', linewidth=3)
    ax4.set_xlabel("Time [min]")
    ax4.set_ylabel("Rate Error [rad/s]")
    ax4.set_title("Angular Velocity Tracking Error", pad=12)
    ax4.legend(loc="upper right")
    
    plt.tight_layout()
    path_p2 = os.path.join(output_folder, "angular_velocity_and_error.png")
    plt.savefig(path_p2, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Success! Plots saved to '{output_folder}/'.")

if __name__ == "__main__":
    plot_attitude_and_rate_performance()