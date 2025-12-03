#!/usr/bin/env python3
"""
Generate comprehensive graphs from SwingPal CSV data.
Usage: python generate_graphs.py <input.csv> <output.png>
"""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Use non-interactive backend
matplotlib.use('Agg')

def parse_csv(filename):
    """Parse CSV file and return data arrays."""
    times = []
    yaws = []
    pitches = []
    rolls = []
    accel_x = []
    accel_y = []
    accel_z = []
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in reader:
            if len(row) >= 7:
                try:
                    time_ms = float(row[0])
                    time_s = time_ms / 1000.0  # Convert to seconds
                    yaw = float(row[1])
                    pitch = float(row[2])
                    roll = float(row[3])
                    ax = float(row[4])
                    ay = float(row[5])
                    az = float(row[6])
                    
                    times.append(time_s)
                    yaws.append(yaw)
                    pitches.append(pitch)
                    rolls.append(roll)
                    accel_x.append(ax)
                    accel_y.append(ay)
                    accel_z.append(az)
                except (ValueError, IndexError):
                    continue
    
    return np.array(times), np.array(yaws), np.array(pitches), np.array(rolls), \
           np.array(accel_x), np.array(accel_y), np.array(accel_z)

def calculate_net_acceleration(ax, ay, az):
    """Calculate net acceleration."""
    return np.sqrt(ax**2 + ay**2 + az**2)

def calculate_net_jerk(net_acc, times):
    """Calculate net jerk (derivative of net acceleration)."""
    jerk = np.diff(net_acc) / np.diff(times)
    jerk_times = times[1:]
    return jerk, jerk_times

def calculate_angular_velocities(yaws, pitches, rolls, times):
    """Calculate angular velocities (derivatives of angles)."""
    yaw_rate = np.diff(yaws) / np.diff(times)
    pitch_rate = np.diff(pitches) / np.diff(times)
    roll_rate = np.diff(rolls) / np.diff(times)
    deriv_times = times[1:]
    return yaw_rate, pitch_rate, roll_rate, deriv_times

def generate_graphs(input_file, output_file):
    """Generate all graphs and save as PNG."""
    # Parse data
    times, yaws, pitches, rolls, ax, ay, az = parse_csv(input_file)
    
    if len(times) == 0:
        print("Error: No valid data found in CSV file")
        return False
    
    # Calculate derived quantities
    net_acc = calculate_net_acceleration(ax, ay, az)
    jerk, jerk_times = calculate_net_jerk(net_acc, times)
    yaw_rate, pitch_rate, roll_rate, deriv_times = calculate_angular_velocities(yaws, pitches, rolls, times)
    
    # Normalize times to start from 0
    start_time = times[0]
    times_norm = times - start_time
    jerk_times_norm = jerk_times - start_time
    deriv_times_norm = deriv_times - start_time
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 20))
    fig.patch.set_facecolor('#ffffff')
    
    # Use a modern style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Net Acceleration vs Time
    ax1 = plt.subplot(5, 1, 1)
    ax1.plot(times_norm, net_acc, color='#10b981', linewidth=2, label='Net Acceleration')
    ax1.fill_between(times_norm, net_acc, alpha=0.15, color='#10b981')
    ax1.set_xlabel('Time (s)', fontsize=11, fontweight='600')
    ax1.set_ylabel('Acceleration (m/s²)', fontsize=11, fontweight='600')
    ax1.set_title('Net Acceleration vs Time', fontsize=12, fontweight='600', pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)
    
    # 2. Net Jerk vs Time
    ax2 = plt.subplot(5, 1, 2)
    ax2.plot(jerk_times_norm, jerk, color='#3b82f6', linewidth=2, label='Net Jerk')
    ax2.fill_between(jerk_times_norm, jerk, alpha=0.15, color='#3b82f6')
    ax2.set_xlabel('Time (s)', fontsize=11, fontweight='600')
    ax2.set_ylabel('Jerk (m/s³)', fontsize=11, fontweight='600')
    ax2.set_title('Net Jerk vs Time', fontsize=12, fontweight='600', pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)
    
    # 3. Accelerations (X, Y, Z) vs Time
    ax3 = plt.subplot(5, 1, 3)
    ax3.plot(times_norm, ax, color='#ef4444', linewidth=2, label='Accel X', alpha=0.9)
    ax3.plot(times_norm, ay, color='#10b981', linewidth=2, label='Accel Y', alpha=0.9)
    ax3.plot(times_norm, az, color='#3b82f6', linewidth=2, label='Accel Z', alpha=0.9)
    ax3.set_xlabel('Time (s)', fontsize=11, fontweight='600')
    ax3.set_ylabel('Acceleration (m/s²)', fontsize=11, fontweight='600')
    ax3.set_title('Accelerations (X, Y, Z) vs Time', fontsize=12, fontweight='600', pad=10)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=9)
    
    # 4. Angles (Yaw, Pitch, Roll) vs Time
    ax4 = plt.subplot(5, 1, 4)
    ax4.plot(times_norm, yaws, color='#f59e0b', linewidth=2, label='Yaw', alpha=0.9)
    ax4.plot(times_norm, pitches, color='#8b5cf6', linewidth=2, label='Pitch', alpha=0.9)
    ax4.plot(times_norm, rolls, color='#ec4899', linewidth=2, label='Roll', alpha=0.9)
    ax4.set_xlabel('Time (s)', fontsize=11, fontweight='600')
    ax4.set_ylabel('Angle (degrees)', fontsize=11, fontweight='600')
    ax4.set_title('Angles (Yaw, Pitch, Roll) vs Time', fontsize=12, fontweight='600', pad=10)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=9)
    
    # 5. Angular Velocities vs Time
    ax5 = plt.subplot(5, 1, 5)
    ax5.plot(deriv_times_norm, yaw_rate, color='#f59e0b', linewidth=2, label='Yaw Rate', alpha=0.9)
    ax5.plot(deriv_times_norm, pitch_rate, color='#8b5cf6', linewidth=2, label='Pitch Rate', alpha=0.9)
    ax5.plot(deriv_times_norm, roll_rate, color='#ec4899', linewidth=2, label='Roll Rate', alpha=0.9)
    ax5.set_xlabel('Time (s)', fontsize=11, fontweight='600')
    ax5.set_ylabel('Angular Velocity (deg/s)', fontsize=11, fontweight='600')
    ax5.set_title('Angular Velocities (Yaw, Pitch, Roll) vs Time', fontsize=12, fontweight='600', pad=10)
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='upper right', fontsize=9)
    
    # Adjust layout
    plt.tight_layout(pad=2.0)
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Graphs saved to {output_file}")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_graphs.py <input.csv> <output.png>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    success = generate_graphs(input_file, output_file)
    sys.exit(0 if success else 1)
