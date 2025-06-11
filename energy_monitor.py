"""
File description:

Auxiliary file for measuring the execution time and CPU utilization.
"""
import threading
import time
import psutil

# Global variables for resource usage measurement
thread_running = None  # Flag to control the measurement thread
cpu_thread = None  # Thread object for CPU measurement
start_time = None  # Start time of the measurement
avg_cpu_percentage = None  # Average CPU percentage during the measurement

#Data related to the machine used to calculate energy
MAX_POWER=135
STATIC_POWER=93.7
TARGET_MIPS=5320
LOAD_RELATED_POWER=MAX_POWER-STATIC_POWER

#Data related to the machine where experiments are running
MACHINE_MIPS=138039

#Energy in Kj
def calculate_energy(util_percent, time):
    return MAX_POWER * (MACHINE_MIPS*util_percent*time/TARGET_MIPS)*0.001