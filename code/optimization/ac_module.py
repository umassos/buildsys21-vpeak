
# NikkO AC Module

import math

def calc_load_per_hour(area, AC_efficiency, ambient_temp, setpoint_temp, U=0.3):
	# print("{} {} {} {}".format(area, AC_efficiency, ambient_temp, setpoint_temp))
	width = length = math.sqrt(area)
	A = estimate_A(width=feet_to_meter(width), length=feet_to_meter(length), height=feet_to_meter(8))
	cooling_load_per_hour = estimate_cooling_load(U, A, F_to_C(ambient_temp), F_to_C(setpoint_temp))
	# AC_efficiency = estimate_AC_efficiency(6000,600)
	load_needed_per_hour = cooling_load_per_hour / AC_efficiency
	# Remove negative load (outside is colder)
	if load_needed_per_hour < 0:
		load_needed_per_hour = 0
	# RETURN IN kWh
	return load_needed_per_hour/1000 

def estimate_A(width, length, height):
	# unit is meter
	# 4 sides + roof
	return (width * height) * 2 + (length * height) * 2 + (width * length)

def estimate_cooling_load(U, A, To, Ti):
	# Q = U * A * (To - Ti)
	return U * A * (To -Ti)

def estimate_AC_efficiency(BTU, watts):
	# Efficiency at 100% is BTU 12000 using 3516 Watts
	return 3516/watts * BTU/12000

def F_to_C(F):
	return (5/9) * (F - 32)

def C_to_F(C):
	return (C*9/5) + 32

def feet_to_meter(ft):
	return ft*0.3048