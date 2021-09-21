#!/usr/bin/env python3
# Author: Phuthipong, 2021
# Organization: LASS UMASS Amherst
import sys
import time
import argparse
import ac_module
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from multiprocessing import Pool, freeze_support

np.set_printoptions(threshold=sys.maxsize)

AC_PATH = "/mnt/nfs/work1/shenoy/phuthipong/hge_data/ac-data/2019/"
EV_PATH = "/mnt/nfs/work1/shenoy/phuthipong/hge_data/ev-data/"
PAIR_FILE = "/mnt/nfs/work1/shenoy/phuthipong/hge_data/transformer-data/pair_2019.csv"
DETAIL_FILE = "/mnt/nfs/work1/shenoy/phuthipong/hge_data/code/optimization/meters_transformers_ev_0_3.csv"
PEAK_DAY_FILE = "/mnt/nfs/work1/shenoy/phuthipong/hge_data/code/optimization/daily_peak.csv"
PEAK_HOUR_FILE = "/mnt/nfs/work1/shenoy/phuthipong/hge_data/code/optimization/hourly_predict.csv"
TRANSFORMER_PATH = "/mnt/nfs/work1/shenoy/phuthipong/hge_data/transformer-data/2019/"
OUTPUT_PATH = "/mnt/nfs/work1/shenoy/phuthipong/hge_data/optimization-data/tmp/"
DEFAULT_SETPOINT = 74
DEFAULT_BATTERY_SIZE = 13.5*0.7 #kWh
DEFAULT_DISCHARGE_RATE = 5 #kWh
HOUR_LIMIT = 15
AC_EFFICIENCY = 3
TRANSFORMER_THRESHOLD = 0.65
STOP_THRESHOLD = -1
USE_PRED = True
BATTERY_COST = 1
EV_COST = 1
AC_COST = 1
DISABLED_BATTERY = False
DISABLED_EV = False
DISABLED_AC = False
AC_MAX_CHANGE_TEMP = 10
AC_DAILY_HOUR_LIMIT = 5
EV_DAILY_HOUR_LIMIT = 5
BATTERY_DAILY_HOUR_LIMIT = 5
DEFAULT_U_VALUE = 0.3

def run_optimize_one_transformer(transformer_demand, target_demand, number_of_home, batteries, acs, evs, resource_cost, weather, ac_max_reduction_temp=3):
    # Parameters
    ambient_temp = weather["ambient_temperature"]

    # battery level each home at time t
    battery_level = batteries["level"]
    battery_max_discharge_rate = batteries["discharge_rate"]
    battery_hour_limit = batteries["hour_limit"]

    # ac trace / each home / at time t
    ac_load = acs["load"]
    ac_temp = acs["setpoint"]
    ac_hour_limit = acs["hour_limit"]

    # ev tace / each home
    ev_load = evs["load"]
    ev_hour_limit = evs["hour_limit"]

    # binary variable
    controllable_battery = [(battery_hour_limit[i] > 0.001) and (batteries["daily_hour_limit"][i] > 0.001) for i in range(number_of_home)] # model.addVars(number_of_home, vtype=GRB.BINARY, name="controllable_battery")
    # make sure it's summer too
    controllable_ac = [(ac_hour_limit[i] > 0.001) and (acs["controllable"][i] == 1) and (acs["daily_hour_limit"][i] > 0.001) for i in range(number_of_home)] # model.addVars(number_of_home, vtype=GRB.BINARY, name="controllable_ac")
    controllable_ev = [(ev_hour_limit[i] > 0.001) and (evs["daily_hour_limit"][i] > 0.001) for i in range(number_of_home)] # model.addVars(number_of_home, vtype=GRB.BINARY, name="controllable_ev")

    battery_cost = resource_cost["battery"]
    ac_cost = resource_cost["ac"]
    ev_cost = resource_cost["ev"]

    ac_saving_per_temp = [ac_module.calc_load_per_hour(acs["area"][i], AC_EFFICIENCY, ambient_temp, ambient_temp-1, acs["u_values"][i]) for i in range(number_of_home)]
    max_ac_temp_change = [0]*number_of_home
    if acs["mode"] == "cooling":
        max_ac_temp_change = [max(0,min(ac_max_reduction_temp, ambient_temp-acs["setpoint"][i])) for i in range(number_of_home)]
    elif acs["mode"] == "heating":
        max_ac_temp_change = [max(0,min(ac_max_reduction_temp, acs["setpoint"][i]-ambient_temp)) for i in range(number_of_home)]        
    # create model
    model = gp.Model('OneTransformerModel')

    # Decision Variables
    output_battery = model.addVars(number_of_home, vtype=GRB.CONTINUOUS, name="output_battery")
    output_ac = model.addVars(number_of_home, vtype=GRB.CONTINUOUS, name="output_ac")
    output_ev = model.addVars(number_of_home, vtype=GRB.CONTINUOUS, name="output_ev")
    temperature_diff_ac = model.addVars(number_of_home, vtype=GRB.INTEGER, name="temperature_diff_ac")

    # Constraints
    sum_battery = gp.quicksum(output_battery[i]*controllable_battery[i] for i in range(number_of_home))
    sum_ac = gp.quicksum(output_ac[i]*controllable_ac[i] for i in range(number_of_home))
    sum_ev = gp.quicksum(output_ev[i]*controllable_ev[i] for i in range(number_of_home))
    model.addConstr((sum_battery + sum_ac + sum_ev) >= (transformer_demand - target_demand))

    # # ac constraint, NEED UPDATE
    # model.addConstrs(output_ac[i] <= ac_load[i] for i in range(number_of_home)
    model.addConstrs(output_ac[i] == (ac_saving_per_temp[i]*temperature_diff_ac[i]) for i in range(number_of_home))
    model.addConstrs(temperature_diff_ac[i] <= max_ac_temp_change[i] for i in range(number_of_home))
    # must be positive
    model.addConstrs(output_ac[i] >= 0 for i in range(number_of_home))

    # # ev constraint
    model.addConstrs(output_ev[i] <= ev_load[i] for i in range(number_of_home))
    # must be positive
    model.addConstrs(output_ev[i] >= 0 for i in range(number_of_home))

    # # battery constraint
    # # less than discharge rate
    model.addConstrs(output_battery[i] <= battery_max_discharge_rate[i] for i in range(number_of_home))
    # # less than energy in battery
    model.addConstrs(output_battery[i] <= battery_level[i]*12 for i in range(number_of_home))
    # must be positive
    model.addConstrs(output_battery[i] >= 0 for i in range(number_of_home))

    # objective minimize
    sum_battery_obj = gp.quicksum(output_battery[i]*battery_cost for i in range(number_of_home))
    sum_ac_obj = gp.quicksum(output_ac[i]*ac_cost for i in range(number_of_home))
    sum_ev_obj = gp.quicksum(output_ev[i]*ev_cost for i in range(number_of_home))
    model.setObjective(sum_battery_obj + sum_ac_obj + sum_ev_obj)

    model.optimize()

    # save results
    if model.solCount == 0:
        print("Model is infeasible, use all resources that has left")
        batteries["used"] = [min(battery_level[i]*12,battery_max_discharge_rate[i]) if controllable_battery[i] > 0.001 else 0 for i in range(number_of_home)]
        acs["used"] = [ac_saving_per_temp[i]*max_ac_temp_change[i] if controllable_ac[i] > 0.001 else 0 for i in range(number_of_home)]
        evs["used"] = [ev_load[i] if controllable_ev[i] > 0.001 else 0 for i in range(number_of_home)]
        
    else:
        for i in range(number_of_home):
            print("{} {} {}".format(output_battery[i].x, output_ac[i].x, output_ev[i].x))

        batteries["used"] = [output_battery[i].x for i in range(number_of_home)]
        acs["used"] = [output_ac[i].x for i in range(number_of_home)]
        evs["used"] = [output_ev[i].x for i in range(number_of_home)]

        print("max_ac_temp_change: {}".format(max_ac_temp_change))
        print("temperature_diff_ac: {}".format(temperature_diff_ac))

    batteries["level"] = [batteries["level"][i]-(batteries["used"][i]/12) for i in range(number_of_home)]
    batteries["hour_limit"] = [batteries["hour_limit"][i]-((batteries["used"][i]>0)/12) for i in range(number_of_home)]
    acs["hour_limit"] = [acs["hour_limit"][i]-((acs["used"][i]>0)/12) for i in range(number_of_home)]
    evs["hour_limit"] = [evs["hour_limit"][i]-((evs["used"][i]>0)/12) for i in range(number_of_home)]

    batteries["daily_hour_limit"] = [batteries["daily_hour_limit"][i]-((batteries["used"][i]>0)/12) for i in range(number_of_home)]
    acs["daily_hour_limit"] = [acs["daily_hour_limit"][i]-((acs["used"][i]>0)/12) for i in range(number_of_home)]
    evs["daily_hour_limit"] = [evs["daily_hour_limit"][i]-((evs["used"][i]>0)/12) for i in range(number_of_home)]

    return (batteries, acs, evs)

def run_everybody_shave():
    # LOAD TRANSFORMER-HOUSE PAIR
    pair_df = load_transformer_house_pair(PAIR_FILE)
    # LOAD DETAIL AND HOUSE-EV 
    detail_df = load_transformer_house_detail(DETAIL_FILE)

    transformer_list = pair_df.groupby("Transformer").count().index.values

    # LOAD PEAK DAY AND TIME
    peak_days_df = load_predicted_peak_day(PEAK_DAY_FILE)
    peak_hours_df = load_predicted_peak_hour(PEAK_HOUR_FILE)
    temperature_list = resampling_temperature(peak_hours_df)

    if USE_PRED:
        peak_days_list = peak_days_df[peak_days_df["bin_pred"] == True].index.date
    else:
        peak_days_list = peak_days_df[peak_days_df["bin_gt"] == True].index.date

    gt_peak_hour, pred_peak_hour  = find_peak_time_of_each_day(peak_hours_df, top_k=5)

    if USE_PRED:
        peak_hour = pred_peak_hour
    else:
        peak_hour = gt_peak_hour

    # SET RESOURCE PRIORITY (FROM ARGUMENTS)
    resource_cost = {"battery":BATTERY_COST, "ev":EV_COST, "ac":AC_COST}

    # set summer and winter months
    summer_months = [6, 7, 8]
    winter_months = [12, 1, 2]

    # RUN IN MULTITHREAD
    with Pool() as pool:
        pool.starmap(shave_one_transformer, [(tran_id, detail_df, pair_df, peak_days_list, peak_hour, summer_months, winter_months, temperature_list, resource_cost) for tran_id in transformer_list])

    # DONE

def shave_one_transformer(tran_id, detail_df, pair_df, peak_days_list, peak_hour, summer_months, winter_months, temperature_list, resource_cost):
    # RUN ONE TRANSFORMER AT A TIME
    print(tran_id)
    # LOAD TRANSFORMER TRACE
    transformer_trace = load_transformer_trace(tran_id)
    # LOAD TRANSFORMER CAPACITY
    transformer_capacity = get_transformer_capcity(tran_id, detail_df)  

    # load all house, ac, ev trace data
    house_ids = find_houses_in_tranformer(pair_df, tran_id)
    ac_traces = load_ac_traces(house_ids)
    ev_traces = load_ev_traces(house_ids, detail_df)
    battery_available = load_battery_available(house_ids, detail_df)
    # load house area
    house_areas = get_house_area(house_ids, detail_df)
    setpoints = get_setpoint(house_ids, detail_df)
    u_values = get_uvalues(house_ids, detail_df)

    number_of_home = len(house_ids)
    # create battery for each house
    batteries = create_batteries(number_of_home, HOUR_LIMIT, DEFAULT_BATTERY_SIZE, DEFAULT_DISCHARGE_RATE)
    acs = create_acs(number_of_home, HOUR_LIMIT, setpoints=setpoints, uvalues=u_values)
    evs = create_evs(number_of_home, HOUR_LIMIT)

    print("SETPOINTS")
    print(acs["setpoint"])
    # prepare output string
    output_string = 'dt,demand,target,t_limit,ac_load,ac_calc_load,ev_load,battery_shaved,ac_shaved,ev_shaved,sum_shaved,is_peak,is_shave,ac_shaved_count,ev_shaved_count,battery_shaved_count\n'

    # LOOP ONE TIME STEP AT A TIME (HOUR LEVEL)
    i_counter = 0
    current_month = -1
    current_day = -1
    is_peak = False
    is_shave = False
    # used = True
    for index, row in transformer_trace.iterrows():

        # t4 = time.time()
        if index.month in summer_months:
            transformer_limit = transformer_capacity*0.9
            acs["mode"] = "cooling"
        elif index.month in winter_months:
            transformer_limit = transformer_capacity*0.95
            acs["mode"] = "heating"
        else:
            acs["mode"] = "off"

        
        # UPDATE NEW MONTH, RESET QUOTA
        if current_month != index.month:
            if DISABLED_EV:
                evs["hour_limit"] = [0]*number_of_home                
            else:
                evs["hour_limit"] = [HOUR_LIMIT]*number_of_home
            # batteries["hour_limit"] = [HOUR_LIMIT]*number_of_home
            if DISABLED_BATTERY:
                batteries["hour_limit"] = [0]*number_of_home
            else:
                batteries["hour_limit"] = []
                for h_id in house_ids:
                    if battery_available[h_id]:
                        batteries["hour_limit"].append(HOUR_LIMIT)
                    else:
                        batteries["hour_limit"].append(0)
            if DISABLED_AC:
                acs["hour_limit"] = [0]*number_of_home
            else:
                acs["hour_limit"] = [HOUR_LIMIT]*number_of_home
            current_month = index.month

        # UPDATE NEW DAY, RESET SOME RESOURCES
        if current_day != index.day:
            batteries["level"] = []
            for h_id in house_ids:
                if battery_available[h_id]:
                    batteries["level"].append(DEFAULT_BATTERY_SIZE)
                else:
                    batteries["level"].append(0)
            current_day = index.day

            batteries["daily_hour_limit"] = [BATTERY_DAILY_HOUR_LIMIT]*number_of_home
            acs["daily_hour_limit"] = [AC_DAILY_HOUR_LIMIT]*number_of_home
            evs["daily_hour_limit"] = [EV_DAILY_HOUR_LIMIT]*number_of_home

        # INIT RESOURCE
        ambient_temp = temperature_list[i_counter]
        acs["load"] = []
        evs["load"] = []
        acs["load_calc"] = []
        acs["area"] = []
        acs["controllable"] = []
        for m, h_id in enumerate(house_ids):

            acs["load"].append(ac_traces[h_id][i_counter])
            ac_load = 0
            if acs["mode"] == "cooling":
                ac_load = ac_module.calc_load_per_hour(house_areas[h_id], AC_EFFICIENCY, ambient_temp, acs["setpoint"][m], acs["u_values"][m])
            elif acs["mode"] == "heating":
                ac_load = ac_module.calc_load_per_hour(house_areas[h_id], AC_EFFICIENCY, acs["setpoint"][m], ambient_temp, acs["u_values"][m])
            acs["load_calc"].append(ac_load)
            acs["area"].append(house_areas[h_id])
            if index.month in summer_months or index.month in winter_months:
                acs["controllable"].append(1)
            else:
                acs["controllable"].append(0)

            if h_id in ev_traces:
                evs["load"].append(ev_traces[h_id][i_counter])
            else:
                evs["load"].append(0)

        # reset used             
        batteries["used"] = [0]*number_of_home
        acs["used"] = [0]*number_of_home
        evs["used"] = [0]*number_of_home

        # LOAD TRANSFORMER DEMAND
        transformer_demand = transformer_trace[transformer_trace.index == index]["power"].values[0] + sum(evs["load"])
        target_demand = transformer_limit*TRANSFORMER_THRESHOLD

        weather = {"ambient_temperature":ambient_temp}

        # CHECK WHETHER IT IS TIME TO SHAVE PEAK OR NOT
        # IF PEAK DAY AND PEAK HOUR
        is_peak = False
        is_shave = False
        if index.date() in peak_days_list and peak_hour[i_counter] == True:
            is_peak = True
            # LOAD NECESSARY DATA AT EACH TIME STEP
            # RUN OPTIMIZE ONLY WHEN DEMAND > TARGET
            if transformer_demand > target_demand:
                is_shave = True
                if STOP_THRESHOLD < 0:
                    batteries, acs, evs = run_optimize_one_transformer(transformer_demand, target_demand, number_of_home, batteries, acs, evs, resource_cost, weather, AC_MAX_CHANGE_TEMP)
                else:
                    batteries, acs, evs = run_optimize_one_transformer(transformer_demand, transformer_demand*(1-STOP_THRESHOLD), number_of_home, batteries, acs, evs, resource_cost, weather, AC_MAX_CHANGE_TEMP)

        output_string += string_output(transformer_demand, target_demand, batteries, acs, evs, tran_id, index, is_peak, is_shave, transformer_limit)
        i_counter += 1

    # SAVE OUTPUT
    with open('{}{}.csv'.format(OUTPUT_PATH, tran_id), 'w') as f:
        f.write(output_string)
        


def load_transformer_house_pair(path):
    return pd.read_csv(path)

def load_transformer_house_detail(path):
    df = pd.read_csv(path)
    df["Transformer Shop Number"] = [s.replace('/','_') for s in df["Transformer Shop Number"].values]
    return df

def load_transformer_trace(tran_id):
    # MAKE IT HOURLY AND FILL THE BLANK
    load_df = pd.read_csv("{}{}.csv".format(TRANSFORMER_PATH, tran_id), index_col=0)
    load_df.index = pd.to_datetime(load_df.index)
    load_df = load_df.groupby(by=load_df.index.date).resample('5T').max().fillna(method='ffill').fillna(method='bfill')
    load_df.index = load_df.index.get_level_values('datetime')
    load_df.index = load_df.index.tz_localize('UTC').tz_convert('US/Eastern')
    # remove first 5 hours
    # load_df = load_df.iloc[12*5:]
    print("LOAD TRAN_ID:{} ROW COUNT:{}".format(tran_id, load_df.shape[0]))
    return load_df

def find_houses_in_tranformer(pair_df, tran_id):
    # RETURN LIST OF HOUSES IN TRAN_ID
    return pair_df[pair_df["Transformer"] == tran_id]["ANONID"].values

def load_ac_traces(house_ids):
    d = {}
    for h_id in house_ids:
        df = pd.read_csv("{}{}.csv".format(AC_PATH, h_id), index_col=0)
        df.index = pd.to_datetime(df.index, unit='s')
        df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
        # convert from hourly to 5 minutes
        df = df.resample('5T').pad()
        d[h_id] = df["AC_power"].values
    return d

def load_ev_traces(house_ids, house_pair):
    d = {}
    for h_id in house_ids:
        ev_filename = house_pair[house_pair["ANONID"] == h_id]["ev_file"].values[0]
        if pd.isnull(ev_filename):
            continue
        df = pd.read_csv("{}{}".format(EV_PATH, ev_filename), index_col=0)   
        d[h_id] = df["power"].values
    return d

def load_battery_available(house_ids, house_pair):
    d = {}
    for h_id in house_ids:
        has_battery = house_pair[house_pair["ANONID"] == h_id]["has_battery"].values[0]      
        d[h_id] = has_battery
    return d

def create_batteries(number_of_home, hour_limit, default_battery_size, default_discharge_rate):
    d = {"hour_limit":[hour_limit]*number_of_home, "size":[default_battery_size]*number_of_home, "level":[default_battery_size]*number_of_home}
    d["discharge_rate"] = [default_discharge_rate]*number_of_home
    return d

def create_acs(number_of_home, hour_limit, default_setpoint=74, setpoints=None, uvalues=None):
    if setpoints is None:
        d = {"hour_limit":[hour_limit]*number_of_home, "setpoint": [default_setpoint]*number_of_home, "u_values":[DEFAULT_U_VALUE]*number_of_home}
    else:
        d = {"hour_limit":[hour_limit]*number_of_home, "setpoint": setpoints, "u_values":uvalues}
    return d

def create_evs(number_of_home, hour_limit):
    d = {"hour_limit":[hour_limit]*number_of_home}
    return d

def load_predicted_peak_day(file):
    df = pd.read_csv(file, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize('US/Eastern')
    return df

def load_predicted_peak_hour(file):
    df = pd.read_csv(file, index_col=0)
    df.index = pd.to_datetime(df.index, unit='s')
    df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
    return df

def resampling_temperature(peak_hours_df):

    temperature_df = peak_hours_df.loc[:,"temperature"].resample('5T').pad()
    return temperature_df.values

def get_transformer_capcity(tran_id, detail_df):
    return detail_df[detail_df["Transformer Shop Number"] == tran_id]["Transformer kVA"].values[0]

def get_house_area(house_ids, detail_df):
    d = {}
    for h_id in house_ids:        
        d[h_id] = detail_df[detail_df["ANONID"] == h_id]["area"].values[0]
    return d

def get_setpoint(house_ids, detail_df):
    arr = []
    for h_id in house_ids:        
        arr.append(detail_df[detail_df["ANONID"] == h_id]["setpoint"].values[0])
    return arr   

def get_uvalues(house_ids, detail_df):
    arr = []
    for h_id in house_ids:   
        year_built = detail_df[detail_df["ANONID"] == h_id]["built"].values[0]
        u = 1.25
        if year_built >= 2000: # good insulation
            u = 0.25
        elif year_built >= 1970: # ok insulation
            u = 0.5
        elif year_built >= 1920: # bad insulation
            u = 0.75

        arr.append(u)
    return arr  

def find_peak_time_of_each_day(peak_hours_df, top_k=5):
    gt_peak = np.full((peak_hours_df.shape[0]), False)
    pred_peak = np.full((peak_hours_df.shape[0]), False)

    for i in range(0, peak_hours_df.shape[0], 24):
        gt = np.array(peak_hours_df["gt"].values[i:i+24])
        pd = np.array(peak_hours_df["pred"].values[i:i+24])
        gt_idx = gt.argsort()[::-1][:top_k] + i
        pd_idx = pd.argsort()[::-1][:top_k] + i
        pred_peak[pd_idx] = True
        gt_peak[gt_idx] = True

    # scale 5 minutes 
    five_min_pred_peak = []
    five_min_gt_peak = []
    for i in range(len(pred_peak)):
        five_min_pred_peak += [pred_peak[i]] * 12
        five_min_gt_peak += [gt_peak[i]] * 12

    return (five_min_gt_peak, five_min_pred_peak) # (gt_peak, pred_peak)

def string_output(transformer_demand, target_demand, batteries, acs, evs, tran_id, dt, is_peak, is_shave, transformer_limit):
    sum_battery = sum(batteries["used"])
    sum_ac = sum(acs["used"])
    sum_ev = sum(evs["used"])
    all_sum = sum_battery + sum_ac + sum_ev
    evs_load = sum(evs["load"])
    acs_load = sum(acs["load"])
    acs_calc_load = sum(acs["load_calc"])

    ac_shaved_count = sum([1 if b > 0 else 0 for b in batteries["used"]])
    ev_shaved_count = sum([1 if b > 0 else 0 for b in evs["used"]])
    battery_shaved_count = sum([1 if b > 0 else 0 for b in acs["used"]])
    to_write = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(dt, transformer_demand, target_demand, transformer_limit, acs_load, acs_calc_load, evs_load, sum_battery, sum_ac, sum_ev, all_sum, is_peak, is_shave, ac_shaved_count,ev_shaved_count,battery_shaved_count)

    return to_write

def write_output(transformer_demand, target_demand, batteries, acs, evs, tran_id, dt):
    to_write = string_output(transformer_demand, target_demand, batteries, acs, evs, tran_id, dt)
    with open('{}{}.csv'.format(OUTPUT_PATH, tran_id), 'a') as f:
        f.write(to_write)

def write_newfile(tran_id):
    with open('{}{}.csv'.format(OUTPUT_PATH, tran_id), 'w') as f:
        f.write('demand,target,ev_load,ac_load,battery_shaved,ac_shaved,ev_shaved,sum_shaved\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate readings for each transformer.')

    parser.add_argument('--ac_path', dest='ac_path', action='store', help='ac_path', required=False)
    parser.add_argument('--ev_path', dest='ev_path', action='store', help='ac_path', required=False)
    parser.add_argument('--detail_file', dest='detail_file', action='store', help='detail_file', required=False)
    parser.add_argument('--peakday_file', dest='peakday_file', action='store', help='peakday_file', required=False)
    parser.add_argument('--peakhour_file', dest='peakhour_file', action='store', help='peakhour_file', required=False)
    parser.add_argument('--output_path', dest='output_path', action='store', help='output_path', required=False)
    parser.add_argument('--use_pred', dest='use_pred', action='store_true')
    parser.add_argument('--use_gt', dest='use_pred', action='store_false')
    parser.set_defaults(use_pred=True)
    parser.add_argument('--trans_load_thresold', dest='trans_load_thresold', type=float)
    parser.add_argument('--stop_thresold', dest='stop_thresold', type=float)

    parser.add_argument('--battery_cost', dest='battery_cost', type=float)
    parser.add_argument('--ev_cost', dest='ev_cost', type=float)
    parser.add_argument('--ac_cost', dest='ac_cost', type=float)

    parser.add_argument('--disable_battery', dest='disable_battery', action='store_true')
    parser.add_argument('--disable_ev', dest='disable_ev', action='store_true')
    parser.add_argument('--disable_ac', dest='disable_ac', action='store_true')

    parser.add_argument('--ac_max_change_temp', dest='ac_max_change_temp', type=float)

    parser.add_argument('--ac_daily_hour_limit', dest='ac_daily_hour_limit', type=float)
    parser.add_argument('--ev_daily_hour_limit', dest='ev_daily_hour_limit', type=float)
    parser.add_argument('--battery_daily_hour_limit', dest='battery_daily_hour_limit', type=float)

    args = parser.parse_args()

    if args.ac_path is not None:
        AC_PATH = args.ac_path

    if args.ev_path is not None:
        EV_PATH = args.ev_path

    if args.detail_file is not None:
        DETAIL_FILE = args.detail_file

    if args.peakday_file is not None:
        PEAK_DAY_FILE = args.peakday_file

    if args.peakhour_file is not None:
        PEAK_HOUR_FILE = args.peakhour_file

    if args.output_path is not None:
        OUTPUT_PATH = args.output_path

    if args.use_pred is not None:
        USE_PRED = args.use_pred

    if args.trans_load_thresold is not None:
        TRANSFORMER_THRESHOLD = args.trans_load_thresold
    print("trans_load_thresold={}".format(TRANSFORMER_THRESHOLD))

    if args.stop_thresold is not None:
        STOP_THRESHOLD = args.stop_thresold
    print("stop_thresold={}".format(STOP_THRESHOLD))

    if args.battery_cost is not None:
        BATTERY_COST = args.battery_cost

    if args.ac_cost is not None:
        AC_COST = args.ac_cost

    if args.ev_cost is not None:
        EV_COST = args.ev_cost

    print("battery_cost: {}, ac_cost: {}, ev_cost: {}".format(BATTERY_COST, AC_COST, EV_COST))

    if args.disable_battery is not None:
        DISABLED_BATTERY = args.disable_battery

    if args.disable_ev is not None:
        DISABLED_EV = args.disable_ev

    if args.disable_battery is not None:
        DISABLED_AC = args.disable_ac

    print("disable - battery: {}, ev: {}, ac: {}".format(DISABLED_BATTERY, DISABLED_EV, DISABLED_AC))

    if args.ac_max_change_temp is not None:
        AC_MAX_CHANGE_TEMP = args.ac_max_change_temp

    print("ac_max_change_temp={}".format(AC_MAX_CHANGE_TEMP))

    if args.ac_daily_hour_limit is not None:
        AC_DAILY_HOUR_LIMIT = args.ac_daily_hour_limit

    if args.ev_daily_hour_limit is not None:
        EV_DAILY_HOUR_LIMIT = args.ev_daily_hour_limit

    if args.battery_daily_hour_limit is not None:
        BATTERY_DAILY_HOUR_LIMIT = args.battery_daily_hour_limit

    print("ac_daily_hour_limit: {}, ev_daily_hour_limit: {}, battery_daily_hour_limit: {}".format(AC_DAILY_HOUR_LIMIT, EV_DAILY_HOUR_LIMIT, BATTERY_DAILY_HOUR_LIMIT))

    freeze_support()
    run_everybody_shave()
