
import csv
import numpy as np
import pandas as pd

np.random.seed(0)

#-----------------------Read csv files
def Read_CSV(new_var, file_name, file_dtype, file_delimiter):

    file_rows = new_var.shape[0]
    file_columns = new_var.shape[1]    
    
    Index = 0
    Output = np.zeros([file_rows, file_columns], dtype = file_dtype)
    with open(file_name, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter = file_delimiter) 
    
        for row in reader:
            Output[Index] = row
            Index = Index + 1
    
    return Output
                

#---------------- Inputs
Num_of_blood_groups = 8
Max_Shelf_life = 42
Duration_of_Scenario = 6
num_of_Years = 3
Active_Scenarios = 2 * num_of_Years + 1

Flag_stock = 1 # Flag_stock = 1: data, = 2: variable, 
Flag_life = 2  # Flag_life = 1: percentages, = 2: pre-calculated values.

#---------------- Cost vector: first eight for order of one unit from different groups, 
#                              follwoed by inventory and outdate cost

Cost_vector = [24, 23, 22, 21, 19, 18, 17, 16, 1, 13]
Order_Cost_Vector = Cost_vector[0:8]
Emergency_Cost_Vector = Cost_vector[0:8]
for i in range(8):
    Emergency_Cost_Vector[i] = Emergency_Cost_Vector[i] + 1

#---------------- Compatibility matrix
Compatibility = np.zeros([Num_of_blood_groups, Num_of_blood_groups], dtype=np.int64)
file_name = 'Data/Compatibility.txt'
Compatibility = Read_CSV(Compatibility, file_name, int, ',')                  

#---------------- Empirical choices
Empirical = np.zeros([Num_of_blood_groups, Num_of_blood_groups], dtype=np.int64)
file_name = 'Data/Empirical.txt'
Empirical = Read_CSV(Empirical, file_name, float, ',')

#---------------- Over_order choices
Over_order = np.zeros([Num_of_blood_groups, Num_of_blood_groups], dtype=np.int64)
file_name = 'Data/Over_order.txt'
Over_order = Read_CSV(Over_order, file_name, float, ',')

#-----------------------------------------
Blood_Demands = np.zeros([Num_of_blood_groups, 6], dtype=np.int64)
file_name = 'Data/Total_Demands.txt'
Blood_Demands = Read_CSV(Blood_Demands, file_name, int, '\t')

#-----------------------------------------
DayPercentage_0 = np.zeros([8,7], dtype = float)
file_name = 'Data/Day_Percentage.txt'
DayPercentage_0 = Read_CSV(DayPercentage_0, file_name, float, '\t')

#-----------------------------------------
Week_Coefficient = np.zeros([52,1], dtype = float)
file_name = 'Data/Week_Coefficient.txt'
Week_Coefficient = Read_CSV(Week_Coefficient, file_name, float, '\t')

#-----------------------------------------
DayPercentage = np.zeros([8,364], dtype = float)
for i in range(52):
    DayPercentage[:,7*i:7*i+7] = (DayPercentage_0/52) * Week_Coefficient[i]
      

#-----------------------------------------    
Hospital_sizes = [5,4,3,2,1,0]

for h in Hospital_sizes: 
        
    #---------------- Initial Inventory  
    def inputData(Num_of_blood_groups, Max_Shelf_life, num_of_Years, Duration_of_Scenario, Active_Scenarios):
        
              
        Initial_Inventory = pd.read_excel('Data/Initial_Inventory_%s.xlsx' % (h), index_col=0, skiprows = 0)  
        Initial_Inventory = Initial_Inventory.to_numpy() 
              
        RealizedDemands = pd.read_excel('Data/RealizedDemands_%s.xlsx' % (h), index_col=0, skiprows = 0)  
        RealizedDemands = RealizedDemands.to_numpy() 
        
        Demand0 = RealizedDemands[:, 7]
            
        for i in range(num_of_Years):
            preDemand[i,:,:] = pd.read_excel('Data/PreDemand_%s.xlsx' % (h), str(i+1), index_col=0, skiprows = 0)  
        
        
        for i in range(Active_Scenarios - 1):
            Index = int(np.ceil((i + 1)/2))
            for k in range(Duration_of_Scenario):
                for j in range(Num_of_blood_groups):
                    if i % 2 == 0:
                        Demand[j, k+1, 1, i] = preDemand[Index-1, j, k+1]
                        Demand_II[j, k+1, i] = preDemand[Index-1, j, k+1]
                    else:
                        Demand[j, k+1, 1, i] = preDemand[Index-1, j, k + 7+1]
                        Demand_II[j, k+1, i] = preDemand[Index-1, j, k + 7+1]
        
        for k in range(Duration_of_Scenario):
            for j in range(Num_of_blood_groups):
                Demand[j, k+1, 1, Active_Scenarios - 1] = RealizedDemands[j, k+1]
                Demand_II[j, k+1, Active_Scenarios - 1] = RealizedDemands[j, k+1]
                
        for i in range(Active_Scenarios):
            for k in range(Duration_of_Scenario):
                for j in range(Num_of_blood_groups):
                    Demand[j, k+1, 0, i] = min(Demand[j, k+1, 1, :])
                    Demand[j, k+1, 2, i] = max(Demand[j, k+1, 1, :])
        
        return Initial_Inventory, RealizedDemands, Demand0, preDemand, Demand, Demand_II
    
    #----------------------------------------        
    Stock_M = pd.read_excel('Data/Stock_M_%s.xlsx' % (h), index_col=None, header = None)  
    Stock_M = Stock_M.to_numpy(dtype=np.int64) 
    Stock_T = pd.read_excel('Data/Stock_T_%s.xlsx' % (h), index_col=None, header = None)  
    Stock_T = Stock_T.to_numpy(dtype=np.int64)         
    Stock_over_M = pd.read_excel('Data/Stock_over_M_%s.xlsx' % (h), index_col=None, header = None)  
    Stock_over_M = Stock_over_M.to_numpy(dtype=np.int64) 
    Stock_over_T = pd.read_excel('Data/Stock_over_T_%s.xlsx' % (h), index_col=None, header = None)  
    Stock_over_T = Stock_over_T.to_numpy(dtype=np.int64)  
    
    Horizon = 364
    
    units_M = np.zeros([Horizon, Num_of_blood_groups, Max_Shelf_life, np.max(Stock_M[:]) + 1], dtype=np.int64)       
    for hz in range(Horizon):
        for r in range(Num_of_blood_groups):
            for phi in range(1, int(Stock_M[r]) + 1):
                if r <= 1:
                    My_units = np.random.standard_normal(phi) * 6 + 25
                else:
                    My_units = np.random.standard_normal(phi) * 6 + 32
                My_units2 = np.round(My_units)
                for m in range(Max_Shelf_life):
                    units_M[hz, r, m, phi] = np.count_nonzero(My_units2 == m)
                if np.sum(units_M[hz, r, :, phi]) != phi:
                    deviation = phi - np.sum(units_M[hz, r, :, phi])
                    if r <= 1:
                        units_M[hz, r, 25, phi] = units_M[hz, r, 25, phi] + deviation
                    else:
                        units_M[hz, r, 32, phi] = units_M[hz, r, 32, phi] + deviation
     
            
    units_T = np.zeros([Horizon, Num_of_blood_groups, Max_Shelf_life, np.max(Stock_T[:]) + 1], dtype=np.int64)       
    for hz in range(Horizon):
        for r in range(Num_of_blood_groups):
            for phi in range(1, int(Stock_T[r]) + 1):
                if r <= 1:
                    My_units = np.random.standard_normal(phi) * 6 + 25
                else:
                    My_units = np.random.standard_normal(phi) * 6 + 32
                My_units2 = np.round(My_units)
                for m in range(Max_Shelf_life):
                    units_T[hz, r, m, phi] = np.count_nonzero(My_units2 == m)
                if np.sum(units_T[hz, r, :, phi]) != phi:
                    deviation = phi - np.sum(units_T[hz, r, :, phi])
                    if r <= 1:
                        units_T[hz, r, 25, phi] = units_T[hz, r, 25, phi] + deviation
                    else:
                        units_T[hz, r, 32, phi] = units_T[hz, r, 32, phi] + deviation
    
    
    units_over_M = np.zeros([Horizon, Num_of_blood_groups, Max_Shelf_life, np.max(Stock_over_M[:]) + 1], dtype=np.int64)       
    for hz in range(Horizon):
        for r in range(Num_of_blood_groups):
            for phi in range(1, int(Stock_over_M[r]) + 1):
                if r <= 1:
                    My_units = np.random.standard_normal(phi) * 6 + 25
                else:
                    My_units = np.random.standard_normal(phi) * 6 + 32
                My_units2 = np.round(My_units)
                for m in range(Max_Shelf_life):
                    units_over_M[hz, r, m, phi] = np.count_nonzero(My_units2 == m)
                if np.sum(units_over_M[hz, r, :, phi]) != phi:
                    deviation = phi - np.sum(units_over_M[hz, r, :, phi])
                    if r <= 1:
                        units_over_M[hz, r, 25, phi] = units_over_M[hz, r, 25, phi] + deviation
                    else:
                        units_over_M[hz, r, 32, phi] = units_over_M[hz, r, 32, phi] + deviation
         
     
    units_over_T = np.zeros([Horizon, Num_of_blood_groups, Max_Shelf_life, np.max(Stock_over_T[:]) + 1], dtype=np.int64)       
    for hz in range(Horizon):
        for r in range(Num_of_blood_groups):
            for phi in range(1, int(Stock_over_T[r]) + 1):
                if r <= 1:
                    My_units = np.random.standard_normal(phi) * 6 + 25
                else:
                    My_units = np.random.standard_normal(phi) * 6 + 32
                My_units2 = np.round(My_units)
                for m in range(Max_Shelf_life):
                    units_over_T[hz, r, m, phi] = np.count_nonzero(My_units2 == m)
                if np.sum(units_over_T[hz, r, :, phi]) != phi:
                    deviation = phi - np.sum(units_over_T[hz, r, :, phi])
                    if r <= 1:
                        units_over_T[hz, r, 25, phi] = units_over_T[hz, r, 25, phi] + deviation
                    else:
                        units_over_T[hz, r, 32, phi] = units_over_T[hz, r, 32, phi] + deviation
    
    
    Initial_Inventory = np.zeros([Num_of_blood_groups, Max_Shelf_life], dtype=np.int64)    
    RealizedDemands = np.zeros([Num_of_blood_groups, 2 * 364], dtype=np.int64)    
    Demand0 = np.zeros([Num_of_blood_groups, 1], dtype=np.int64) 
    preDemand = np.zeros([num_of_Years + 1, Num_of_blood_groups, 2 * 364], dtype=np.int64)                
    Demand = np.zeros([Num_of_blood_groups, Duration_of_Scenario + 1, 3, Active_Scenarios], dtype=np.int64)
    Demand_II = np.zeros([Num_of_blood_groups, Duration_of_Scenario + 1, Active_Scenarios], dtype=np.int64)
    
    #=============================================================================
    Initial_Inventory, RealizedDemands, Demand0, preDemand, Demand, Demand_II = inputData(Num_of_blood_groups, Max_Shelf_life, 
                                  num_of_Years, Duration_of_Scenario, Active_Scenarios)
    
    import SAA

    SAA.RollingHorizon(h, Num_of_blood_groups, Max_Shelf_life, Cost_vector, 
                            Order_Cost_Vector, Emergency_Cost_Vector, Compatibility, Hospital_sizes, 
                            Blood_Demands, Initial_Inventory, RealizedDemands, Demand0, Stock_M, 
                            Stock_T, Horizon, units_M , units_T, Flag_stock, Flag_life,
                            Duration_of_Scenario, num_of_Years, Active_Scenarios,
                            preDemand, Demand_II)
    
    #=============================================================================    
    Initial_Inventory, RealizedDemands, Demand0, preDemand, Demand, Demand_II = inputData(Num_of_blood_groups, Max_Shelf_life, 
                                  num_of_Years, Duration_of_Scenario, Active_Scenarios)
    
    import MyopicPolicy

    MyopicPolicy.RollingHorizon(h, Num_of_blood_groups, Max_Shelf_life, Cost_vector, 
                            Order_Cost_Vector, Emergency_Cost_Vector, Compatibility, Hospital_sizes, 
                            Blood_Demands, Initial_Inventory, RealizedDemands, Demand0, Stock_M, 
                            Stock_T, Horizon, units_M , units_T)
    
    #=============================================================================
    Initial_Inventory, RealizedDemands, Demand0, preDemand, Demand, Demand_II = inputData(Num_of_blood_groups, Max_Shelf_life, 
                                  num_of_Years, Duration_of_Scenario, Active_Scenarios)
    
    import NoSubstitution

    NoSubstitution.RollingHorizon(h, Num_of_blood_groups, Max_Shelf_life, Cost_vector, 
                            Order_Cost_Vector, Emergency_Cost_Vector, Compatibility, Hospital_sizes, 
                            Blood_Demands, Initial_Inventory, RealizedDemands, Demand0, Stock_M, 
                            Stock_T, Horizon, units_M , units_T)
    
    #=============================================================================
    Initial_Inventory, RealizedDemands, Demand0, preDemand, Demand, Demand_II = inputData(Num_of_blood_groups, Max_Shelf_life, 
                                  num_of_Years, Duration_of_Scenario, Active_Scenarios)
    
    import EmpiricalPolicy

    EmpiricalPolicy.RollingHorizon(h, Num_of_blood_groups, Max_Shelf_life, Cost_vector, 
                            Order_Cost_Vector, Emergency_Cost_Vector, Compatibility, Hospital_sizes, 
                            Blood_Demands, Initial_Inventory, RealizedDemands, Demand0, Stock_M, 
                            Stock_T, Horizon, units_M , units_T, Empirical)
    
    #=============================================================================
    Initial_Inventory, RealizedDemands, Demand0, preDemand, Demand, Demand_II = inputData(Num_of_blood_groups, Max_Shelf_life, 
                                  num_of_Years, Duration_of_Scenario, Active_Scenarios)

    import OverOrder

    OverOrder.RollingHorizon(h, Num_of_blood_groups, Max_Shelf_life, Cost_vector, 
                            Order_Cost_Vector, Emergency_Cost_Vector, Compatibility, Hospital_sizes, 
                            Blood_Demands, Initial_Inventory, RealizedDemands, Demand0, Stock_over_M, 
                            Stock_over_T, Horizon, units_over_M , units_over_T, Over_order)
    #=============================================================================


