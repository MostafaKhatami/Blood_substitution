

#                         Mostafa Khatami
#           As part of project: Blood Substitution Decision at Hospitals


import numpy as np
import os
import pandas as pd

np.random.seed(0)
            
#----------------- Rolling horizon

def RollingHorizon(h, Num_of_blood_groups, Max_Shelf_life, Cost_vector, 
                            Order_Cost_Vector, Order_Cost_Vector_2, Compatibility, Hospital_sizes, 
                            Blood_Demands, Initial_Inventory_E, RealizedDemands, Demand0_E, Stock_M, 
                            Stock_T, Horizon, units_M , units_T):
    
    #-----------------Generating output files
    MyOutputList = ['OF', 'Sub_To', 'Sub_From', 'Tran', 'Inv_Gr', 
                    'Inv_Shlf', 'Inv_Shlf', 'Emegcy', 'Tran_Life']

    for i in MyOutputList:
        with open('Outputs/%s_%s_E.xlsx' % (i, h), 'w') as file:
            os.chmod('Outputs/%s_%s_E.xlsx' % (i, h), 0o755)
                
    #---------------- Rolling horizon no-substitution:
    Horizon_ObjectiveFunction_E = np.zeros([Horizon, 4], dtype=float)
    Horizon_EmergencyOrders_E = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
    Horizon_Inventory_Groups_E = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
    Horizon_Inventory_Shelf_E = np.zeros([Horizon, Max_Shelf_life], dtype=np.int64)
    Horizon_Transfusion_Life_E = np.zeros([Horizon, Max_Shelf_life], dtype=np.int64)
    Horizon_Inventory_E = np.zeros([Num_of_blood_groups, Max_Shelf_life], dtype=np.int64)
    Horizon_RegularOrders_E = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)

    for z in range(Horizon):
        
        Horizon_units_M = np.zeros([Num_of_blood_groups, Max_Shelf_life, np.max(Stock_M[:]) + 1], dtype=np.int64)       
        Horizon_units_M[:] = units_M[z, :, :, :]

        Horizon_units_T = np.zeros([Num_of_blood_groups, Max_Shelf_life, np.max(Stock_T[:]) + 1], dtype=np.int64)       
        Horizon_units_T[:] = units_T[z, :, :, :]    
                
        Order_Size_E = np.zeros([Num_of_blood_groups], dtype=np.int64)
                            
        if z > 0 and z % 7 == 0:
            for r in range(Num_of_blood_groups):
                Order_Size = np.zeros([Max_Shelf_life], dtype=np.int64)
                if Stock_M[r] > sum(Initial_Inventory_E[r, :]):
                    Order_Size_E[r] = Stock_M[r] - sum(Initial_Inventory_E[r, :])
                else:
                    Order_Size_E[r] = 0
                    
                for i in range(Max_Shelf_life):
                    Order_Size[i] = Horizon_units_M[r, i, Order_Size_E[r]]
                for i in range(Max_Shelf_life):
                    Initial_Inventory_E[r, i] = Initial_Inventory_E[r, i] + Order_Size[i]
                    
        if z > 0 and z % 7 == 3:
            for r in range(Num_of_blood_groups):
                Order_Size = np.zeros([Max_Shelf_life], dtype=np.int64)
                if Stock_T[r] > sum(Initial_Inventory_E[r, :]):
                    Order_Size_E[r] = Stock_T[r] - sum(Initial_Inventory_E[r, :])
                else:
                    Order_Size_E[r] = 0
                    
                for i in range(Max_Shelf_life):
                    Order_Size[i] = Horizon_units_T[r, i, Order_Size_E[r]]
                for i in range(Max_Shelf_life):
                    Initial_Inventory_E[r, i] = Initial_Inventory_E[r, i] + Order_Size[i]
                        
        Horizon_RegularOrders_E[z, :] = Order_Size_E[:]
                
        for r in range(Num_of_blood_groups):
            Required_Units = Demand0_E[r]
            Available_Units = sum(Initial_Inventory_E[r, :])
            if Available_Units > Required_Units:
                Horizon_EmergencyOrders_E[z, r] = 0
                Remained_Units = Required_Units
                j = 0
                while Remained_Units > 0 and j <= 41:
                    if Initial_Inventory_E[r, j] > 0:
                        Duduced_Units = min(Initial_Inventory_E[r, j], Remained_Units)
                        Initial_Inventory_E[r, j] = Initial_Inventory_E[r, j] - Duduced_Units
                        Remained_Units = Remained_Units - Duduced_Units
                        Horizon_Transfusion_Life_E[z, j] = Horizon_Transfusion_Life_E[z, j] + Duduced_Units
                    j = j + 1
            else:
                Horizon_EmergencyOrders_E[z, r] = Required_Units - Available_Units
                for j in range(Max_Shelf_life):
                    Horizon_Transfusion_Life_E[z, j] = Horizon_Transfusion_Life_E[z, j] + Initial_Inventory_E[r, j]
                Initial_Inventory_E[r, :] = 0
        
        Horizon_Inventory_E[:] = Initial_Inventory_E
        
        #--------------Calculating Objective Function and Outputs
        Regular_order = sum(Order_Cost_Vector * Horizon_RegularOrders_E[z, :])
        Emergency_Cost = sum(Order_Cost_Vector_2 * Horizon_EmergencyOrders_E[z, :])
        Inventory_Cost = Cost_vector[8] * sum(sum(Horizon_Inventory_E))
        Outdate_Cost = Cost_vector[9] * sum(Horizon_Inventory_E[:, 0])
        Horizon_ObjectiveFunction_E[z, :] = [Regular_order, Emergency_Cost, Inventory_Cost, Outdate_Cost]
        
        for i in range(Num_of_blood_groups):
            Horizon_Inventory_Groups_E[z, i] = sum(Horizon_Inventory_E[i, :])
            
        for i in range(Max_Shelf_life):
            Horizon_Inventory_Shelf_E[z, i] = sum(Horizon_Inventory_E[:, i])
            

        df = pd.DataFrame(Horizon_ObjectiveFunction_E)
        df.to_excel('Outputs/OF_%s_E.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Inventory_Groups_E)
        df.to_excel('Outputs/Inv_Gr_%s_E.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Inventory_Shelf_E)
        df.to_excel('Outputs/Inv_Shlf_%s_E.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_EmergencyOrders_E)
        df.to_excel('Outputs/Emegcy_%s_E.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Transfusion_Life_E)
        df.to_excel('Outputs/Tran_Life_%s_E.xlsx' % (h), header=False, index=False)
        
        # ------------ Updating initial inventory  with new units
        for i in range(Max_Shelf_life - 1):
            Initial_Inventory_E[:, i] = Initial_Inventory_E[:, i + 1]
        Initial_Inventory_E[:, 41] = 0
        
        for r in range(Num_of_blood_groups):
            Horizon_RegularOrders_E[z, r] = Order_Size_E[r]
            
        Demand0_E = RealizedDemands[:, z + 8]    
    
    
    life = np.array([np.sum(Horizon_Transfusion_Life_E, axis=0)])
    Horizon_Transfusion_Life_E = np.concatenate((Horizon_Transfusion_Life_E, life), axis=0)
    
    df = pd.DataFrame(Horizon_Transfusion_Life_E)
    df.to_excel('Outputs/Tran_Life_%s_E.xlsx' % (h), header=False, index=False)
    
    
    TotalCost = np.zeros([Num_of_blood_groups], dtype=np.int64)
    TotalCost[0] = sum(sum(Horizon_ObjectiveFunction_E))
        
    TotalEmerg = np.zeros([Num_of_blood_groups], dtype=np.int64)
    for i in range(Num_of_blood_groups):
        TotalEmerg[i] = sum(Horizon_EmergencyOrders_E[:,i])
    
    df0 = pd.DataFrame(TotalCost)
    df1 = pd.DataFrame(TotalEmerg)
    
    with pd.ExcelWriter('Outputs/Summary_%s_E.xlsx' % (h)) as writer:  
        df0.to_excel(writer, header=False, index=False, sheet_name='TotalCost')
        df1.to_excel(writer, header=False, index=False, sheet_name='TotalEmerg')
        
        
 