

#                         Mostafa Khatami
#           As part of project: Blood Substitution Decision at Hospitals


import numpy as np
import os
import pandas as pd

np.random.seed(0)
            
#----------------- Rolling horizon

def RollingHorizon(h, Num_of_blood_groups, Max_Shelf_life, Cost_vector, 
                            Order_Cost_Vector, Order_Cost_Vector_2, Compatibility, Hospital_sizes, 
                            Blood_Demands, Initial_Inventory_P, RealizedDemands, Demand0_P, Stock_M, 
                            Stock_T, Horizon, units_M , units_T, Empirical):
    
    #-----------------Generating output files
    MyOutputList = ['OF', 'Sub_To', 'Sub_From', 'Tran', 'Inv_Gr', 
                    'Inv_Shlf', 'Inv_Shlf', 'Emegcy', 'Tran_Life']

    for i in MyOutputList:
        with open('Outputs/%s_%s_P.xlsx' % (i, h), 'w') as file:
            os.chmod('Outputs/%s_%s_P.xlsx' % (i, h), 0o755)
                
    #---------------- Rolling horizon empirical rule:
    Horizon_ObjectiveFunction_P = np.zeros([Horizon, 4], dtype=float)
    Horizon_EmergencyOrders_P = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
    Horizon_Substitutions_To_P = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
    Horizon_Substitutions_From_P = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
    Horizon_Transfusion_P = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
    Horizon_Inventory_Groups_P = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
    Horizon_Inventory_Shelf_P = np.zeros([Horizon, Max_Shelf_life], dtype=np.int64)
    Horizon_Transfusion_Life_P = np.zeros([Horizon, Max_Shelf_life], dtype=np.int64)
    Horizon_Inventory_P = np.zeros([Num_of_blood_groups, Max_Shelf_life], dtype=np.int64)
    Horizon_RegularOrders_P = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
    
    for z in range(Horizon):
        
        Horizon_units_M = np.zeros([Num_of_blood_groups, Max_Shelf_life, np.max(Stock_M[:]) + 1], dtype=np.int64)       
        Horizon_units_M[:] = units_M[z, :, :, :]

        Horizon_units_T = np.zeros([Num_of_blood_groups, Max_Shelf_life, np.max(Stock_T[:]) + 1], dtype=np.int64)       
        Horizon_units_T[:] = units_T[z, :, :, :]    
                                                       
        Order_Size_P = np.zeros([Num_of_blood_groups], dtype=np.int64)

        
        if z > 0 and z % 7 == 0:
            for r in range(Num_of_blood_groups):
                Order_Size = np.zeros([Max_Shelf_life], dtype=np.int64)
                if Stock_M[r] > sum(Initial_Inventory_P[r, :]):
                    Order_Size_P[r] = Stock_M[r] - sum(Initial_Inventory_P[r, :])
                else:
                    Order_Size_P[r] = 0
                
                for i in range(Max_Shelf_life):
                    Order_Size[i] = Horizon_units_M[r, i, Order_Size_P[r]]
                for i in range(Max_Shelf_life):
                    Initial_Inventory_P[r, i] = Initial_Inventory_P[r, i] + Order_Size[i]
                    
        if z > 0 and z % 7 == 3:
            for r in range(Num_of_blood_groups):
                Order_Size = np.zeros([Max_Shelf_life], dtype=np.int64)
                if Stock_T[r] > sum(Initial_Inventory_P[r, :]):
                    Order_Size_P[r] = Stock_T[r] - sum(Initial_Inventory_P[r, :])
                else:
                    Order_Size_P[r] = 0
                    
                for i in range(Max_Shelf_life):
                    Order_Size[i] = Horizon_units_T[r, i, Order_Size_P[r]]
                for i in range(Max_Shelf_life):
                    Initial_Inventory_P[r, i] = Initial_Inventory_P[r, i] + Order_Size[i]
                        
        Horizon_RegularOrders_P[z, :] = Order_Size_P[:]
        
        Substitution_P = np.zeros([Num_of_blood_groups, Num_of_blood_groups], dtype=np.int64)
        Required_Units_P = np.zeros([Num_of_blood_groups], dtype=np.int64)
        Available_Units_P = np.zeros([Num_of_blood_groups], dtype=np.int64)
        Remained_Units_P = np.zeros([Num_of_blood_groups], dtype=np.int64)
        for r in range(Num_of_blood_groups):
            Required_Units_P[r] = Demand0_P[r]
            Available_Units_P[r] = sum(Initial_Inventory_P[r, :])
            
            if Available_Units_P[r] > Required_Units_P[r]:
                Substitution_P[r, r] = Required_Units_P[r]
                Remained_Units_P[r] = Required_Units_P[r]
                j = 0
                while Remained_Units_P[r] > 0 and j <= 41:
                    if Initial_Inventory_P[r, j] > 0:
                        Duduced_Units = min(Initial_Inventory_P[r, j], Remained_Units_P[r])
                        Initial_Inventory_P[r, j] = Initial_Inventory_P[r, j] - Duduced_Units
                        Remained_Units_P[r] = Remained_Units_P[r] - Duduced_Units
                        Horizon_Transfusion_Life_P[z, j] = Horizon_Transfusion_Life_P[z, j] + Duduced_Units
                    j = j + 1
            else:
                if r == 0:
                    Substitution_P[0, 0] = Available_Units_P[0]
                    Horizon_EmergencyOrders_P[z, 0] = Required_Units_P[0] - Available_Units_P[0]
                    for j in range(Max_Shelf_life):
                        Horizon_Transfusion_Life_P[z, j] = Horizon_Transfusion_Life_P[z, j] + Initial_Inventory_P[r, j]
                    Initial_Inventory_P[0, :] = 0
                else:
                    Substitution_P[r, r] = Available_Units_P[r]
                    Remained_Units_P[r] = Required_Units_P[r] - Available_Units_P[r]
                    for j in range(Max_Shelf_life):
                        Horizon_Transfusion_Life_P[z, j] = Horizon_Transfusion_Life_P[z, j] + Initial_Inventory_P[r, j]
                    Initial_Inventory_P[r, :] = 0
        
        for r in range(1, Num_of_blood_groups):
            if Remained_Units_P[r] > 0:
                s = 1
                while Remained_Units_P[r] > 0  and s <= 6:
                    Priority = np.argwhere(Empirical[r,:] == s)
                    if len(Priority) == 1:
                        Substitution_Available = sum(sum(Initial_Inventory_P[Priority[0], :]))
                        if Substitution_Available > 0:
                            if Substitution_Available > Remained_Units_P[r]:
                                Substitution_P[Priority[0], r] = Remained_Units_P[r]
                                j = 0
                                while Remained_Units_P[r] > 0 and j <= 41:
                                    Duduced_Units = min(Initial_Inventory_P[Priority[0], j], Remained_Units_P[r])
                                    Initial_Inventory_P[Priority[0], j] = Initial_Inventory_P[Priority[0], j] - Duduced_Units
                                    Remained_Units_P[r] = Remained_Units_P[r] - Duduced_Units
                                    Horizon_Transfusion_Life_P[z, j] = Horizon_Transfusion_Life_P[z, j] + Duduced_Units
                                    j = j + 1
                            else:
                                Substitution_P[Priority[0], r] = Substitution_Available
                                Remained_Units_P[r] = Remained_Units_P[r] - Substitution_Available
                                for j in range(Max_Shelf_life):
                                    Horizon_Transfusion_Life_P[z, j] = Horizon_Transfusion_Life_P[z, j] + Initial_Inventory_P[Priority[0], j]
                                Initial_Inventory_P[Priority[0], :] = 0
                    s = s + 1              
            Horizon_EmergencyOrders_P[z, r] = Remained_Units_P[r]                
        
        Horizon_Inventory_P[:] = Initial_Inventory_P
        
        #--------------Calculating Objective Function and Outputs
        Regular_order = sum(Order_Cost_Vector * Horizon_RegularOrders_P[z, :])
        Emergency_Cost = sum(Order_Cost_Vector_2 * Horizon_EmergencyOrders_P[z, :])
        Inventory_Cost = Cost_vector[8] * sum(sum(Horizon_Inventory_P))
        Outdate_Cost = Cost_vector[9] * sum(Horizon_Inventory_P[:, 0])
        Horizon_ObjectiveFunction_P[z, :] = [Regular_order, Emergency_Cost, Inventory_Cost, Outdate_Cost]
        
        for i in range(Num_of_blood_groups):
            Horizon_Substitutions_To_P[z, i] = sum(Substitution_P[i, :])
            Horizon_Substitutions_From_P[z, i] = sum(Substitution_P[:, i])
            Horizon_Transfusion_P[z, i] = Substitution_P[i, i]
            Horizon_Inventory_Groups_P[z, i] = sum(Horizon_Inventory_P[i, :])
            
        for i in range(Max_Shelf_life):
            Horizon_Inventory_Shelf_P[z, i] = sum(Horizon_Inventory_P[:, i])
            

        df = pd.DataFrame(Horizon_ObjectiveFunction_P)
        df.to_excel('Outputs/OF_%s_P.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Substitutions_To_P)
        df.to_excel('Outputs/Sub_To_%s_P.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Substitutions_From_P)
        df.to_excel('Outputs/Sub_From_%s_P.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Transfusion_P)
        df.to_excel('Outputs/Tran_%s_P.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Inventory_Groups_P)
        df.to_excel('Outputs/Inv_Gr_%s_P.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Inventory_Shelf_P)
        df.to_excel('Outputs/Inv_Shlf_%s_P.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_EmergencyOrders_P)
        df.to_excel('Outputs/Emegcy_%s_P.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Transfusion_Life_P)
        df.to_excel('Outputs/Tran_Life_%s_P.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Substitutions_To_P - Horizon_Transfusion_P)
        df.to_excel('Outputs/Sub_To_exc_%s_P.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Substitutions_From_P - Horizon_Transfusion_P)
        df.to_excel('Outputs/Sub_From_exc_%s_P.xlsx' % (h), header=False, index=False)
        
        # ------------ Updating initial inventory with new units
        for i in range(Max_Shelf_life - 1):
            Initial_Inventory_P[:, i] = Initial_Inventory_P[:, i + 1]
        Initial_Inventory_P[:, 41] = 0
        
        for r in range(Num_of_blood_groups):
            Horizon_RegularOrders_P[z, r] = Order_Size_P[r]
            
        Demand0_P = RealizedDemands[:, z + 8]

    
    life = np.array([np.sum(Horizon_Transfusion_Life_P, axis=0)])
    Horizon_Transfusion_Life_P = np.concatenate((Horizon_Transfusion_Life_P, life), axis=0)

    df = pd.DataFrame(Horizon_Transfusion_Life_P)
    df.to_excel('Outputs/Tran_Life_%s_P.xlsx' % (h), header=False, index=False)
    
    TotalCost = np.zeros([Num_of_blood_groups], dtype=np.int64)
    TotalCost[0] = sum(sum(Horizon_ObjectiveFunction_P))
    
    TotalTransfusion = np.zeros([Num_of_blood_groups], dtype=np.int64)
    for i in range(Num_of_blood_groups):
        TotalTransfusion[i] = sum(Horizon_Transfusion_P[:,i])
        
    TotalSubTo = np.zeros([Num_of_blood_groups], dtype=np.int64)
    for i in range(Num_of_blood_groups):
        TotalSubTo[i] = sum(Horizon_Substitutions_To_P[:,i])
    
    TotalSubFrom = np.zeros([Num_of_blood_groups], dtype=np.int64) 
    for i in range(Num_of_blood_groups):
        TotalSubFrom[i] = sum(Horizon_Substitutions_From_P[:,i])
        
    TotalEmerg = np.zeros([Num_of_blood_groups], dtype=np.int64)
    for i in range(Num_of_blood_groups):
        TotalEmerg[i] = sum(Horizon_EmergencyOrders_P[:,i])
    
    df0 = pd.DataFrame(TotalCost)
    df1 = pd.DataFrame(TotalTransfusion)
    df2 = pd.DataFrame(TotalSubTo)
    df3 = pd.DataFrame(TotalSubFrom)
    df4 = pd.DataFrame(TotalEmerg)
    
    with pd.ExcelWriter('Outputs/Summary_%s_P.xlsx' % (h)) as writer:  
        df0.to_excel(writer, header=False, index=False, sheet_name='TotalCost')
        df1.to_excel(writer, header=False, index=False, sheet_name='TotalTransfusion')
        df2.to_excel(writer, header=False, index=False, sheet_name='TotalSubTo')
        df3.to_excel(writer, header=False, index=False, sheet_name='TotalSubFrom')
        df4.to_excel(writer, header=False, index=False, sheet_name='TotalEmerg')
        
        
 