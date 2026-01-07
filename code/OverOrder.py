

#                         Mostafa Khatami
#           As part of project: Blood Substitution Decision at Hospitals


import numpy as np
import os
import pandas as pd

np.random.seed(0)
            
#----------------- Rolling horizon

def RollingHorizon(h, Num_of_blood_groups, Max_Shelf_life, Cost_vector, 
                            Order_Cost_Vector, Order_Cost_Vector_2, Compatibility, Hospital_sizes, 
                            Blood_Demands, Initial_Inventory_P2, RealizedDemands, Demand0_P2, Stock_over_M, 
                            Stock_over_T, Horizon, units_over_M , units_over_T, Over_order):
    
    #-----------------Generating output files
    MyOutputList = ['OF', 'Sub_To', 'Sub_From', 'Tran', 'Inv_Gr', 
                    'Inv_Shlf', 'Inv_Shlf', 'Emegcy', 'Tran_Life']

    for i in MyOutputList:
        with open('Outputs/%s_%s_P2.xlsx' % (i, h), 'w') as file:
            os.chmod('Outputs/%s_%s_P2.xlsx' % (i, h), 0o755)
                
    #---------------- Rolling horizon empirical rule:
    Horizon_ObjectiveFunction_P2 = np.zeros([Horizon, 4], dtype=float)
    Horizon_EmergencyOrders_P2 = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
    Horizon_Substitutions_To_P2 = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
    Horizon_Substitutions_From_P2 = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
    Horizon_Transfusion_P2 = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
    Horizon_Inventory_Groups_P2 = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
    Horizon_Inventory_Shelf_P2 = np.zeros([Horizon, Max_Shelf_life], dtype=np.int64)
    Horizon_Transfusion_Life_P2 = np.zeros([Horizon, Max_Shelf_life], dtype=np.int64)
    Horizon_Inventory_P2 = np.zeros([Num_of_blood_groups, Max_Shelf_life], dtype=np.int64)
    Horizon_RegularOrders_P2 = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
    
    for z in range(Horizon):
        
        Horizon_units_M = np.zeros([Num_of_blood_groups, Max_Shelf_life, np.max(Stock_over_M[:]) + 1], dtype=np.int64)       
        Horizon_units_M[:] = units_over_M[z, :, :, :]

        Horizon_units_T = np.zeros([Num_of_blood_groups, Max_Shelf_life, np.max(Stock_over_T[:]) + 1], dtype=np.int64)       
        Horizon_units_T[:] = units_over_T[z, :, :, :]    
        
        Order_Size_P2 = np.zeros([Num_of_blood_groups], dtype=np.int64)

        
        if z > 0 and z % 7 == 0:
            for r in range(Num_of_blood_groups):
                Order_Size = np.zeros([Max_Shelf_life], dtype=np.int64)
                if Stock_over_M[r] > sum(Initial_Inventory_P2[r, :]):
                    Order_Size_P2[r] = Stock_over_M[r] - sum(Initial_Inventory_P2[r, :])
                else:
                    Order_Size_P2[r] = 0
                    
                for i in range(Max_Shelf_life):
                    Order_Size[i] = Horizon_units_M[r, i, Order_Size_P2[r]]
                for i in range(Max_Shelf_life):
                    Initial_Inventory_P2[r, i] = Initial_Inventory_P2[r, i] + Order_Size[i]
                    
        if z > 0 and z % 7 == 3:
            for r in range(Num_of_blood_groups):
                Order_Size = np.zeros([Max_Shelf_life], dtype=np.int64)
                if Stock_over_T[r] > sum(Initial_Inventory_P2[r, :]):
                    Order_Size_P2[r] = Stock_over_T[r] - sum(Initial_Inventory_P2[r, :])
                else:
                    Order_Size_P2[r] = 0
                    
                for i in range(Max_Shelf_life):
                    Order_Size[i] = Horizon_units_T[r, i, Order_Size_P2[r]]
                for i in range(Max_Shelf_life):
                    Initial_Inventory_P2[r, i] = Initial_Inventory_P2[r, i] + Order_Size[i]
                        
        Horizon_RegularOrders_P2[z, :] = Order_Size_P2[:]
        
        Substitution_P2 = np.zeros([Num_of_blood_groups, Num_of_blood_groups], dtype=np.int64)
        Required_Units_P2 = np.zeros([Num_of_blood_groups], dtype=np.int64)
        Available_Units_P2 = np.zeros([Num_of_blood_groups], dtype=np.int64)
        Remained_Units_P2 = np.zeros([Num_of_blood_groups], dtype=np.int64)
        for r in range(Num_of_blood_groups):
            Required_Units_P2[r] = Demand0_P2[r]
            Available_Units_P2[r] = sum(Initial_Inventory_P2[r, :])
            
            if Available_Units_P2[r] > Required_Units_P2[r]:
                Substitution_P2[r, r] = Required_Units_P2[r]
                Remained_Units_P2[r] = Required_Units_P2[r]
                j = 0
                while Remained_Units_P2[r] > 0 and j <= 41:
                    if Initial_Inventory_P2[r, j] > 0:
                        Duduced_Units = min(Initial_Inventory_P2[r, j], Remained_Units_P2[r])
                        Initial_Inventory_P2[r, j] = Initial_Inventory_P2[r, j] - Duduced_Units
                        Remained_Units_P2[r] = Remained_Units_P2[r] - Duduced_Units
                        Horizon_Transfusion_Life_P2[z, j] = Horizon_Transfusion_Life_P2[z, j] + Duduced_Units
                    j = j + 1
            else:
                if r == 0:
                    Substitution_P2[0, 0] = Available_Units_P2[0]
                    Horizon_EmergencyOrders_P2[z, 0] = Required_Units_P2[0] - Available_Units_P2[0]
                    for j in range(Max_Shelf_life):
                        Horizon_Transfusion_Life_P2[z, j] = Horizon_Transfusion_Life_P2[z, j] + Initial_Inventory_P2[r, j]
                    Initial_Inventory_P2[0, :] = 0
                else:
                    Substitution_P2[r, r] = Available_Units_P2[r]
                    Remained_Units_P2[r] = Required_Units_P2[r] - Available_Units_P2[r]
                    for j in range(Max_Shelf_life):
                        Horizon_Transfusion_Life_P2[z, j] = Horizon_Transfusion_Life_P2[z, j] + Initial_Inventory_P2[r, j]
                    Initial_Inventory_P2[r, :] = 0
        
        for r in range(1, Num_of_blood_groups):
            if Remained_Units_P2[r] > 0:
                s = 1
                while Remained_Units_P2[r] > 0  and s <= 6:
                    Priority = np.argwhere(Over_order[r,:] == s)
                    if len(Priority) == 1:
                        Substitution_Available = sum(sum(Initial_Inventory_P2[Priority[0], :]))
                        if Substitution_Available > 0:
                            if Substitution_Available > Remained_Units_P2[r]:
                                Substitution_P2[Priority[0], r] = Remained_Units_P2[r]
                                j = 0
                                while Remained_Units_P2[r] > 0 and j <= 41:
                                    Duduced_Units = min(Initial_Inventory_P2[Priority[0], j], Remained_Units_P2[r])
                                    Initial_Inventory_P2[Priority[0], j] = Initial_Inventory_P2[Priority[0], j] - Duduced_Units
                                    Remained_Units_P2[r] = Remained_Units_P2[r] - Duduced_Units
                                    Horizon_Transfusion_Life_P2[z, j] = Horizon_Transfusion_Life_P2[z, j] + Duduced_Units
                                    j = j + 1
                            else:
                                Substitution_P2[Priority[0], r] = Substitution_Available
                                Remained_Units_P2[r] = Remained_Units_P2[r] - Substitution_Available
                                for j in range(Max_Shelf_life):
                                    Horizon_Transfusion_Life_P2[z, j] = Horizon_Transfusion_Life_P2[z, j] + Initial_Inventory_P2[Priority[0], j]
                                Initial_Inventory_P2[Priority[0], :] = 0
                    s = s + 1              
            Horizon_EmergencyOrders_P2[z, r] = Remained_Units_P2[r]                
        
        Horizon_Inventory_P2[:] = Initial_Inventory_P2
        
        #--------------Calculating Objective Function and Outputs
        Regular_order = sum(Order_Cost_Vector * Horizon_RegularOrders_P2[z, :])
        Emergency_Cost = sum(Order_Cost_Vector_2 * Horizon_EmergencyOrders_P2[z, :])
        Inventory_Cost = Cost_vector[8] * sum(sum(Horizon_Inventory_P2))
        Outdate_Cost = Cost_vector[9] * sum(Horizon_Inventory_P2[:, 0])
        Horizon_ObjectiveFunction_P2[z, :] = [Regular_order, Emergency_Cost, Inventory_Cost, Outdate_Cost]
        
        for i in range(Num_of_blood_groups):
            Horizon_Substitutions_To_P2[z, i] = sum(Substitution_P2[i, :])
            Horizon_Substitutions_From_P2[z, i] = sum(Substitution_P2[:, i])
            Horizon_Transfusion_P2[z, i] = Substitution_P2[i, i]
            Horizon_Inventory_Groups_P2[z, i] = sum(Horizon_Inventory_P2[i, :])
            
        for i in range(Max_Shelf_life):
            Horizon_Inventory_Shelf_P2[z, i] = sum(Horizon_Inventory_P2[:, i])
            

        df = pd.DataFrame(Horizon_ObjectiveFunction_P2)
        df.to_excel('Outputs/OF_%s_P2.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Substitutions_To_P2)
        df.to_excel('Outputs/Sub_To_%s_P2.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Substitutions_From_P2)
        df.to_excel('Outputs/Sub_From_%s_P2.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Transfusion_P2)
        df.to_excel('Outputs/Tran_%s_P2.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Inventory_Groups_P2)
        df.to_excel('Outputs/Inv_Gr_%s_P2.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Inventory_Shelf_P2)
        df.to_excel('Outputs/Inv_Shlf_%s_P2.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_EmergencyOrders_P2)
        df.to_excel('Outputs/Emegcy_%s_P2.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Transfusion_Life_P2)
        df.to_excel('Outputs/Tran_Life_%s_P2.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Substitutions_To_P2 - Horizon_Transfusion_P2)
        df.to_excel('Outputs/Sub_To_exc_%s_P2.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Substitutions_From_P2 - Horizon_Transfusion_P2)
        df.to_excel('Outputs/Sub_From_exc_%s_P2.xlsx' % (h), header=False, index=False)
        
        # ------------ Updating initial inventory with new units
        for i in range(Max_Shelf_life - 1):
            Initial_Inventory_P2[:, i] = Initial_Inventory_P2[:, i + 1]
        Initial_Inventory_P2[:, 41] = 0
        
        for r in range(Num_of_blood_groups):
            Horizon_RegularOrders_P2[z, r] = Order_Size_P2[r]
            
        Demand0_P2 = RealizedDemands[:, z + 8]        
    
    life = np.array([np.sum(Horizon_Transfusion_Life_P2, axis=0)])
    Horizon_Transfusion_Life_P2 = np.concatenate((Horizon_Transfusion_Life_P2, life), axis=0)
    
    df = pd.DataFrame(Horizon_Transfusion_Life_P2)
    df.to_excel('Outputs/Tran_Life_%s_P2.xlsx' % (h), header=False, index=False)        
    
    TotalCost = np.zeros([Num_of_blood_groups], dtype=np.int64)
    TotalCost[0] = sum(sum(Horizon_ObjectiveFunction_P2))
    
    TotalTransfusion = np.zeros([Num_of_blood_groups], dtype=np.int64)
    for i in range(Num_of_blood_groups):
        TotalTransfusion[i] = sum(Horizon_Transfusion_P2[:,i])
        
    TotalSubTo = np.zeros([Num_of_blood_groups], dtype=np.int64)
    for i in range(Num_of_blood_groups):
        TotalSubTo[i] = sum(Horizon_Substitutions_To_P2[:,i])
    
    TotalSubFrom = np.zeros([Num_of_blood_groups], dtype=np.int64) 
    for i in range(Num_of_blood_groups):
        TotalSubFrom[i] = sum(Horizon_Substitutions_From_P2[:,i])
        
    TotalEmerg = np.zeros([Num_of_blood_groups], dtype=np.int64)
    for i in range(Num_of_blood_groups):
        TotalEmerg[i] = sum(Horizon_EmergencyOrders_P2[:,i])
    
    df0 = pd.DataFrame(TotalCost)
    df1 = pd.DataFrame(TotalTransfusion)
    df2 = pd.DataFrame(TotalSubTo)
    df3 = pd.DataFrame(TotalSubFrom)
    df4 = pd.DataFrame(TotalEmerg)
    
    with pd.ExcelWriter('Outputs/Summary_%s_P2.xlsx' % (h)) as writer:  
        df0.to_excel(writer, header=False, index=False, sheet_name='TotalCost')
        df1.to_excel(writer, header=False, index=False, sheet_name='TotalTransfusion')
        df2.to_excel(writer, header=False, index=False, sheet_name='TotalSubTo')
        df3.to_excel(writer, header=False, index=False, sheet_name='TotalSubFrom')
        df4.to_excel(writer, header=False, index=False, sheet_name='TotalEmerg')
        
        
 