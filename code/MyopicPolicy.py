

#                         Mostafa Khatami
#           As part of project: Blood Substitution Decision at Hospitals

from gurobipy import *
import numpy as np
import os
import pandas as pd

np.random.seed(0)


def CallBack(mod, where): 
    Elapsed_Time = mod.cbGet(GRB.Callback.RUNTIME)      

    if  Elapsed_Time >= 900:
        if where == GRB.Callback.MIP:
            solcnt = mod.cbGet(GRB.Callback.MIP_SOLCNT)
            if solcnt >= 1:
                mod.terminate()
                
        if where == GRB.Callback.MIPSOL:
            solcnt = mod.cbGet(GRB.Callback.MIPSOL_SOLCNT)
            if solcnt >= 1:
                mod.terminate()
                
        if where == GRB.Callback.MIPNODE:
            solcnt = mod.cbGet(GRB.Callback.MIPNODE_SOLCNT)
            if solcnt >= 1:
                mod.terminate()
    
        
#-----------------------Building the robust model with SAA------------------------
def Myopic(Initial_Inventory, Demand0, Compatibility, Cost_vector, Cost_vector_2):

    mod = Model ("Myopic")

    # ---------------------Inputs:
    BloodGroups = 8
    
    MatrixDimensions = Initial_Inventory.shape
    Remaining_Life = MatrixDimensions[1]
    
    Order_Cost = np.zeros(8)
    Order_Cost[0:8] = Cost_vector[0:8]
    Emergency_Cost = np.zeros(8)
    Emergency_Cost[0:8] = Cost_vector_2[0:8]
    Inventory_Cost = Cost_vector[8]
    Outdate_Cost = Cost_vector[9]
    
    # ---------Generating variables:
    
    # q represents the number of units of group r, with age m, that is used 
    # for patients of group s at time period t
    q = {}
    for r in range(BloodGroups):
        for s in range(BloodGroups):
            for m in range(Remaining_Life):
                q[r, s, m] = mod.addVar(vtype=GRB.INTEGER)
    
    #--------------------------------
    # g represents the number of emergency-ordered units of group r
    g = {}
    for r in range(BloodGroups):
        g[r] = mod.addVar(vtype=GRB.INTEGER)
    
    a = {}
    for r in range(BloodGroups):
        for m in range(Remaining_Life):
            a[r, m] = mod.addVar(vtype=GRB.INTEGER)
            
    #--------------------------------
    # Auxiliary binary variables
    z = {}
    for r in range(BloodGroups):
        for m in range(1, Remaining_Life):
            z[r, m] = mod.addVar(vtype=GRB.BINARY)
       
    mod.update()
    
    # -------------- Generating constraints:
    
    for s in range(BloodGroups):
        mod.addConstr(quicksum(quicksum(q[r, s, m] for m in range(Remaining_Life)) 
            for r in range(BloodGroups) if Compatibility[r, s] == 1 and r != s) + g[s] == Demand0[s])
                            
    for r in range(BloodGroups):
        for m in range(Remaining_Life):
            mod.addConstr(quicksum(q[r, s, m] for s in range(BloodGroups) if Compatibility[r, s] == 1 and r != s) <= Initial_Inventory[r, m])
    
    for r in range(BloodGroups):
        for m in range(Remaining_Life):
            mod.addConstr(a[r, m] == Initial_Inventory[r, m] - quicksum(q[r, s, m] for s in range(BloodGroups) if Compatibility[r, s] == 1 and r != s))
                              
    for r in range(BloodGroups):
        for m in range(1, Remaining_Life):
            mod.addConstr(quicksum(q[r, s, m] for s in range(BloodGroups) if Compatibility[r, s] == 1 and r != s) <= 100000 * (1 - z[r, m]))
    
    for r in range(BloodGroups):
        for m in range(1, Remaining_Life):
            mod.addConstr(quicksum(Initial_Inventory[r, new_m] - quicksum(q[r, s, new_m] for s in range(BloodGroups) if Compatibility[r, s] == 1 and r != s) for new_m in range(m)) <= 100000 * z[r, m])
    
    # -------------- Setting the objective function
    
    mod.setObjective(quicksum((Emergency_Cost[r]) * g[r] for r in range(BloodGroups))
                         + quicksum(quicksum(quicksum((Order_Cost[r] - Inventory_Cost) * q[r, s, m] for m in range(1, Remaining_Life)) 
                             for r in range(BloodGroups)) for s in range(BloodGroups) if Compatibility[r, s] == 1 and r != s)
                        - Outdate_Cost *
                        (quicksum(quicksum(q[r, s, 1] for r in range(BloodGroups)) for s in range(BloodGroups) if Compatibility[r, s] == 1 and r != s)),
                        GRB.MINIMIZE)

    mod.optimize()
    
    # -------------- Generating reports
    
    # -------------- Q variable
    OutputQ0 = np.zeros([8,8], dtype=np.int64)
    for r in range(8):
        for s in range(BloodGroups):
            if s != r:
                TOutputQ0 = 0
                for m in range(42):
                    TOutputQ0 = TOutputQ0 + q[r,s,m].x
                OutputQ0[r,s] = TOutputQ0
    
    OutputQM = np.zeros([42], dtype=np.int64)
    for m in range(Remaining_Life):
        TOutputQM = 0
        for r in range(8):
            for s in range(8):
                if s != r:
                    TOutputQM = TOutputQM + q[r,s,m].x
        OutputQM[m] = TOutputQM
    
    OutputA0 = np.zeros([8,42], dtype=np.int64)
    for m in range(42):
        for r in range(8):
            OutputA0[r,m] = a[r,m].x
            
    # -------------- G variable                  
    OutputG0 = np.zeros([8], dtype=np.int64)
    for r in range(8):
        OutputG0[r] = g[r].x

    # --------------  Generating outputs                 
    if mod.status != 3 and mod.status != 1:
        OF = mod.objVal
    else:
        OF = 99999999
            
    return OF, OutputG0, OutputQ0, OutputA0, OutputQM



            
#----------------- Rolling horizon

def RollingHorizon(h, Num_of_blood_groups, Max_Shelf_life, Cost_vector, 
                            Order_Cost_Vector, Order_Cost_Vector_2, Compatibility, Hospital_sizes, 
                            Blood_Demands, Initial_Inventory, RealizedDemands, Demand0, Stock_M, 
                            Stock_T, Horizon, units_M , units_T):
    
    #-----------------Generating output files
    MyOutputList = ['OF', 'Sub_To', 'Sub_From', 'Tran', 'Inv_Gr', 
                    'Inv_Shlf', 'Inv_Shlf', 'Emegcy', 'Tran_Life']

    for i in MyOutputList:
        with open('Outputs/%s_%s_M.xlsx' % (i, h), 'w') as file:
            os.chmod('Outputs/%s_%s_M.xlsx' % (i, h), 0o755)
                
    #---------------- Rolling horizon myopic:
    Horizon_ObjectiveFunction = np.zeros([Horizon, 4], dtype=float)
    Horizon_EmergencyOrders = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
    Horizon_Substitutions_To = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
    Horizon_Substitutions_From = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
    Horizon_Transfusion = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
    Horizon_Inventory_Groups = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
    Horizon_Inventory_Shelf = np.zeros([Horizon, Max_Shelf_life], dtype=np.int64)
    Horizon_Transfusion_Life = np.zeros([Horizon, Max_Shelf_life], dtype=np.int64)
    Horizon_RegularOrders = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
         
    for z in range(Horizon):
        
        invent = np.zeros([Num_of_blood_groups], dtype=int)
        for r in range(Num_of_blood_groups):
            invent[r] = sum(Initial_Inventory[r,:])
                
        Horizon_units_M = np.zeros([Num_of_blood_groups, Max_Shelf_life, np.max(Stock_M[:]) + 1], dtype=np.int64)       
        Horizon_units_M[:] = units_M[z, :, :, :]

        Horizon_units_T = np.zeros([Num_of_blood_groups, Max_Shelf_life, np.max(Stock_T[:]) + 1], dtype=np.int64)       
        Horizon_units_T[:] = units_T[z, :, :, :]
        
        
        Order_Size_myopic = np.zeros([Num_of_blood_groups], dtype=np.int64)
          

        if z > 0 and z % 7 == 0:
            for r in range(Num_of_blood_groups):
                Order_Size = np.zeros([Max_Shelf_life], dtype=np.int64)
                if Stock_M[r] > sum(Initial_Inventory[r, :]):
                    Order_Size_myopic[r] = Stock_M[r] - sum(Initial_Inventory[r, :])
                else:
                    Order_Size_myopic[r] = 0
                    
                for i in range(Max_Shelf_life):
                    Order_Size[i] = Horizon_units_M[r, i, Order_Size_myopic[r]]
                for i in range(Max_Shelf_life):
                    Initial_Inventory[r, i] = Initial_Inventory[r, i] + Order_Size[i]
                    
        if z > 0 and z % 7 == 3:
            for r in range(Num_of_blood_groups):
                Order_Size = np.zeros([Max_Shelf_life], dtype=np.int64)
                if Stock_T[r] > sum(Initial_Inventory[r, :]):
                    Order_Size_myopic[r] = Stock_T[r] - sum(Initial_Inventory[r, :])
                else:
                    Order_Size_myopic[r] = 0
                    
                for i in range(Max_Shelf_life):
                    Order_Size[i] = Horizon_units_T[r, i, Order_Size_myopic[r]]
                for i in range(Max_Shelf_life):
                    Initial_Inventory[r, i] = Initial_Inventory[r, i] + Order_Size[i]
                            
        Horizon_RegularOrders[z, :] = Order_Size_myopic[:]
        
        for r in range(Num_of_blood_groups):
            if Demand0[r] > 0:
                if sum(Initial_Inventory[r, :]) >= Demand0[r]:
                    Horizon_Transfusion[z, r] = Horizon_Transfusion[z, r] + Demand0[r]
                    ii = 0
                    while Demand0[r] > 0 and ii <= 41:
                        if Initial_Inventory[r, ii] > 0:
                            if Initial_Inventory[r, ii] <= Demand0[r]:
                                Demand0[r] = Demand0[r] - Initial_Inventory[r, ii]
                                Horizon_Transfusion_Life[z, ii] = Horizon_Transfusion_Life[z, ii] + Initial_Inventory[r, ii]
                                Initial_Inventory[r, ii] = 0
                            else:
                                Initial_Inventory[r, ii] = Initial_Inventory[r, ii] - Demand0[r]
                                Horizon_Transfusion_Life[z, ii] = Horizon_Transfusion_Life[z, ii] + Demand0[r]
                                Demand0[r] = 0
                        ii = ii + 1
                
                else:
                    Horizon_Transfusion[z, r] = Horizon_Transfusion[z, r] + sum(Initial_Inventory[r, :])
                    ii = 0
                    while sum(Initial_Inventory[r, :]) > 0 and ii <= 41:
                        if Initial_Inventory[r, ii] > 0:
                            Demand0[r] = Demand0[r] - Initial_Inventory[r, ii]
                            Horizon_Transfusion_Life[z, ii] = Horizon_Transfusion_Life[z, ii] + Initial_Inventory[r, ii]
                            Initial_Inventory[r, ii] = 0
                        ii = ii + 1
        
        invent = np.zeros([Num_of_blood_groups], dtype=int)
        for r in range(Num_of_blood_groups):
            invent[r] = sum(Initial_Inventory[r,:])
                
        Horizon_Inventory = np.zeros([Num_of_blood_groups, Max_Shelf_life], dtype=np.int64)
        
        if sum(Demand0[:]) > 0:
            
            Answer_myopic = Myopic(Initial_Inventory, Demand0, Compatibility, Cost_vector, Order_Cost_Vector_2)
                   
            Horizon_EmergencyOrders[z, :] = Answer_myopic[1]
            Horizon_Substitutions = Answer_myopic[2]
            Horizon_Inventory = Answer_myopic[3]
            Horizon_Transfusion_Life[z, :] = Horizon_Transfusion_Life[z, :] + Answer_myopic[4]

            Demand0 = RealizedDemands[:, z+8]
    
            #--------------Calculating Objective Function and Outputs
            Regular_order = sum(Order_Cost_Vector * Horizon_RegularOrders[z, :])
            Emergency_Cost = sum(Order_Cost_Vector_2 * Horizon_EmergencyOrders[z, :])
            Inventory_Cost = Cost_vector[8] * sum(sum(Horizon_Inventory))
            Outdate_Cost = Cost_vector[9] * sum(Horizon_Inventory[:, 0])
            Horizon_ObjectiveFunction[z, :] = [Regular_order, Emergency_Cost, Inventory_Cost, Outdate_Cost]
            
            for i in range(Num_of_blood_groups):
                Horizon_Substitutions_To[z, i] = sum(Horizon_Substitutions[i, :])
                Horizon_Substitutions_From[z, i] = sum(Horizon_Substitutions[:, i])
                Horizon_Inventory_Groups[z, i] = sum(Horizon_Inventory[i, :])
                
            for i in range(Max_Shelf_life):
                Horizon_Inventory_Shelf[z, i] = sum(Horizon_Inventory[:, i])
    
    
        else:
            Horizon_Inventory[:] = Initial_Inventory  
            Demand0 = RealizedDemands[:, z+8]
            
            Regular_order = sum(Order_Cost_Vector * Horizon_RegularOrders[z, :])
            Emergency_Cost = 0
            Inventory_Cost = Cost_vector[8] * sum(sum(Horizon_Inventory))
            Outdate_Cost = Cost_vector[9] * sum(Horizon_Inventory[:, 0])
            Horizon_ObjectiveFunction[z, :] = [Regular_order, Emergency_Cost, Inventory_Cost, Outdate_Cost]
            
        
        df = pd.DataFrame(Horizon_ObjectiveFunction)
        df.to_excel('Outputs/OF_%s_M.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Substitutions_To)
        df.to_excel('Outputs/Sub_To_%s_M.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Substitutions_From)
        df.to_excel('Outputs/Sub_From_%s_M.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Transfusion)
        df.to_excel('Outputs/Tran_%s_M.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Inventory_Groups)
        df.to_excel('Outputs/Inv_Gr_%s_M.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Inventory_Shelf)
        df.to_excel('Outputs/Inv_Shlf_%s_M.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_EmergencyOrders)
        df.to_excel('Outputs/Emegcy_%s_M.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Transfusion_Life)
        df.to_excel('Outputs/Tran_Life_%s_M.xlsx' % (h), header=False, index=False)
        
        
        Initial_Inventory[:] = Horizon_Inventory
        for i in range(Max_Shelf_life - 1):
            Initial_Inventory[:, i] = Initial_Inventory[:, i + 1]
        Initial_Inventory[:, 41] = 0 
            
    
    life = np.array([np.sum(Horizon_Transfusion_Life, axis=0)])
    Horizon_Transfusion_Life = np.concatenate((Horizon_Transfusion_Life, life), axis=0)
    
    df = pd.DataFrame(Horizon_Transfusion_Life)
    df.to_excel('Outputs/Tran_Life_%s_M.xlsx' % (h), header=False, index=False)
    
    
    TotalCost = np.zeros([Num_of_blood_groups], dtype=np.int64)
    TotalCost[0] = sum(sum(Horizon_ObjectiveFunction))
    
    TotalTransfusion = np.zeros([Num_of_blood_groups], dtype=np.int64)
    for i in range(Num_of_blood_groups):
        TotalTransfusion[i] = sum(Horizon_Transfusion[:,i])
        
    TotalSubTo = np.zeros([Num_of_blood_groups], dtype=np.int64)
    for i in range(Num_of_blood_groups):
        TotalSubTo[i] = sum(Horizon_Substitutions_To[:,i])
    
    TotalSubFrom = np.zeros([Num_of_blood_groups], dtype=np.int64) 
    for i in range(Num_of_blood_groups):
        TotalSubFrom[i] = sum(Horizon_Substitutions_From[:,i])
        
    TotalEmerg = np.zeros([Num_of_blood_groups], dtype=np.int64)
    for i in range(Num_of_blood_groups):
        TotalEmerg[i] = sum(Horizon_EmergencyOrders[:,i])
    
    df0 = pd.DataFrame(TotalCost)
    df1 = pd.DataFrame(TotalTransfusion)
    df2 = pd.DataFrame(TotalSubTo)
    df3 = pd.DataFrame(TotalSubFrom)
    df4 = pd.DataFrame(TotalEmerg)
    
    with pd.ExcelWriter('Outputs/Summary_%s_M.xlsx' % (h)) as writer:  
        df0.to_excel(writer, header=False, index=False, sheet_name='TotalCost')
        df1.to_excel(writer, header=False, index=False, sheet_name='TotalTransfusion')
        df2.to_excel(writer, header=False, index=False, sheet_name='TotalSubTo')
        df3.to_excel(writer, header=False, index=False, sheet_name='TotalSubFrom')
        df4.to_excel(writer, header=False, index=False, sheet_name='TotalEmerg')
        
        
 