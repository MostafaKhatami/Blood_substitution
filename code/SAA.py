

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
def Model_II(rollingz, Stock_M, Stock_T, Initial_Inventory, Demand0, Demand, Compatibility, Cost_vector, Cost_vector_2, Flag_stock, Flag_life, Horizon_units_M, Horizon_units_T):
    
    #Stock_M = Stock_T
    mod = Model ("Substitution_II")

    # ---------------------Inputs:
    BloodGroups = 8
    MatrixDimensions = Demand.shape
    Periods = MatrixDimensions[1]
    Scenarios = MatrixDimensions[2]
    MatrixDimensions = Initial_Inventory.shape
    Remaining_Life = MatrixDimensions[1]
    
    Order_Cost = np.zeros(8)
    Order_Cost[0:8] = Cost_vector[0:8]
    Emergency_Cost = np.zeros(8)
    Emergency_Cost[0:8] = Cost_vector_2[0:8]
    Inventory_Cost = Cost_vector[8]
    Outdate_Cost = Cost_vector[9]
    
    dayofweek = rollingz % 7 + 1
    
    if dayofweek == 1:
        Monday = 100
        Thursday = 3
        
    if dayofweek == 2:
        Monday = 6
        Thursday = 2
        
    if dayofweek == 3:
        Monday = 5
        Thursday = 1
        
    if dayofweek == 4:
        Monday = 4
        Thursday = 100
        
    if dayofweek == 5:
        Monday = 3
        Thursday = 6
        
    if dayofweek == 6:
        Monday = 2
        Thursday = 5
        
    if dayofweek == 7:
        Monday = 1
        Thursday = 4
        
    # ---------Generating variables:
    
    # q represents the number of units of group r, with age m, that is used 
    # for patients of group s at time period t
    q0 = {}
    for r in range(BloodGroups):
        for s in range(BloodGroups):
            for m in range(Remaining_Life):
                q0[r, s, m] = mod.addVar(vtype=GRB.INTEGER)
                        
    q = {}
    for r in range(BloodGroups):
        for s in range(BloodGroups):
            for t in range(1, Periods):
                for m in range(Remaining_Life):
                    for i in range(Scenarios):
                        q[r, s, t, m, i] = mod.addVar(vtype=GRB.INTEGER)
    
    #--------------------------------
    # g represents the number of emergency-ordered units of group r
    g0 = {}
    for r in range(BloodGroups):
        g0[r] = mod.addVar(vtype=GRB.INTEGER)
            
    g = {}
    for r in range(BloodGroups):
        for t in range(1, Periods):
            for i in range(Scenarios):
                g[r, t, i] = mod.addVar(vtype=GRB.INTEGER)
    
    #--------------------------------
    # O (the Capital letter) represents the total number of units of group r,
    # in the order placed at period t, while o (the small letter) represents 
    # those units with remaining shelf life m
    bigO_M = {}
    bigO_T = {}
    for r in range(BloodGroups):
        for t in range(1, Periods):
            if t == Monday:
                for i in range(Scenarios):
                    bigO_M[r, t, i] = mod.addVar(vtype=GRB.INTEGER)
            
            if t == Thursday:
                for i in range(Scenarios):
                    bigO_T[r, t, i] = mod.addVar(vtype=GRB.INTEGER)
    
    o_M = {}
    o_T = {}
    for r in range(BloodGroups):
        for t in range(1, Periods):
            if t == Monday:
                for m in range(Remaining_Life):
                    for i in range(Scenarios):
                        o_M[r, t, m, i] = mod.addVar(vtype=GRB.INTEGER)
            if t == Thursday:
                for m in range(Remaining_Life):
                    for i in range(Scenarios):
                        o_T[r, t, m, i] = mod.addVar(vtype=GRB.INTEGER)

                            
    #--------------------------------
    # v represents the available inventory level of group r with remaining 
    # shelf life m at the beggining of time period t                
    v0 = {}
    for r in range(BloodGroups):
        for m in range(Remaining_Life):
            v0[r, m] = mod.addVar(vtype=GRB.INTEGER)

    v = {}
    for r in range(BloodGroups):
        for t in range(1, Periods):
            for m in range(Remaining_Life):
                for i in range(Scenarios):
                    v[r, t, m, i] = mod.addVar(vtype=GRB.INTEGER)
    
    #--------------------------------
    # a represents the available inventory level of group r with remaining 
    # shelf life m at the end of time period t 
    a0 = {}
    for r in range(BloodGroups):
        for m in range(Remaining_Life):
            a0[r, m] = mod.addVar(vtype=GRB.INTEGER)
                
    a = {}
    for r in range(BloodGroups):
        for t in range(1, Periods):
            for m in range(Remaining_Life):
                for i in range(Scenarios):
                    a[r, t, m, i] = mod.addVar(vtype=GRB.INTEGER)
    
    #--------------------------------
    # Auxiliary binary variables
    x0 = {}
    for r in range(BloodGroups):
        x0[r] = mod.addVar(vtype=GRB.BINARY)
    
    x = {}
    for r in range(BloodGroups):
        for t in range(1, Periods):
            for i in range(Scenarios):
                x[r, t, i] = mod.addVar(vtype=GRB.BINARY)
    
    y0 = {}
    for r in range(BloodGroups):
        y0[r] = mod.addVar(vtype=GRB.BINARY)
    
    y = {}
    for r in range(BloodGroups):
        for t in range(1, Periods):
            for i in range(Scenarios):
                y[r, t, i] = mod.addVar(vtype=GRB.BINARY)
         
    z0 = {}
    for r in range(BloodGroups):
        for m in range(1, Remaining_Life):
            z0[r, m] = mod.addVar(vtype=GRB.BINARY)
    
    z = {}
    for r in range(BloodGroups):
        for t in range(1, Periods):
            for m in range(1, Remaining_Life):
                for i in range(Scenarios):
                    z[r, t, m, i] = mod.addVar(vtype=GRB.BINARY)
    
    if Flag_stock == 2:
        Stock_var = {}
        for r in range(BloodGroups):
            Stock_var[r] = mod.addVar(vtype=GRB.INTEGER)
    
    if Flag_life == 2:
       w_M = {}
       w_T = {}
       for r in range(BloodGroups):
           for phi in range(int(Stock_M[r]) + 1):
               for i in range(Scenarios):
                   w_M[r, phi, i] = mod.addVar(vtype=GRB.BINARY) 
           
           for phi in range(int(Stock_T[r]) + 1):
               for i in range(Scenarios):
                   w_T[r, phi, i] = mod.addVar(vtype=GRB.BINARY) 
    
   
    auxil_M = {}
    auxil_T = {}
    for r in range(BloodGroups):
        for i in range(Scenarios):
            auxil_M[r, i] = mod.addVar(lb=-GRB.INFINITY, vtype=GRB.INTEGER)
            auxil_T[r, i] = mod.addVar(lb=-GRB.INFINITY, vtype=GRB.INTEGER)
                
                
                
    mod.update()
    
    
    # -------------- Generating constraints:
    
    for s in range(BloodGroups):
        mod.addConstr(quicksum(quicksum(q0[r, s, m] for m in range(Remaining_Life)) 
            for r in range(BloodGroups) if Compatibility[r, s] == 1) + g0[s] == Demand0[s])
                
    for i in range(Scenarios):
        for s in range(BloodGroups):
            for t in range(1, Periods):
                mod.addConstr(quicksum(quicksum(q[r, s, t, m, i] for m in range(Remaining_Life)) 
                    for r in range(BloodGroups) if Compatibility[r, s] == 1) + g[s, t, i] == Demand[s, t, i])
    
    for r in range(BloodGroups):
        for m in range(Remaining_Life):
            mod.addConstr(v0[r, m] == Initial_Inventory[r, m])    

    if Flag_life == 2:
        for i in range(Scenarios):
            for r in range(BloodGroups):
                for t in range(1, Periods):
                    
                    if t == Monday:
                        if Flag_stock == 1:
                            if t > 1:
                                mod.addConstr(auxil_M[r, i] == int(Stock_M[r]) - quicksum(a[r, t-1, m, i] for m in range(Remaining_Life)))
                                mod.addConstr(bigO_M[r, t, i] == max_(0, auxil_M[r, i]))
                            if t == 1:
                                mod.addConstr(auxil_M[r, i] == int(Stock_M[r]) - quicksum(a0[r, m] for m in range(Remaining_Life)))
                                mod.addConstr(bigO_M[r, t, i] == max_(0, auxil_M[r, i]))
                                
                        elif Flag_stock == 2:
                            if t > 1:
                                mod.addConstr(bigO_M[r, t, i] == Stock_var[r] - quicksum(a[r, t-1, m, i] for m in range(Remaining_Life)))
                            if t == 1:
                                mod.addConstr(bigO_M[r, t, i] == Stock_var[r] - quicksum(a0[r, m] for m in range(Remaining_Life)))
                            
                        mod.addConstr(quicksum(w_M[r, phi, i] for phi in range(int(Stock_M[r]) + 1)) == 1)
                        mod.addConstr(bigO_M[r, t, i] == quicksum(phi * w_M[r, phi, i] for phi in range(int(Stock_M[r]) + 1)))
                        
                        for m in range(Remaining_Life - 1):
                            mod.addConstr(o_M[r, t, m, i] == quicksum(Horizon_units_M[r, m, phi] * w_M[r, phi, i] for phi in range(int(Stock_M[r]) + 1)))
                            
                            if t > 1:
                                mod.addConstr(v[r, t, m, i] == a[r, t-1, m+1, i] + o_M[r, t, m, i])
                            if t == 1:
                                mod.addConstr(v[r, t, m, i] == a0[r, m+1] + o_M[r, t, m, i])
                                
                        mod.addConstr(o_M[r, t, Remaining_Life - 1, i] == bigO_M[r, t, i] - quicksum(o_M[r, t, m_prime, i] for m_prime in range(Remaining_Life - 1)))
                        mod.addConstr(v[r, t, Remaining_Life - 1, i] == o_M[r, t, Remaining_Life - 1, i])
                    
                    
                    elif t == Thursday:
                        if Flag_stock == 1:
                            if t > 1:
                                mod.addConstr(auxil_T[r, i] == int(Stock_T[r]) - quicksum(a[r, t-1, m, i] for m in range(Remaining_Life)))
                                mod.addConstr(bigO_T[r, t, i] == max_(0, auxil_T[r, i]))
                            if t == 1:
                                mod.addConstr(auxil_T[r, i] == int(Stock_T[r]) - quicksum(a0[r, m] for m in range(Remaining_Life)))
                                mod.addConstr(bigO_T[r, t, i] == max_(0, auxil_T[r, i]))
                                
                        elif Flag_stock == 2:
                            if t > 1:
                                mod.addConstr(bigO_T[r, t, i] == Stock_var[r] - quicksum(a[r, t-1, m, i] for m in range(Remaining_Life)))
                            if t == 1:
                                mod.addConstr(bigO_T[r, t, i] == Stock_var[r] - quicksum(a0[r, m] for m in range(Remaining_Life)))
                            
                        mod.addConstr(quicksum(w_T[r, phi, i] for phi in range(int(Stock_T[r]) + 1)) == 1)
                        mod.addConstr(bigO_T[r, t, i] == quicksum(phi * w_T[r, phi, i] for phi in range(int(Stock_T[r]) + 1)))
                        
                        for m in range(Remaining_Life - 1):
                            mod.addConstr(o_T[r, t, m, i] == quicksum(Horizon_units_T[r, m, phi] * w_T[r, phi, i] for phi in range(int(Stock_T[r]) + 1)))
                            
                            if t > 1:
                                mod.addConstr(v[r, t, m, i] == a[r, t-1, m+1, i] + o_T[r, t, m, i])
                            if t == 1:
                                mod.addConstr(v[r, t, m, i] == a0[r, m+1] + o_T[r, t, m, i])
                                
                        mod.addConstr(o_T[r, t, Remaining_Life - 1, i] == bigO_T[r, t, i] - quicksum(o_T[r, t, m_prime, i] for m_prime in range(Remaining_Life - 1)))
                        mod.addConstr(v[r, t, Remaining_Life - 1, i] == o_T[r, t, Remaining_Life - 1, i])
                        
                    else:
                        for m in range(Remaining_Life - 1):
                            if t == 1:
                                mod.addConstr(v[r, t, m, i] == a0[r, m+1])
                            elif t > 1:
                                mod.addConstr(v[r, t, m, i] == a[r, t-1, m+1, i])
                                
                        mod.addConstr(v[r, t, Remaining_Life - 1, i] == 0)
            
                            
    for r in range(BloodGroups):
        for m in range(Remaining_Life):
            mod.addConstr(quicksum(q0[r, s, m] for s in range(BloodGroups) if Compatibility[r, s] == 1) <= v0[r, m])
    
    for i in range(Scenarios):
        for r in range(BloodGroups):
            for t in range(1, Periods):
                for m in range(Remaining_Life):
                    mod.addConstr(quicksum(q[r, s, t, m, i] for s in range(BloodGroups) if Compatibility[r, s] == 1) <= v[r, t, m, i])
                        
    for r in range(BloodGroups):
        for m in range(Remaining_Life):
            mod.addConstr(a0[r, m] == v0[r, m] - quicksum(q0[r, s, m] for s in range(BloodGroups) if Compatibility[r, s] == 1))
    
    for i in range(Scenarios):
        for r in range(BloodGroups):
            for t in range(1, Periods):
                for m in range(Remaining_Life):
                    mod.addConstr(a[r, t, m, i] == v[r, t, m, i] - quicksum(q[r, s, t, m, i] for s in range(BloodGroups) if Compatibility[r, s] == 1))         
                
    for s in range(BloodGroups):
        mod.addConstr(quicksum(quicksum(q0[r, s, m] for m in range(Remaining_Life)) for r in range(BloodGroups) if Compatibility[r, s] == 1 and r != s) <= 100000 * (1 - x0[s]))

    for i in range(Scenarios):
        for s in range(BloodGroups):
            for t in range(1, Periods):
                mod.addConstr(quicksum(quicksum(q[r, s, t, m, i] for m in range(Remaining_Life)) for r in range(BloodGroups) if Compatibility[r, s] == 1 and r != s) <= 100000 * (1 - x[s, t, i]))

    for s in range(BloodGroups):
        mod.addConstr(quicksum(a0[s, m] for m in range(Remaining_Life)) <= 100000 * x0[s])

    for i in range(Scenarios):
        for s in range(BloodGroups):
            for t in range(1, Periods):
                mod.addConstr(quicksum(a[s, t, m, i] for m in range(Remaining_Life)) <= 100000 * x[s, t, i])
    
    for r in range(BloodGroups):
        mod.addConstr(quicksum(quicksum(q0[r, s, m] for m in range(Remaining_Life)) for s in range(BloodGroups) if Compatibility[r, s] == 1 and s != r) <= 100000 * (1 - y0[r]))

    for i in range(Scenarios):
        for r in range(BloodGroups):
            for t in range(1, Periods):
                mod.addConstr(quicksum(quicksum(q[r, s, t, m, i] for m in range(Remaining_Life)) for s in range(BloodGroups) if Compatibility[r, s] == 1 and s != r) <= 100000 * (1 - y[r, t, i]))
    
    for r in range(BloodGroups):
        mod.addConstr(g0[r] <= 100000 * y0[r])

    for i in range(Scenarios):
        for r in range(BloodGroups):
            for t in range(1, Periods):
                mod.addConstr(g[r, t, i] <= 100000 * y[r, t, i])
    
    for r in range(BloodGroups):
        for m in range(1, Remaining_Life):
            mod.addConstr(quicksum(q0[r, s, m] for s in range(BloodGroups) if Compatibility[r, s] == 1) <= 100000 * (1 - z0[r, m]))
    
    for i in range(Scenarios):
        for r in range(BloodGroups):
            for m in range(1, Remaining_Life):
                for t in range(1, Periods):
                    mod.addConstr(quicksum(q[r, s, t, m, i] for s in range(BloodGroups) if Compatibility[r, s] == 1) <= 100000 * (1 - z[r, t, m, i]))
                    
    for r in range(BloodGroups):
        for m in range(1, Remaining_Life):
            mod.addConstr(quicksum(v0[r, new_m] - quicksum(q0[r, s, new_m] for s in range(BloodGroups) if Compatibility[r, s] == 1) for new_m in range(m)) <= 100000 * z0[r, m])
    
           
    for i in range(Scenarios):
        for r in range(BloodGroups):
            for m in range(1, Remaining_Life):
                for t in range(1, Periods):
                    mod.addConstr(quicksum(v[r, t, new_m, i] - quicksum(q[r, s, t, new_m, i] for s in range(BloodGroups) if Compatibility[r, s] == 1) for new_m in range(m)) <= 100000 * z[r, t, m, i])
                      
    # -------------- Setting the objective function
    
    mod.setObjective(quicksum((Emergency_Cost[r]) * g0[r] for r in range(BloodGroups))
                         + Inventory_Cost * quicksum(quicksum(a0[r, m] for m in range(Remaining_Life))
                         for r in range(BloodGroups)) + Outdate_Cost * quicksum(a0[r, 0] for r in range(BloodGroups))
                        + (1/float(Scenarios)) * 
                        quicksum(quicksum(Order_Cost[r] * (quicksum(bigO_M[r, t, i] for t in range(1, Periods) if t == Monday) + 
                                                  quicksum(bigO_T[r, t, i] for t in range(1, Periods) if t == Thursday)) for r in range(BloodGroups)) +
                        quicksum((Emergency_Cost[r]) * (quicksum(g[r, t, i] for t in range(1, Periods)))
                        for r in range(BloodGroups)) 
                        + Inventory_Cost * (quicksum(quicksum(quicksum(a[r, t, m, i] for m in range(Remaining_Life))
                        for t in range(1, Periods)) for r in range(BloodGroups))) 
                        + Outdate_Cost *
                        (quicksum(quicksum(a[r, t, 0, i] for t in range(1, Periods)) for r in range(BloodGroups)))  
                        for i in range(Scenarios)),
                        GRB.MINIMIZE)

    mod.optimize()
    
    if mod.status == 3:
        mod.computeIIS()
        mod.write("iismodel.ilp")
 
    # -------------- Generating reports
    
    # -------------- Q variable
    OutputQ0 = np.zeros([8,8], dtype=np.int64)
    for r in range(8):
        for s in range(BloodGroups):
            TOutputQ0 = 0
            for m in range(42):
                TOutputQ0 = TOutputQ0 + q0[r,s,m].x
            OutputQ0[r,s] = TOutputQ0
    
    OutputQM = np.zeros([42], dtype=np.int64)
    for m in range(Remaining_Life):
        TOutputQM = 0
        for r in range(8):
            for s in range(8):
                TOutputQM = TOutputQM + q0[r,s,m].x
        OutputQM[m] = TOutputQM
        
    # -------------- G variable                  
    OutputG0 = np.zeros([8], dtype=np.int64)
    for r in range(8):
        OutputG0[r] = g0[r].x
    
    # -------------- A variable                  
    OutputA0 = np.zeros([8,42], dtype=np.int64)
    for m in range(42):
        for r in range(8):
            OutputA0[r,m] = a0[r,m].x  
    
    # --------------  Generating outputs                 
    if mod.status != 3 and mod.status != 1:
        OF = mod.objVal
    else:
        OF = 99999999
            
    return OF, OutputG0, OutputQ0, OutputA0, OutputQM  

            
#----------------- Rolling horizon

def RollingHorizon(h, Num_of_blood_groups, Max_Shelf_life, Cost_vector, 
                            Order_Cost_Vector, Order_Cost_Vector_2, Compatibility, Hospital_sizes, 
                            Blood_Demands, Initial_Inventory_II, RealizedDemands, Demand0_II, Stock_M, 
                            Stock_T, Horizon, units_M , units_T, Flag_stock, Flag_life,
                            Duration_of_Scenario, num_of_Years, Active_Scenarios,
                            preDemand, Demand_II):
    
    #-----------------Generating output files
    MyOutputList = ['OF', 'Sub_To', 'Sub_From', 'Tran', 'Inv_Gr', 
                    'Inv_Shlf', 'Inv_Shlf', 'Emegcy', 'Tran_Life']

    for i in MyOutputList:
        with open('Outputs/%s_%s_II.xlsx' % (i, h), 'w') as file:
            os.chmod('Outputs/%s_%s_II.xlsx' % (i, h), 0o755)
                
    #---------------- Rolling horizon empirical rule:
    Horizon_ObjectiveFunction_II = np.zeros([Horizon, 4], dtype=float)
    Horizon_EmergencyOrders_II = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
    Horizon_Substitutions_To_II = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
    Horizon_Substitutions_From_II = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
    Horizon_Transfusion_II = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
    Horizon_Inventory_Groups_II = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
    Horizon_Inventory_Shelf_II = np.zeros([Horizon, Max_Shelf_life], dtype=np.int64)
    Horizon_Transfusion_Life_II = np.zeros([Horizon, Max_Shelf_life], dtype=np.int64)
    Horizon_ObjectiveFunction_II_2 = np.zeros([Horizon, 1], dtype=float)
    Horizon_RegularOrders_II = np.zeros([Horizon, Num_of_blood_groups], dtype=np.int64)
         
    for z in range(Horizon):
        
        print("z is equal to", z)
        
        Horizon_units_M = np.zeros([Num_of_blood_groups, Max_Shelf_life, np.max(Stock_M[:]) + 1], dtype=np.int64)       
        Horizon_units_M[:] = units_M[z, :, :, :]

        Horizon_units_T = np.zeros([Num_of_blood_groups, Max_Shelf_life, np.max(Stock_T[:]) + 1], dtype=np.int64)       
        Horizon_units_T[:] = units_T[z, :, :, :]
        
        Order_Size_II = np.zeros([Num_of_blood_groups], dtype=np.int64)
          
        if z > 0 and z % 7 == 0:
            for r in range(Num_of_blood_groups):
                Order_Size = np.zeros([Max_Shelf_life], dtype=np.int64)
                if Stock_M[r] > sum(Initial_Inventory_II[r, :]):
                    Order_Size_II[r] = Stock_M[r] - sum(Initial_Inventory_II[r, :])
                else:
                    Order_Size_II[r] = 0
                    
                for i in range(Max_Shelf_life):
                    Order_Size[i] = Horizon_units_M[r, i, Order_Size_II[r]]
                for i in range(Max_Shelf_life):
                    Initial_Inventory_II[r, i] = Initial_Inventory_II[r, i] + Order_Size[i]
        
        if z > 0 and z % 7 == 3:
            for r in range(Num_of_blood_groups):
                Order_Size = np.zeros([Max_Shelf_life], dtype=np.int64)
                if Stock_T[r] > sum(Initial_Inventory_II[r, :]):
                    Order_Size_II[r] = Stock_T[r] - sum(Initial_Inventory_II[r, :])
                else:
                    Order_Size_II[r] = 0
                    
                for i in range(Max_Shelf_life):
                    Order_Size[i] = Horizon_units_T[r, i, Order_Size_II[r]]
                for i in range(Max_Shelf_life):
                    Initial_Inventory_II[r, i] = Initial_Inventory_II[r, i] + Order_Size[i]
                            
        Horizon_RegularOrders_II[z, :] = Order_Size_II[:]
        
        Answer_II = Model_II(z, Stock_M, Stock_T, Initial_Inventory_II, Demand0_II, Demand_II, Compatibility, Cost_vector, Order_Cost_Vector_2, Flag_stock, Flag_life, Horizon_units_M, Horizon_units_T)
        
        Horizon_ObjectiveFunction_II_2[z, 0] = Answer_II[0]
        
        Horizon_EmergencyOrders_II[z, :] = Answer_II[1]
        Horizon_Substitutions_II = Answer_II[2]
        Horizon_Inventory_II = Answer_II[3]
        Horizon_Transfusion_Life_II[z, :] = Answer_II[4]
        
        Initial_Inventory_II = Answer_II[3]
        Demand0_II = RealizedDemands[:, z+8]   #changed

        #--------------Calculating Objective Function and Outputs
        Regular_order = sum(Order_Cost_Vector * Horizon_RegularOrders_II[z, :])
        Emergency_Cost = sum(Order_Cost_Vector_2 * Horizon_EmergencyOrders_II[z, :])
        Inventory_Cost = Cost_vector[8] * sum(sum(Horizon_Inventory_II))
        Outdate_Cost = Cost_vector[9] * sum(Horizon_Inventory_II[:, 0])
        Horizon_ObjectiveFunction_II[z, :] = [Regular_order, Emergency_Cost, Inventory_Cost, Outdate_Cost]
        
        for i in range(Num_of_blood_groups):
            Horizon_Substitutions_To_II[z, i] = sum(Horizon_Substitutions_II[i, :])
            Horizon_Substitutions_From_II[z, i] = sum(Horizon_Substitutions_II[:, i])
            Horizon_Transfusion_II[z, i] = Horizon_Substitutions_II[i, i]
            Horizon_Inventory_Groups_II[z, i] = sum(Horizon_Inventory_II[i, :])
            
        for i in range(Max_Shelf_life):
            Horizon_Inventory_Shelf_II[z, i] = sum(Horizon_Inventory_II[:, i])

        
        df = pd.DataFrame(Horizon_ObjectiveFunction_II)
        df.to_excel('Outputs/OF_%s_II.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Substitutions_To_II)
        df.to_excel('Outputs/Sub_To_%s_II.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Substitutions_From_II)
        df.to_excel('Outputs/Sub_From_%s_II.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Transfusion_II)
        df.to_excel('Outputs/Tran_%s_II.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Inventory_Groups_II)
        df.to_excel('Outputs/Inv_Gr_%s_II.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Inventory_Shelf_II)
        df.to_excel('Outputs/Inv_Shlf_%s_II.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_EmergencyOrders_II)
        df.to_excel('Outputs/Emegcy_%s_II.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Transfusion_Life_II)
        df.to_excel('Outputs/Tran_Life_%s_II.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_ObjectiveFunction_II_2)
        df.to_excel('Outputs/OF_Model_%s_II.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Substitutions_To_II - Horizon_Transfusion_II)
        df.to_excel('Outputs/Sub_To_exc_%s_II.xlsx' % (h), header=False, index=False)
        
        df = pd.DataFrame(Horizon_Substitutions_From_II - Horizon_Transfusion_II)
        df.to_excel('Outputs/Sub_From_exc_%s_II.xlsx' % (h), header=False, index=False)
        
        #--------------Updating Inventory

        for i in range(Active_Scenarios - 1):
            Index = int(np.ceil((i + 1)/2))
            for k in range(Duration_of_Scenario):
                for j in range(Num_of_blood_groups):
                    if i % 2 == 0:
                        Demand_II[j, k+1, i] = preDemand[Index-1, j, k+1 + z+1]
                    else:
                        Demand_II[j, k+1, i] = preDemand[Index-1, j, k + 7+1 + z+1]
        
        for k in range(Duration_of_Scenario):
            for j in range(Num_of_blood_groups):
                Demand_II[j, k+1, Active_Scenarios - 1] = RealizedDemands[j, k+1 + z+1]
         
        for r in range(Num_of_blood_groups):
            Horizon_RegularOrders_II[z, r] = Order_Size_II[r]
         
            
        for i in range(Max_Shelf_life - 1):
            Initial_Inventory_II[:, i] = Initial_Inventory_II[:, i + 1]
        Initial_Inventory_II[:, 41] = 0    
    
    
    life = np.array([np.sum(Horizon_Transfusion_Life_II, axis=0)])
    Horizon_Transfusion_Life_II = np.concatenate((Horizon_Transfusion_Life_II, life), axis=0)
    
    df = pd.DataFrame(Horizon_Transfusion_Life_II)
    df.to_excel('Outputs/Tran_Life_%s_II.xlsx' % (h), header=False, index=False)
    
    TotalCost = np.zeros([Num_of_blood_groups], dtype=np.int64)
    TotalCost[0] = sum(sum(Horizon_ObjectiveFunction_II))
    
    TotalTransfusion = np.zeros([Num_of_blood_groups], dtype=np.int64)
    for i in range(Num_of_blood_groups):
        TotalTransfusion[i] = sum(Horizon_Transfusion_II[:,i])
        
    TotalSubTo = np.zeros([Num_of_blood_groups], dtype=np.int64)
    for i in range(Num_of_blood_groups):
        TotalSubTo[i] = sum(Horizon_Substitutions_To_II[:,i])
    
    TotalSubFrom = np.zeros([Num_of_blood_groups], dtype=np.int64) 
    for i in range(Num_of_blood_groups):
        TotalSubFrom[i] = sum(Horizon_Substitutions_From_II[:,i])
        
    TotalEmerg = np.zeros([Num_of_blood_groups], dtype=np.int64)
    for i in range(Num_of_blood_groups):
        TotalEmerg[i] = sum(Horizon_EmergencyOrders_II[:,i])
    
    df0 = pd.DataFrame(TotalCost)
    df1 = pd.DataFrame(TotalTransfusion)
    df2 = pd.DataFrame(TotalSubTo)
    df3 = pd.DataFrame(TotalSubFrom)
    df4 = pd.DataFrame(TotalEmerg)
    
    with pd.ExcelWriter('Outputs/Summary_%s_II.xlsx' % (h)) as writer:  
        df0.to_excel(writer, header=False, index=False, sheet_name='TotalCost')
        df1.to_excel(writer, header=False, index=False, sheet_name='TotalTransfusion')
        df2.to_excel(writer, header=False, index=False, sheet_name='TotalSubTo')
        df3.to_excel(writer, header=False, index=False, sheet_name='TotalSubFrom')
        df4.to_excel(writer, header=False, index=False, sheet_name='TotalEmerg')
        
        
 