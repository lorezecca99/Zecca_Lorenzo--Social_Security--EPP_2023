import numpy as np
import matplotlib.pyplot as plt

#Set up experiment
def SS(data,iterations):
    """ The function, and the project as a whole, replicates the work 
    from Conesa and Krueger (1999) and assesses a social security reform.
    We consider a discrete time overlapping generations model, 
    where the economy is populated by a continuum with given mass 
    growing at a constant rate "n" of ex-ante identical individuals.
    The function computes the two steady states: 
    one in which the government runs a social security system, financed 
    through taxes on labor; and another one, where the there is no public 
    pension system, and eranings from labor are not taxed. In this project, we will cover only 
    the comparion between the two steady states, neglecting the transition dynamics analysis """

    def Reform(R):
        if  R== 0:
            tau = 0.11
            k0 = 3.3254
            l0 = 0.3414
            

        else:
            tau = 0.0
            k0 = 4.2434
            l0 = 0.3565
        return [tau,k0,l0]

    # Demographics
    J = 66                        # life-span
    JR = 46                       # age of retirement
    tR = J - JR + 1               # length of retirement
    tW = JR - 1                   # length of working life
    n = 0.012                     # Population growth
    z = [3.0, 0.5]                # Labor efficiency unit
    # Preferences
    beta = 0.97                   # discount factor
    sigma = 2                     # coefficient of relative risk aversion
    gamma = 0.42                  # weight on consumption

    # Production
    alpha = 0.36                  # production elasticity of capital
    delta = 0.06                  # rate of depreciation

    # Idiosyncratic productivities
    N = 2                         # number of shock realizations at each age

    # Age-efficiency profile
    eff = data #'eff_profiles_py.txt'
    #eff=eff[:,1]
    # Distribution of newborns over shocks
    z_init = np.array([0.2037, 0.7963])

    # Transition matrix 
    tran = np.zeros((2, 2))
    tran[0, 0] = 0.9261
    tran[0, 1] = 1.0 - 0.9261
    tran[1, 1] = 0.9811
    tran[1, 0] = 1.0 - 0.9811

    # Measure of each generation
    mass = np.ones(J)
    for j in range(1, J):
        mass[j] = mass[j-1] / (1 + n)
    mass = mass / np.sum(mass)

    # Capital grid
    maxkap = 30                              # maximum value of capital grid
    minkap = 0.001                            # minimum value of capital grid
    nk = 200                                  # number of grid points
    inckap = (maxkap - minkap) / (nk - 1)     # distance between points
    #kap = np.linspace(minkap, maxkap, nk)
    aux=np.arange(1, 201).reshape((1, 200))
    kap= minkap+inckap*(aux-1); 


    for R in range(2):
        tau,k0,l0=Reform(R)
        ## Auxiliary parameters
        ## Tolerance levels for capital, labor and pension benefits
        tolk=1e-4
        tollab=1e-4

        nq=iterations                                # Max number of iterations
        q=0                                  # Counter for iterations

        k1=k0+10
        l1=l0+10

    # Initializations for backward induction
        vR = np.zeros((nk, tR))  # value function of retired agents
        kapRopt = np.ones((nk, tR))  # optimal savings of retired agents (store INDEX of k' in capital grid, not k' itself!)

        vW = np.zeros((N, nk, tW))  # value function of workers
        kapWopt = np.ones((N, nk, tW))  # optimal savings of workers

        labopt = np.ones((N, nk, tW))  # optimal labor supply
    

        neg=-1e10;                              # very small number


        while q < nq and (abs(k1 - k0) > tolk or abs(l1 - l0) > tollab):
            q += 1
        
            print('\nIteration {} out of {} \n'.format(q, nq))

            # Prices
            r0 = alpha * (k0 ** (alpha - 1)) * (l0 ** (1 - alpha)) - delta
            w0 = (1 - alpha) * (k0 ** alpha) * (l0 ** (-alpha))
        
            # Pension benefit
            b = tau * w0 * l0 / sum(mass[JR-1:])
        
            ##########################################################
            ## BACKWARD INDUCTION
            ##########################################################
        
            # Retired households
        
            # Last period utility
            cons = (1 + r0) * kap + b  # last period consumption (vector!)
            util = (cons ** ((1 - sigma) * gamma)) / (1 - sigma)  # last period utility (vector!)
            vR[:, tR-1] = util  # last period indirect utility (vector!)
        
            for i in range(tR - 1):  # age
                for j in range(nk):  # assets today
            
                    # Initialize right-hand side of Bellman equation
                    vmin = neg
                    l = -1
                
                    # Loop over all k's in the capital grid to find the value,
                    # which gives max of the right-hand side of Bellman equation
                
                    while l < nk-1:  # assets tomorrow
                        l += 1
                        kap0 = kap[0,j]  # current asset holdings
                        kap1 = kap[0,l]  # future asset holdings
                    
                        # Instantaneous utility
                        cons = (1 + r0) * kap0 + b - kap1
                    
                        if cons <= 0:
                            util = neg
                            l = nk-1
                        else:
                            util = (cons ** ((1 - sigma) * gamma)) / (1 - sigma)
                    
                        # Right-hand side of Bellman equation
                        v0 = util + beta * vR[l, i + 1]
                    
                        # Store indirect utility and optimal saving 
                        if v0 > vmin:
                            vR[j, i] = v0
                            kapRopt[j, i] = l
                            vmin = v0
            # Working households
            for i in range(tW):           # age
                for e in range(N):              # productivity shock
                    for j in range(nk):         # assets today

                    # Initialize right-hand side of Bellman equation
                        vmin = neg
                        l = -1

                    # Loop over all k's in the capital grid to find the value,
                    # which gives max of the right-hand side of Bellman equation
                        while l < nk-1:  # assets tomorrow
                            l += 1

                            kap0 = kap[0,j]   # current asset holdings
                            kap1 = kap[0,l]   # future asset holdings

                        # Optimal labor supply
                            lab = (gamma*(1-tau)*z[e]*eff[i]*w0-(1-gamma)*((1+r0)*kap0-kap1))/((1-tau)*w0*z[e]*eff[i])

                        # Check feasibility of labor supply
                            if lab > 1:
                                lab = 1
                            elif lab < 0:
                                lab = 0

                    # Instantaneous utility
                            cons = (1+r0)*kap0+(1-tau)*w0*z[e]*eff[i]*lab-kap1

                            if cons <= 0:
                                util = neg
                                l = nk-1
                            else:
                                util = (((cons**gamma)*(1-lab)**(1-gamma))**(1-sigma))/(1-sigma)

                        # Right-hand side of Bellman equation
                            if i == tW-1:       # retired next period
                                v0 = util + beta*vR[l, 1]
                            else:
                                v0 = util + beta*(tran[e, 0]*vW[0, l, i+1]+tran[e, 1]*vW[1, l, i+1])

                    # Store indirect utility, optimal saving and labor
                            if v0 > vmin:
                                vW[e, j, i] = v0
                                kapWopt[e, j, i] = l
                                labopt[e, j, i] = lab
                                vmin = v0

            ## Aggregate capital stock and employment

            #Initializations

            kgen=np.zeros(J)       # Aggregate capital for each generation

             # Distribution of workers over capital and shocks for each working cohort
            gkW=np.zeros((N,nk,tW))

             ## Newborns

            gkW[0,0,0]=z_init[0]*mass[0]  # Mass of high shock agents at age 1 
            gkW[1,0,0]=z_init[1]*mass[0] # Mass of low shock agents at age 1 

            #Distribution of agents over capital for each cohort (pooling togetherboth productivity shocks)
            gk=np.zeros((nk,J))     
            gk[0,0]=mass[0] # Distribution of newborns over capital   

            # Aggregate labor supply by generation
            labgen=np.zeros(tW)

            #Distribution of retirees over capital
            gkR=np.zeros((nk,tR))
        
            #Workers
            for ind_age in range(tW):            # iterations over cohorts
                for ind_k in range(nk):         #current asset holdings
                    for ind_e in range(N):      # current shock
                        ind_kk = int(kapWopt[ind_e, ind_k, ind_age])  # optimal saving (as index in asset grid)

                        for ind_ee in range(N):  # tomorrow's shock

                            if ind_age < tW-1:
                                gkW[ind_ee, ind_kk, ind_age+1] += tran[ind_e, ind_ee] * gkW[ind_e, ind_k, ind_age] / (1+n)
                            elif ind_age == tW-1:  # need to be careful because workers transit to retirees
                                gkR[ind_kk, 0] += tran[ind_e, ind_ee] * gkW[ind_e, ind_k, ind_age] / (1+n)
                # Aggregate labor by age
                labgen[ind_age] = 0

                for ind_k in range(nk):
                    for ind_e in range(N):
                        labgen[ind_age] += z[ind_e] * eff[ind_age] * labopt[ind_e, ind_k, ind_age] * gkW[ind_e, ind_k, ind_age] 
                        
                # Aggregate capital by age (for workers)
                for ind_k in range(nk):
                    if ind_age < tW-1:
                        gk[ind_k, ind_age+1] = np.sum(gkW[:, ind_k, ind_age+1])
                    else:
                        gk[ind_k, ind_age+1] = gkR[ind_k, 0]

                kgen[ind_age+1]=kap @ gk[:, ind_age+1]

            ## Retirees
            for ind_age in range(tR-1): # iterations over cohort #tR-2
                for ind_k in range(nk):          # current asset holdings
                    ind_kk = int(kapRopt[ind_k, ind_age])  # optimal saving (as index in asset grid)
                    gkR[ind_kk, ind_age+1] += gkR[ind_k, ind_age]/(1+n)

                # Distribution by capital and age
                gk[:, tW+ind_age+1] = gkR[:, ind_age+1]
                # Aggregate capital by age
                kgen[tW+ind_age+1] = kap @ gk[:, tW+ind_age+1]

            k1 = np.sum(kgen)
            l1 = np.sum(labgen)

        # Update the guess on capital and labor
            k0 = 0.95 * k0 + 0.05 * k1
            l0 = 0.95 * l0 + 0.05 * l1



        # Display equilibrium results
        #print('      k0         l0       w         r         b   ')
        #print(np.array([k0, l0, w0, r0, b]))

        # Prices
        r0 = alpha * (k0**(alpha-1)) * (l0**(1-alpha)) - delta
        w0 = (1-alpha) * (k0**alpha) * (l0**(-alpha))

        if R==0:
            kgen0=kgen
            labgen0=labgen
            r0_0=r0
            w0_0=w0
            K_0=k0
            L_0=l0
            b_0=b
            labopt_0=labopt
            V0_0=np.mean(vW[:,0,0])
        else:
            kgen1=kgen
            labgen1=labgen
            r0_1=r0
            w0_1=w0
            K_1=k0
            L_1=l0
            b_1=b
            labopt_1=labopt
            V0_1=np.mean(vW[:,0,0])
    #Computing (average) earnings by age
    earningsW_0 = np.zeros((N,nk,tW))
    earningsW_1 = np.zeros((N,nk,tW))

    for d in range(tW):
        for ii in range(nk):
            for jj in range(N):
                earningsW_0[jj,ii,d] = z[jj]*eff[d]*labopt_0[jj,ii,d]
                earningsW_1[jj,ii,d] = z[jj]*eff[d]*labopt_1[jj,ii,d]
    
    earnings_0 = np.mean(earningsW_0, 0) #earningsW_0 is a 2x200x45
    earnings_0 = np.mean(earnings_0, 0) #45x1, initial ss

    earnings_1 = np.mean(earningsW_1, 0) #earningsW_1 is a 2x200x45
    earnings_1 = np.mean(earnings_1, 0) #45x1, final ss
    #Computing (average) consumption by age
    consgen_0=np.zeros((J))
    consgen_0[J-1]=(1+r0_0)*kgen0[J-1]+b_0*mass[J-1]
    consgen_1=np.zeros((J))
    consgen_1[J-1]=(1+r0_1)*kgen1[J-1]+b_1*mass[J-1]

    for j in range(tW,J-1):
        consgen_0[j]=(1+r0_0)*kgen0[j]+b_0*mass[j]-kgen0[j+1]
        consgen_1[j]=(1+r0_1)*kgen1[j]+b_1*mass[j]-kgen1[j+1]
    for j in range(0, tW):
        consgen_0[j] = (1+r0_0)*kgen0[j] + (1-tau)*w0_0*(z[0]*z_init[0] + z[1]*z_init[1])*eff[j]*labgen0[j] - kgen0[j+1]
        consgen_1[j] = (1+r0_1)*kgen1[j] + (1-tau)*w0_1*(z[0]*z_init[0] + z[1]*z_init[1])*eff[j]*labgen0[j] - kgen1[j+1]
    for j in range(J):
        consgen_0[j]=consgen_0[j]/mass[j]
        consgen_1[j]=consgen_1[j]/mass[j]
    #Computing (average) savings by age, where kgen initially stands for "capital accumulated by each generation"
    for j in range(J):
        kgen0[j] = kgen0[j] / mass[j]
        kgen1[j] = kgen1[j] / mass[j]
    #Computing (average) labor supply by age, where labgen initially stands for "labor supplied by each generation"
    for j in range(tW):
        labgen0[j] = labgen0[j] / mass[j]
        labgen1[j] = labgen1[j] / mass[j]

    # Computing welfare effects of the reform
    """Let us now study the pension system reform in terms of welfare effects for each generation. 
        To do so, we use the consumption equivalent variation (CEV). Such an index measures 
        how much (in percent) the consumption of a new-born individual has to be increased, in all future periods
        (keeping leisure unchanged), in the old steady state so that his/her future utility equals 
        that under the policy reform. For instance, if the CEV for an individual in the initial 
        steady state economy is 1.0%, it means that he/she prefers an economy where the
        policy is put into place and needs to be compensated with a rise in consumption by 1.0% in all future periods
        with leisure constant at the initial steady-state itself."""
    
    CEV=(V0_1/V0_0)**(1/(gamma*(1-sigma)))-1

    return [kgen0, kgen1, labgen0, labgen1,r0_0,
            w0_0, K_0,L_0,b_0, r0_1, 
            w0_1,K_1,L_1,b_1, earnings_0,
            earnings_1,consgen_0, consgen_1,V0_0, V0_1,CEV]
print('Both sets of iterations completed')

def plot_SS_K(kgen0,kgen1):
    """(Average) Savings by age."""
    J = 66                        # life-span
    #JR = 46                       # age of retirement
    #tR = J - JR + 1               # length of retirement
    #tW = JR - 1                   # length of working life
    plt.figure(1)
    plt.plot(range(20, J+20), kgen0, 'b-', markersize=14, linewidth=1.5)
    plt.plot(range(20, J+20), kgen1, 'r--', markersize=14, linewidth=1.5)
    plt.xlabel('Age')
    plt.legend(['Initial SS', 'Final SS (reform)'])
    plt.grid(True)
    return plt.figure(1)

def plot_SS_L(labgen0,labgen1):
    """(Average) Labor (effective) supply by age."""
    #J = 66                        # life-span
    JR = 46                       # age of retirement
    #tR = J - JR + 1               # length of retirement
    tW = JR - 1                   # length of working life
    plt.figure(2)
    plt.plot(range(20, tW+20), labgen0, 'b-', markersize=14, linewidth=1.5)
    plt.plot(range(20, tW+20), labgen1, 'r--', markersize=14, linewidth=1.5)
    plt.xlabel('Age')
    plt.ylabel('Labor')
    plt.legend(['Initial SS', 'Final SS (reform)'])
    plt.grid(True)
    return plt.figure(2)

def plot_SS_E(earnings_0,earnings_1):
    """(Average) Earnings before tax from labor by age."""
    #J = 66                        # life-span
    JR = 46                       # age of retirement
    #tR = J - JR + 1               # length of retirement
    tW = JR - 1                   # length of working life
    plt.figure(3)
    plt.plot(range(20, tW+20), earnings_0, 'b-', markersize=14, linewidth=1.5)
    plt.plot(range(20, tW+20), earnings_1, 'r--', markersize=14, linewidth=1.5)
    plt.xlabel('Age')
    plt.ylabel('Earnings')
    plt.legend(['Initial SS', 'Final SS (reform)'])
    plt.grid(True)
    return plt.figure(3)

def plot_SS_C(cons_0,cons_1):
    """(Average) Consumption by age."""
    J = 66                        # life-span
    #JR = 46                       # age of retirement
    #tR = J - JR + 1               # length of retirement
    #tW = JR - 1                   # length of working life
    plt.figure(4)
    plt.plot(range(20, J+20), cons_0, 'b-', markersize=14, linewidth=1.5)
    plt.plot(range(20, J+20), cons_1, 'r--', markersize=14, linewidth=1.5)
    plt.xlabel('Age')
    plt.ylabel('Consumption')
    plt.legend(['Initial SS', 'Final SS (reform)'])
    plt.grid(True)
    return plt.figure(4)