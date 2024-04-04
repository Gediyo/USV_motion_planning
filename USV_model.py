from gnc import Smtrx,Hmtrx,m2c,crossFlowDrag,sat
import numpy as np
import math
import sys


# Class Vehicle
class frigate:
    """
    frigate()                               Step input, rudder angle
    frigate('headingAutopilot',U,psi_d)     Heading autopilot, speed (m/s)
                                                    and desired heading(deg)
    """        
    def __init__(self, U = 5.0, r = 0.0):

        # Check if the speed U is within the valid range
        if (U < 5.0 or U > 12.0):
            sys.exit('The speed U should be between 5-12 m/s')          
            
        self.ref = r
                   
        # Initialize the ship model
        self.name = "Frigate"
        self.L = 100.0        # Length      
        self.deltaMax = 30   # max rudder angle (deg)  
        self.DdeltaMax = 10  # max rudder rate (deg/s)        
        self.nu  = np.array([ U, 0, 0, 0, 0, 0],float)  # velocity vector  
        self.u_actual = np.array([0],float)             # control input vector              

        self.controls = ['Rudder angle (deg)']

        
        # ROV Zefakkel (Van Amerongen 1982) 
        self.n1 = 1
        self.n3 = 0.4
        
        # interpolate to find K, T and n3 as a function of U
        U_data = np.array([ 6.0, 9.0, 12.0 ],float)
        K_data = np.array([ 0.08, 0.18, 0.23 ],float)
        T_data = np.array([ 20.0, 27.0, 21.0 ],float) 
        n3_data = np.array([ 0.4, 0.6, 0.3 ],float)         
        

        U = self.nu[0]
        self.K  = np.interp( U, U_data, K_data )
        self.T  = np.interp( U, U_data, T_data )
        self.n3 = np.interp( U, U_data, n3_data )
    
        # Heading autopilot  
        self.e_int = 0.0         # integral state, initial value
        self.wn = 0.4            # PID pole placement parameters
        self.zeta = 1
        
        # Reference model
        self.r_max = 1 * math.pi / 180   # maximum yaw rate 
        self.psi_d = 0                   # position, velocity and acc. states
        self.r_d = 0
        self.a_d = 0
        self.wn_d = self.wn / 5
        self.zeta_d = 1       

        
    def dynamics(self,nu,rudder_angle,sampleTime):
    

        # States and inputs
        delta   = rudder_angle       # rudder angle (rad)
        r       = nu[5]          # yaw velcity (rad/sec)
                            
        # Dynamics
        r_dot = (1 / self.T) * ( self.K * delta - self.n3 * r**3 - self.n1 * r )
        nu_dot = np.array( [0, 0, 0, 0, 0, r_dot], float)
        
        # Forward Euler integration [k+1]
        nu  = nu + sampleTime * nu_dot
        
        return nu
    
    
# Class Vehicle
class otter:
    """
    otter()                          Step inputs for n1 and n2
    otter('headingAutopilot',psi_d)  Heading autopilot, desired yaw angle (deg)
    """        
    def __init__(self,r = 0, V_current = 0, beta_current = 0):         
                
  
      
        self.ref = r
        self.V_c = V_current
        self.beta_c = beta_current
        self.tauX = 120                       # surge force (N)
                    
        # Initialize the Otter USV model
        self.T_n = 1.0                          # propeller time constants (s)          
        self.L = 2.0                            # Length (m)
        self.B = 1.08                           # beam (m)           
        self.nu = np.array([0, 0, 0, 0, 0, 0], float) # velocity vector    
        self.u_actual = np.array([0, 0], float)   # propeller revolution states
        self.name = "Otter USV"     
        
        self.controls = [ 
            'Left propeller shaft speed (rad/s)', 
             'Right propeller shaft speed (rad/s)' ]
        self.dimU = len(self.controls)  

        # Constants   
        g   = 9.81                            # acceleration of gravity (m/s^2)
        rho = 1025                            # density of water
        
        m = 55.0                              # mass (kg)
        mp = 25.0                             # Payload (kg) 
        self.m_total = m + mp
        rp = np.array([0,   0, -0.35],float)  # location of payload (m)        
        rg = np.array([0.2, 0, -0.2],float)   # CG for hull only (m)  
        rg = (m*rg + mp*rp) / (m + mp)        # CG corrected for payload 
        self.S_rg = Smtrx(rg) 
        self.H_rg = Hmtrx(rg)
        self.S_rp = Smtrx(rp)        

        R44 = 0.4 * self.B                    # radii of gyration (m)
        R55 = 0.25 * self.L
        R66 = 0.25 * self.L
        T_yaw = 1.0                            # time constant in yaw (s)
        Umax = 6 * 0.5144                     # max forward speed (m/s)

        # Data for one pontoon
        self.B_pont  = 0.25  # beam of one pontoon (m)
        y_pont  = 0.395      # distance from centerline to waterline centroid (m)
        Cw_pont = 0.75       # waterline area coefficient (-)
        Cb_pont = 0.4        # block coefficient, computed from m = 55 kg
        
        # Inertia dyadic, volume displacement and draft
        nabla = (m + mp) / rho                                    # volume
        self.T = nabla / (2 * Cb_pont * self.B_pont*self.L)       # draft      
        Ig_CG = m * np.diag( np.array( [R44**2, R55**2, R66**2]) )
        self.Ig = Ig_CG - m * self.S_rg @ self.S_rg - mp * self.S_rp @ self.S_rp  

        # Experimental propeller data including lever arms
        self.l1 = -y_pont                    # lever arm, left propeller (m)
        self.l2 = y_pont                     # lever arm, right propeller (m)
        self.k_pos = 0.02216/2               # Positive Bollard, one propeller 
        self.k_neg = 0.01289/2               # Negative Bollard, one propeller 
        self.n_max =  math.sqrt((0.5*24.4 * g)/self.k_pos) # max. prop. rev.
        self.n_min = -math.sqrt((0.5*13.6 * g)/self.k_neg) # min. prop. rev. 

        # MRB_CG = [ (m+mp) * I3  O3      (Fossen 2021, Chapter 3)
        #               O3       Ig ]        
        MRB_CG = np.zeros( (6,6) )
        MRB_CG[0:3, 0:3] = (m + mp) * np.identity(3) 
        MRB_CG[3:6, 3:6] = self.Ig
        MRB = self.H_rg.T @ MRB_CG @ self.H_rg

        # Hydrodynamic added mass (best practice)  
        Xudot = -0.1 * m   
        Yvdot = -1.5 * m
        Zwdot = -1.0 * m
        Kpdot = -0.2 * self.Ig[0,0]
        Mqdot = -0.8 * self.Ig[1,1]
        Nrdot = -1.7 * self.Ig[2,2]        
        
        self.MA = -np.diag([Xudot, Yvdot, Zwdot, Kpdot, Mqdot, Nrdot]) 
        
        # System mass matrix
        self.M = MRB + self.MA
        self.Minv = np.linalg.inv(self.M)
        
        # Hydrostatic quantities (Fossen 2021, Chapter 4)
        Aw_pont = Cw_pont * self.L * self.B_pont # waterline area, one pontoon 
        I_T = 2 * (1/12)*self.L * self.B_pont**3 * \
            (6*Cw_pont**3/((1+Cw_pont)*(1+2*Cw_pont))) + 2 * Aw_pont * y_pont**2
        I_L = 0.8 * 2 * (1/12) * self.B_pont * self.L**3
        KB = (1/3) * ( 5 * self.T / 2 - 0.5 * nabla / (self.L * self.B_pont) )
        BM_T = I_T/nabla        # BM values
        BM_L = I_L/nabla
        KM_T = KB + BM_T        # KM values
        KM_L = KB + BM_L
        KG = self.T - rg[2]
        GM_T = KM_T - KG        # GM values
        GM_L = KM_L - KG

        G33 = rho * g * (2 * Aw_pont)      # spring stiffness
        G44 = rho * g * nabla * GM_T
        G55 = rho * g * nabla * GM_L
        G_CF = np.diag([0, 0, G33, G44, G55, 0])  # spring stiff. matrix in CF
        LCF = -0.2
        H = Hmtrx(np.array([LCF, 0.0, 0.0]))    # transform G_CF from CF to CO 
        self.G = H.T @ G_CF @ H

        # Natural frequencies
        w3 = math.sqrt( G33 / self.M[2,2] )       
        w4 = math.sqrt( G44 / self.M[3,3] )
        w5 = math.sqrt( G55 / self.M[4,4] )

        # Linear damping terms (hydrodynamic derivatives)
        Xu = -24.4 * g / Umax             # specified using the maximum speed        
        Yv = 0
        Zw = -2 * 0.3 * w3 * self.M[2,2]  # specified using relative damping
        Kp = -2 * 0.2 * w4 * self.M[3,3]
        Mq = -2 * 0.4 * w5 * self.M[4,4]
        Nr = -self.M[5,5] / T_yaw         #specified by the time constant T_yaw       
        
        self.D = -np.diag([Xu, Yv, Zw, Kp, Mq, Nr])   
  
        # Trim: theta = -7.5 deg corresponds to 13.5 cm less height aft 
        self.trim_moment = 0
        self.trim_setpoint = 280;   
        
        # Propeller configuration/input matrix 
        B = self.k_pos * \
            np.array([
                [1, 1],
                [-self.l1, -self.l2] ])
        self.Binv = np.linalg.inv(B)
    
        # Heading autopilot
        self.e_int = 0           # integral state   
        self.wn = 1.2            # PID pole placement
        self.zeta = 0.8
        
        # Reference model
        self.r_max = 10 * math.pi / 180   # maximum yaw rate 
        self.psi_d = 0           # angle, angular rate and angular acc. states
        self.r_d = 0
        self.a_d = 0
        self.wn_d = self.wn / 5  # desired natural frequency in yaw
        self.zeta_d = 1          # desired relative damping ratio
        

    def dynamics(self,eta,nu,u_actual,sampleTime):
        """
        [nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime) integrates 
        the Otter USV equations of motion using Euler's method.
        """   

        # Input vector
        n = np.array( [u_actual[0], u_actual[1]] )  

        # Current velocities
        u_c = self.V_c * math.cos(self.beta_c - eta[5])  # current surge vel., eta[5] = yaw angle
        v_c = self.V_c * math.sin(self.beta_c - eta[5])  # current sway vel.

        nu_c = np.array([u_c,v_c,0,0,0,0],float)     # current velocity vector
        nu_r = nu - nu_c                             # relative velocity vector
        
        # Rigid body and added mass Coriolis and centripetal matrices 
        # CRB_CG = [ (m+mp) * Smtrx(nu2)          O3   (Fossen 2021, Chapter 6)
        #              O3                   -Smtrx(Ig*nu2)  ] 
        CRB_CG = np.zeros( (6,6) )
        CRB_CG[0:3, 0:3] =  self.m_total * Smtrx(nu[3:6])
        CRB_CG[3:6, 3:6] = -Smtrx( np.matmul(self.Ig,nu[3:6]) )    
        CRB = self.H_rg.T @ CRB_CG @ self.H_rg  # transform CRB from CG to CO 
 
        CA = m2c(self.MA, nu_r)
        CA[5,0] = 0 # assume that the Munk moment in yaw can be neglected
        CA[5,1] = 0 # if nonzero, must be balanced by adding nonlinear damping

        C = CRB + CA      
        
        # Ballast
        g_0 = np.array([0.0, 0.0, 0.0, 0.0, self.trim_moment, 0.0])   

        # Control forces and moments - with propeller revolution saturation 
        thrust = np.zeros(2)
        for i in range(0,2):
            
            n[i] = sat(n[i],self.n_min,self.n_max) # saturation, physical limits

            if n[i] > 0:                                    # positive thrust              
                thrust[i] = self.k_pos * n[i] * abs(n[i])  
            else:                                           # negative thrust 
                thrust[i] = self.k_neg * n[i] * abs(n[i]) 

        # Control forces and moments
        tau = np.array( [thrust[0] + thrust[1], 0, 0, 0, 0, 
                       -self.l1 * thrust[0] - self.l2 * thrust[1] ])

        # Hydrodynamic linear damping + nonlinear yaw damping
        tau_damp = -np.matmul(self.D, nu_r)  
        tau_damp[5] = tau_damp[5] - 10 * self.D[5,5] * abs(nu_r[5]) * nu_r[5]

        # State derivatives (with dimension)
        tau_crossflow = crossFlowDrag(self.L,self.B_pont,self.T,nu_r)
        sum_tau = tau + tau_damp + tau_crossflow - np.matmul(C,nu_r) \
            - np.matmul(self.G,eta) - g_0
       
        nu_dot = np.matmul(self.Minv, sum_tau)   # USV dynamics
        
        # Forward Euler integration [k+1]
        nu = nu + sampleTime * nu_dot
    

        return nu

