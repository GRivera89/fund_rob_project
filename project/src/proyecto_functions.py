import numpy as np
from copy import copy

cos=np.cos; sin=np.sin; pi=np.pi

def Trasl(x,y,z):
    T = np.array([[1, 0, 0, x],
                  [0, 1, 0, y], 
                  [0, 0, 1, z],
                  [0, 0, 0, 1]])
    return T

def Rotx(ang):
    Tx = np.array([[1, 0, 0, 0],
                   [0, np.cos(ang), -np.sin(ang), 0],
                   [0, np.sin(ang),  np.cos(ang), 0],
                   [0, 0, 0, 1]])
    return Tx

def Rotz(ang):
    Tz = np.array([[np.cos(ang), -np.sin(ang), 0, 0],
                    [np.sin(ang),  np.cos(ang), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    return Tz

def Roty(ang):
    Ty = np.array([[np.cos(ang), 0, np.sin(ang), 0],
                    [0, 1, 0, 0],
                    [-np.sin(ang),  0, np.cos(ang), 0],
                    [0, 0, 0, 1]])
    return Ty


def dh(d, theta, a, alpha):
    """
    Calcular la matriz de transformacion homogenea asociada con los parametros
    de Denavit-Hartenberg.
    Los valores d, theta, a, alpha son escalares.
    """
    # Escriba aqui la matriz de transformacion homogenea en funcion de los valores de d, theta, a, alpha
    cth = np.cos(theta);    sth = np.sin(theta)
    ca = np.cos(alpha);  sa = np.sin(alpha)
    T = np.array([[cth, -ca*sth,  sa*sth, a*cth],
                    [sth,  ca*cth, -sa*cth, a*sth],
                    [0,        sa,     ca,      d],
                    [0,         0,      0,      1]])
    return T
    
    

def fkine_tm900(q):
    """
    Calcular la cinematica directa del robot UR5 dados sus valores articulares. 
    q es un vector numpy de la forma [q1, q2, q3, q4, q5, q6]
    """

    # Matrices DH (completar), emplear la funcion dh con los parametros DH para cada articulacion
    T1 = dh( 0.1452,pi+q[0],0,pi/2);
    T2 = dh( -0.146,pi/2+q[1],0.429,0);
    T3 = dh( 0.1297,q[2],0.4115,0);
    T4 = dh( -0.106,pi/2+q[3],0,pi/2);
    T5 = dh( 0.106,pi+q[4],0,pi/2);
    T6 = dh( 0.1132,pi/2+q[5],0,0);
    # Efector final con respecto a la base
    T = T1 @ T2 @ T3 @ T4 @ T5 @ T6
    return T

def jacobian_kukakr6(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion. Retorna una matriz de 3x6 y toma como
    entrada el vector de configuracion articular q=[q1, q2, q3, q4, q5, q6]
    """
    # Crear una matriz 3x6
    J = np.zeros((3,6))
    # Transformacion homogenea inicial (usando q)
    T = fkine_tm900(q)
    # Iteracion para la derivada de cada columna
    for i in range(6):
        # Copiar la configuracion articular inicial
        dq = copy(q)
        # Incrementar la articulacion i-esima usando un delta
        dq[i] = dq[i] + delta
        # Transformacion homogenea luego del incremento (q+delta)
        dT = fkine_tm900(dq) 
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        col_i = 1/delta * (dT[0:3,3]-T[0:3,3])
        J[:,i] = col_i
    return J


def ikine_kukakr6(xdes, q0):
    """
    Calcular la cinematica inversa de UR5 numericamente a partir de la configuracion articular inicial de q0. 
    Emplear el metodo de newton
    """
    epsilon  = 0.001
    max_iter = 1000
    delta    = 0.00001

    q  = copy(q0)
    for i in range(max_iter):
        # Main loop
        J = jacobian_kukakr6(q, delta)
        T = fkine_tm900(q)
        f = T[0:3, 3]
        e = xdes - f
        q = q + np.dot(np.linalg.pinv(J), e)
        if (np.linalg.norm(e) < epsilon):
            for i in range(6):
                while q[i] > 6.28:
                    q[i] = q[i]-6.28
                while q[i] < -6.28:
                    q[i] = q[i]+6.28
            break
    
    return q

def ik_gradient_kukakr6(xdes, q0):
    """
    Calcular la cinematica inversa de UR5 numericamente a partir de la configuracion articular inicial de q0. 
    Emplear el metodo gradiente
    """
    epsilon  = 0.001
    max_iter = 1000
    delta    = 0.00001
    alfa     = 0.5
    pos_reached = False

    q  = copy(q0)
    for i in range(max_iter):
        # Main loop
        J = jacobian_kukakr6(q, delta)
        T = fkine_tm900(q)
        f = T[0:3, 3]
        e = xdes - f
        q = q + alfa*np.dot(J.T, e)
        if (np.linalg.norm(e) < epsilon):
            pos_reached = True
            break
    if pos_reached:
        print("Position reached")
    else:
        print("Position not reached")
    return q