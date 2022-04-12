from ortools.sat.python import cp_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from scipy.stats import pearsonr
#===Cargar las diferentes bases de datos===
pcac_mac_gpi_clientes = pd.read_csv('./Adjuntos_Modelo_Capacidad/datos_modelo_capacidad/pcac_mac_gpi_clientes.csv')
pcac_mac_gpi_ecas = pd.read_csv('./Adjuntos_Modelo_Capacidad/datos_modelo_capacidad/pcac_mac_gpi_ecas.csv')
pcac_oportunidades_comer = pd.read_csv('./Adjuntos_Modelo_Capacidad/datos_modelo_capacidad/pcac_oportunidades_comer.csv')
pcac_mac_gpi_tenencia_prod = pd.read_csv('./Adjuntos_Modelo_Capacidad/datos_modelo_capacidad/pcac_mac_gpi_tenencia_prod.csv')
pcac_planta_comercial2 = pd.read_csv('./Adjuntos_Modelo_Capacidad/datos_modelo_capacidad/pcac_planta_comercial2.csv')
pcac_encuesta = pd.read_csv('./Adjuntos_Modelo_Capacidad/datos_modelo_capacidad/pcac_encuesta.csv')
pcac_capacidad_gerentes = pd.read_csv('./Adjuntos_Modelo_Capacidad/datos_modelo_capacidad/pcac_capacidad_gerentes.csv')

#===Obtener el numero de clientes, gerentes y ejecutivos===
num_gerentes = len(pcac_planta_comercial2['cod_gte_inv'].value_counts()) #Se asegura que no hayan valores repetidos
num_clientes = len(pcac_mac_gpi_clientes['num_doc_cli'].value_counts())
num_ejecutivos = len(pcac_mac_gpi_clientes['cod_ejec_bco'].value_counts())

#===Calcular los tiempos promedios de cada operacion====
tiempos_todos_producto = pcac_encuesta.groupby(['cod_producto'])['total_promedio_tiempo_min_x_actividad'].median().sort_values(ascending=False) #Se calcula la mediana por ser mas robusta
num_oport_clientes = pcac_oportunidades_comer.groupby(['num_doc_cli', 'cod_producto']).size().unstack(fill_value=0)

#=Crear un vector con los tiempos de los productos que se encuentran en las oportunidades comerciales, si el cod_producto no tiene un tiempo asociado, se toma la mediana=
tiempos_oport_producto = [tiempos_todos_producto.loc[f] if f in tiempos_todos_producto.index else tiempos_todos_producto.median()  for f in num_oport_clientes.keys()]

#=Calcular el tiempo de atencion de cada cliente segun los productos ofertados, si el cliente no tiene producto ofertados, se le asignara la mediana de los tiempos=
aux_tiempo_cliente=[]
for cliente in pcac_mac_gpi_clientes['num_doc_cli'].values:
    if (cliente in num_oport_clientes.index):
        aux_tiempo_cliente.append(np.dot(num_oport_clientes.loc[cliente].values,tiempos_oport_producto))
    else:
         aux_tiempo_cliente.append(np.NaN)

aux_tiempo_cliente = np.array(aux_tiempo_cliente)
aux_tiempo_cliente[np.isnan(aux_tiempo_cliente)] = np.nanmedian(aux_tiempo_cliente)
pcac_mac_gpi_clientes['tiempo']=aux_tiempo_cliente.tolist()

#===Crear un dataframe con la informacion de los ejecutivos, que contenga el numero de clientes, numero de clientes A+B, ratio A+B, tiempo total de los clientes==

ejecutivos_info = pcac_mac_gpi_clientes.groupby(['cod_ejec_bco', 'marca_mac_inv']).size().unstack(fill_value = 0)
ejecutivos_info['num_clientes'] =  (ejecutivos_info['A']+ejecutivos_info['B']+ejecutivos_info['C'])
ejecutivos_info['AB'] = (ejecutivos_info['A']+ejecutivos_info['B'])
ejecutivos_info['AB_relacion'] = ejecutivos_info['AB']/ ejecutivos_info['num_clientes']#Se calcula la relacion de clientes categoria A y B
ejecutivos_info['score'] = pcac_mac_gpi_clientes.groupby(['cod_ejec_bco'])['score_modelo'].mean() #Este Score puede ser el profit o ganancia
ejecutivos_info['region'] = pcac_mac_gpi_clientes.groupby(['cod_ejec_bco', 'cod_region_ejec_bco']).size().index.get_level_values(1).values #Se toma el codigo de la region de cada ejecutivo
ejecutivos_info['tiempo_total_clientes']=pcac_mac_gpi_clientes.groupby(['cod_ejec_bco'])['tiempo'].sum() # Se suma el tiempo de todos los clientes de cada ejecutivo

ejecutivos_info.sort_values(by = 'AB_relacion',ascending = False,inplace = True) #Se organiza segun la relacion de clientes categoria A y B

#========Creacion del modelo y matrices para la solucion usando Google ORTools y CP-SAT
#====Matriz de la funcion de costo=====
#Esta matriz tiene el numero de de clientes A+B que cada ejecutivo asignara a un gerente. En este caso, es igual independientemente del gerente

J = [[ejecutivos_info['AB'].iloc[ejecutivo] for gerente in range(num_gerentes)] for ejecutivo in range(num_ejecutivos)]

#====Matrices de restricciones====
#===Matriz de tiempos====
#Esta Matriz contiene el tiempo que le agregaria cada ejecutivo a un gerente. En este caso, es igual independientemente del gerente

T = [[ejecutivos_info['tiempo_total_clientes'].iloc[ejecutivo] for gerente in range(num_gerentes)] for ejecutivo in range(num_ejecutivos)]

#===Matriz de zona====
#Esta Matriz contiene 1 si el ejecutivo y el gerente son de la misma zona, 2 si no. Se utiliza 2 si no son de la misma zona, para crear la restriccion que la zona del
#ejecutivo y el gerente deba ser igual (1) o el ejecutivo no asignado (0).Para el orden de los gerentes, se utiliza el archivo pcac_capacidad_gerentes.

Z = [[1  if ejecutivos_info['region'].iloc[ejecutivo]==pcac_capacidad_gerentes['cod_region_gte_inv'].iloc[gerente] else 2 for gerente in range(num_gerentes)] for ejecutivo in range(num_ejecutivos)]

#=====Definicion del modelo, variables, funcional de costo y restricciones======
#===Creacion del modelo usando CP-SAT===
Bancolombia = cp_model.CpModel()

#===Definicion de la variable de decision=====
#Esta variable binaria de dimensiones num_ejecutivos*num_gerentes. El valor de 1 en x[i][j] significa que el ejecutivo i sera asignado al gerente j

x = [[Bancolombia.NewBoolVar(f'x[{gerente},{ejecutivo}]') for gerente in range(num_gerentes)] for ejecutivo in range(num_ejecutivos)]

#===Definicion de las restricciones=====
# Cada ejecutivo puede ser asignado unicamente a un gerente
for ejecutivo in range(num_ejecutivos):
    Bancolombia.AddAtMostOne(x[ejecutivo][gerente] for gerente in range(num_gerentes))

# El ejecutivo solo puede ser asignado a un gerente de su misma zona. 
for ejecutivo in range(num_ejecutivos):
    #Bancolombia.AddAtMostOne(x[ejecutivo][gerente]*Z[ejecutivo][gerente] for gerente in range(num_gerentes))
    Bancolombia.Add(sum(x[ejecutivo][gerente]*Z[ejecutivo][gerente] for gerente in range(num_gerentes))<=1)

# El tiempo total asignado a un gerente, no puede ser mayor al tiempo disponible
for gerente in range(num_gerentes):
    Bancolombia.Add(sum(int(T[ejecutivo][gerente])*x[ejecutivo][gerente] for ejecutivo in range(num_ejecutivos)) <= pcac_capacidad_gerentes['tiempo_restante'].iloc[gerente])


#====Definicion de la funcion objetivo====
#La funcion objetivo es la sumatoria del producto entre la matriz J y la variable de asignacion x. En este caso, se busca maximizar dicha funcion objetivo
Funcion_objetivo = []
for ejecutivo in range(num_ejecutivos):
    for gerente in range(num_gerentes):
       Funcion_objetivo.append(J[ejecutivo][gerente]*x[ejecutivo][gerente])
Bancolombia.Maximize(sum(Funcion_objetivo)) 

#====Resolver el problema de optimizacion propuesto=====
solver = cp_model.CpSolver()
status = solver.Solve(Bancolombia)

#====Crear una matriz con la solucion====
#Por facilidad es mejor tener una matriz con las asignaciones
X_matriz = [[solver.BooleanValue(x[ejecutivo][gerente]) for gerente in range(num_gerentes)]for ejecutivo in range(num_ejecutivos)]

#=Guardar la matriz de asignacion usando json=
with open("X_matriz", "w") as fp:  
    json.dump(X_matriz, fp)
#=Cargar la matriz de asignacion usando json=
# with open("test", "r") as fp:
#      X_matriz = json.load(fp)

#=Guardar el valor de la funcion de costo usando json=
J_value = solver.ObjectiveValue()
with open("J_value", "w") as fp:  
    json.dump(J_value, fp)
#=Cargar el valor de la funcion de costo usando json=
# with open("J_value", "r") as fp:
#      J_value = json.load(fp)

#==Verificar que un ejecutivo sea asignado unicamente a un gerente==
errores_multiple_asignacion = 0
ejecutivos_asignados = 0
for ejecutivo in range(num_ejecutivos):
    if sum(X_matriz[ejecutivo][:])==1:
        ejecutivos_asignados += 1
    elif sum(X_matriz[ejecutivo][:])>1:
        errores_multiple_asignacion += 1
        
#==Verificar que la restriccion de tiempo se cumple===

errores_tiempo_gerentes = 0
tiempos_restantes_gerentes=[]
for gerente in range(num_gerentes):
    aux = [f*t for f,t in zip(X_matriz[:][gerente],T[:][gerente])]
    tiempos_restantes_gerentes.append(pcac_capacidad_gerentes['tiempo_restante'].iloc[gerente]-sum(aux))
    if sum(aux)>pcac_capacidad_gerentes['tiempo_restante'].iloc[gerente]:
        errores_tiempo_gerentes += 1
   
#==Verificar restriccion de zonas===
errores_zonas = 0
for ejecutivo in range(num_ejecutivos):
        if 1 in X_matriz[ejecutivo][:]:
            if Z[ejecutivo][X_matriz[ejecutivo][:].index(1)]>1:
                errores_zonas += 1 
            
            
#===Crear archivo de asignacion de clientes solo si no hay errores==

#Columnas siguiendo archivo de muestra
template = pd.read_csv('./Adjuntos_Modelo_Capacidad/resultado_prueba.csv',delimiter=',')
lista_clientes_asignados = {name: [] for name in list(template.columns)}
lista_clientes_no_asignados = {name: [] for name in list(template.columns)}
if (errores_zonas==0) and (errores_multiple_asignacion==0) and (errores_tiempo_gerentes==0):
    for ejecutivo in range(num_ejecutivos):
        #Validar si el ejecutivo fue asignado
        num_clientes_ejecutivo = len(pcac_mac_gpi_clientes[pcac_mac_gpi_clientes['cod_ejec_bco']==ejecutivos_info.index[ejecutivo]]['num_doc_cli'].values)
        if sum(X_matriz[ejecutivo][:])==1:
            cod_gte_inv = pcac_capacidad_gerentes['cod_gte_inv'].iloc[X_matriz[ejecutivo][:].index(1)]
            aux_cod_gte_inv = [cod_gte_inv for i in range(num_clientes_ejecutivo)]
            num_doc_gte_inv = pcac_planta_comercial2[pcac_planta_comercial2['cod_gte_inv']==cod_gte_inv]['num_doc_gte_inv'].values[0]
            aux_num_doc_gte_inv = [num_doc_gte_inv for i in range(num_clientes_ejecutivo)]
            lista_clientes_asignados['num_doc_cli'].extend(pcac_mac_gpi_clientes[pcac_mac_gpi_clientes['cod_ejec_bco']==ejecutivos_info.index[ejecutivo]]['num_doc_cli'].values)
            lista_clientes_asignados['cod_tipo_doc_cli'].extend(pcac_mac_gpi_clientes[pcac_mac_gpi_clientes['cod_ejec_bco']==ejecutivos_info.index[ejecutivo]]['cod_tipo_doc_cli'].values)
            lista_clientes_asignados['cod_ejec_bco'].extend(pcac_mac_gpi_clientes[pcac_mac_gpi_clientes['cod_ejec_bco']==ejecutivos_info.index[ejecutivo]]['cod_ejec_bco'].values)
            lista_clientes_asignados['cod_gte_inv'].extend(aux_cod_gte_inv)
            lista_clientes_asignados['num_doc_gte_inv'].extend(aux_num_doc_gte_inv)
            
        else:
            aux_cod_gte_inv = [None for i in range(num_clientes_ejecutivo)]
            aux_num_doc_gte_inv = [None for i in range(num_clientes_ejecutivo)]
            lista_clientes_no_asignados['num_doc_cli'].extend(pcac_mac_gpi_clientes[pcac_mac_gpi_clientes['cod_ejec_bco']==ejecutivos_info.index[ejecutivo]]['num_doc_cli'].values)
            lista_clientes_no_asignados['cod_tipo_doc_cli'].extend(pcac_mac_gpi_clientes[pcac_mac_gpi_clientes['cod_ejec_bco']==ejecutivos_info.index[ejecutivo]]['cod_tipo_doc_cli'].values)
            lista_clientes_no_asignados['cod_ejec_bco'].extend(pcac_mac_gpi_clientes[pcac_mac_gpi_clientes['cod_ejec_bco']==ejecutivos_info.index[ejecutivo]]['cod_ejec_bco'].values)
            lista_clientes_no_asignados['cod_gte_inv'].extend(aux_cod_gte_inv)
            lista_clientes_no_asignados['num_doc_gte_inv'].extend(aux_num_doc_gte_inv)
else:
     print('No se cumplen todas las restricciones')       

            
pd.DataFrame.from_dict(lista_clientes_asignados).to_csv('./resultado_prueba.csv',sep=',',index=False)
pd.DataFrame.from_dict(lista_clientes_no_asignados).to_csv('./lista_no_asignados.csv',sep=',',index=False)
          
