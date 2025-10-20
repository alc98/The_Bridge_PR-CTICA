import numpy as np
############################################################################################################################<3

# Crear tablero base 
tablero = np.arange(1, 226).reshape(15, 15).astype(object)

# Colocar barcos 
barco1 = [(0, 1), (1, 1)]
barco2 = [(1, 3), (1, 4), (1, 5), (1, 6)]

for barco in [barco1, barco2]:
    for pieza in barco:
        tablero[pieza] = "BBB"

############################################################################################################################<3

# Función para recibir disparos 
def recibir_disparo(tablero_dos, coordenada):
    if tablero_dos[coordenada] == "BBB":
        tablero_dos[coordenada] = "X0X"
        print("💥 ¡Tocado!")
    elif tablero_dos[coordenada] in ["X0X", "AAA"]:
        print("⚠️ Ya has disparado aquí, elige otra casilla.")
    else:
        tablero_dos[coordenada] = "000"
        print("🌊 Agua")


recibir_disparo(tablero, (0, 1))  # Tocado
recibir_disparo(tablero, (2, 2))  # Agua
recibir_disparo(tablero, (0, 1))  # Repetido


print("\n🔹 Tablero después de los disparos:\n")

for fila in tablero:
    print(" ".join(f"{str(x).zfill(3) if isinstance(x, int) else f'{x:>3}'}" for x in fila))



############################################################################################################################<3

# Función para reparar barcos y limpiar aguas 
def arregla_barcos(tablero):
    for i in range(tablero.shape[0]):
        for j in range(tablero.shape[1]):
            valor = tablero[i, j]
            if valor == "X0X":  # Repara barcos dañados
                tablero[i, j] = "BBB"
            elif valor == "000":  # Limpia aguas y repone número original
                tablero[i, j] = str(i * tablero.shape[1] + j + 1).zfill(3)
    return tablero

# Aplicar reparación 
tablero_reparado = arregla_barcos(tablero)


print("\n🔹 Tablero reparado y limpio:\n")
for fila in tablero_reparado:
    print(" ".join(f"{str(x).zfill(3) if isinstance(x, int) else f'{x:>3}'}" for x in fila))

############################################################################################################################<3

#Funcion para que se coloquen los barcos correctamente dentro del tablero sin iterar con los limites
def coloca_barco_plus(tablero, barco):
    tablero_temp = tablero.copy()
    num_max_filas, num_max_columnas = tablero.shape

    for pieza in barco:
        fila, columna = pieza

        if fila < 0 or fila >= num_max_filas:
            print(f"❌ No puedo poner la pieza {pieza}: fuera del tablero (fila inválida)")
            return False
        if columna < 0 or columna >= num_max_columnas:
            print(f"❌ No puedo poner la pieza {pieza}: fuera del tablero (columna inválida)")
            return False

        if tablero[fila, columna] in ["BBB", "X0X"]:
            print(f"⚠️ No puedo poner la pieza {pieza}: ya hay un barco en esa posición")
            return False

        tablero_temp[fila, columna] = "BBB"

    print(f"✅ Barco colocado en posiciones: {barco}")
    return tablero_temp

############################################################################################################################<3

def coloca_barco_plus(tablero, barco):

    tablero_temp = tablero.copy()
    num_max_filas, num_max_columnas = tablero.shape

    for pieza in barco:
        fila, columna = pieza

        if fila < 0 or fila >= num_max_filas:
            print(f"❌ No puedo poner la pieza {pieza}: fuera del tablero (fila inválida)")
            print("⛔ Volviendo al tablero original...")
            return tablero  # ← Mantiene el original

        if columna < 0 or columna >= num_max_columnas:
            print(f"❌ No puedo poner la pieza {pieza}: fuera del tablero (columna inválida)")
            print("⛔ Volviendo al tablero original...")
            return tablero  

        if tablero[fila, columna] in ["BBB", "X0X"]:
            print(f"⚠️ No puedo poner la pieza {pieza}: ya hay un barco en esa posición")
            print("⛔ Volviendo al tablero original...")
            return tablero  

        tablero_temp[fila, columna] = "BBB"

    print(f"✅ Barco colocado correctamente en posiciones: {barco}")
    return tablero_temp


# Intentar colocar un nuevo barco (correcto) 
barco3 = [(3, 10), (3, 11), (3, 12)]
tablero = coloca_barco_plus(tablero, barco3)

# Intentar colocar un barco fuera del tablero (incorrecto) 
barco_mal = [(14, 15), (14, 16)]  # 16 está fuera del tablero
tablero = coloca_barco_plus(tablero, barco_mal)

print("\n🔹 Tablero final:\n")

for fila in tablero:
    print(" ".join(f"{str(x).zfill(3) if isinstance(x, int) else f'{x:>3}'}" for x in fila))


############################################################################################################################<3

# Nueva función: crear barco aleatorio y colocarlo 
def crea_barco_aleatorio(tablero, eslora=4, num_intentos=100):

    num_max_filas, num_max_columnas = tablero.shape

    for intento in range(num_intentos):
        barco = []

        # Escoger posición inicial
        pieza_original = (random.randint(0, num_max_filas - 1), random.randint(0, num_max_columnas - 1))
        orientacion = random.choice(["N", "S", "E", "O"])

        fila, columna = pieza_original
        barco.append(pieza_original)

        # Construir barco según orientación
        for _ in range(eslora - 1):
            if orientacion == "N":
                fila -= 1
            elif orientacion == "S":
                fila += 1
            elif orientacion == "E":
                columna += 1
            elif orientacion == "O":
                columna -= 1
            barco.append((fila, columna))

        # Intentar colocar barco
        tablero_temp = coloca_barco_plus(tablero, barco)
        if not np.array_equal(tablero_temp, tablero):  # 
            return tablero_temp

        print(f"🔁 Intento {intento+1}/{num_intentos}: no se pudo colocar el barco. Reintentando...")

    print("🚫 No se pudo colocar el barco tras varios intentos.")
    return tablero

############################################################################################################################<3

tablero = crea_barco_aleatorio(tablero, eslora = 6)

############################################################################################################################<3

print(tablero)


############################################################################################################################<3
############################################################################################################################<3
############################################################################################################################<3
############################################################################################################################<3