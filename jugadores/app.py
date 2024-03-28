import streamlit as st
import numpy as np
from keras.models import load_model
from joblib import load
import pymongo
import os
import datetime
import time
from dotenv import load_dotenv
load_dotenv()

model = load_model("jugadores_bestmodel.keras")
scaler = load("jugadores_scaler.joblib")

def guardar_datos_en_bbdd(muestra_nueva):
    client = pymongo.MongoClient(
        f"mongodb+srv://{os.environ.get('MONGO_USER')}:{os.environ.get('MONGO_PASSWORD')}@cluster0.2fbpmxe.mongodb.net/"  # DEFINE CLUSTER
    )
    db = client["tesis"]  # CHANGE DB NAME
    json_insert = {"dateTime": datetime.datetime.now(), "values": muestra_nueva}
    db["jugadores"].insert_one(json_insert)


# API PREDICCIÓN CON RED NEURONAL
def predice_clases(datos):
    st.write(datos)
    datos_st = scaler.transform(datos)
    st.write(datos_st)

    # Ejemplo de una función simple (puedes cambiar esto)
    resultado = model.predict(datos_st)
    predicted_classes = np.argmax(resultado, axis=1)

    return predicted_classes[0]


punctuations = {
    "Siempre": 5,
    "Casi siempre": 4,
    "A menudo": 3,
    "Casi nunca": 2,
    "Nunca": 1,
    None: 0,
    "Totalmente de acuerdo": 5,
    "De acuerdo": 4,
    "Ni de acuerdo ni en desacuerdo": 3,
    "En desacuerdo": 2,
    "Totalmente en desacuerdo": 1,
}


opciones = ["Siempre", "Casi siempre", "A menudo", "Casi nunca", "Nunca"]
opciones_satisfaccion = [
    "Totalmente en desacuerdo",
    "En desacuerdo",
    "Ni de acuerdo ni en desacuerdo",
    "De acuerdo",
    "Totalmente de acuerdo",
]


# Títulos y subtítulos
st.title(
    "Cuestionario de Conducta Apropiada en la Educación Física y el Deporte y habilidades sociales."
)
st.write("Doctorando: Prof. Jesús Martínez")

with st.expander("Datos cualitativos"):
    sexo = st.radio("Elige el sexo", ["Hombre", "Mujer"], index=None, horizontal=True)
    grupo = st.radio(
        "Elige el grupo", ["Futbol", "Baloncesto"], index=None, horizontal=True
    )
    categ_deportiva = st.radio(
        "Elige la categoría deportiva",
        ["Hasta benjamin", "cadete y juvenil", "adultos"],
        index=None,
        horizontal=True,
    )


with st.expander("Destrezas apropiadas a ganar"):
    # Crear variables para almacenar los resultados
    destrezas_ganar = [None] * 5

    # Preguntas y opciones
    preguntas_ganar = [
        "¿Evitas criticar al que perdió o a los que perdieron?",
        "¿Aceptas complementos de otros al ganar?",
        "¿Aportas sugerencias a otros de manera respetuosa?",
        "¿Demuestras apreciación a oponentes y compañeros de equipo?",
        "¿Te recompensa a ti mismo y te mantienes motivado sin reírte de los demás?",
    ]

    st.write("Durante la clase de educación física...")
    for i, pregunta in enumerate(preguntas_ganar):
        respuesta = st.radio(pregunta, opciones, key=i + 5, index=None, horizontal=True)

        # Guardar resultados en la lista correspondiente
        destrezas_ganar[i] = punctuations[respuesta]


with st.expander("Destrezas apropiadas a perder"):

    # Crear variables para almacenar los resultados
    destrezas_perder = [None] * 5

    # Preguntas y opciones
    preguntas_perder = [
        "¿Felicitas al ganador al perder un partido o un juego?",
        "¿Te mantienes calmado y positivo?",
        "¿Ignoras la burla de otros compañeros?",
        "¿Evitas culpar a tus compañeros de equipo por una mala ejecución personal?",
        "¿Respetas tus propios materiales (ej. Raqueta de tenis) y los materiales de otros compañeros?",
    ]

    st.write("Durante la clase de educación física...")
    for i, pregunta in enumerate(preguntas_perder):
        respuesta = st.radio(pregunta, opciones, key=i, index=None, horizontal=True)

        # Guardar resultados en la lista correspondiente
        destrezas_perder[i] = punctuations[respuesta]

with st.expander("Destrezas apropiadas durante el juego"):
    # Crear variables para almacenar los resultados
    destrezas_juego = [None] * 5

    # Preguntas y opciones
    preguntas_juego = [
        "¿Sigues las reglas de juego en todo momento?",
        "¿Haces comentarios positivos de la actuación de otros durante el juego?",
        "¿Ayudas a otros durante el juego de ser necesario? (ej. Ayuda a otros a levantarse del suelo luego de una caída)",
        "¿Respetas el nivel de habilidad de los demás sin menospreciarles o burlarte de ellos?",
        "¿Eres un buen miembro del equipo trabajando colaborativamente (no querer jugar solo)?",
    ]

    st.write("Durante la clase de educación física...")
    for i, pregunta in enumerate(preguntas_juego, start=11):
        respuesta = st.radio(pregunta, opciones, key=i, index=None, horizontal=True)

        # Guardar resultados en la lista correspondiente
        destrezas_juego[i - 11] = punctuations[respuesta]

with st.expander("Destrezas del juego justo"):
    # Crear variables para almacenar los resultados
    destrezas_justo = [None] * 5

    # Preguntas y opciones
    preguntas_justo = [
        "¿Participas con entusiasmo e intensidad?",
        "¿Realizas tu mejor esfuerzo para mejorar en tus destrezas y nivel de actividad física?",
        "¿Controlas tu conducta en todo momento?",
        "¿Tratas de resolver conflictos de manera adecuada?",
        "¿Respetas la decisión de algún compañero o maestro que asuma la posición de árbitro (oficial)?",
    ]

    st.write("Durante la clase de educación física...")
    for i, pregunta in enumerate(preguntas_justo, start=16):
        respuesta = st.radio(pregunta, opciones, key=i, index=None, horizontal=True)

        # Guardar resultados en la lista correspondiente
        destrezas_justo[i - 16] = punctuations[respuesta]

with st.expander("Habilidad Social"):
    # Crear variables para almacenar los resultados
    habilidad_social = [None] * 12

    # Preguntas y opciones
    preguntas_habilidad_social = [
        "¿Interactúas de manera adecuada con tus compañeros/as de clase?",
        "¿Escuchas cuando alguien te habla?",
        "¿Mantienes un contacto visual cuando alguien te habla?",
        "¿Sigues las órdenes del profesor?",
        "¿Usas un tono de voz apropiado?",
        "¿Aprendes a manejar situaciones de burla, enojo y malentendido?",
        "¿Compartes el equipamiento durante las clases?",
        "¿Usas un lenguaje apropiado y correcto en las clases?",
        "¿Ayudas a otros cuando lo necesitan?",
        "¿Sonríes adecuadamente?",
        "¿Respetas los turnos de intervención de tus compañeros?",
        "¿Tienes gestos apropiados con tus compañeros?",
    ]

    for i, pregunta in enumerate(preguntas_habilidad_social, start=21):
        respuesta = st.radio(pregunta, opciones, key=i, index=None, horizontal=True)

        # Guardar resultados en la lista correspondiente
        habilidad_social[i - 21] = punctuations[respuesta]

with st.expander("Satisfacción con la vida"):
    # Crear variables para almacenar los resultados
    satisfaccion_vida = [None] * 5

    # Preguntas y opciones
    preguntas_satisfaccion_vida = [
        "En la mayoría de los aspectos mi vida es como yo quiero que sea",
        "Las circunstancias de mi vida son muy buenas",
        "Estoy satisfecho con mi vida",
        "Hasta ahora he conseguido de la vida las cosas que considero importantes",
        "Si pudiera vivir mi vida otra vez no cambiaría casi nada",
    ]

    for i, pregunta in enumerate(preguntas_satisfaccion_vida, start=1):
        respuesta = st.radio(
            pregunta,
            opciones_satisfaccion,
            key="satis" + str(i),
            index=None,
            horizontal=True,
        )

        # Guardar resultados en la lista correspondiente
        satisfaccion_vida[i - 1] = punctuations[respuesta]

with st.expander("Miedo a los errores"):
    # Crear variables para almacenar los resultados
    miedo_errores = [None] * 12

    # Preguntas y opciones
    preguntas_miedo_errores = [
        "Si fallo en el trabajo, en la escuela o en casa soy un fracaso como persona",
        "Debo preocuparme si cometo un error",
        "Si fallo en parte, es tan malo como fracasar completamente",
        "Incluso cuando hago algo muy cuidadosamente, a menudo siento que no lo he hecho del todo bien",
        "La gente probablemente tendrá peor opinión de mí si cometo un error",
        "Si no hago las cosas tan bien como el resto de personas quiere decir que soy inferior a ellas",
        "Si no lo hago bien siempre, la gente no me respetará",
        "En general tengo dudas acerca de lo que hago",
        "Tiendo a atrasarme en mi trabajo porque repito las cosas una y otra vez",
        "Me lleva mucho tiempo hacer las cosas correctamente",
        "Cuantos menos errores cometa más personas me querrán",
        "Siento que nunca cumpliré las expectativas de mis padres",
    ]

    for i, pregunta in enumerate(preguntas_miedo_errores):
        respuesta = st.radio(
            pregunta,
            opciones_satisfaccion,
            key="miedo" + str(i),
            index=None,
            horizontal=True,
        )

        # Guardar resultados en la lista correspondiente
        miedo_errores[i] = punctuations[respuesta]

with st.expander("Influencias Paternas"):
    # Crear variables para almacenar los resultados
    influencias_paternas = [None] * 8

    # Preguntas y opciones
    preguntas_influencias_paternas = [
        "Mis padres me fijaron metas muy altas",
        "Cuando era niño/a fui castigado por no hacer las cosas a la perfección",
        "Mis padres nunca intentaron entender mis errores",
        "Mis padres querían que yo fuera el/la mejor en todo",
        "Para mi familia solo son buenos los resultados excelentes",
        "Mis padres han esperado grandes cosas de mí",
        "Pienso que nunca llegaré a satisfacer las expectativas de mis padres",
        "Mis padres siempre han tenido más expectativas sobre mi futuro que yo",
    ]

    # Índices de las preguntas seleccionadas
    for i, pregunta in enumerate(preguntas_influencias_paternas):
        respuesta = st.radio(
            pregunta,
            opciones_satisfaccion,
            key="infl" + str(i),
            index=None,
            horizontal=True,
        )

        # Guardar resultados en la lista correspondiente
        influencias_paternas[i] = punctuations[respuesta]

with st.expander("Expectativas al logro"):
    # Crear variables para almacenar los resultados
    expectativas_logro = [None] * 9

    # Preguntas y opciones
    preguntas_expectativas_logro = [
        "Si no me fijo metas muy elevadas probablemente acabaré siendo una persona de segunda categoría",
        "Es importante para mí ser absolutamente competente en todo lo que hago.",
        "Me fijo metas más elevadas que la mayoría de la gente",
        "Si alguien hace mejor que yo las cosas en el trabajo, en la escuela o en casa me siento como si hubiera fracasado completamente",
        "Soy muy bueno/a concentrando mis esfuerzos para alcanzar una meta",
        "Odio no ser el/la mejor en todo lo que hago",
        "Me propongo metas excesivamente altas",
        "Otras personas parecen conformarse con menos que yo",
        "Espero hacer las cosas de cada día mejor que la mayoría de la gente",
    ]

    for i, pregunta in enumerate(
        preguntas_expectativas_logro,
    ):
        respuesta = st.radio(
            pregunta,
            opciones_satisfaccion,
            key="expectativas" + str(i),
            index=None,
            horizontal=True,
        )

        # Guardar resultados en la lista correspondiente
        expectativas_logro[i] = punctuations[respuesta]

with st.expander("Organización"):
    # Crear variables para almacenar los resultados
    organizacion = [None] * 6

    # Preguntas y opciones
    preguntas_organizacion = [
        "La organización es muy importante para mí",
        "Soy una persona ordenada",
        "Intento ser una persona organizada",
        "Trato de ser una persona ordenada",
        "La limpieza tiene mucha importancia para mí",
        "Soy una persona organizada",
    ]
    opciones_organizacion = [
        "Totalmente en desacuerdo",
        "En desacuerdo",
        "Ni de acuerdo ni en desacuerdo",
        "De acuerdo",
        "Totalmente de acuerdo",
    ]

    for i, pregunta in enumerate(preguntas_organizacion):
        respuesta = st.radio(
            pregunta,
            opciones_organizacion,
            key="org" + str(i),
            index=None,
            horizontal=True,
        )

        # Guardar resultados en la lista correspondiente
        organizacion[i - 2] = punctuations[respuesta]


# Mostrar los resultados totales de destrezas a perder y ganar
total_destrezas_perder = sum(destrezas_perder)
total_destrezas_ganar = sum(destrezas_ganar)
total_destrezas_juego = sum(destrezas_juego)
total_destrezas_justo = sum(destrezas_justo)
total_habilidad_social = sum(habilidad_social)
total_satisfaccion_vida = sum(satisfaccion_vida)
total_miedo_errores = sum(miedo_errores)
total_influencias_paternas = sum(influencias_paternas)
total_expectativas_logro = sum(expectativas_logro)
total_organizacion = sum(organizacion)

new_experiment = np.array(
    [
        total_influencias_paternas,
        total_organizacion,
        total_expectativas_logro,
        total_miedo_errores,
        total_destrezas_perder,
        total_destrezas_ganar,
        total_destrezas_juego,
        total_destrezas_justo,
        total_habilidad_social,
        total_satisfaccion_vida,
    ]
).reshape(1, -1)

col1,  col2 = st.columns(2)

with col1:
    # Botón para activar la función y mostrar el resultado
    if st.button("Calcular perfil", type="primary"):

        # si no se ha contestado todo el formulario
        if any([v == 0.0 for v in new_experiment[0]]):
            st.warning(
                "Debes responder todas las preguntas del formulario antes de poder calcular el perfil"
            )

        else:

            # Llamada a la API
            prediccion = predice_clases(new_experiment)

            # Mostrar el resultado y una foto dependiendo del resultado
            st.write(f"Resultado de la función x: {prediccion}")

            # Mostrar foto según el resultado (ejemplo simple)
            if prediccion == 0:
                # st.image("imagen_positiva.jpg", caption="Resultado positivo", use_column_width=True)
                st.header("Nos enontramos delante del perfil 1")
                st.write("Generalmente son:")
                st.write(
                    "Mas chicos, jóvenes y de futbol con alto perfeccionismo, y moderadas competencias deportivas. Moderadas habilidades sociales y satisfacción con la vida"
                )
            elif prediccion == 1:
                st.header("Nos enontramos delante del perfil 2")
                st.write("Generalmente son:")
                st.write(
                    "Mas chicas, adultas y jugadoras de baloncesto, con bajos perfeccionismo excepto organización y altas habilidades deportivas. Altas habilidades sociales y satisfacción con la vida"
                )
            elif prediccion == 2:
                st.header("Nos enontramos delante del perfil 3")
                st.write("Generalmente son:")
                st.write(
                    "No se asocia a ningún grupo, ni por genero ni grupo y categoría, tienen bajas competencias deportivas, y sobre todo bajas habilidades sociales y satisfacción con la vida. Perfeccionismo moderado bajo. "
                )
            # st.image("imagen_negativa.jpg", caption="Resultado negativo", use_column_width=True)

with col2:
    #st.write("Desea guardar el perfil?")
    if st.button("Guardar perfil", type="primary"):

        # si no se ha contestado todo el formulario
        if any([v == 0.0 for v in new_experiment[0]]):
            st.warning(
                "Debes responder todas las preguntas del formulario antes de poder guardar el perfil"
            )

        else:
            new_value = [sexo, grupo, categ_deportiva] + [float(v) for v in new_experiment[0]]
            guardar_datos_en_bbdd(new_value)
            st.info("Perfil guardado correctamente")
