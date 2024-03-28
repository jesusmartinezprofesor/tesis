import streamlit as st
import numpy as np
from keras.models import load_model
from joblib import load
import os
import datetime
import pymongo
import time
from dotenv import load_dotenv
load_dotenv()

model = load_model("ninos_bestmodel.keras")
scaler = load("ninos_scaler.joblib")


def guardar_datos_en_bbdd(muestra_nueva):
    client = pymongo.MongoClient(
        f"mongodb+srv://{os.environ.get('MONGO_USER')}:{os.environ.get('MONGO_PASSWORD')}@cluster0.2fbpmxe.mongodb.net/"  # DEFINE CLUSTER
    )
    db = client["tesis"]  # CHANGE DB NAME
    json_insert = {"dateTime": datetime.datetime.now(), "values": muestra_nueva}
    db["ninos"].insert_one(json_insert)


# Define la función x (puedes personalizarla según tus necesidades)
def funcion_x(datos):
    st.write(datos)
    datos_st = scaler.transform(datos)
    st.write(datos_st)

    # Ejemplo de una función simple (puedes cambiar esto)
    resultado = model.predict(datos_st)
    predicted_classes = np.argmax(resultado, axis=1)

    return predicted_classes[0]


punctuations = {
    "Muchas veces": 5,
    "A veces": 3,
    "Nunca": 1,
    None: 0,
    "Totalmente de acuerdo": 5,
    "De acuerdo": 4,
    "Ni de acuerdo ni en desacuerdo": 3,
    "En desacuerdo": 2,
    "Totalmente en desacuerdo": 1,
}


# opciones = ["Siempre", "Casi siempre", "A menudo", "Casi nunca", "Nunca"]
opciones = [
    "Totalmente en desacuerdo",
    "En desacuerdo",
    "Ni de acuerdo ni en desacuerdo",
    "De acuerdo",
    "Totalmente de acuerdo",
]

opciones_pos_neg = [
    "Nunca",
    "A veces",
    "Muchas veces",
]


# Títulos y subtítulos
st.title("Cuestionario Niños.")
st.write("Doctorando: Prof. Jesús Martínez")

with st.expander("Datos cualitativos"):
    sexo = st.radio("Elige el sexo", ["Niño", "Niña"], index=None, horizontal=True)
    peso = st.radio(
        "Elige el grupo", ["peso normal", "sobre peso"], index=None, horizontal=True
    )
    edad = st.number_input(
        "Introduce edad", min_value=0, max_value=120, step=1
    )


with st.expander("Test de Perfeccionismos"):
    # Crear variables para almacenar los resultados
    presion_externa = [None] * 8

    # Preguntas y opciones
    preguntas_padres = [
        "Mis padres o profesores me castigan o riñen cuando no hago las cosas perfectamente",
        "Mis padres o profesores me exigen tanto que pienso que nunca los voy a poder satisfacer",
        "Mis padres no aceptan los errores que yo pueda cometer",
        "Haga lo que haga mis padres siempre me critican",
        "Mis padres y profesores ven más los errores que cometo que los aciertos que tengo",
        "Haga lo que haga no voy a poder satisfacer los deseos de mis padres",
        "En mi familia sólo se valora al mejor",
        "En mi familia todos tienen que hacer las cosas perfectamente",
    ]

    st.write(
        "Por favor, piensa en cómo eres en clase de Educación Física, en lo que sientes, y elige una respuesta rodeando con un círculo el número que más se acerque a tu opinión."
    )
    for i, pregunta in enumerate(preguntas_padres):
        respuesta = st.radio(
            pregunta, opciones, key=i + 10, index=None, horizontal=True
        )

        # Guardar resultados en la lista correspondiente
        presion_externa[i] = punctuations[respuesta]

    # Crear variables para almacenar los resultados
    auto_exigencia = [None] * 8

    # Preguntas y opciones
    preguntas_rendimiento = [
        "Intento ser el mejor en todo lo que hago",
        "Para mí es muy importante ser el mejor en aquello que hago (estudiar, hacer un deporte…)",
        "Aunque otros me digan que hice bien las tareas, pienso que podría haberlas hecho todavía mejor",
        "Mis padres quieren que sea el mejor en todo lo que hago",
        "Mis padres esperan que mi futuro sea brillante",
        "Haga lo que haga mis padres o profesores siempre esperan más de mí",
        "Mis padres esperan que yo llegue a estudiar en la Universidad",
        "No me gusta ser ni el segundo, quiero ser el primero",
    ]

    for i, pregunta in enumerate(preguntas_rendimiento):
        respuesta = st.radio(
            pregunta, opciones, key=i + 28, index=None, horizontal=True
        )

        # Guardar resultados en la lista correspondiente
        auto_exigencia[i] = punctuations[respuesta]

    # Crear variables para almacenar los resultados
    autovaloracion = [None] * 9

    # Preguntas y opciones
    preguntas_autovaloracion = [
        "Si no soy el mejor en las cosas que hago me siento mal",
        "Cuando no hago las cosas tan bien como quiero, siento que no valgo para nada",
        "Si saco un cinco o seis en un examen, pienso que soy un mal estudiante",
        "Me siento nervioso cuando veo un error en mi trabajo",
        "Cuando suspendo un examen o hago mal un trabajo, pienso que no valgo para nada",
        "Cuando cometo un error (estudiando, haciendo deporte,…) me siento muy mal",
        "Si la gente se da cuenta que cometo algún error pensará que no valgo.",
        "Si hago algo peor que mis compañeros, significa que soy inferior a ellos.",
        "Después de entregar un examen o un trabajo dudo que lo haya hecho bien.",
    ]

    for i, pregunta in enumerate(preguntas_autovaloracion):
        respuesta = st.radio(
            pregunta, opciones, key=i + 40, index=None, horizontal=True
        )

        # Guardar resultados en la lista correspondiente
        autovaloracion[i] = punctuations[respuesta]

with st.expander("Test de competencia motriz percibida."):

    # Crear variables para almacenar los resultados
    experiencia_personal = [None] * 8

    # Preguntas y opciones
    preguntas_actividades = [
        "Si fracasé en una actividad antes no creo que nunca vaya a hacerlo bien.",
        "Si he intentado esta actividad antes y no lo he hecho bien no creo que lo pueda hacer bien en clase de EF.",
        "No creo que haga bien las actividades de EF que no me gustan.",
        "No creo que me salgan bien las actividades para las que no tengo habilidad.",
        "Creo que se me dan bien las actividades que práctico fuera del colegio.",
        "Si no me ha salido la actividad bien antes, no creo que me salga bien en clase de EF.",
        "No creo que puedan salir bien las actividades que no hayas practicado antes.",
        "No creo que se me dé bien una actividad si no logro anotar ningún tanto cuando la practico.",
    ]

    for i, pregunta in enumerate(preguntas_actividades):
        respuesta = st.radio(
            pregunta, opciones, key=i + 50, index=None, horizontal=True
        )

        # Guardar resultados en la lista correspondiente
        experiencia_personal[i] = punctuations[respuesta]

    # Crear variables para almacenar los resultados
    companeros = [None] * 3

    # Preguntas y opciones
    preguntas_amigos = [
        "Si mis amigos me dicen que se me da algo bien, entonces creo que soy bueno en esa actividad.",
        "Sé que soy bueno en las actividades de EF porque mis amigos me lo dicen.",
        "No creo que sea bueno en una actividad si mis amigos no me lo dicen.",
    ]

    for i, pregunta in enumerate(preguntas_amigos):
        respuesta = st.radio(
            pregunta, opciones, key=i + 60, index=None, horizontal=True
        )

        # Guardar resultados en la lista correspondiente
        companeros[i] = punctuations[respuesta]

    # Crear variables para almacenar los resultados
    profesor = [None] * 4

    # Preguntas y opciones
    preguntas_profesor_practica = [
        "Si el profesor no sabe cómo explicar una actividad no creo que yo lo pueda hacer bien cuando la intente practicar.",
        "Si las instrucciones del profesor no tienen sentido no creo que yo pueda realizar la actividad correctamente.",
        "Si no practico una técnica en clase lo suficiente no creo que pueda hacerla bien.",
        "Cuando mi profesor me deja tomar decisiones en una actividad creo que me saldrá mejor.",
    ]

    for i, pregunta in enumerate(preguntas_profesor_practica):
        respuesta = st.radio(
            pregunta, opciones, key=i + 70, index=None, horizontal=True
        )

        # Guardar resultados en la lista correspondiente
        profesor[i] = punctuations[respuesta]

with st.expander("Test de afectividad"):
    positivas = [None] * 10

    # Preguntas y opciones
    preguntas_autoevaluacion = [
        "Me intereso por la gente o las cosas",
        "Soy una persona animada, suelo emocionarme",
        "Siento que tengo vitalidad o energía",
        "Me entusiasmo (por cosas, personas, etc.)",
        "Me siento orgulloso/a (de algo), satisfecho/a",
        "Soy un/a chico/a despierto/a, <<despabilado/a>>",
        "Me siento inspirado/a",
        "Soy un/a chico/a decidido/a",
        "Soy una persona atenta, esmerada",
        "Soy un/a chico/a activo/a",
    ]

    for i, pregunta in enumerate(preguntas_autoevaluacion):
        respuesta = st.radio(
            pregunta, opciones_pos_neg, key=i + 80, index=None, horizontal=True
        )

        # Guardar resultados en la lista correspondiente
        positivas[i] = punctuations[respuesta]

    # Crear variables para almacenar los resultados
    negativas = [None] * 10

    # Preguntas y opciones
    preguntas_autoevaluacion_2 = [
        "Me siento tenso/a, agobiado/a, con sensación de estrés",
        "Me siento disgustado/a o molesto/a",
        "Me siento culpable",
        "Soy un/a chico/a asustadizo/a",
        "Estoy enfadado/a o furioso/a",
        "Tengo mal humor (me altero o irrito)",
        "Soy vergonzoso/a",
        "Me siento nervioso/a",
        "Siento sensaciones corporales de estar intranquilo/a o preocupado/a",
        "Siento miedo",
    ]

    for i, pregunta in enumerate(preguntas_autoevaluacion_2):
        respuesta = st.radio(
            pregunta, opciones_pos_neg, key=i + 100, index=None, horizontal=True
        )

        # Guardar resultados en la lista correspondiente
        negativas[i] = punctuations[respuesta]

# Mostrar los resultados totales de destrezas a perder y ganar
total_presion_externa = sum(presion_externa)
total_auto_exigencia = sum(auto_exigencia)
total_autovaloracion = sum(autovaloracion)
total_experiencia_personal = sum(experiencia_personal)
total_companeros = sum(companeros)
total_profesor = sum(profesor)
total_positivas = sum(positivas)
total_negativas = sum(negativas)

new_experiment = np.array(
    [
        total_presion_externa,
        total_auto_exigencia,
        total_autovaloracion,
        total_experiencia_personal,
        total_companeros,
        total_profesor,
        total_positivas,
        total_negativas,
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
            # Aplicar la función x al vector de datos
            resultado_funcion_x = funcion_x(new_experiment)

            # Mostrar el resultado y una foto dependiendo del resultado
            st.write(f"Resultado de la función x: {resultado_funcion_x}")

            # Mostrar  el resultado
            if resultado_funcion_x == 0:
                st.header("Nos enontramos delante del perfil 1")
                st.write("Generalmente son:")
                st.write(
                    "Puntúan bajo en las tres dimensiones de la competencia motriz percibida, y también bajas en las puntuaciones del perfeccionismo y la afectividad positiva."
                )
            elif resultado_funcion_x == 1:
                st.header("Nos enontramos delante del perfil 2")
                st.write("Generalmente son:")
                st.write(
                    "Puntuaciones mayores en las dimensiones de competencia motriz percibida, experiencia personal y profesor, y también alta, pero más moderada en compañero. Fueron los que puntuaron más alto en todas las dimensiones del perfeccionismo y en afectividad negativa, y puntuaron bajo en aspectos positivos de la afectividad."
                )
            elif resultado_funcion_x == 2:
                st.header("Nos enontramos delante del perfil 3")
                st.write("Generalmente son:")
                st.write(
                    "Mayores puntuaciones en la dimensión de competencia motriz percibida compañeros, en las otras dos dimensiones, profesor y experiencia personal sus valores eran neutros en torno a la media. Este grupo a su vez se caracterizó por tener las mayores puntuaciones en afectividad positiva y también puntuaciones altas en autoexperiencia, pero bajas en autovaloración y afectividad negativa. "
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
            new_value = [sexo, peso, edad] + [float(v) for v in new_experiment[0]]
            guardar_datos_en_bbdd(new_value)
            st.info("Perfil guardado correctamente")
