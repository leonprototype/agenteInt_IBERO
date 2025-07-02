"""Demo script for searching teacher info in Pinecone using integrated embedding and reranking."""

from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()
# Initialize client with API key from environment
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = "teachers-demo"
namespace_name = "teachers"

if not pc.has_index(index_name):
    # create index with automatic embedding for the "info" field
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "llama-text-embed-v2",
            "field_map": {"text": "info"}
        }
    )

    teachers = [
        {"_id": "t1", "name": "Dra. Maria López", "office": "J-101", "position": "Coordinadora del Departamento de Ingeniería en Sistemas", "background": "Doctora en Ciencias de la Computación por el MIT. Investiga inteligencia artificial.", "info": "La Dra. Maria López es la coordinadora del Departamento de Ingeniería en Sistemas. Su oficina es J-101. Tiene un doctorado en Ciencias de la Computación por el MIT e investiga inteligencia artificial."},
        {"_id": "t2", "name": "Ing. Carlos Ramírez", "office": "B-203", "position": "Profesor del Departamento de Electrónica", "background": "Maestría en Ingeniería Eléctrica. Especialista en proyectos de robótica.", "info": "El Ing. Carlos Ramírez es profesor del Departamento de Electrónica. Su oficina es B-203. Tiene una maestría en Ingeniería Eléctrica y se enfoca en robótica."},
        {"_id": "t3", "name": "Mtra. Ana Torres", "office": "C-105", "position": "Profesora del Departamento de Ciencias Básicas y Matemáticas", "background": "Maestría en Matemáticas Aplicadas. Amplia experiencia enseñando cálculo.", "info": "La Mtra. Ana Torres imparte clases en el Departamento de Ciencias Básicas y Matemáticas y tiene la oficina C-105. Posee una maestría en Matemáticas Aplicadas y gran experiencia enseñando cálculo."},
        {"_id": "t4", "name": "Dr. Jorge Hernández", "office": "D-210", "position": "Coordinador de Investigación", "background": "Doctorado en Física. Lidera proyectos de energía renovable en la universidad.", "info": "El Dr. Jorge Hernández es el coordinador de investigación con oficina D-210. Cuenta con un doctorado en Física y lidera proyectos de energía renovable."},
        {"_id": "t5", "name": "Lic. Sofía Pérez", "office": False, "position": "Profesora del Departamento de Humanidades", "background": "Licenciatura en Filosofía. Coordina eventos de difusión cultural.", "info": "La Lic. Sofía Pérez es profesora del Departamento de Humanidades. No tiene oficina asignada. Posee una licenciatura en Filosofía y coordina eventos de difusión cultural."},
        {"_id": "t6", "name": "Dr. Raúl Gómez", "office": "E-302", "position": "Director del Programa de Posgrado", "background": "Doctorado en Ingeniería Mecánica. Más de 20 artículos publicados.", "info": "El Dr. Raúl Gómez dirige el programa de posgrado desde la oficina E-302. Tiene un doctorado en Ingeniería Mecánica y más de veinte artículos publicados."},
        {"_id": "t7", "name": "Ing. Laura Morales", "office": "A-220", "position": "Coordinadora de Laboratorio", "background": "Experta en ciencia de materiales con maestría en Ingeniería Química.", "info": "La Ing. Laura Morales gestiona los laboratorios universitarios. Su oficina es A-220. Es experta en ciencia de materiales y tiene una maestría en Ingeniería Química."},
        {"_id": "t8", "name": "Mtro. Pedro Castillo", "office": "F-101", "position": "Profesor del Departamento de Administración de Empresas", "background": "MBA por la Universidad de Texas. Consultor de negocios locales.", "info": "El Mtro. Pedro Castillo imparte clases en el Departamento de Administración de Empresas con oficina F-101. Cuenta con un MBA de la Universidad de Texas y fue consultor de negocios locales."},
        {"_id": "t9", "name": "Dra. Elena Suárez", "office": "G-114", "position": "Directora de Educación Continua", "background": "Doctorado en Educación. Desarrolla programas de capacitación profesional.", "info": "La Dra. Elena Suárez dirige educación continua desde la oficina G-114. Posee un doctorado en Educación y desarrolla programas de capacitación profesional."},
        {"_id": "t10", "name": "Dr. Miguel Navarro", "office": "H-215", "position": "Profesor del Departamento de Ingeniería Biomédica", "background": "Doctorado en Ingeniería Biomédica con investigación en imagenología médica.", "info": "El Dr. Miguel Navarro es profesor de Ingeniería Biomédica, oficina H-215. Tiene un doctorado en Ingeniería Biomédica y realiza investigación en imagenología médica."},
        {"_id": "t11", "name": "Ing. Gabriela Flores", "office": "A-120", "position": "Coordinadora de Proyectos Estudiantiles", "background": "Maestría en Ingeniería de Software. Guía proyectos de innovación estudiantil.", "info": "La Ing. Gabriela Flores coordina proyectos estudiantiles desde la oficina A-120. Cuenta con una maestría en Ingeniería de Software y guía proyectos de innovación."},
        {"_id": "t12", "name": "Dr. Luis Mendoza", "office": "C-301", "position": "Profesor de Ciencias Ambientales", "background": "Doctorado en Ciencias Ambientales. Estudia el impacto del cambio climático.", "info": "El Dr. Luis Mendoza imparte Ciencias Ambientales en la oficina C-301. Tiene un doctorado en Ciencias Ambientales y estudia los impactos del cambio climático."},
        {"_id": "t13", "name": "Mtra. Patricia Salinas", "office": "D-115", "position": "Coordinadora del Centro de Idiomas", "background": "Maestría en Lingüística. Supervisa iniciativas de enseñanza de idiomas.", "info": "La Mtra. Patricia Salinas coordina el Centro de Idiomas desde la oficina D-115. Posee una maestría en Lingüística y supervisa iniciativas de enseñanza de idiomas."},
        {"_id": "t14", "name": "Lic. Andrés Vega", "office": False, "position": "Profesor de Diseño Gráfico", "background": "Licenciatura en Diseño Gráfico. Diseñador profesional con proyectos premiados.", "info": "El Lic. Andrés Vega imparte Diseño Gráfico y no tiene oficina asignada. Es diseñador profesional con proyectos premiados."},
        {"_id": "t15", "name": "Dra. Mónica Ruiz", "office": "B-110", "position": "Jefa del Departamento de Psicología", "background": "Doctorado en Psicología Clínica. Especialista en desarrollo adolescente.", "info": "La Dra. Mónica Ruiz dirige el Departamento de Psicología, oficina B-110. Tiene un doctorado en Psicología Clínica y se especializa en desarrollo adolescente."},
        {"_id": "t16", "name": "Ing. Jorge Palacios", "office": "E-207", "position": "Profesor de Ingeniería Civil", "background": "Experto en análisis estructural con maestría en Ingeniería Civil.", "info": "El Ing. Jorge Palacios es profesor de Ingeniería Civil, oficina E-207. Es experto en análisis estructural y posee una maestría en Ingeniería Civil."},
        {"_id": "t17", "name": "Mtra. Karla Rivera", "office": "F-204", "position": "Coordinadora del Programa de Artes", "background": "Maestría en Bellas Artes. Organiza exposiciones y talleres.", "info": "La Mtra. Karla Rivera coordina el Programa de Artes desde la oficina F-204. Tiene una maestría en Bellas Artes y organiza exposiciones."},
        {"_id": "t18", "name": "Dr. Fernando Aguilar", "office": "G-211", "position": "Profesor del Departamento de Historia", "background": "Doctorado en Historia. Ha publicado varios libros sobre historia latinoamericana.", "info": "El Dr. Fernando Aguilar enseña en el Departamento de Historia, oficina G-211. Posee un doctorado en Historia y ha publicado libros sobre Latinoamérica."},
        {"_id": "t19", "name": "Lic. Daniela Ortiz", "office": "H-102", "position": "Profesora de Mercadotecnia", "background": "Licenciatura en Mercadotecnia. Experiencia en agencias de marketing digital.", "info": "La Lic. Daniela Ortiz es profesora de Mercadotecnia con oficina H-102. Tiene una licenciatura en Mercadotecnia y trabajó en agencias de marketing digital."},
        {"_id": "t20", "name": "Dr. Julio Martínez", "office": "I-221", "position": "Director del Centro de Investigación", "background": "Doctorado en Química. Lidera equipos de investigación multidisciplinarios.", "info": "El Dr. Julio Martínez dirige el Centro de Investigación desde la oficina I-221. Cuenta con un doctorado en Química y lidera equipos multidisciplinarios."},
        {"_id": "t21", "name": "Ing. Beatriz Núñez", "office": "J-305", "position": "Profesora de Ingeniería Mecánica", "background": "Maestría en Ingeniería Industrial. Experiencia en la industria automotriz.", "info": "La Ing. Beatriz Núñez enseña Ingeniería Mecánica con oficina J-305. Tiene una maestría en Ingeniería Industrial y trabajó en la industria automotriz."},
        {"_id": "t22", "name": "Mtro. Ricardo Díaz", "office": "K-110", "position": "Director de Asuntos Estudiantiles", "background": "Maestría en Administración Educativa. Antiguo consejero estudiantil.", "info": "El Mtro. Ricardo Díaz dirige Asuntos Estudiantiles desde la oficina K-110. Posee una maestría en Administración Educativa y fue consejero estudiantil."},
        {"_id": "t23", "name": "Dra. Isabel Fuentes", "office": "L-214", "position": "Profesora del Departamento de Química", "background": "Doctorado en Química Orgánica. Investiga materiales sustentables.", "info": "La Dra. Isabel Fuentes es profesora del Departamento de Química con oficina L-214. Tiene un doctorado en Química Orgánica e investiga materiales sustentables."},
        {"_id": "t24", "name": "Lic. Adrián Reyes", "office": False, "position": "Coordinador Deportivo", "background": "Licenciatura en Educación Física. Organiza eventos deportivos en el campus.", "info": "El Lic. Adrián Reyes es el Coordinador Deportivo sin oficina asignada. Posee una licenciatura en Educación Física y organiza eventos deportivos en el campus."},
        {"_id": "t25", "name": "Dra. Lorena Gutiérrez", "office": "M-220", "position": "Profesora de Arquitectura", "background": "Doctorado en Arquitectura con énfasis en diseño sustentable.", "info": "La Dra. Lorena Gutiérrez imparte Arquitectura en la oficina M-220. Cuenta con un doctorado en Arquitectura y se enfoca en diseño sustentable."},
        {"_id": "t26", "name": "Ing. Óscar Valdez", "office": "N-109", "position": "Supervisor de Talleres", "background": "Maestría en Diseño Industrial. Supervisa los laboratorios de fabricación estudiantil.", "info": "El Ing. Óscar Valdez es el supervisor de talleres con oficina N-109. Tiene una maestría en Diseño Industrial y supervisa los laboratorios de fabricación."},
        {"_id": "t27", "name": "Mtra. Verónica Chávez", "office": "O-112", "position": "Profesora de Economía", "background": "Maestría en Economía. Especialista en análisis de política económica.", "info": "La Mtra. Verónica Chávez enseña Economía desde la oficina O-112. Posee una maestría en Economía y se especializa en análisis de políticas."},
        {"_id": "t28", "name": "Dr. Roberto Luna", "office": "P-217", "position": "Coordinador de Investigación Biomédica", "background": "Doctorado en Ciencias Biomédicas. Lidera colaboraciones de investigación clínica.", "info": "El Dr. Roberto Luna coordina la Investigación Biomédica en la oficina P-217. Tiene un doctorado en Ciencias Biomédicas y lidera colaboraciones clínicas."},
        {"_id": "t29", "name": "Mtro. Sergio Castro", "office": "Q-208", "position": "Profesor de Literatura", "background": "Maestría en Literatura Hispánica. Reconocido por estudios de poesía contemporánea.", "info": "El Mtro. Sergio Castro imparte Literatura con oficina Q-208. Cuenta con una maestría en Literatura Hispánica y estudia poesía contemporánea."},
        {"_id": "t30", "name": "Lic. Mariana López", "office": "R-306", "position": "Coordinadora de Servicio Social", "background": "Licenciatura en Trabajo Social. Enlaza a los estudiantes con proyectos de servicio.", "info": "La Lic. Mariana López coordina el Servicio Social desde la oficina R-306. Posee una licenciatura en Trabajo Social y enlaza a los estudiantes con proyectos de servicio."},
        {"_id": "t31", "name": "Dr. Alberto Cuevas", "office": "S-111", "position": "Profesor de Matemáticas", "background": "Doctorado en Matemáticas con especialidad en teoría de probabilidades.", "info": "El Dr. Alberto Cuevas es profesor de Matemáticas con oficina S-111. Tiene un doctorado en Matemáticas y se especializa en teoría de probabilidades."},
        {"_id": "t32", "name": "Ing. Araceli Pineda", "office": "T-118", "position": "Profesora asistente de Redes de Computadoras", "background": "Maestría en Telecomunicaciones. Participó en proyectos de seguridad de redes.", "info": "La Ing. Araceli Pineda es profesora asistente de Redes de Computadoras con oficina T-118. Posee una maestría en Telecomunicaciones y trabajó en proyectos de seguridad."},
        {"_id": "t33", "name": "Mtra. Cecilia Vargas", "office": "U-204", "position": "Profesora de Ciencias Políticas", "background": "Maestría en Ciencias Políticas. Investiga instituciones democráticas.", "info": "La Mtra. Cecilia Vargas imparte Ciencias Políticas desde la oficina U-204. Cuenta con una maestría en Ciencias Políticas e investiga instituciones democráticas."},
        {"_id": "t34", "name": "Dr. Tomás Juárez", "office": "V-101", "position": "Coordinador del Club de Robótica", "background": "Doctorado en Mecatrónica. Entrena a estudiantes para competencias de robótica.", "info": "El Dr. Tomás Juárez coordina el Club de Robótica desde la oficina V-101. Tiene un doctorado en Mecatrónica y entrena a estudiantes para competencias."},
        {"_id": "t35", "name": "Lic. Paula Sandoval", "office": False, "position": "Profesora de Periodismo", "background": "Licenciatura en Periodismo. Exeditora de noticias con premios regionales.", "info": "La Lic. Paula Sandoval enseña Periodismo y no tiene oficina asignada. Posee una licenciatura en Periodismo y fue editora de noticias ganadora de premios regionales."},
        {"_id": "t36", "name": "Dr. Enrique Silva", "office": "W-209", "position": "Director de Estudios de Posgrado", "background": "Doctorado en Economía. Supervisa programas de maestría y doctorado.", "info": "El Dr. Enrique Silva es el Director de Estudios de Posgrado en la oficina W-209. Cuenta con un doctorado en Economía y supervisa programas de posgrado."},
        {"_id": "t37", "name": "Ing. Teresa Rojas", "office": "X-213", "position": "Profesora de Ingeniería de Software", "background": "Maestría en Ciencias de la Computación. Especialista en arquitectura de software.", "info": "La Ing. Teresa Rojas imparte Ingeniería de Software en la oficina X-213. Tiene una maestría en Ciencias de la Computación y se especializa en arquitectura de software."},
        {"_id": "t38", "name": "Mtro. Héctor Ramírez", "office": "Y-201", "position": "Coordinador de Asesoría Académica", "background": "Maestría en Psicología Educativa. Brinda orientación a estudiantes de primer ingreso.", "info": "El Mtro. Héctor Ramírez coordina la Asesoría Académica con oficina Y-201. Posee una maestría en Psicología Educativa y orienta a estudiantes de primer ingreso."},
        {"_id": "t39", "name": "Lic. Nancy Estrada", "office": "Z-115", "position": "Profesora de Contabilidad", "background": "Licenciatura en Contaduría. Experta en auditoría financiera.", "info": "La Lic. Nancy Estrada es profesora de Contabilidad con oficina Z-115. Tiene una licenciatura en Contaduría y es experta en auditoría financiera."},
        {"_id": "t40", "name": "Dr. Rodrigo Sánchez", "office": "A-104", "position": "Investigador en Biotecnología", "background": "Doctorado en Biotecnología. Trabaja en innovaciones agrícolas.", "info": "El Dr. Rodrigo Sánchez es investigador en Biotecnología, oficina A-104. Cuenta con un doctorado en Biotecnología y trabaja en innovaciones agrícolas."}
    ]

    # add records to the new index
    index = pc.Index(index_name)
    index.upsert_records(namespace_name, teachers)

index = pc.Index(index_name)
query = input("Enter search term: ")

results = index.search(
    namespace=namespace_name,
    query={
        "top_k": 5,
        "inputs": {"text": query}
    },
    rerank={
        "model": "bge-reranker-v2-m3",
        "top_n": 5,
        "rank_fields": ["info"]
    }
)

for hit in results["result"]["hits"]:
    fields = hit["fields"]
    print(
        f"id: {hit['_id']}, score: {round(hit['_score'], 2)}, "
        f"name: {fields['name']}, office: {fields['office']}, "
        f"position: {fields['position']}"
    )
