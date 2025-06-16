# agenteInt_IBERO

Este proyecto implementa un modelo de n8n cuyo objetivo es crear un asistente de IA para los estudiantes de la Universidad Iberoamericana de Puebla. El asistente busca concentrar la mayor cantidad posible de información de la institución para responder dudas de forma rápida y precisa.

## Descripción del asistente

El agente de IA está pensado para conocer detalles como:

- La oferta académica o **planes de estudio** de cada carrera.
- Los horarios de los profesores y su ubicación para indicar en qué salón o área se encuentran.
- Información general de los docentes, tales como grados académicos, clases que imparten y puestos dentro de la universidad.
- Las áreas y procesos administrativos a los que puede acudir un alumno para realizar trámites o solicitar documentación.
- La localización de los salones mediante su código alfanumérico (por ejemplo, saber que el departamento de ingeniería está en el salón **A203**).
- Datos generales de la universidad, como la fecha de fundación o quién ocupa el cargo de rector.

## Arquitectura propuesta

Para obtener y entregar la información correcta, se contempla el uso de:

1. **Base de datos**: alojada en la nube o en un servidor local, con tablas que agrupan información de profesores, horarios, aulas, planes de estudio y otros datos relevantes.
2. **Agente de IA con n8n**: configurado con la estructura de la base de datos para comunicarse con ella y elaborar las respuestas adecuadas.
3. **Interacción estudiante–agente**: el alumno realiza una consulta, el agente consulta la base de datos y devuelve una respuesta clara.

Este repositorio contiene los recursos iniciales de la implementación. El proyecto se encuentra en desarrollo y tiene como meta facilitar el acceso a la información de la universidad para toda la comunidad estudiantil.
