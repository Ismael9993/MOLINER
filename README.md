# Extractor de Términos y Definiciones

Sistema de extracción automática de conocimiento terminológico a partir de corpus textuales en español. Identifica pares **[Término → Definición]** utilizando análisis sintáctico con spaCy y exporta los resultados a un archivo Excel listo para uso como glosario o base de datos terminológica. 

*Este es un proyecto personal que se encuentra en sus primeras etapas, por lo que se encuentra en constante cambio y mejora. Los resultados pueden variar dependiendo de la calidad y cantidad de corpus. Se busca mejorar la calidad de los resultados.*
---

## Características principales

- **Detección de estructuras definitorias** — identifica verbos como *ser*, *definir*, *denominar*, *entender*, etc.
- **Pre-segmentación de títulos e índices** — las líneas en MAYÚSCULAS se aíslan antes del análisis para evitar que contaminen los resultados.
- **Manejo de voz pasiva refleja** — detecta construcciones del tipo *"X se denomina Y"* e invierte correctamente los roles Término/Definición.
- **Validación por núcleo nominal** — aplica una lista negra sobre el sustantivo central del término, independientemente de su longitud, eliminando capturas genéricas como *"El caso más grave"*.
- **Filtro de densidad léxica** — descarta oraciones sin verbo, demasiado cortas o con mayoría de tokens en mayúsculas.
- **Umbral estricto de definición** — exige un mínimo de 30 caracteres y la presencia de al menos un sustantivo en la definición.
- **Modo dual de salida** — genera columnas `[Término, Definiciones]` o `[Término, Verbo, Definiciones]` según el parámetro `incluir_verbo`.

---

## Estructura del proyecto

```
Extractor de terminos_definiciones/
│
├── Term_Extractor_Geco3.py       # Extractor que trabaja con corpus de GECO3 (UNAM)
├── Term_Extractor_Local_Docs.py  # Extractor que trabaja con documentos locales (.txt / .pdf)
│
├── geco3_client/                 # Paquete local: cliente de autenticación y descarga de GECO3
│
├── docs/                         # Coloca aquí tus documentos locales (.txt o .pdf)
├── data/                         # Datos intermedios generados durante el procesamiento
│
├── config.json                   # Credenciales de GECO3 (NO se sube al repositorio)
├── requirements.txt
└── .gitignore
```

---

## Requisitos

- Python **3.10+**
- Las dependencias listadas en `requirements.txt`

### Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd "Extractor de terminos_definiciones"

# 2. Crear y activar el entorno virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar el modelo de lenguaje de spaCy
python -m spacy download es_core_news_lg
```

### Dependencia opcional (PDFs)

Si vas a procesar documentos `.pdf` con `Term_Extractor_Local_Docs.py`:

```bash
pip install pdfplumber
```

---

## Uso

### Modo 1 — Corpus de GECO3

```bash
python Term_Extractor_Geco3.py
```

El script te guiará interactivamente para:
1. Autenticarse en la plataforma GECO3.
2. Elegir un corpus de la lista disponible.
3. Filtrar documentos por metadatos (área, lengua, etc.).
4. Seleccionar los documentos a procesar.
5. Elegir el nombre del archivo Excel de salida.
6. Elegir si incluir la columna `Verbo` en la salida.

**Configuración de credenciales** — crea un archivo `config.json` en la raíz del proyecto:

```json
{
  "base_url":     "http://www.geco.unam.mx/geco3/",
  "anon_user":    "tu_usuario",
  "anon_pass":    "tu_contraseña",
  "app_name":     "nombre_de_tu_app",
  "app_password": "password_de_la_app"
}
```

---

### Modo 2 — Documentos locales

```bash
python Term_Extractor_Local_Docs.py
```

1. Coloca tus archivos `.txt` o `.pdf` en la carpeta `docs/`.
2. El script mostrará la lista de documentos disponibles con su tamaño.
3. Elige cuáles procesar (o presiona Enter para procesar todos).
4. Especifica el nombre del archivo Excel de salida.
5. Elige si incluir la columna `Verbo`.

---

## Arquitectura del motor de extracción

```
Texto del corpus
      │
      ▼
┌─────────────────────────┐
│  limpiar_texto_avanzado │  ← Detecta y aísla títulos/índices con [TITULO]
└─────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│   _es_oracion_valida    │  ← Filtra oraciones sin verbo, muy cortas o en mayúsculas
└─────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│  Verbo definitorio      │  ← ser, definir, denominar, entender, identificar...
└─────────────────────────┘
      │
      ├─── ¿Estructura inversa? (se denomina / se llama)
      │         │ Sí → Término a la DERECHA del verbo
      │         │ No → Término a la IZQUIERDA del verbo
      ▼
┌─────────────────────────┐
│   normalizar_termino    │  ← Minúsculas, deduplic., poda de artículos/numeraciones
└─────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│  _validar_nucleo_nominal│  ← Blacklist aplicada al sustantivo central del término
└─────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│  Validación definición  │  ← Mínimo 30 chars + al menos un sustantivo
└─────────────────────────┘
      │
      ▼
   Excel (.xlsx)
```

---

## Formato de salida

| Término | Definiciones |
|---|---|
| Sífilis congénita | Infección que se transmite de madre a hijo durante el embarazo... |
| Holoprosencefalia | Malformación cerebral que resulta de la falta de división... |
| Tricomoniasis | Infección de transmisión sexual causada por el parásito... |

Con `incluir_verbo=True` se agrega una columna `Verbo` entre las dos anteriores.

---

## Dependencias

| Paquete | Versión mínima | Uso |
|---|---|---|
| `spacy` | 3.8 | Motor NLP y análisis sintáctico |
| `es_core_news_lg` | — | Modelo de español (descarga con spaCy) |
| `pandas` | 3.0 | Construcción y exportación del DataFrame |
| `openpyxl` | 3.1 | Motor de escritura `.xlsx` |
| `requests` | 2.31 | Comunicación HTTP con GECO3 |
| `pdfplumber` | 0.11 *(opcional)* | Extracción de texto desde PDFs |

---

## Licencia

Este proyecto es de uso personal para el desarrollo de proyectos académicos y de investigación.  
Nace como una necesidad para mejorar la construcción de diccionarios terminológicos especializados. 
