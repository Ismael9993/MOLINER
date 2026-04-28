import os
import re
import sys
import json
import subprocess
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import spacy
from spacy.lang.es.stop_words import STOP_WORDS




# ---------------------------------------------------------------------------
# CONSTANTES DEL EXTRACTOR
# ---------------------------------------------------------------------------

# Verbos que invierten los roles: término a la DERECHA del verbo
VERBOS_INVERSOS: set = {"denominar", "conocer", "llamar", "designar", "apellidar"}

# Verbos cuya presencia indica una definición legítima
VERBOS_DEFINITORIOS: List[str] = [
    "ser", "definir", "conocer", "entender", "identificar", "denominar",
]

# Núcleos nominales genéricos que no constituyen términos válidos
BLACKLIST_NUCLEOS: set = {
    "caso", "ejemplo", "vía", "manera", "forma", "tipo", "parte",
    "aspecto", "cosa", "hecho", "vez", "tiempo", "lugar", "situación",
    "resultado", "proceso", "problema", "año", "mes", "día", "síntoma",
    "transmisión",
}

# Palabras que indican perífrasis verbal (se ignoran como verbo principal)
VERBOS_MODALES: set = {"poder", "deber", "soler", "querer", "ir"}

# Palabras que invalidan una definición cuando aparecen como verbo principal
VERBOS_PROHIBIDOS: set = {"haber", "existir", "tener", "parecer"}
PALABRAS_HAY: set = {"hay", "hubo", "había", "habrá", "existen"}

# Primeras palabras post-verbo que indican definición válida con 'ser'
KEYWORDS_DEFINITORIAS: set = {
    "entendido", "considerado", "definido", "visto", "la", "el", "un",
    "una", "aquello", "capacidad", "proceso", "enfermedad", "infección",
}

# Participios pasivos que invalidan el patrón con 'ser'
VERBOS_ACCION_PASIVA: set = {
    "aislado", "identificado", "creado", "modificado", "estudiado",
}

# Adjetivos y determinantes que indican un sujeto incompleto
PALABRAS_BASURA_INICIO: set = {
    "otro", "otra", "otros", "otras", "último", "últimos",
    "primer", "primero", "cada", "alguno", "nuevo", "nueva",
}


# ---------------------------------------------------------------------------
# CLASE PRINCIPAL: TermExtractor
# ---------------------------------------------------------------------------

class TermExtractor:
    """
    Extractor de pares [Término - Definición] a partir de texto en español.

    Utiliza análisis sintáctico de spaCy para identificar estructuras
    definitorio-nominales. Incluye lógica mejorada para:
    - Distinguir títulos/índices de contenido real.
    - Manejar estructuras inversas (voz pasiva refleja).
    - Validar términos por su núcleo nominal.
    """

    def __init__(self, nlp_model: spacy.language.Language) -> None:
        """
        Args:
            nlp_model: Modelo de lenguaje de spaCy ya cargado.
        """
        self.nlp = nlp_model
        self.nlp.max_length = 1_500_000

    # Limpieza de texto 

    def limpiar_texto_avanzado(self, texto: str) -> str:
        """
        Limpia el texto del corpus preservando fronteras semánticas.

        MEJORA: Procesa línea a línea antes de unir el texto. Las líneas
        que parecen títulos o cabeceras de índice se marcan con [TITULO] para
        que el filtro de oraciones las descarte más adelante, evitando que
        spaCy las fusione con el contenido siguiente.

        Args:
            texto: Texto crudo descargado del corpus.

        Returns:
            Texto normalizado listo para procesar con spaCy.
        """
        lineas = texto.splitlines()
        resultado: List[str] = []

        for linea in lineas:
            linea_strip = linea.strip()
            if not linea_strip:
                continue

            # Heurística de título: corto, todo MAYÚSCULAS, sin puntuación fuerte
            es_titulo = (
                len(linea_strip) < 65
                and linea_strip == linea_strip.upper()
                and not any(c in linea_strip for c in [".", ",", ";", "?", "!"])
            )
            resultado.append("[TITULO]." if es_titulo else linea_strip)

        texto_unido = " ".join(resultado)

        # Limpieza estándar
        texto_unido = re.sub(r"https?://\S+", "", texto_unido)
        texto_unido = re.sub(r"\S+@\S+", "", texto_unido)
        texto_unido = re.sub(r"\b\d+\b", "", texto_unido)
        return re.sub(r"\s+", " ", texto_unido).strip()

    # Filtro de oraciones 

    def _es_oracion_valida(self, sent: spacy.tokens.Span) -> bool:
        """
        Descarta oraciones que probablemente son títulos, fragmentos de índice
        o ruido estructural del documento.

        MEJORA: Añade validación por densidad léxica mínima y presencia de
        verbo, capas de seguridad que el filtro de tokens individuales no cubre.

        Args:
            sent: Oración de spaCy a evaluar.

        Returns:
            True si la oración tiene contenido válido para extraer.
        """
        # Descartar marcadores de título insertados en la limpieza
        if "[TITULO]" in sent.text:
            return False

        tokens_validos = [t for t in sent if not t.is_punct and not t.is_space]

        # Mínimo de tokens para considerarla una oración real
        if len(tokens_validos) < 5:
            return False

        # Debe tener al menos un verbo
        if not any(t.pos_ in {"VERB", "AUX"} for t in tokens_validos):
            return False

        # Si más del 50 % de los tokens están en MAYÚSCULAS, es un índice
        tokens_mayus = sum(1 for t in tokens_validos if t.text.isupper() and len(t.text) > 2)
        if tokens_mayus / len(tokens_validos) > 0.5:
            return False

        return True

    #   Detección de estructura inversa   

    def _es_estructura_inversa(
        self,
        verbo: spacy.tokens.Token,
        sent: spacy.tokens.Span,
    ) -> bool:
        """
        Detecta si el verbo está en construcción pasiva refleja ('se denomina').

        MEJORA: Para verbos como 'denominar' o 'llamar' con clítico 'se',
        el término técnico está a la DERECHA del verbo, no a la izquierda.
        Ejemplo: "La unión de hemisferios se denomina holoprosencefalia."
                 <-- definición -->   <--TÉRMINO -->

        Args:
            verbo: Token del verbo definitorio.
            sent: Oración completa.

        Returns:
            True si se detecta construcción con inversión de roles.
        """
        if verbo.lemma_ not in VERBOS_INVERSOS:
            return False

        # Buscar clitíco 'se' antes del verbo en la misma oracion
        tiene_se = any(
            t.text.lower() == "se" and t.i < verbo.i
            for t in sent
        )
        return tiene_se

    def _extraer_termino_post_verbo(
        self,
        verbo: spacy.tokens.Token,
        sent: spacy.tokens.Span,
    ) -> str:
        """
        Extrae el término cuando está a la DERECHA del verbo (estructura inversa).

        Args:
            verbo: Token del verbo definitorio.
            sent: Oración completa.

        Returns:
            Texto crudo del término candidato.
        """
        doc = verbo.doc
        inicio = verbo.i + 1

        # Saltar 'como' si es la palabra inmediatamente siguiente
        if inicio < sent.end and doc[inicio].lemma_ == "como":
            inicio += 1

        tokens_termino: List[spacy.tokens.Token] = []
        for t in doc[inicio:sent.end]:
            if t.text in {".", ",", ";", ":"} or t.pos_ == "VERB":
                break
            tokens_termino.append(t)
            if len(tokens_termino) >= 5:
                break

        return "".join(t.text_with_ws for t in tokens_termino).strip()

    # Normalización del término

    def normalizar_termino(self, texto: str) -> str:
        """
        Normaliza y limpia el texto de un término candidato.

        Aplica en cascada: minúsculas, eliminación de numeraciones,
        paréntesis, guiones, adverbios en -mente, deduplicación de palabras
        y eliminación de artículos/preposiciones al inicio y final.

        Args:
            texto: Texto crudo del término candidato.

        Returns:
            Término normalizado y capitalizado, o cadena vacía si no es válido.
        """
        t = texto.lower().strip()

        # Eliminar numeraciones de lista (ej: "1. ", "iv) ", "a) ")
        t = re.sub(r"^[a-z0-9]{1,3}[\)\.]\s+", "", t)

        # Eliminar contenido entre paréntesis/corchetes
        t = re.sub(r"\[.*?\]|\(.*?\)", "", t)
        t = re.sub(r"[\[\]\(\)\{\}]", "", t)

        # Reemplazar guiones y viñetas por espacio
        t = re.sub(r"[â€”â€“_â€¢Â·]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()

        # Eliminar adverbios terminados en -mente al inicio
        t = re.sub(r"^\w+mente\s+", "", t)

        # Deduplicación de palabras (corrige "sífilis la sífilis")
        palabras = t.split()
        vistas: List[str] = []
        for p in palabras:
            if p not in vistas:
                vistas.append(p)
        t = " ".join(vistas)

        # Eliminar artículos y preposiciones al inicio/final (3 pasadas)
        patron_inicio = r"^(que|puesto que|puesto|el|la|los|las|un|una|unos|unas|de|del|y|o|en|con|para|por|a|pero|estas|esta|este)\s+"
        patron_final  = r"\s+(que|el|la|los|las|un|una|unos|unas|de|del|y|o|en|con|para|por|a|pero|estas|esta|este|se|no)$"
        for _ in range(3):
            t = re.sub(patron_inicio, "", t)
            t = re.sub(patron_final, "", t)

        # Limpiar puntuación residual al inicio/final
        t = re.sub(r"^[:,\.\-\s]+|[:,\.\-\s]+$", "", t)

        return t.strip().capitalize()

    # Validación del término por núcleo nominal
    def _validar_nucleo_nominal(self, doc_termino: spacy.tokens.Doc) -> bool:
        """
        Valida el término analizando su núcleo nominal.

        MEJORA: A diferencia del enfoque anterior (blacklist solo para términos
        de 1 palabra), aquí se identifica el sustantivo más relevante del
        término y se evalúa la blacklist independientemente de la longitud.
        Esto evita que pasen términos como "El caso más grave" o "Un tipo de...".

        Args:
            doc_termino: Documento spaCy del término ya normalizado.

        Returns:
            True si el término tiene un núcleo nominal válido.
        """
        # Obtener todos los sustantivos del término
        sustantivos = [t for t in doc_termino if t.pos_ in {"NOUN", "PROPN"}]
        if not sustantivos:
            return False  # Sin sustantivo real → inválido

        # El núcleo es el último sustantivo (más cercano al verbo en la oración original)
        nucleo = sustantivos[-1]

        if nucleo.lemma_.lower() in BLACKLIST_NUCLEOS:
            return False

        return True

    # Extracción principal
    def extraer_a_excel(
        self,
        contenidos: List[Dict[str, str]],
        nombre_archivo_salida: str,
        incluir_verbo: bool = False,
    ) -> None:
        """
        Procesa los textos del corpus y exporta los pares término-definición a Excel.

        MEJORA UNIFICADA: El parámetro incluir_verbo reemplaza los dos scripts
        anteriores. False (por defecto) = columnas [Término, Definiciones],
        equivalente a Term_Extrac.py. True = [Término, Verbo, Definiciones],
        equivalente a Extractor_TermDef.py.

        Args:
            contenidos: Lista de dicts con claves 'nombre' y 'texto'.
            nombre_archivo_salida: Ruta del archivo .xlsx a generar.
            incluir_verbo: Si True, incluye la columna 'Verbo' en la salida.
        """
        todas_las_tripletas: List[Dict[str, str]] = []

        for item in contenidos:
            texto_limpio = self.limpiar_texto_avanzado(item["texto"])
            # Fragmentar en chunks de 100 000 caracteres para no saturar RAM
            chunks = [texto_limpio[i:i + 100_000] for i in range(0, len(texto_limpio), 100_000)]

            for chunk in chunks:
                doc = self.nlp(chunk)

                for sent in doc.sents:

                    # FILTRO 1: Calidad mínima de la oración
                    if not self._es_oracion_valida(sent):
                        continue

                    # FILTRO 2: Localizar verbo definitorio
                    verbo = next(
                        (t for t in sent if t.lemma_ in VERBOS_DEFINITORIOS), None
                    )
                    if not verbo:
                        continue

                    # Descartar verbos prohibidos y existenciales
                    if verbo.lemma_ in VERBOS_PROHIBIDOS:
                        continue
                    if verbo.text.lower() in PALABRAS_HAY:
                        continue

                    # Descartar perífrasis modales (puede ser, debe ser...)
                    if verbo.i > sent.start:
                        token_anterior = sent.doc[verbo.i - 1]
                        if token_anterior.lemma_ in VERBOS_MODALES:
                            continue

                    # Descartar verbos auxiliares dentro de un VP mayor
                    if verbo.dep_ == "aux" or (
                        verbo.head.pos_ == "VERB" and verbo.head.i != verbo.i
                    ):
                        continue

                    # Validación adicional para el verbo 'ser'
                    if verbo.lemma_ == "ser":
                        post = [t for t in sent if t.i > verbo.i]
                        if not post:
                            continue
                        primera = post[0]
                        if (
                            primera.pos_ not in {"DET", "NOUN"}
                            and primera.lemma_ not in KEYWORDS_DEFINITORIAS
                        ):
                            continue
                        if primera.lemma_ in VERBOS_ACCION_PASIVA:
                            continue

                    # EXTRACCIÓN CON DETECCIÓN DE ESTRUCTURA INVERSA
                    if self._es_estructura_inversa(verbo, sent):
                        # Estructura inversa: término a la derecha
                        termino_raw = self._extraer_termino_post_verbo(verbo, sent)
                        definicion_raw = "".join(
                            t.text_with_ws for t in sent if t.i < verbo.i
                        ).strip()
                    else:
                        # Estructura estándar: término a la izquierda
                        posibles = [t for t in sent if t.i < verbo.i]
                        if len(posibles) > 5:
                            posibles = posibles[-5:]

                        tokens_sujeto: List[spacy.tokens.Token] = []
                        encontro_sustantivo = False

                        for t in reversed(posibles):
                            # Corte por título en MAYÚSCULAS
                            if t.text.isupper() and len(t.text) > 3:
                                break
                            # Corte por verbo/auxiliar (evita arrastre de perífrasis)
                            if t.pos_ in {"VERB", "AUX"}:
                                break
                            # Corte gramatical
                            if t.text.lower() in {"que", "puesto", ",", ".", ";", ":", "Â¿", "Â¡"}:
                                break
                            # Corte de adverbio pegado al verbo
                            if t.pos_ == "ADV" and t.i == verbo.i - 1:
                                break

                            if t.pos_ in {"NOUN", "PROPN"}:
                                encontro_sustantivo = True
                            tokens_sujeto.insert(0, t)

                        if not encontro_sustantivo:
                            continue

                        termino_raw = "".join(t.text_with_ws for t in tokens_sujeto)
                        tokens_post = [t for t in sent if t.i > verbo.i]
                        definicion_raw = "".join(t.text_with_ws for t in tokens_post).strip()

                    # NORMALIZACIÓN DEL TÉRMINO
                    termino = self.normalizar_termino(termino_raw)

                    # Filtro de longitud máxima (Términos multipalabra aceptables)
                    if not termino or len(termino.split()) > 5:
                        continue

                    # Validación por núcleo nominal (MEJORA CLAVE)
                    doc_termino = self.nlp(termino)

                    # Bloqueo de adjetivos/determinantes basura al inicio
                    if doc_termino and doc_termino[0].lemma_.lower() in PALABRAS_BASURA_INICIO:
                        continue

                    # Bloqueo de verbos auxiliares internos
                    if any(t.lemma_ in {"haber", "ser", "estar", "hacer"} for t in doc_termino):
                        continue

                    # Validación por núcleo nominal (reemplaza blacklist simple)
                    if not self._validar_nucleo_nominal(doc_termino):
                        continue

                    # LIMPIEZA DE LA DEFINICIÓN
                    definicion = re.sub(
                        r"^(como|es|son)\s+", "", definicion_raw, flags=re.IGNORECASE
                    )
                    definicion = re.sub(r"^[:,\s]+", "", definicion).split(". ")[0]

                    # MEJORA: umbral más estricto + verificar sustantivo en definición
                    doc_def = self.nlp(definicion)
                    tiene_sustantivo_def = any(t.pos_ in {"NOUN", "PROPN"} for t in doc_def)
                    if len(definicion) < 30 or not tiene_sustantivo_def:
                        continue

                    todas_las_tripletas.append({
                        "Término":      termino,
                        "Verbo":        verbo.lemma_.lower(),
                        "Definiciones": definicion.capitalize(),
                    })

        # GUARDADO
        if not todas_las_tripletas:
            print("No se encontraron términos para exportar.")
            return

        df = pd.DataFrame(todas_las_tripletas).drop_duplicates(subset=["Término"])

        columnas = ["Término", "Verbo", "Definiciones"] if incluir_verbo else ["Término", "Definiciones"]
        df[columnas].to_excel(nombre_archivo_salida, index=False)
        print(f"Excel generado: '{nombre_archivo_salida}' con {len(df)} términos.")


# ---------------------------------------------------------------------------
# LECTURA DE DOCUMENTOS LOCALES
# ---------------------------------------------------------------------------

# Extensiones soportadas
EXTENSIONES_SOPORTADAS = {".txt", ".pdf"}


def _leer_txt(ruta: str) -> str:
    """Lee un archivo .txt con detección automática de encoding."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(ruta, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"No se pudo leer '{ruta}' con ningún encoding conocido.")


def _leer_pdf(ruta: str) -> str:
    """
    Lee un archivo .pdf extrayendo texto página por página.
    Requiere: pip install pdfplumber
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "Para leer PDFs instala pdfplumber:  pip install pdfplumber"
        )
    paginas: List[str] = []
    with pdfplumber.open(ruta) as pdf:
        for pagina in pdf.pages:
            texto = pagina.extract_text()
            if texto:
                paginas.append(texto)
    return "\n".join(paginas)


def listar_archivos_locales(carpeta_docs: str) -> List[str]:
    """
    Lista los archivos con extensión soportada dentro de carpeta_docs.

    Args:
        carpeta_docs: Ruta a la carpeta docs/.

    Returns:
        Lista de rutas absolutas de archivos encontrados.
    """
    if not os.path.isdir(carpeta_docs):
        print(f"La carpeta '{carpeta_docs}' no existe.")
        return []

    archivos = sorted(
        os.path.join(carpeta_docs, f)
        for f in os.listdir(carpeta_docs)
        if os.path.splitext(f)[1].lower() in EXTENSIONES_SOPORTADAS
    )
    return archivos


def elegir_archivos_locales(archivos: List[str]) -> List[str]:
    """
    Muestra los archivos disponibles y permite seleccionar cuáles procesar.

    Args:
        archivos: Lista de rutas absolutas.

    Returns:
        Subconjunto seleccionado por el usuario.
    """
    print(f"\n{len(archivos)} documentos encontrados en docs/:\n")
    for i, ruta in enumerate(archivos, 1):
        nombre = os.path.basename(ruta)
        tam_kb = os.path.getsize(ruta) // 1024
        print(f"  {i}. {nombre}  ({tam_kb} KB)")

    indices_str = input(
        "\nElige documentos (ej: 1,3,5) o presiona Enter para procesar todos: "
    ).strip()

    if not indices_str:
        return archivos

    try:
        indices = [int(x.strip()) - 1 for x in indices_str.split(",") if x.strip().isdigit()]
        seleccionados = [archivos[i] for i in indices if 0 <= i < len(archivos)]
        if not seleccionados:
            print("Selección inválida. Se procesarán todos.")
            return archivos
        return seleccionados
    except ValueError:
        print("Entrada no válida. Se procesarán todos.")
        return archivos


def leer_documentos_locales(rutas: List[str]) -> List[Dict[str, str]]:
    """
    Lee el contenido de cada archivo y lo devuelve en el formato que
    espera TermExtractor: lista de dicts con claves 'nombre' y 'texto'.

    Args:
        rutas: Lista de rutas absolutas de archivos a leer.

    Returns:
        Lista de dicts {'nombre': str, 'texto': str}.
    """
    textos: List[Dict[str, str]] = []
    for ruta in rutas:
        nombre = os.path.basename(ruta)
        ext = os.path.splitext(ruta)[1].lower()
        try:
            print(f"  → Leyendo: {nombre}")
            if ext == ".txt":
                contenido = _leer_txt(ruta)
            elif ext == ".pdf":
                contenido = _leer_pdf(ruta)
            else:
                print(f"    Extensión '{ext}' no soportada. Omitiendo.")
                continue

            if not contenido.strip():
                print(f"    El archivo '{nombre}' está vacío. Omitiendo.")
                continue

            textos.append({"nombre": nombre, "texto": contenido})
        except ImportError as e:
            print(f"    Dependencia faltante para '{nombre}': {e}")
        except Exception as e:
            print(f"    Error al leer '{nombre}': {e}")
    return textos


def solicitar_nombre_archivo() -> str:
    """
    Solicita al usuario el nombre del archivo Excel de salida.

    Returns:
        Nombre de archivo validado con extensión .xlsx.
    """
    nombre = input(
        "\nNombre del archivo Excel de salida (Enter = 'terminos_definiciones.xlsx'): "
    ).strip()
    if not nombre:
        return "terminos_definiciones.xlsx"
    if not nombre.lower().endswith(".xlsx"):
        nombre += ".xlsx"
    return nombre


# ---------------------------------------------------------------------------
# BLOQUE DE EJECUCIÓN PRINCIPAL
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Verificar e inicializar modelo spaCy ---
    modelo_nombre = "es_core_news_lg"
    if not spacy.util.is_package(modelo_nombre):
        print(f"Descargando modelo {modelo_nombre}...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", modelo_nombre])

    nlp_instancia = spacy.load(modelo_nombre)
    print(f"Modelo '{modelo_nombre}' cargado correctamente.")

    # --- Localizar carpeta docs/ en la raíz del proyecto ---
    # Se busca docs/ relativa al directorio donde está este script.
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    carpeta_docs = os.path.join(directorio_script, "docs")

    print(f"\nBuscando documentos en: {carpeta_docs}")

    archivos_disponibles = listar_archivos_locales(carpeta_docs)
    if not archivos_disponibles:
        print(
            f"No se encontraron archivos .txt o .pdf en '{carpeta_docs}'.\n"
            "Crea la carpeta 'docs/' en la raíz del proyecto y coloca tus documentos ahí."
        )
        sys.exit(0)

    # --- Selección de archivos ---
    archivos_seleccionados = elegir_archivos_locales(archivos_disponibles)

    # --- Lectura del contenido ---
    print(f"\nLeyendo {len(archivos_seleccionados)} documento(s)...")
    textos = leer_documentos_locales(archivos_seleccionados)

    if not textos:
        print("No se pudo leer ningún documento. Terminando.")
        sys.exit(0)

    # --- Modo de salida ---
    modo = input("\n¿Incluir columna 'Verbo' en el Excel? (s/n, Enter=no): ").strip().lower()
    incluir_verbo = modo == "s"

    nombre_salida = solicitar_nombre_archivo()

    # --- Extracción ---
    print(f"\nExtrayendo términos de {len(textos)} documento(s)...\n")
    extractor = TermExtractor(nlp_instancia)
    extractor.extraer_a_excel(textos, nombre_salida, incluir_verbo=incluir_verbo)


