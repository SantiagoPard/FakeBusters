import streamlit as st
import torch #libreria para modelos de deepLearning
import torch.nn.functional as F #  Es un módulo dentro de PyTorch que tiene funciones matemáticas comunes para redes neuronales(softmax)
from transformers import AutoTokenizer, AutoModelForSequenceClassification  #conversion de texto a tokens y carga modelo ya entrenado para clasidicacion de teXTO
import pandas as pd

# --- Dispositivo y modelo ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #Detecta si hay una GPU disponible (más rápida para deep learning); si no, usa CPU
ckpt = "Narrativaai/fake-news-detection-spanish" #repositorio modelo
tokenizer = AutoTokenizer.from_pretrained(ckpt) #convierte el texto en tokens
model = AutoModelForSequenceClassification.from_pretrained(
    ckpt,
    output_attentions=True # permite obtener los pesos de atención para hacer explicaciones visuale
).to(device) #carga del modelo ya entrenado
model.eval() #modo evaluacion ya que no se entrenara

# --- Función para predicció ---
def get_prediction_and_attention(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True) #conbierte los tokens a tensores
    # Mover los tensores al dispositivo
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs) #Se hace la predicción sin guardar nada para entrenamiento

    logits = outputs.logits #resultado crudo del modelo
    probs = F.softmax(logits, dim=-1)[0] #convierte en probabilidades los resultados
    label_idx = torch.argmax(probs).item() #selecciona la clase de mayor probabilidad
    label = model.config.id2label[label_idx]
    confidence = probs[label_idx].item() #seguridad del resuldtado del modelo

    # Explicabilidad con atenciones
    attentions = outputs.attentions
    last_attn = attentions[-1][0]       # (heads, seq_len, seq_len)
    avg_attn = last_attn.mean(dim=0)    # (seq_len, seq_len)
    cls_attn = avg_attn[0]              # (seq_len,) importancia de cada toquen
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]) # palabras procesadas
    importances = cls_attn.cpu().numpy()
    importances = importances / importances.sum()

    return label, confidence, tokens, importances

# --- Interfaz de usuario Streamlit ---
st.set_page_config(page_title="Detector de Noticias Falsas", layout="wide")

# Crear dos columnas: una para el logo y utilidad (izquierda), otra para el contenido (derecha)
col1, col2 = st.columns([1, 4])  # Puedes ajustar el tamaño relativo

with col1:
    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center;">
            <img src="https://i.postimg.cc/kBtt2stc/logo.png" width="120">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("### ¿Qué es esta página?")
    st.markdown(
        """
        Esta aplicación utiliza inteligencia artificial para analizar noticias escritas en español y detectar si son **falsas o verdaderas**.
        Solo tienes que ingresar el texto de la noticia y el modelo te dará un resultado.
        """
    )

with col2:
    # Botones para cambiar de vista
    st.markdown("""
        <style>
        .button-container {
            margin-bottom: 20px;
        }
        .button-container button {
            background-color: #007bff;
            border: none;
            color: white;
            padding: 10px 20px;
            margin-right: 10px;
            font-size: 16px;
            cursor: pointer;
            border-radius: 5px;
        }
        .button-container button:hover {
            background-color: #0056b3;
        }
        </style>
    """, unsafe_allow_html=True)

    # Usamos st.session_state para controlar la vista activa
    if "vista" not in st.session_state:
        st.session_state.vista = "prediccion"

    # Crear botones estilo HTML usando markdown + js
    btn_prediccion = st.button("🔍 Predicción")
    btn_pricing = st.button("💰 Pricing")

    if btn_prediccion:
        st.session_state.vista = "prediccion"
    if btn_pricing:
        st.session_state.vista = "pricing"

    # Mostrar la vista según st.session_state.vista
    if st.session_state.vista == "prediccion":
        st.title("📰 Detector de Noticias Falsas (Español)")
        st.markdown("Ingresa el texto de una noticia y analiza si podría ser **falsa o verdadera**.")
        text_input = st.text_area("✏️ Escribe o pega aquí tu noticia:", height=200)

        if st.button("Analizar"):
            if not text_input.strip():
                st.warning("Por favor, ingresa un texto para analizar.")
            else:
                with st.spinner("Analizando..."):
                    label, confidence, tokens, importances = get_prediction_and_attention(text_input)

                if label.lower() == "real":
                    veracidad = confidence
                    emoji = "✅"
                    mensaje = "Parece VERDADERA"
                    color = "green"
                else:
                    veracidad = 1 - confidence
                    emoji = "❌"
                    mensaje = "Parece FALSA"
                    color = "red"

                porcentaje = int(veracidad * 100)

                st.markdown("---")
                st.markdown(f"<h3 style='color:{color}'>{emoji} {mensaje}</h3>", unsafe_allow_html=True)
                st.progress(veracidad)
                st.info(
                    "🧠 *Nota:* Este modelo no es perfecto. Úsalo como herramienta de apoyo, no como veredicto final.")
                
                # top_tokens = [
                #     (token.replace("▁", "").replace("##", ""), float(score))
                #     for token, score in zip(tokens, importances)
                #     if token.isalpha()
                # ]
                # top_tokens.sort(key=lambda x: x[1], reverse=True)
                # keywords = [t for t, _ in top_tokens[:5]]
                #
                # st.markdown("💡 El modelo se fijó especialmente en:")
                # st.markdown(", ".join(f"{kw}" for kw in keywords))
                #
                # with st.expander("🔎 Ver importancia de todos los tokens"):
                #     df = pd.DataFrame({
                #         "Token": [t for t, _ in top_tokens],
                #         "Importancia": [s for _, s in top_tokens]
                #     })
                #     st.bar_chart(df.set_index("Token"))
    else:  # vista pricing
        st.header("💰 Planes y Precios")
        st.markdown("Elige el plan que más se adapte a tus necesidades:")

        col_plan1, col_plan2, col_plan3 = st.columns(3)

        with col_plan1:
            st.markdown(f"""
                   <div style="
                       border: 2px solid #007bff;
                       border-radius: 15px;
                       padding: 20px;
                       background-color: #141414;
                       text-align: center;
                       box-shadow: 2px 2px 8px rgba(0, 123, 255, 0.3);
                   ">
                       <div style="color: #007bff; font-weight: bold; font-size: 24px; margin-bottom: 15px;">Gratis</div>
                       <ul style="list-style-type:none; padding-left: 0;">
                           <li>✅ Hasta 5 análisis diarios</li>
                           <li>✅ Acceso básico al modelo</li>
                           <li>✅ Soporte comunidad</li>
                       </ul>
                       <h3 style="color:#007bff;">$0 / mes</h3>
                   </div>
                   """, unsafe_allow_html=True)

        with col_plan2:
            st.markdown(f"""
                   <div style="
                       border: 2px solid #007bff;
                       border-radius: 15px;
                       padding: 20px;
                       background-color: #141414;
                       text-align: center;
                       box-shadow: 2px 2px 8px rgba(0, 123, 255, 0.3);
                   ">
                       <div style="color: #007bff; font-weight: bold; font-size: 24px; margin-bottom: 15px;">Básico</div>
                       <ul style="list-style-type:none; padding-left: 0;">
                           <li>✅ Hasta 50 análisis diarios</li>
                           <li>✅ Acceso prioritario al modelo</li>
                           <li>✅ Soporte por correo</li>
                       </ul>
                       <h3 style="color:#007bff;">$9.99 / mes</h3>
                   </div>
                   """, unsafe_allow_html=True)

        with col_plan3:
            st.markdown(f"""
                   <div style="
                       border: 2px solid #007bff;
                       border-radius: 15px;
                       padding: 20px;
                       background-color: #141414;
                       text-align: center;
                       box-shadow: 2px 2px 8px rgba(0, 123, 255, 0.3);
                   ">
                       <div style="color: #007bff; font-weight: bold; font-size: 24px; margin-bottom: 15px;">Premium</div>
                       <ul style="list-style-type:none; padding-left: 0;">
                           <li>✅ Análisis ilimitados</li>
                           <li>✅ Acceso a características avanzadas</li>
                           <li>✅ Soporte prioritario</li>
                       </ul>
                       <h3 style="color:#007bff;">$29.99 / mes</h3>
                   </div>
                   """, unsafe_allow_html=True)