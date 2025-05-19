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
st.set_page_config(page_title="Detector de Noticias Falsas", layout="centered")
st.title("📰 Detector de Noticias Falsas (Español)")
st.markdown("Ingresa el texto de una noticia y analiza si podría ser **falsa o verdadera**.")

text_input = st.text_area("✏️ Escribe o pega aquí tu noticia:", height=200)

if st.button("🔍 Analizar"):
    if not text_input.strip():
        st.warning("Por favor, ingresa un texto para analizar.")
    else:
        with st.spinner("Analizando..."):
            label, confidence, tokens, importances = get_prediction_and_attention(text_input)

        # Interpretar confianza como porcentaje de veracidad
        if label.lower() == "real":
            veracidad = confidence
            emoji = "✅"
            mensaje = "Parece **VERDADERA**"
        else:
            veracidad = 1 - confidence
            emoji = "❌"
            mensaje = "Parece **FALSA**"

        porcentaje = int(veracidad * 100)

        # Mostrar resultado
        st.markdown("---")
        st.markdown(f"### {emoji} Resultado del análisis")
        st.markdown(f"**{mensaje}** con un {porcentaje}% de confianza")
        st.progress(veracidad)

        # Extraer y mostrar palabras clave
        top_tokens = [
            (token.replace("▁", "").replace("##", ""), float(score))
            for token, score in zip(tokens, importances)
            if token.isalpha()
        ]
        top_tokens.sort(key=lambda x: x[1], reverse=True)
        keywords = [t for t, _ in top_tokens[:5]]

        st.markdown("💡 El modelo se fijó especialmente en:")
        st.markdown(", ".join(f"{kw}" for kw in keywords))

        # Detalles en expander
        with st.expander("🔎 Ver importancia de todos los tokens"):
            df = pd.DataFrame({
                "Token": [t for t, _ in top_tokens],
                "Importancia": [s for _, s in top_tokens]
            })
            st.bar_chart(df.set_index("Token"))
