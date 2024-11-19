import os
import requests
from dotenv import load_dotenv
import numpy as np
from collections import defaultdict
import hashlib

# Charger les variables d'environnement depuis .env
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# Vérification de la clé API
if not api_key:
    raise ValueError("La clé API OpenAI n'est pas définie. Veuillez la définir dans le fichier .env.")

# Configuration des fichiers pour chaque agent
agent_files = {
    "agent1": os.getenv("AGENT1_FILE_PATH", "agent1_data.txt"),
    "agent2": os.getenv("AGENT2_FILE_PATH", "agent2_data.txt")
}

# Initialisation des mémoires épisodiques et globales par agent et par utilisateur
episodic_memory = defaultdict(lambda: defaultdict(list))
global_memory = defaultdict(lambda: defaultdict(lambda: {"topics": set(), "style_preference": None, "user_preferences": {}}))

# Cache pour les embeddings
embedding_cache = {}

# Dictionnaire des traits de personnalité pour chaque agent
agent_personalities = {
    "agent1": {
        "name": "Alex",
        "personality_traits": {
            "openness": 0.8,
            "conscientiousness": 0.7,
            "extraversion": 0.6,
            "agreeableness": 0.9,
            "neuroticism": 0.2
        },
        "default_emotion": "happy"
    },
    "agent2": {
        "name": "Jordan",
        "personality_traits": {
            "openness": 0.5,
            "conscientiousness": 0.9,
            "extraversion": 0.4,
            "agreeableness": 0.6,
            "neuroticism": 0.3
        },
        "default_emotion": "calm"
    }
}

def generate_openai_response(api_key, model, messages):
    """
    Génère une réponse en utilisant l'API OpenAI Chat Completion.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 1.0,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.5
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as http_err:
        print(f"Erreur HTTP lors de l'appel à l'API OpenAI : {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Erreur lors de l'appel à l'API OpenAI : {req_err}")
    except Exception as e:
        print(f"Une erreur inattendue s'est produite : {e}")
    return "Erreur dans la génération de la réponse."

def read_and_split_txt(file_path, chunk_size=500):
    """
    Lire un fichier texte et le diviser en morceaux plus petits.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        return chunks
    except FileNotFoundError:
        print(f"Fichier non trouvé : {file_path}")
        return []
    except UnicodeDecodeError:
        print(f"Erreur d'encodage lors de la lecture du fichier : {file_path}")
        return []
    except Exception as e:
        print(f"Une erreur s'est produite lors de la lecture du fichier : {e}")
        return []

def get_embedding(text, api_key):
    """
    Obtenir l'embedding pour un texte via l'API OpenAI, avec mise en cache.
    """
    # Utiliser un hash du texte comme clé de cache
    text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    if text_hash in embedding_cache:
        return embedding_cache[text_hash]

    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "text-embedding-ada-002",
        "input": text
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        embedding = result['data'][0]['embedding']
        # Stocker dans le cache
        embedding_cache[text_hash] = embedding
        return embedding
    except requests.exceptions.RequestException as e:
        print(f"Une erreur s'est produite lors de l'appel à l'API OpenAI : {e}")
        return None
    except KeyError:
        print("Erreur lors de la récupération des embeddings.")
        return None

def cosine_similarity(vec1, vec2):
    """Calculer la similarité cosinus entre deux vecteurs."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if not np.any(vec1) or not np.any(vec2):
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve_documents(query, file_path, api_key, max_chunks=5):
    """
    Récupérer les documents pertinents d'un fichier texte basé sur une requête.
    """
    chunks = read_and_split_txt(file_path)
    if not chunks:
        return []

    query_embedding = get_embedding(query, api_key)
    if query_embedding is None:
        return []

    similarities = []
    for chunk in chunks:
        chunk_embedding = get_embedding(chunk, api_key)
        if chunk_embedding is not None:
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((chunk, similarity))

    sorted_chunks = sorted(similarities, key=lambda x: x[1], reverse=True)
    relevant_chunks = [chunk for chunk, sim in sorted_chunks[:max_chunks]]
    return relevant_chunks

def analyze_user_sentiment(user_input):
    """
    Analyse le sentiment de l'entrée de l'utilisateur en utilisant l'API OpenAI.
    """
    prompt = f"Analyse le sentiment du texte suivant et réponds avec un seul mot (Positif, Neutre, Négatif) :\n\n\"{user_input}\""
    messages = [{"role": "user", "content": prompt}]

    sentiment = generate_openai_response(api_key, "gpt-3.5-turbo", messages)
    if sentiment:
        return sentiment.strip().lower()
    return "neutral"

def update_agent_emotion(agent, user_sentiment):
    """
    Met à jour l'émotion de l'agent en fonction du sentiment de l'utilisateur.
    """
    if user_sentiment == "positif":
        agent_emotion = "happy"
    elif user_sentiment == "négatif":
        agent_emotion = "concerned"
    else:
        agent_emotion = agent_personalities[agent]["default_emotion"]

    return agent_emotion

def generate_personality_system_message(agent, agent_emotion):
    """
    Génère le message système en fonction de la personnalité et de l'émotion de l'agent.
    """
    personality = agent_personalities.get(agent, {})
    name = personality.get("name", "Agent")
    traits = personality.get("personality_traits", {})
    openness = traits.get("openness", 0.5)
    conscientiousness = traits.get("conscientiousness", 0.5)
    extraversion = traits.get("extraversion", 0.5)
    agreeableness = traits.get("agreeableness", 0.5)
    neuroticism = traits.get("neuroticism", 0.5)

    system_message = (
        f"Ton nom est {name}. Tu te sens {agent_emotion}.\n"
        f"Traits de personnalité :\n"
        f"- Ouverture : {openness}\n"
        f"- Conscience : {conscientiousness}\n"
        f"- Extraversion : {extraversion}\n"
        f"- Agréabilité : {agreeableness}\n"
        f"- Neuroticisme : {neuroticism}\n"
        f"Réponds à l'utilisateur en conséquence, en adaptant ton style et ton ton."
    )
    return system_message

def store_interaction_in_memory(agent, user, user_query, agent_response):
    """
    Stocke l'interaction dans la mémoire épisodique et met à jour la mémoire globale pour un utilisateur spécifique.
    """
    episodic_memory[agent][user].append((user_query, agent_response))
    if len(episodic_memory[agent][user]) > 10:  # Augmentation de la limite pour plus d'historique
        episodic_memory[agent][user].pop(0)

    # Mise à jour des tendances de la mémoire globale
    global_memory[agent][user]["topics"].add(user_query)
    # Mise à jour des préférences de style si détecté
    if "example" in agent_response.lower():
        global_memory[agent][user]["style_preference"] = "example-driven"

    # Mise à jour des préférences utilisateur détectées
    if "je préfère des réponses courtes" in user_query.lower():
        global_memory[agent][user]["user_preferences"]["response_length"] = "short"

def summarize_conversation(agent, user):
    """
    Résume les dernières interactions stockées dans la mémoire épisodique de l'agent pour un utilisateur spécifique.
    """
    interactions = episodic_memory[agent][user]
    if not interactions:
        return ""

    # Créer un texte à partir des dernières interactions
    conversation_text = "\n".join([f"Utilisateur : {q}\nAgent : {a}" for q, a in interactions[-10:]])
    prompt = f"Résume la conversation suivante entre l'utilisateur et l'agent de manière concise, en mettant en évidence les principaux sujets et les préférences exprimées par l'utilisateur :\n\n{conversation_text}\n\nRésumé :"
    messages = [{"role": "user", "content": prompt}]

    summary = generate_openai_response(api_key, "gpt-3.5-turbo", messages)
    if summary:
        return summary.strip()
    return ""

def retrieve_conversation_context(query, agent, user, api_key):
    """
    Récupère les résumés pertinents des conversations passées basés sur la requête pour un utilisateur spécifique.
    """
    # Obtenir le résumé de la conversation
    summary = summarize_conversation(agent, user)
    if not summary:
        return ""

    # Obtenir les embeddings pour la requête et le résumé
    query_embedding = get_embedding(query, api_key)
    summary_embedding = get_embedding(summary, api_key)

    if query_embedding is None or summary_embedding is None:
        return ""

    similarity = cosine_similarity(query_embedding, summary_embedding)

    # Si la similarité est au-dessus d'un seuil, inclure le résumé
    threshold = 0.3  # Seuil ajusté pour être plus inclusif
    if similarity >= threshold:
        return summary
    else:
        return ""

def retrieve_personalized_context(agent, user):
    """
    Génère un contexte personnalisé basé sur la mémoire globale pour un utilisateur spécifique.
    """
    personalized_info = ""
    user_memory = global_memory[agent][user]
    if user_memory["style_preference"]:
        personalized_info += f"L'agent préfère des réponses {user_memory['style_preference']}.\n"

    user_prefs = user_memory.get("user_preferences", {})
    if user_prefs:
        prefs = ", ".join([f"{key} : {value}" for key, value in user_prefs.items()])
        personalized_info += f"Préférences utilisateur détectées : {prefs}.\n"

    return personalized_info

def count_tokens(text):
    """
    Estime le nombre de tokens dans un texte.
    """
    return int(len(text) / 4)

def adjust_response_length(agent, user, response):
    """
    Ajuste la longueur de la réponse en fonction des préférences utilisateur détectées pour un utilisateur spécifique.
    """
    user_prefs = global_memory[agent][user].get("user_preferences", {})
    if user_prefs.get("response_length") == "short":
        # Troncature de la réponse pour la raccourcir
        return response[:200] + "..."
    return response

def rag_system(api_key, model, query, agent="agent1", user="user1"):
    """
    Exécute un système RAG en utilisant l'API OpenAI pour un agent et un utilisateur spécifiques avec une gestion avancée de la personnalité.
    """
    file_path = agent_files.get(agent)
    if not file_path:
        return f"Aucun fichier trouvé pour l'agent {agent}."

    # Analyser le sentiment de l'utilisateur
    user_sentiment = analyze_user_sentiment(query)

    # Mettre à jour l'émotion de l'agent
    agent_emotion = update_agent_emotion(agent, user_sentiment)

    # Générer le message système basé sur la personnalité et l'émotion de l'agent
    system_message = generate_personality_system_message(agent, agent_emotion)

    # Récupérer les documents pertinents du fichier texte
    retrieved_documents = retrieve_documents(query, file_path, api_key)
    documents_context = "\n".join(retrieved_documents) if retrieved_documents else ""

    # Récupérer le contexte pertinent des conversations passées
    conversation_summary = retrieve_conversation_context(query, agent, user, api_key)
    conversation_context = f"Résumé des conversations précédentes :\n{conversation_summary}" if conversation_summary else ""

    # Ajouter un contexte personnalisé basé sur la mémoire globale
    personalized_context = retrieve_personalized_context(agent, user)

    # Combiner tous les contextes
    context_parts = [documents_context, conversation_context, personalized_context]
    context = "\n\n".join([part for part in context_parts if part])

    # Vérifier le nombre de tokens pour éviter de dépasser les limites de l'API
    total_tokens = count_tokens(system_message + context + query)
    max_allowed_tokens = 4096 - 1000  # Marge pour la réponse
    if total_tokens > max_allowed_tokens:
        # Tronquer le contexte si nécessaire
        context = context[:int(max_allowed_tokens * 4)]  # Approximation

    # Générer la réponse avec le message système
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"{query}\n\nContexte :\n{context}"}
    ]

    response = generate_openai_response(api_key, model, messages)

    if response:
        # Stocker l'interaction dans la mémoire
        store_interaction_in_memory(agent, user, query, response)

        # Ajuster la réponse en fonction des préférences utilisateur
        response = adjust_response_length(agent, user, response)

    return response if response else "Désolé, une erreur est survenue lors de la génération de la réponse."

# Exemple d'utilisation
if __name__ == "__main__":
    # Supposons que nous ayons plusieurs utilisateurs : user1, user2
    user_id = "user1"
    agent_response = rag_system(api_key, model="gpt-3.5-turbo", query="Je me sens un peu perdu avec les concepts de la mécanique quantique.", agent="agent1", user=user_id)
    print(f"Réponse pour {user_id} :\n{agent_response}\n")

    user_id = "user2"
    agent_response = rag_system(api_key, model="gpt-3.5-turbo", query="Pouvez-vous m'expliquer la théorie de la relativité ?", agent="agent1", user=user_id)
    print(f"Réponse pour {user_id} :\n{agent_response}")
