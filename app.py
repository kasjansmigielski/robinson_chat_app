import os
import streamlit as st
from openai import OpenAI
from dotenv import dotenv_values
import json
from pathlib import Path
import numpy as np
import faiss
import boto3
import re
from nltk.tokenize import sent_tokenize # do tokenizacji na słowa

FILE_PATH = 'robinson-crusoe-lema.txt' # TEKST PO LEMATYZACJI
# FILE_PATH = 'robinson-crusoe.txt' # TESKT BEZ LEMATYZACJI

#################### KOMUNIKACJA Z LLM: OpenAI / AWS - START #####################

# Konfiguracja klienta AWS
client = boto3.client('bedrock-runtime', region_name='eu-central-1')
MODEL_ID_RETRIEVAL = 'amazon.titan-embed-text-v1' 
MODEL_ID_GENERATION = 'amazon.titan-text-express-v1'

# Konfiguracja klienta OpenAI
env = dotenv_values('.env')
openai_client = OpenAI(api_key=env['OPENAI_API_KEY'])

# Ceny modeli: LLM
model_pricings = {
    'gpt-4o':{
        'input_tokens' : 5.00 / 1_000_000, #per token --> 0,005 $ / 1000 tokens
        'output_tokens' : 15.00 / 1_000_000 #per token --> 0,015 $ / 1000 tokens
    },
    'gpt-4o-mini':{
        'input_tokens' : 0.150 / 1_000_000, #per token --> 0,00015 $ / 1000 tokens
        'output_tokens' : 0.600 / 1_000_000 #per token --> 0,0006 $ / 1000 tokens
    },
    'amazon-titan-text-express-v1':{
        'input_tokens' : 0.3 / 1_000_000, #per token --> 0,0003 $ / 1000 tokens
        'output_tokens' : 0.863 / 1_000_000 #per token --> 0,000863 $ / 1000 tokens
    },

}

# Dodanie listy z modelami
DEFAULT_MODEL_INDEX=1
models = list(model_pricings.keys())
if 'model' not in st.session_state:
    st.session_state['model'] = models[DEFAULT_MODEL_INDEX] #jeśli nie ma - przypisz 1 model z listy


USD_TO_PLN = 4.02
PRICING = model_pricings[st.session_state['model']]


# Sekrety - tylko po wdrożeniu app na Streamlit Cloud !!

# if 'OPENAI_API_KEY' in st.secrets:
#     env['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# openai_client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

#################### KOMUNIKACJA Z LLM / AWS - KONIEC #################


#################### USTAWIENIA (FUNKCJE) RAG - START ##########################


###################### CHUNKOWANIE - START #####################################

# 1. Chunkowanie - na akapity (grupowanie 5 akapitów)
def split_text_by_paragraphs(full_text, group_size=5):
    # Podziel tekst na akapity
    paragraphs = full_text.split('\n\n')
    # Odfiltruj niepotrzebne akapity i pozbądź się nadmiarowych spacji
    filtered_paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip() != '' and 'Rozdział' not in paragraph]

    # Grupowanie akapitów
    grouped_paragraphs = []
    for i in range(0, len(filtered_paragraphs), group_size):
        # Połącz grupę akapitów jako jeden tekst
        group = '\n\n'.join(filtered_paragraphs[i:i + group_size])
        grouped_paragraphs.append(group)

    return grouped_paragraphs

######################

# 2. Chunkowanie - na rozdziały (SZYBSZA METODA -> tylko 14 chunków)
def split_text_by_chapters(full_text):

    if 'Rozdział' in full_text:
        chapters = full_text.split('Rozdział')
    elif 'rozdział' in full_text: # dla new_full_text -> po lemmatyzacji
        chapters = full_text.split('rozdział')
    
    if chapters[0].strip() == '':
        chapters = chapters[1:]
    chapters = ['Rozdział' + chapter for chapter in chapters]
    
    return chapters

######################

# 3. Chunkowanie - na stałą liczbę znaków
def split_text_by_fixed_length(full_text, max_length=500):
    return [full_text[i:i + max_length] for i in range(0, len(full_text), max_length)]

######################

# 4. Chunkowanie - z wykorzystaniem window sliding
def split_text_with_sliding_window(full_text, window_size=300, step=100):
    words = full_text.split()
    chunks = []
    for i in range(0, len(words), step):
        chunk = words[i:i + window_size]
        if chunk:
            chunks.append(' '.join(chunk))
    return chunks

###################### CHUNKOWANIE - KONIEC ##################################### 

# 5. Chunkowanie dynamiczne - na stałą liczbę zdań
def dynamic_chunking(text, max_sentences=20):
    # Tokenizacja na zdania
    sentences = sent_tokenize(text) # prosta tokenizacja (podział na słowa)

    dynamic_chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(current_chunk) >= max_sentences:
            dynamic_chunks.append(" ".join(current_chunk))
            current_chunk = []

    # Dodanie ostatniego fragmentu
    if current_chunk:
        dynamic_chunks.append(" ".join(current_chunk))

    return dynamic_chunks

######################

# Wczytywanie tekstu (już po lematyzacji)
def read_text(FILE_PATH):
    with open(FILE_PATH, 'r', encoding='utf-8') as file:
        full_text = file.read()
        
    return full_text

# Generowanie Embeddings
@st.cache_resource
def get_embeddings(chunks):

    if isinstance(chunks, str):
        chunks = [chunks]

    embeddings = []
    
    for chunk in chunks:
        request_body = {
            'inputText' : chunk
        }

        response = client.invoke_model(
            modelId=MODEL_ID_RETRIEVAL,
            body=json.dumps(request_body)
        )

        model_response = json.loads(response['body'].read())

        embedding = model_response['embedding']
        embeddings.append(embedding)
        
    return embeddings

# Umieszczanie Embeddings w Faiss
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) 
    
    normalized_embeddings = embeddings / norms
    
    return normalized_embeddings

def embeddings_to_faiss_cosine(embeddings):
    embedding_array = np.array(embeddings).astype('float32') # float32 - to wydajniejszych obliczeń
    
    normalized_embeddings = normalize_embeddings(embedding_array)

    embedding_size = len(normalized_embeddings[0])
    index = faiss.IndexFlatIP(embedding_size)  # Index dla iloczynu skalarnego (IP = Inner Product), 

    index.add(normalized_embeddings)

    return index # Zwracamy indeks, który umożliwia przeszukiwanie najbliższych sąsiddów wykorzystując cosinus similarity

# Wyszukiwanie najbliższych sąsiadów (k-NN)
def retrieve_knn(query_embedding, index, chunks, k=5):
    
    distances, indices = index.search(np.array(query_embedding).reshape(1,-1).astype('float32'), k)

    retrieved_chunks = [chunks[i] for i in indices[0]]

    retrieved_similarities = distances[0] # Zbierz wartości cosinus similarity dla poszczególnych k-NN (chunków)

    return retrieved_chunks, retrieved_similarities

# Progowanie dynamiczne
similarity_metrics = []

def update_threshold(successful_similarities, min_threshold=0.6, max_threshold=0.8):
    if not successful_similarities:
        return 0.6  # Domyślna wartość
        
    average_similarity = sum(successful_similarities) / len(successful_similarities)

    return min(max(average_similarity, min_threshold), max_threshold) # Trzymanie się zakresu: min(0.6) - max(0.8)


#################### USTAWIENIA (FUNKCJE) RAG - KONIEC #########################


#################### USTAWIENIA CHATU - START ##########################


# Generowanie odpowiedzi przez LLM z wbudowaną pamięcią

#@st.cache_resource
def generate_response_by_llm(context, memory):

    # Post-processing tekstu (odpowiedzi) - zapewnienie żeby model nie urywał tekstu i skończył zawsze na końcu zdania

    def trim_to_last_sentence(response):
    # Znajdź wszystkie pełne zdania zakończone kropką bezpośrednio po literach
        sentences = re.findall(r"[^.!?\d]+\.", response)

        # Jeżeli znaleziono przynajmniej jedno takie zdanie, zwróć ostatnie z nich
        if sentences:
            return ''.join(sentences)

        # Gdy nie znaleziono, zwróć cały teks
        return response


    # def trim_to_last_sentence(response):
    #     if '.' in response:
    #         return response[:response.rfind('.') + 1] # rfing('.') + 1 -> zwraca ostatnie wystąpienie kropki w zdaniu (włącznie z nią)
    #     return response 

    complete_response = ""

    # Opisz charakter LLM (jego świadomość)
    messages = [
            {
                'role' : 'system',
                'content' : st.session_state['chatbot_personality']
            },
        ]
    # Dodaj wszystkie wiadomości do czatu
    for message in memory:
        messages.append({
            'role' : message['role'],
            'content' : message['content']
        })

    # Dodaj najnowszą wiadomość użytkownika
    messages.append({
        'role' : 'user',
        'content' : context
    })

    # WYBRANY MODEL == OPENAI GPT
    usage = {} # koszty użycia

    # complete_response = ""

    if st.session_state['model'] in ['gpt-4o', 'gpt-4o-mini']:

        with st.spinner('Generowanie odpowiedzi...'):
            # Łączenie z modelem gpt-4o
            response = openai_client.chat.completions.create(
                model=st.session_state['model'],
                temperature=0.7,
                max_tokens = 200,
                top_p = 0.9,
                messages = messages
            )

        complete_response = trim_to_last_sentence(response.choices[0].message.content)

        # Koszty korzystania z OpenAI
        if response.usage:
            usage = {
                'prompt_tokens' : response.usage.prompt_tokens,
                'completion_tokens' : response.usage.completion_tokens,
                'total_tokens' : response.usage.total_tokens
            }

    # WYBRANY MODEL == AMAZON TITAN

    elif st.session_state['model'] == 'amazon-titan-text-express-v1':

        request_body = {
        "inputText": context, 
        "textGenerationConfig": {
        "maxTokenCount": 200, 
        "temperature": 0.7, 
        "topP": 0.9, 
        "stopSequences": []
            }
        }

        with st.spinner('Generowanie odpowiedzi...'):

            response = client.invoke_model(
            modelId=MODEL_ID_GENERATION,
            body=json.dumps(request_body)
            )

        model_response = json.loads(response['body'].read())
        complete_response = trim_to_last_sentence(model_response['results'][0]['outputText'])
    

        if 'results' in model_response:
            #complete_response = trim_to_last_sentence(model_response['results'][0]['outputText'])
            input_tokens = model_response.get('inputTextTokenCount', 0)
            output_tokens = model_response['results'][0].get('tokenCount', 0)

            usage = {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens
            }

    return {
        'role' : 'assistant',
        'content' : complete_response,
        'usage': usage,
    }


#
# CONVERSATION HISTORY AND DATABASE
#

DEFAULT_PERSONALITY = """
Jesteś znawcą lektury "Robinson Crusoe" i poprawnie odpowiadasz na zadane pytania.
""".strip()

DB_PATH = Path('db')
DB_CONVERSATIONS_PATH = DB_PATH / 'conversations'
# db/
# ├── current.json
# ├── conversations/
# │   ├── 1.json
# │   ├── 2.json
# │   └── ...

#funkcja ładująca elementy do pamięci session state
def load_conversation_to_session_state(conversation):
    st.session_state['id'] = conversation['id']
    st.session_state['name'] = conversation['name']
    st.session_state['messages'] = conversation['messages']
    st.session_state['chatbot_personality'] = conversation['chatbot_personality']

# WCZYTUJEMY NASZE DANE ZE STRUKTURY

def load_current_conversation():
#sprawdzamy czy istnieje nasza struktura - jeśli NIE, inicjalizujemy ją
    if not DB_PATH.exists():
        DB_PATH.mkdir()
        DB_CONVERSATIONS_PATH.mkdir()
        conversation_id = 1
        conversation = {
            'id' : conversation_id,
            'name' : 'Konwersacja 1',
            'chatbot_personality' : DEFAULT_PERSONALITY,
            'messages' : []
        }
#następnie tworzymy NOWE pliki w naszej strukturze (nowa konwersacja)
        with open(DB_CONVERSATIONS_PATH / f'{conversation_id}.json', 'w') as f:
            f.write(json.dumps(conversation))
#ta konwersacja staje się od razu aktualną
        with open(DB_PATH / 'current.json', 'w') as f:
            f.write(json.dumps({
                'current_conversation_id' : conversation_id
            }))
#jeśli struktura już ISTNIEJE, odczytujemy z niej aktualną konwersację
    else:
#sprawdzamy, która konwersacja jest aktualna
        with open(DB_PATH / 'current.json', 'r') as f:
            data = json.loads(f.read())
            conversation_id = data['current_conversation_id']
#wczytujemy aktualną konwersacje
        with open(DB_CONVERSATIONS_PATH / f'{conversation_id}.json', 'r') as f:
            conversation = json.loads(f.read())

    load_conversation_to_session_state(conversation)

#ZAPISUJEMY NOWE INFORMACJE DO KONWERSACJI

#nowe wiadomości
def save_current_conversation_messages(): 
    conversation_id = st.session_state['id']
    new_messages = st.session_state['messages']
#odczytujemy treść wskazanej konwersacji
    with open(DB_CONVERSATIONS_PATH / f'{conversation_id}.json', 'r') as f:
        conversation = json.loads(f.read())
#nadpisujemy ją poprzez nowe informacje
    with open(DB_CONVERSATIONS_PATH / f'{conversation_id}.json', 'w') as f:
        f.write(json.dumps({
            **conversation,
            'messages' : new_messages
        }))

#nowa nazwa konwersacji
def save_current_conversation_name():
    conversation_id = st.session_state['id']
    new_conversation_name = st.session_state['new_conversation_name']
    #odczytujemy treść wskazanej konwersacji
    with open(DB_CONVERSATIONS_PATH / f'{conversation_id}.json', 'r') as f:
        conversation = json.loads(f.read())
    #nadpisujemy ją poprzez nowe informacje
    with open(DB_CONVERSATIONS_PATH / f'{conversation_id}.json', 'w') as f:
        f.write(json.dumps({
            **conversation,
            'name' : new_conversation_name
        }))

#nowa osobowość
def save_current_conversation_personality():
    conversation_id = st.session_state['id']
    new_chatbot_personality = st.session_state['new_chatbot_personality']
#odczytujemy treść wskazanej konwersacji
    with open(DB_CONVERSATIONS_PATH / f'{conversation_id}.json', 'r') as f:
        conversation = json.loads(f.read())
#nadpisujemy ją poprzez nowe informacje
    with open(DB_CONVERSATIONS_PATH / f'{conversation_id}.json', 'w') as f:
        f.write(json.dumps({
            **conversation,
            'chatbot_personality' : new_chatbot_personality
        }))

#TWORZYMY NOWA KONWERSACJE

def create_new_conversation():
#szukamy id dla naszej kolejnej konwersacji
    conversation_ids = []
#iterujemy po wszystkich plikach .json
    for p in DB_CONVERSATIONS_PATH.glob('*.json'):
#dodajemy do listy wyciągniete nazwy bez rozszerzeń = id
        conversation_ids.append(int(p.stem))
#znajdujemy id dla naszej nowej konwersacji
    conversation_id = max(conversation_ids) + 1
#inicjalizujemy zawartość nowej konwersacji
    personality = DEFAULT_PERSONALITY
#dzięki poniższej instrukcji zostanie wczytana ostatnia zapisana wersja osobowości chatu
    if 'chatbot_personality' in st.session_state and st.session_state['chatbot_personality']:
        personality = st.session_state['chatbot_personality']
    conversation = {
        'id' : conversation_id,
        'name' : f'Konwersacja {conversation_id}',
        'chatbot_personality' : personality,
        'messages' : []
    }
#tworzymy nową konwersacje
    with open(DB_CONVERSATIONS_PATH / f'{conversation_id}.json', 'w') as f:
        f.write(json.dumps(conversation))
#która od razu staje się aktualna
    with open(DB_PATH / 'current.json', 'w') as f:
        f.write(json.dumps({
            'current_conversation_id' : conversation_id
        }))
#dane z nowej konwersacji ładujemy do session state
    load_conversation_to_session_state(conversation)
    st.rerun() #na koniec wymuszamy restart aplikacji

#PRZEŁACZANIE KONWERSACJI MIEDZY SOBA

#najpierw odczytaj mi konwersacje o podanym id
def switch_conversation(conversation_id):
    with open(DB_CONVERSATIONS_PATH / f'{conversation_id}.json', 'r') as f:
        conversation = json.loads(f.read())
#następnie przełącz ją na obecną konwersacje
    with open(DB_PATH / 'current.json', 'w') as f:
        f.write(json.dumps({
            'current_conversation_id' : conversation_id
        }))
    load_conversation_to_session_state(conversation)
    st.rerun()

#WYLISTOWANIE STWORZONYCH KONWERSACJI

def list_conversations():
    conversations = [] #inicjalizujemy pustą listę wszystkich konwersacji
    for p in DB_CONVERSATIONS_PATH.glob('*.json'):
        with open(p, 'r') as f:
            conversation = json.loads(f.read())
            conversations.append({
                'id' : conversation['id'],
                'name' : conversation['name']
            })
    return conversations #zwracamy listę wszystkich konwersacji

############# USTAWIENIA CHATU - KONIEC #######################


############# MAIN PROGRAM - START ##########################

# Wczytywanie historii rozmów z naszej bazy danych (utworzonej przez nas struktury)
load_current_conversation()

# Tytuł aplikacji
st.title('Robinson Chatbot 🏝️')

# Zapamiętywanie starych wiadomości z czatu
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Ustawienia zapytań i odpowiedzi -> zastosowanie dwóch metod: 1) GOTOWE PROMPTY , 2) MOŻLIWOŚĆ ZADANIA PYTANIA NA CZACIE
user_input = st.chat_input('O co chcesz spytać?')

# Lista promptów
prompts = [
   # 'Wybierz prompt',
    'Podobnie jak Robinson w "Cast Away", który musiał stworzyć codzienną rutynę, aby przeżyć na bezludnej wyspie, opisz, jak Robinson Crusoe zorganizował swoje codzienne życie na wyspie, aby przetrwać.',
    'Podobnie jak w historii "Piętaszka", gdzie bohater musiał znaleźć sposoby na radzenie sobie z lokalnymi niebezpieczeństwami, opisz, w jaki sposób Robinson Crusoe radził sobie z zagrożeniami ze strony dzikich mieszkańców wyspy.',
    'Podobnie jak Tom Hanks w "Cast Away", który przeżywał ciężkie chwile osamotnienia, opisz, jakie były najtrudniejsze momenty samotności, z którymi zmagał się Robinson, i jak sobie z nimi radził.',
    'Podobnie jak bohaterowie odnajdujący nowe wartości po powrocie z dramatycznych przygód, opisz, jakie przemiany osobowości i poglądów przeszedł Robinson po ostatecznym powrocie z wyspy do Anglii?',
    'Podobnie jak inne postacie literackie, które przetrwały samemu na wyspie dzięki szczególnym cechom charakteru, opisz, jakie cechy charakteru pozwoliły Robinsonowi na przetrwanie i co czyniło go wyjątkowym w porównaniu z innymi rozbitkami z literatury.',
    'Jakie cechy charakteru pozwoliły Robinsonowi na przetrwanie na wyspie i co czyniło go wyjątkowym w porównaniu z innymi rozbitkami z literatury ? '
]

selected_prompt = st.sidebar.selectbox('Wybierz gotowy prompt', prompts)

generate = st.sidebar.button('Generuj odpowiedź')

# Decyzja, którą metodę zapytania użyć
if user_input:
    prompt = user_input
elif selected_prompt and generate:
    prompt = selected_prompt
else:
    prompt = None

# Implementacja RAG
full_text = read_text(FILE_PATH)

# Wybór metody chunkowania
# chunks = split_text_by_chapters(full_text)
# chunks = split_text_by_paragraphs(full_text, 5)
# chunks = split_text_by_fixed_length(full_text, 500)
# chunks = split_text_with_sliding_window(full_text, 300, 100)
chunks = dynamic_chunking(full_text, 20)

embeddings = get_embeddings(chunks)

index = embeddings_to_faiss_cosine(embeddings)

# Zapytanie użytkownika
if prompt:

    prompt_embedding = get_embeddings(prompt)
    normalized_prompt_embedding = normalize_embeddings(prompt_embedding)

    retrieved_chunks, retrieved_similarities = retrieve_knn(normalized_prompt_embedding, index, chunks, k=5)

    context = prompt + " " + " ".join(retrieved_chunks)

    user_message = {'role' : 'user', 'content' : prompt}
    with st.chat_message('user'): # Dodanie wiadomości do czatu
        st.markdown(user_message['content'])
        selected_prompt = 'Wybierz gotowy prompt'

# Zapisanie zapytania użytkownika w pamięci
    st.session_state['messages'].append(user_message)
                                        
# Odpowiedź LLM 
    with st.chat_message('assistant'):

        # Wybór MODELU LLM !!!!
        chatbot_message = generate_response_by_llm(
            prompt, memory = st.session_state['messages'][-10:]) # Chat zapamiętuje 10 ostatnich wiadomości !!
        st.markdown(chatbot_message['content'])
        # st.info(st.session_state['model'])


# Zapisanie odpowiedzi LLM
    st.session_state['messages'].append(chatbot_message)

    save_current_conversation_messages() # Zapisanie wszystkich wiadomości

############# MAIN PROGRAM - KONIEC ##########################
    
############ PASEK BOCZNY - START #############################

with st.sidebar:

    # Wyświetlenie ustawionego aktualnie modelu czatu gpt
    st.write('---')
    st.write(f'Aktualny model: {st.session_state["model"]}')

    # Dodanie opcji wyboru modelu
    selected_model = st.selectbox("Wybrany model", models, index=DEFAULT_MODEL_INDEX)
    st.session_state['model'] = selected_model
    PRICING = model_pricings[st.session_state['model']]


    # Obliczenie kosztów
    total_cost = 0
    for message in st.session_state['messages']:
        if 'usage' in message:
            usage = message['usage']

            if 'prompt_tokens' in usage and 'completion_tokens' in usage:
                total_cost += usage['prompt_tokens'] * PRICING['input_tokens']
                total_cost += usage['completion_tokens'] * PRICING['output_tokens']

            if 'input_tokens' in usage and 'output_tokens' in usage:
                total_cost += usage['input_tokens'] * PRICING['input_tokens']
                total_cost += usage['output_tokens'] * PRICING['output_tokens']

    # Wyświetlenie kosztów
    c0, c1 = st.columns(2)
    with c0:
        st.metric('Koszt rozmowy (USD): ', f'${total_cost:.4f}')
    with c1:
        st.metric('Koszt rozmowy (PLN): ', f'{total_cost * USD_TO_PLN:.4f}')

    st.write('---')
    # Wyświetlenie osobowości OpenAI - pole do modyfikacji
    st.session_state['chatbot_personality'] = st.text_area(
        'Osobowość chatbota',
        max_chars = 1000,
        height = 200,
        value=st.session_state["chatbot_personality"],
        key = 'new_chatbot_personality',
        on_change = save_current_conversation_personality
    )

    st.write('---')
    # Wyświetlenie nazwy konwersacji
    st.session_state['name'] = st.text_input(
        'Nazwa konwersacji',
        value = st.session_state['name'],
        key = 'new_conversation_name',
        on_change = save_current_conversation_name 
        #on_change - pozwala na przekazanie funkcji jako argumentu
    )

    # Tworzenie nowych konwersacji
    st.subheader('Konwersacje')
    if st.button('Nowa konwersacja'):
        create_new_conversation()

    # Sortowanie konwersacji i możliwość przełączania pomiędzy nimi
    conversations = list_conversations() 
    sorted_conversations = sorted(conversations, key=lambda x: x['id'], reverse=True)
    for conversation in sorted_conversations[:5]: #pokazujemy tylko top 5 konwersacji
        c0, c1 = st.columns([10, 5])
        with c0:
            st.write(conversation['name'])
        with c1:
            if st.button('Załaduj', key= conversation['id'], disabled=conversation['id'] == st.session_state['id']):
                switch_conversation(conversation['id'])


############ PASEK BOCZNY - KONIEC #############################


