import streamlit as st
import pandas as pd
import os
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore
import warnings
import re


# ----------------------------------------------------
# KONFIGURACJA
# ----------------------------------------------------

MODEL_NAME = 'welcome_survey_clustering_pipeline_v1'
DATA = os.path.join(os.path.dirname(__file__), "welcome_survey_simple_v1.csv")

# KOMPLETNY ZESTAW RÓL DLA KLASTRÓW 0–7
CLUSTER_ROLES = {
    0: {
        "name": "Spokojni analitycy",
        "strengths": "cierpliwość, porządkowanie informacji, spokojne i przemyślane podejście do problemów.",
        "good_tasks": "analiza danych, przygotowywanie raportów, sprawdzanie szczegółów, praca indywidualna.",
        "person_desc": (
            "Osoby z tego klastra zwykle lubią mieć czas na zastanowienie i działają raczej spokojnie niż impulsywnie. "
            "Cenią porządek, przewidywalność i dobrze czują się tam, gdzie można coś na spokojnie przeanalizować."
        )
    },
    1: {
        "name": "Kreatywni odkrywcy",
        "strengths": "szukanie nowych rozwiązań, generowanie pomysłów, otwartość na zmiany.",
        "good_tasks": "burze mózgów, tworzenie nowych ofert, praca projektowa, testowanie nowych pomysłów.",
        "person_desc": (
            "Osoby z tego klastra łatwo wpadają na nowe pomysły i szybko się nudzą rutyną. "
            "Dobrze czują się tam, gdzie można eksperymentować, próbować nowych podejść i wychodzić poza schemat."
        )
    },
    2: {
        "name": "Ludzie od relacji",
        "strengths": "budowanie zaufania, komunikacja, wspieranie innych.",
        "good_tasks": "kontakt z klientem, praca zespołowa, onboarding nowych osób, prowadzenie spotkań.",
        "person_desc": (
            "Osoby z tego klastra zwykle lubią być w kontakcie z innymi i szybko wyczuwają nastroje w zespole. "
            "Często to do nich inni przychodzą, gdy trzeba coś wyjaśnić, dogadać lub po prostu pogadać."
        )
    },
    3: {
        "name": "Zadaniowi wykonawcy",
        "strengths": "konsekwencja, domykanie tematów, trzymanie terminów.",
        "good_tasks": "realizacja konkretnych zadań, pilnowanie harmonogramu, wdrażanie ustalonych planów.",
        "person_desc": (
            "Osoby z tego klastra lubią widzieć konkretne efekty swojej pracy i zazwyczaj dowożą to, co zaczęły. "
            "Dobrze czują się w zadaniach z jasno określonym celem i terminem, gdzie wiadomo, co trzeba zrobić."
        )
    },
    4: {
        "name": "Strategiczni planujący",
        "strengths": "patrzenie szerzej, planowanie kroków do przodu, łączenie różnych wątków.",
        "good_tasks": "planowanie projek­tów, układanie harmonogramów, wyznaczanie kierunku działań.",
        "person_desc": (
            "Osoby z tego klastra lubią rozumieć szerszy obraz sytuacji i myśleć o tym, co będzie dalej. "
            "Dobrze czują się, gdy mogą łączyć kropki i układać całość w sensowny plan."
        )
    },
    5: {
        "name": "Elastyczni eksperymentatorzy",
        "strengths": "dostosowywanie się do zmian, szybkie reagowanie, gotowość do próbowania nowych rozwiązań.",
        "good_tasks": "testowanie nowych pomysłów, praca w zmiennym środowisku, zadania wymagające improwizacji.",
        "person_desc": (
            "Osoby z tego klastra zwykle dobrze znoszą zmiany i niepewność. "
            "Lubią sprawdzać „co się stanie, gdy…” i uczą się głównie poprzez działanie."
        )
    },
    6: {
        "name": "Stabilizatorzy zespołu",
        "strengths": "dawanie poczucia bezpieczeństwa, trzymanie się ustalonych zasad, dbanie o stały rytm pracy.",
        "good_tasks": "pilnowanie standardów, opieka nad stałymi procesami, dbanie o porządek organizacyjny.",
        "person_desc": (
            "Osoby z tego klastra często są „cichą podporą” zespołu. "
            "Dobrze czują się tam, gdzie można robić swoje spokojnie i według ustalonych zasad."
        )
    },
    7: {
        "name": "Szybcy reagujący",
        "strengths": "szybkie podejmowanie decyzji, działanie pod presją, ogarnianie sytuacji kryzysowych.",
        "good_tasks": "gaszenie pożarów, reagowanie na nagłe problemy, wsparcie tam, gdzie liczy się czas.",
        "person_desc": (
            "Osoby z tego klastra zwykle dobrze funkcjonują w sytuacjach, w których trzeba działać tu i teraz. "
            "Często przejmują inicjatywę, gdy coś nagle się sypie i trzeba to szybko postawić na nogi."
        )
    }
}

os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
os.environ["PANDAS_PYARROW_VERSION_CHECK"] = "0"
warnings.filterwarnings("ignore", category=UserWarning, module="pandas.compat")


# ----------------------------------------------------
# FUNKCJE CACHE
# ----------------------------------------------------

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_all_participants():
    return pd.read_csv(DATA, sep=';', engine='python')


# ----------------------------------------------------
# FUNKCJA POMOCNICZA – OPIS KLASTRA
# ----------------------------------------------------

def get_cluster_role(cluster_id):
    """
    Zwraca słownik z rolą dla danego klastra.
    Obsługuje przypadek, gdy cluster_id wygląda np. jak 'Cluster 1'.
    """

    text = str(cluster_id)

    # znajdź pierwszą liczbę w tekście, np. w 'Cluster 1' -> '1'
    match = re.search(r"\d+", text)
    if match:
        cid = int(match.group(0))
        base = CLUSTER_ROLES.get(cid)
        if base is not None:
            return base

    # jeśli nic nie znaleźliśmy albo brak w CLUSTER_ROLES – opis domyślny
    return {
        "name": f"Klaster {text}",
        "strengths": "styl pracy jeszcze nieopisany w szczegółach.",
        "good_tasks": "zadania dobierane indywidualnie na podstawie rozmowy i obserwacji.",
        "person_desc": (
            "Ten klaster nie ma jeszcze dopasowanego profilu w narzędziu. "
            "To dobry pretekst, żeby przyjrzeć się bliżej sposobowi pracy osób z tej grupy."
        )
    }


# ----------------------------------------------------
# NAGŁÓWEK
# ----------------------------------------------------

st.title("Kreatywny zespół – poznaj swój styl pracy")
st.markdown(
    """
    Ten prosty test pokazuje, do której grupy stylu pracy jest Ci **najbliżej**.  
    To, **jak odpoczywasz** i jakie masz **małe preferencje** (psy/koty, las/góry itd.),
    często zdradza też **jak podchodzisz do rozwiązywania problemów** i pracy z ludźmi.
    """
)


# ----------------------------------------------------
# SIDEBAR — WYBÓR TRYBU
# ----------------------------------------------------

st.sidebar.header("Co chcesz zobaczyć?")

mode = st.sidebar.radio(
    "Wybierz opcję",
    ["Mój klaster", "Analiza klastrów – poznaj swoje społeczności"]
)


# ----------------------------------------------------
# SIDEBAR — DANE UŻYTKOWNIKA
# ----------------------------------------------------

st.sidebar.header("Powiedz nam coś o sobie")

age = st.sidebar.selectbox(
    "Wiek",
    ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
)

edu_level = st.sidebar.selectbox(
    "Wykształcenie",
    ['Podstawowe', 'Średnie', 'Wyższe', 'Doktorat']
)

fav_animals = st.sidebar.selectbox(
    "Ulubione zwierzęta",
    ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy']
)

fav_place = st.sidebar.selectbox(
    "Ulubione miejsce wypoczynku",
    ['Nad wodą', 'W lesie', 'W górach', 'Inne']
)

gender = st.sidebar.radio(
    "Płeć",
    ['Mężczyzna', 'Kobieta']
)


# ----------------------------------------------------
# TRYB 1 — MÓJ KLASTER
# ----------------------------------------------------

if mode == "Mój klaster":

    st.header("Twój klaster")

    user_df = pd.DataFrame([{
        'age': age,
        'edu_level': edu_level,
        'fav_animals': fav_animals,
        'fav_place': fav_place,
        'gender': gender
    }])

    model = get_model()
    user_cluster = predict_model(model, data=user_df)["Cluster"].values[0]

    st.subheader(f"Najbliżej Ci do klastra: **{user_cluster}**")

    # Opis roli w zespole
    role = get_cluster_role(user_cluster)

    st.markdown("### Jak możesz się sprawdzać w zespole?")
    st.write(f"**Nazwa profilu:** {role['name']}")
    st.write(f"**Mocne strony:** {role['strengths']}")
    st.write(f"**Dobre zadania dla Ciebie:** {role['good_tasks']}")

    if "person_desc" in role:
        st.markdown("### Krótka charakterystyka osób z tego klastra")
        st.write(role["person_desc"])

    st.markdown("### Twoje odpowiedzi z testu")
    st.table(user_df.T.rename(columns={0: "Ty"}))


# ----------------------------------------------------
# TRYB 2 — ANALIZA KLASTRÓW
# ----------------------------------------------------

else:

    st.header("Analiza klastrów – poznaj swoje społeczności")
    st.write("Poniżej zobaczysz statystyki i strukturę wszystkich klastrów.")

    # 1. Dane wszystkich uczestników
    all_df = get_all_participants()

    # 2. Model
    model = get_model()

    # 3. Predykcja klastra dla każdego wpisu
    clustered = predict_model(model, data=all_df)

    # 4. Liczebność klastrów
    cluster_counts = clustered['Cluster'].value_counts().sort_index()

    st.subheader("Liczba osób w każdym klastrze")
    st.write(cluster_counts)

    # -------------------------------------------------------------
    # PROFESJONALNY RAPORT WSZYSTKICH KLASTRÓW
    # -------------------------------------------------------------

    st.subheader("Profesjonalny raport wszystkich klastrów")

    clustered_df = clustered  # DataFrame z kolumną 'Cluster'
    unique_clusters = sorted(clustered_df["Cluster"].unique())

    def profile_cluster(cid):

        st.markdown("---")
        st.markdown(f"## Klaster {cid}")

        sub = clustered_df[clustered_df["Cluster"] == cid]

        count = len(sub)
        total = len(clustered_df)
        pct = round(100 * count / total, 1)

        st.write(f"Liczebność: **{count} osób** ({pct}%)")

        categorical_cols = ["age", "edu_level", "fav_animals", "fav_place", "gender"]

        for col in categorical_cols:
            if col in sub.columns:
                st.write(f"### Rozkład: {col}")
                counts = sub[col].value_counts().reset_index()
                counts.columns = [col, "Liczba"]

                fig = px.bar(
                    counts,
                    x=col,
                    y="Liczba",
                    title=f"Klaster {cid} — {col}",
                    text="Liczba",
                    color=col
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        st.write("### Najczęstsze wartości")
        summary = {
            col: sub[col].value_counts().idxmax()
            for col in categorical_cols
            if col in sub.columns and not sub[col].empty
        }
        st.table(pd.DataFrame.from_dict(summary, orient="index", columns=["Dominanta"]))

        st.write("### Interpretacja")
        try:
            st.write(
                f"Klaster {cid} skupia osoby najczęściej w wieku **{sub['age'].value_counts().idxmax()}**, "
                f"o wykształceniu **{sub['edu_level'].value_counts().idxmax()}**, "
                f"preferujące **{sub['fav_animals'].value_counts().idxmax().lower()}** "
                f"i odpoczywające **{sub['fav_place'].value_counts().idxmax().lower()}**."
            )
        except Exception:
            st.write("Brak danych do pełnej interpretacji.")

        st.write("### Sugerowane predyspozycje i zadania w zespole")
        role = get_cluster_role(cid)

        st.write(f"**Nazwa profilu:** {role['name']}")
        st.write(f"**Mocne strony:** {role['strengths']}")
        st.write(f"**Dobre zadania w zespole:** {role['good_tasks']}")

        if "person_desc" in role:
            st.write("**Jak zwykle wyglądają osoby z tego klastra:**")
            st.write(role["person_desc"])

    for cid in unique_clusters:
        profile_cluster(cid)


# ----------------------------------------------------
# INFO NA DOLE APLIKACJI
# ----------------------------------------------------

st.markdown("---")
st.info(
    "Każda osoba, która wypełni **krótki test po lewej stronie**, "
    "może zobaczyć, do którego **klastra zespołu** jest jej najbliżej. "
    "To narzędzie ma pomagać w **układaniu zespołów**, dobieraniu ludzi do zadań "
    "i lepszym rozumieniu, **jak różne style pracy mogą się uzupełniać**."
)
