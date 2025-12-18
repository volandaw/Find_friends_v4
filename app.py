import streamlit as st
import sklearn
sklearn.set_config(transform_output="default") # Ta linia naprawia błąd ze zdjęcia
from pycaret.clustering import load_model, predict_model
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
        "name": "Strategiczny Opiekun (Strategist & Mentor)",
        "strengths": "stabilność emocjonalna, wysoka etyka pracy, zdolność do mentoringu, dystans do sytuacji kryzysowych.",
        "good_tasks": "nadzór nad procesami, zarządzanie ryzykiem, doradztwo strategiczne, rozstrzyganie konfliktów.",
        "person_desc": (
            "Osoby z tego klastra to 'spokojna siła' zespołu. Łączą dojrzałość (45-54 lata) z solidnym wykształceniem. "
            "Wybór lasu i psów sugeruje naturę lojalną, ceniącą spokój i głębokie relacje zamiast powierzchownego zgiełku. "
            "To naturalni mentorzy, którzy wnoszą do grupy poczucie bezpieczeństwa i szeroką perspektywę."
        )
    },
    1: {
        "name": "Dynamiczny Innowator (The Flow Master)",
        "strengths": "zdolność adaptacji, inicjatywa, kreatywne łączenie faktów, wysoka energia w działaniu.",
        "good_tasks": "wdrażanie innowacji, prowadzenie dynamicznych projektów, budowanie relacji, szybkie prototypowanie.",
        "person_desc": (
            "Osoby z tego klastra to 'silniki' zmian. Łączą wiek największej aktywności (35-44 lata) z otwartym umysłem. "
            "Wybór odpoczynku nad wodą sugeruje naturę elastyczną, która potrzebuje swobody, by generować najlepsze pomysły. "
            "To lojalni partnerzy (psy), którzy nie boją się wypłynąć na głęboką wodę, gdy projekt wymaga odwagi i świeżego spojrzenia."
        )
    },
    2: {
        "name": "Ambitny Zespołowiec (The Mountain Climber)",
        "strengths": "wytrwałość, autentyczność, budowanie lojalnych relacji, odporność na trudne warunki.",
        "good_tasks": "koordynacja projektów, sprzedaż relacyjna, motywowanie zespołu, wsparcie w sytuacjach kryzysowych.",
        "person_desc": (
            "Osoby z tego klastra to ambitna 'młoda krew' zespołu (25-34 lata). "
            "Wybór gór jako miejsca odpoczynku świadczy o charakterze, który lubi wyzwania i nie boi się wysiłku. "
            "W połączeniu z miłością do psów, tworzy to profil osoby niezwykle lojalnej i pomocnej, "
            "która wejdzie na każdy szczyt, o ile będzie mogła to zrobić w dobrym towarzystwie."
        )
    },
    3: {
        "name": "Niezależny Architekt Rozwiązań (The Autonomous Finisher)",
        "strengths": "autonomia, głęboka koncentracja, precyzja, wysoka jakość dostarczanych rozwiązań.",
        "good_tasks": "samodzielne projekty eksperckie, optymalizacja procesów, rozwiązywanie problemów, domykanie kluczowych etapów.",
        "person_desc": (
            "Osoby z tego klastra to specjaliści ceniący niezależność (35-44 lata). "
            "Wybór kotów i gór świadczy o naturze introwertycznego lidera własnej pracy, "
            "który nie potrzebuje ciągłej uwagi, by dowozić perfekcyjne wyniki. "
            "To typ 'wolnego strzelca' wewnątrz organizacji – niezwykle skuteczny, "
            "gdy otrzyma jasny cel i przestrzeń do działania według własnych zasad."
        )
    },
    4: {
        "name": "Pragmatyczny Analityk Systemowy (The Lean Strategist)",
        "strengths": "bezstronność, chłodna ocena sytuacji, optymalizacja procesów, wysoka orientacja na cel.",
        "good_tasks": "audyt procesów, tworzenie strategii, optymalizacja zasobów, planowanie ścieżek krytycznych.",
        "person_desc": (
            "Osoby z tego klastra to umysły nastawione na czystą logikę i strukturę (25-34 lata). "
            "Brak preferencji zwierzęcych w połączeniu z pasją do gór sugeruje profil wybitnie racjonalny, "
            "który odrzuca zbędne sentymenty na rzecz efektywności. To architekci porządku, "
            "którzy potrafią spojrzeć na projekt z dystansu i zaplanować najbardziej optymalną drogę do sukcesu."
        )
    },
    5: {
        "name": "Intuicyjny Nawigator (The Adaptive Mentor)",
        "strengths": "rezyliencja, wysoka inteligencja emocjonalna, mądra elastyczność, zdolność adaptacji.",
        "good_tasks": "zarządzanie zmianą, moderowanie kreatywne, wsparcie w sytuacjach kryzysowych, wdrażanie nowych standardów.",
        "person_desc": (
            "Osoby z tego klastra to doświadczone liderki zmian (45-54 lata). "
            "Wybór wody i psów wskazuje na połączenie empatii z umiejętnością płynnego reagowania na wyzwania. "
            "Ich 'eksperymentowanie' opiera się na fundamencie wiedzy i intuicji, co daje zespołowi "
            "poczucie bezpieczeństwa nawet w zmiennym środowisku. To mistrzynie opanowania, "
            "które wiedzą, jak przeprowadzić grupę przez nieznane wody."
        )
    },
    6: {
        "name": "Strażnik Standardów i Etosu (The Custodian of Excellence)",
        "strengths": "niezłomność merytoryczna, spokój, wysoka etyka pracy, dbałość o detale i jakość.",
        "good_tasks": "kontrola jakości, audyt procesów, tworzenie standardów, opieka nad fundamentami projektów.",
        "person_desc": (
            "Osoby z tego klastra to autorytety merytoryczne (45-54 lata). "
            "Wybór lasu i niespecyficznych zwierząt ('inne') sugeruje osobowość głęboką, "
            "nieco introwertyczną, o bardzo sprecyzowanych wartościach. To oni pilnują, "
            "by zespół nie szedł na skróty. Są fundamentem, który zapewnia trwałość "
            "i wysoką jakość, działając bez pośpiechu, ale z niezwykłą precyzją."
        )
    },
    7: {
        "name": "Skuteczny Strateg Operacyjny (The Crisis Commander)",
        "strengths": "błyskawiczna decyzyjność, wysoka odporność na stres, pragmatyzm, priorytetyzacja.",
        "good_tasks": "zarządzanie kryzysowe, ratowanie opóźnionych projektów, negocjacje, szybkie wdrożenia.",
        "person_desc": (
            "Osoby z tego klastra to specjaliści od zadań specjalnych (35-44 lata). "
            "Połączenie braku preferencji zwierzęcych z miłością do wody sugeruje umysł "
            "wyjątkowo sprawny, nastawiony na szybkie procesowanie informacji i działanie. "
            "To oni przejmują stery, gdy sytuacja staje się trudna. Są mistrzymi 'gaszenia pożarów', "
            "wprowadzając spokój i konkretną strukturę tam, gdzie liczy się każda sekunda."
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
    
    # 1. Przewidywanie klastra - wersja bezpieczna
    try:
        predictions = predict_model(model, data=user_df)
        user_cluster = predictions["Cluster"].iloc[0]
    except Exception:
        # Plan awaryjny (bezpośrednie wywołanie modelu)
        cluster_id = model.predict(user_df)[0]
        user_cluster = f"Cluster {cluster_id}"

    st.subheader(f"Najbliżej Ci do klastra: **{user_cluster}**")

    # 2. Opis roli w zespole
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

    # 3. Przypisanie klastrów do wszystkich osób - wersja odporna na błędy
    try:
        clustered = predict_model(model, data=all_df)
    except Exception:
        # Plan awaryjny dla całej tabeli
        preds = model.predict(all_df)
        clustered = all_df.copy()
        clustered['Cluster'] = [f"Cluster {p}" for p in preds]

    # 4. Liczebność klastrów
    cluster_counts = clustered['Cluster'].value_counts().sort_index()

    st.subheader("Liczba osób w każdym klastrze")
    st.write(cluster_counts)

    # -------------------------------------------------------------
    # PROFESJONALNY RAPORT WSZYSTKICH KLASTRÓW
    # -------------------------------------------------------------

    st.subheader("Profesjonalny raport wszystkich klastrów")

    clustered_df = clustered
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
