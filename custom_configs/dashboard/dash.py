from datetime import datetime
import json
import os
from pathlib import Path

import bcrypt
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import shap
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide")
CONFIG_PATH = "config.json"
PROJECT_ID = "dismissial_predict"
LOG_CSV_PATH = "auth_log.csv"

COLUMNS_TO_SHOW = [
    # 🎯 Целевая информация
    "дата_увольнения",
    # 👤 Личное
    "дата_рождения",
    "возраст",
    "пол",
    # 👨‍👩‍👧‍👦 Семья
    "число_детей",
    "средний_возраст_детей",
    "дети_мальчики",
    "дети_девочки",
    "есть_маленькие_дети",
    # 🏢 Рабочая информация
    "дата_приема_в_1с",
    "стаж",
    "текущая_должность_на_портале",
    "категория",
    "бе",
    "отдел",
    "id_руководителя",
    "подчиненные",
    # 📅 События
    "скоро_др",
    "скоро_годовщина_приема",
    # 💰 Финансы и производные
    "зп_на_ср_зп_по_компании",
    "зп_к_возрасту",
    "зп_к_стажу",
]

info_file = Path(
    "/home/root6/python/dismissal_predict_v2/data/processed/main_all_history_do_not_tuch.csv"
)


def log_auth_event_csv(login: str, status: str):
    import csv
    from datetime import datetime

    log_path = "auth_log.csv"
    headers = ["datetime", "status", "login"]
    row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, login]

    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        writer.writerow(row)


def handle_auth():
    placeholder = st.empty()

    if "login_stage" not in st.session_state:
        st.session_state.login_stage = "username"
    if "login" not in st.session_state:
        st.session_state.login = ""
    if "password_attempts" not in st.session_state:
        st.session_state.password_attempts = 0

    if st.session_state.login_stage == "username":
        with placeholder.container():
            st.title("🔐 Вход")
            with st.form("login_form"):
                login_input = st.text_input("Введите логин")
                submit_login = st.form_submit_button("Ввод")

            if submit_login:
                if login_input not in allowed_users:
                    log_auth_event_csv(login_input, "fail_unknown_user")
                    st.error("⛔ Пользователь не имеет доступа")
                else:
                    st.session_state.login = login_input
                    st.session_state.login_stage = "password"
                    st.rerun()
        st.stop()

    user_record = next(
        (u for u in config["users"] if u["username"] == st.session_state.login), None
    )

    if user_record and "password" in user_record:
        if st.session_state.login_stage != "authenticated":
            with placeholder.container():
                st.title("🔐 Авторизация")
                with st.form("password_form"):
                    password_input = st.text_input("Введите пароль", type="password")
                    submit_pass = st.form_submit_button("Ввод")

                if submit_pass:
                    if bcrypt.checkpw(
                        password_input.encode(), user_record["password"].encode("utf-8")
                    ):
                        st.session_state.login_stage = "authenticated"
                        log_auth_event_csv(st.session_state.login, "success")
                        placeholder.empty()
                        st.rerun()
                    else:
                        st.session_state.password_attempts += 1
                        log_auth_event_csv(st.session_state.login, "fail_wrong_password")
                        st.error("❌ Неверный пароль")
            st.stop()

    elif user_record is None:
        with placeholder.container():
            st.title("🛡 Создание пароля")
            with st.form("new_password_form"):
                new_pass1 = st.text_input("Введите новый пароль", type="password")
                new_pass2 = st.text_input("Повторите пароль", type="password")
                submit_new = st.form_submit_button("Создать")

            if submit_new:
                if new_pass1 != new_pass2:
                    st.error("❌ Пароли не совпадают")
                else:
                    hashed_pw = bcrypt.hashpw(new_pass1.encode(), bcrypt.gensalt()).decode()
                    config["users"].append(
                        {"username": st.session_state.login, "password": hashed_pw}
                    )
                    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                        json.dump(config, f, ensure_ascii=False, indent=2)

                    log_auth_event_csv(st.session_state.login, "created_new_password")
                    st.success("✅ Пароль сохранён.")
                    st.session_state.login_stage = "authenticated"
                    placeholder.empty()
                    st.rerun()
        st.stop()


def render_user_card(user: dict):
    def safe_get(key, fmt=None, suffix=""):
        val = user.get(key)
        if pd.isna(val):
            return "nan"
        try:
            if fmt:
                return fmt.format(val) + suffix
            return str(val) + suffix
        except:
            return "nan"

    def safe_bool(key):
        val = user.get(key)
        return "Да" if val else "Нет"

    st.markdown("### 👤 Карточка сотрудника")
    st.markdown(
        f"""
    <b>📌 Общая информация</b><br>
    Уволен: {safe_get("дата_увольнения")}<br>
    Дата рождения: {safe_get("дата_рождения")} &nbsp;&nbsp;&nbsp;
    Пол: {safe_get("пол")} &nbsp;&nbsp;&nbsp;
    Возраст: {safe_get("возраст", "{}")}<br><br>

    <b>👨‍👩‍👧‍👦 Семья</b><br>
    Детей: {safe_get("число_детей")} &nbsp;&nbsp;&nbsp;
    Мальчики: {safe_get("дети_мальчики")} &nbsp;&nbsp;&nbsp;
    Девочки: {safe_get("дети_девочки")} &nbsp;&nbsp;&nbsp;
    Маленькие дети (<=5): {safe_bool("есть_маленькие_дети")}<br><br>

    <b>💼 Работа</b><br>
    Дата приёма: {safe_get("дата_приема_в_1с")} &nbsp;&nbsp;&nbsp;<br>
    Руководитель: {safe_get("id_руководителя")}<br>
    Стаж: {safe_get("стаж", "{:.1f}")}<br>
    Должность: {safe_get("текущая_должность_на_портале")}<br>
    Категория: {safe_get("категория")} &nbsp;&nbsp;&nbsp;<br>
    БЕ: {safe_get("бе")}<br>
    Отдел: {safe_get("отдел")}<br>
    Подчиненные: {safe_get("подчиненные")}<br><br>

    <b>📅 События</b><br>
    Скоро ДР: {safe_bool("скоро_др")} &nbsp;&nbsp;&nbsp;<br>
    Скоро годовщина приёма: {safe_bool("скоро_годовщина_приема")}<br><br>

    <b>💰 Финансы</b><br>
    ЗП к средней по компании: {safe_get("зп_на_ср_зп_по_компании", "{:.2f}")}<br><br>
    """,
        unsafe_allow_html=True,
    )


def plot_top_departments_by_risk(df_all: pd.DataFrame, latest_date: str, info_file: Path):
    st.subheader("📊 Топ-10 отделов по среднему риску увольнения")

    if not info_file.exists():
        st.warning("Файл с информацией о сотрудниках не найден.")
        return

    try:
        # Загрузка info_file
        df_info = pd.read_csv(info_file)
        df_info.columns = [str(col).strip().lower() for col in df_info.columns]

        if "id" not in df_info.columns:
            st.error("В info_file нет колонки 'id'")
            return
        if "отдел" not in df_info.columns:
            st.error("В info_file нет колонки 'отдел'")
            return

        # Приведение id к строкам → float (сначала для корректности merge)
        df_info["id"] = df_info["id"].astype(str).str.strip().str.lower()
        df_all["id"] = df_all["id"].astype(str).str.strip().str.lower()

        if latest_date not in df_all.columns:
            st.error(f"Колонка '{latest_date}' не найдена в df_all.")
            return

        df_all["id"] = df_all["id"].astype(float)
        df_info["id"] = df_info["id"].astype(float)

        # Слияние
        merged_df = df_all.merge(df_info[["id", "отдел"]], how="left", on="id")

        # Удалим строки без отдела или без latest_date
        merged_df = merged_df.dropna(subset=["отдел", latest_date])

        if merged_df.empty:
            st.warning("Нет данных с ненулевыми отделами и рисками.")
            return

        # Агрегация
        agg_df = (
            merged_df.groupby("отдел")[latest_date]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        agg_df.columns = ["Отдел", "Средний риск"]

        if agg_df.empty:
            st.info("Нет данных для отображения.")
            return

        # Визуализация через matplotlib
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(agg_df["Отдел"], agg_df["Средний риск"], color="skyblue")
        ax.invert_yaxis()
        ax.set_xlabel("Средний риск увольнения")
        ax.set_title("Топ-10 отделов по среднему риску увольнения")
        plt.tight_layout()

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Ошибка при построении графика: {e}")


def show_critical_department_profiles(
    df_all: pd.DataFrame, info_file: Path, latest_date: str, path_all: Path
):
    st.subheader("🏢 Профили критических отделов")

    if not info_file.exists():
        st.warning("Файл с информацией о сотрудниках не найден.")
        return

    try:
        df_info = pd.read_csv(info_file)
        df_info.columns = [str(c).strip().lower() for c in df_info.columns]
        df_info["id"] = df_info["id"].astype(str).str.strip().str.lower()
        df_info["фио"] = df_info["фио"].astype(str).str.strip().str.lower()

        df_all["id"] = df_all["id"].astype(str).str.strip().str.lower()
        df_all["фио"] = df_all["фио"].astype(str).str.strip().str.lower()

        top_departments_df = (
            df_all[["id", "фио", latest_date]]
            .merge(df_info[["id", "отдел", "должность"]], how="left", on="id")
            .dropna(subset=["отдел", latest_date])
        )

        avg_risk_by_dept = (
            top_departments_df.groupby("отдел")[latest_date]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
            .rename(columns={latest_date: "средний_риск"})
        )

        selected_dept = st.selectbox(
            "Выберите отдел из топ-10 по риску:",
            avg_risk_by_dept["отдел"],
            index=0,
        )

        dept_df = top_departments_df[top_departments_df["отдел"] == selected_dept].copy()
        dept_df = dept_df.dropna(subset=[latest_date])

        st.markdown(f"#### 📊 Распределение рисков в отделе: *{selected_dept}*")

        fig_dept = px.histogram(
            dept_df,
            x=latest_date,
            nbins=20,
            title=f"Распределение рисков увольнения – {selected_dept}",
            labels={latest_date: "Риск увольнения"},
        )
        fig_dept.update_layout(xaxis_title="Риск", yaxis_title="Число сотрудников")
        st.plotly_chart(fig_dept, use_container_width=True)

        st.markdown("#### 👥 Сотрудники в отделе")

        try:
            full_df = pd.read_excel(path_all)
            full_df.columns = [c.strip().lower() for c in full_df.columns]
            full_df["фио"] = full_df["фио"].astype(str).str.strip().str.lower()

            dept_df["фио"] = dept_df["фио"].astype(str).str.strip().str.lower()
            dept_df = dept_df.merge(
                full_df[["фио", "уволен", "предсказание_увольнения"]],
                how="left",
                on="фио",
            )
        except Exception as e:
            st.warning(f"⚠ Не удалось загрузить факт/предсказание: {e}")

        # Удаляем дубли по id
        dept_df = dept_df.drop_duplicates(subset="id")

        # Убираем уволенных сотрудников
        dept_df = dept_df[dept_df["уволен"] != 1]

        # Оставляем только нужные столбцы
        dept_df_display = (
            dept_df[["фио", "должность", latest_date]]
            .rename(
                columns={
                    "фио": "ФИО",
                    "должность": "Должность",
                    latest_date: "Риск увольнения",
                }
            )
            .sort_values("Риск увольнения", ascending=False)
            .reset_index(drop=True)
        )

        st.dataframe(dept_df_display, use_container_width=True)

    except Exception as e:
        st.error(f"Ошибка при обработке профиля критических отделов: {e}")


def run_dashboard(excel_file: str, title: str):
    if not Path(excel_file).exists():
        st.error(f"Файл {excel_file} не найден.")
        return

    df = pd.read_excel(excel_file)
    df.columns = [str(col).strip().lower() for col in df.columns]

    # Определим столбцы-даты и приведем к float
    # Преобразуем все названия столбцов-дат в формат дд.мм.гггг

    date_cols = []
    new_columns = []

    for col in df.columns:
        try:
            parsed = datetime.strptime(col.strip(), "%d.%m.%Y")
            date_cols.append(col.strip())
            new_columns.append(col.strip())
        except ValueError:
            new_columns.append(col)

    df.columns = new_columns

    for col in date_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".")
            .str.replace(" ", "")
            .str.replace(r"[^\d.]", "", regex=True)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    sorted_date_cols = sorted(date_cols, key=lambda d: pd.to_datetime(d, dayfirst=True))
    latest_date = sorted_date_cols[-1]

    st.subheader(f"Фильтры ({title})")
    st.info(
        """
            В таблице собраны все сотрудники, в том числе и уволенные:\n
            Факт: 1 - включает уволенных, 0 - включает не уволенных, если снять выбор, то не показывает.\n
            Предсказание: 1 - модель предсказала, что сотрудник уволится, 0 - не уволится.\n
            """
    )
    col_fact, col_pred = st.columns(2)

    with col_fact:
        st.markdown("**Факт:**")
        show_0_dismissed = st.checkbox("Не уволен (0)", value=True, key=f"dismissed_0_{title}")
        show_1_dismissed = st.checkbox("Уволен (1)", value=True, key=f"dismissed_1_{title}")

    with col_pred:
        st.markdown("**Предсказание:**")
        show_0_pred = st.checkbox("Нет увольнения (0)", value=True, key=f"pred_0_{title}")
        show_1_pred = st.checkbox("Есть увольнение (1)", value=True, key=f"pred_1_{title}")

    dismissed_values = []
    if show_0_dismissed:
        dismissed_values.append(0)
    if show_1_dismissed:
        dismissed_values.append(1)

    pred_values = []
    if show_0_pred:
        pred_values.append(0)
    if show_1_pred:
        pred_values.append(1)

    filtered_df = df[
        df["уволен"].isin(dismissed_values) & df["предсказание_увольнения"].isin(pred_values)
    ]

    if info_file.exists():
        df_info = pd.read_csv(info_file)

        # Приведение ФИО к нижнему регистру и очистка пробелов
        df_info["фио"] = df_info["фио"].astype(str).str.strip().str.lower()
        filtered_df.loc[:, "фио"] = filtered_df["фио"].astype(str).str.strip().str.lower()

        # Мерж по ФИО
        filtered_df = filtered_df.merge(df_info[["фио", "должность"]], how="left", on="фио")

        # Удаление должностей: уборщица, офицер
        filtered_df = filtered_df[
            ~filtered_df["должность"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["должность_1", "должность_2"])
        ]
    else:
        st.warning("Файл с должностями не найден. Фильтрация по должностям пропущена.")

    if filtered_df.empty:
        st.info("Нет сотрудников, соответствующих фильтрам.")
        return

    st.subheader(f"Сотрудники с риском увольнения на {latest_date}")
    st.info(
        """
            Выбор строки развернет профиль сотрудника ниже.
        """
    )
    search_name_tab = st.text_input("🔍 Поиск сотрудника по ФИО", key=f"search_{title}")
    if search_name_tab:
        filtered_df = filtered_df[filtered_df["фио"].str.contains(search_name_tab)]

    top_risk_df = (
        filtered_df.sort_values(by=latest_date, ascending=False)[["фио", latest_date]]
        .drop_duplicates(subset="фио", keep="first")  # <-- удаляет дубли
        .rename(columns={latest_date: "риск_увольнения"})
        .reset_index(drop=True)
    )

    gb = GridOptionsBuilder.from_dataframe(top_risk_df)
    gb.configure_selection("single", use_checkbox=False)
    gb.configure_grid_options(autoSizeColumns=True, domLayout="normal")  # для скролла
    grid_options = gb.build()

    grid_response = AgGrid(
        top_risk_df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        height=500,
        allow_unsafe_jscode=True,
        theme="streamlit",
        fit_columns_on_grid_load=True,
    )

    selected_rows = grid_response.get("selected_rows", [])
    if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
        selected_fio = selected_rows.iloc[0]["фио"]
    else:
        selected_fio = None

    if selected_fio:
        emp = df[df["фио"] == selected_fio].iloc[0]
        st.subheader(f"Профиль: {selected_fio.title()}")
        st.metric("Итоговая вероятность увольнения", f"{emp[latest_date]*100:.1f}%")

        st.write("Динамика вероятности:")
        prob_series = pd.Series(emp[sorted_date_cols].values, index=sorted_date_cols)
        prob_series.index = pd.to_datetime(prob_series.index, dayfirst=True, errors="coerce")
        prob_series = prob_series.sort_index()
        st.line_chart(prob_series)

        if info_file.exists():
            info_df = pd.read_csv(info_file)

            info_df["фио"] = info_df["фио"].str.strip().str.lower()
            fio_lower = selected_fio.strip().lower()

            row_info = info_df[info_df["фио"] == fio_lower]

            if not row_info.empty:
                # Копируем значения строки
                info_display = row_info[COLUMNS_TO_SHOW].iloc[0].copy()

                # Обрабатываем id_руководителя
                manager_id = info_display["id_руководителя"]

                if pd.isna(manager_id):
                    manager_label = "nan"
                elif manager_id == -1:
                    manager_label = "-1 (nan)"
                else:
                    # Ищем фио руководителя по его id
                    manager_fio = info_df.loc[info_df["id"] == manager_id, "фио"].values  # type: ignore
                    if len(manager_fio) > 0:
                        manager_label = f"{manager_id} ({manager_fio[0].title()})"
                    else:
                        manager_label = str(manager_id)

                # Заменяем id_руководителя на формат с ФИО
                info_display["id_руководителя"] = manager_label

                # Финальный вывод
                user_info_dict = info_display.to_dict()
                render_user_card(user_info_dict)

            else:
                st.info("Информация о сотруднике в файле не найдена.")
        else:
            st.warning("Файл с информацией о сотрудниках не найден.")

        shap_file = (
            Path("/home/root6/python/dismissal_predict_v2/data/results/result_top_shap.csv")
            if title == "top"
            else Path("/home/root6/python/dismissal_predict_v2/data/results/result_all_shap.csv")
        )

        if shap_file.exists():
            shap_df = pd.read_csv(shap_file)
            row = shap_df[shap_df["фио"] == selected_fio]
            if not row.empty:
                shap_row = row.iloc[0].drop("фио")
                top_factors = shap_row.abs().sort_values(ascending=False).head(5)
                top_features = top_factors.index
                top_values = shap_row[top_features]

                st.subheader("Ключевые факторы риска (по SHAP):")

                col_left, col_right = st.columns([1.5, 3])

                with col_left:
                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    bars = ax.barh(
                        top_features[::-1],
                        top_values[::-1],
                        color=["red" if val > 0 else "blue" for val in top_values[::-1]],
                    )
                    ax.axvline(0, color="black", linewidth=0.8)
                    ax.set_xlabel("SHAP значение (влияние на риск)", fontsize=10)
                    ax.set_ylabel("Признак", fontsize=10)
                    ax.set_title("ТОП факторов по влиянию", fontsize=11)
                    ax.tick_params(axis="both", labelsize=9)
                    fig.tight_layout()

                    st.pyplot(fig, clear_figure=True)
            else:
                st.info(f"SHAP-факторы не найдены для сотрудника: {selected_fio}")
        else:
            st.info("Файл с SHAP-факторами не найден.")


def run_dashboard_summary(path_all, shap_path_all):
    # Загрузка данных
    df_all = pd.read_excel(path_all)

    # Подготовка столбцов и нормализация
    df_all.columns = [str(c).strip().lower() for c in df_all.columns]
    df_all["фио"] = df_all["фио"].astype(str).str.strip().str.lower()

    # Определение последней даты
    date_cols = [col for col in df_all.columns if col.count(".") == 2 and col[:2].isdigit()]
    sorted_date_cols = sorted(date_cols, key=lambda d: pd.to_datetime(d, dayfirst=True))
    latest_date = sorted_date_cols[-1]

    # Гистограмма
    fig = px.histogram(df_all, x=latest_date, nbins=50, title="Распределение рисков увольнения")
    fig.update_layout(xaxis_title="Риск увольнения", yaxis_title="Количество сотрудников")
    st.plotly_chart(fig, use_container_width=True)

    # SHAP-факторы
    st.subheader("Анализ факторов риска (SHAP)")
    if shap_path_all.exists():
        shap_all = pd.read_csv(shap_path_all)

        shap_all.drop(columns=["фио"], inplace=True, errors="ignore")

        # Вычисление средних значений
        mean_positive = (
            shap_all[shap_all > 0].mean().dropna().sort_values(ascending=False).head(10)
        )
        mean_negative = shap_all[shap_all < 0].mean().dropna().sort_values().head(10)

        # Формирование графиков
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**ТОП-10 факторов на увольнение (SHAP > 0)**")
            fig1 = px.bar(
                mean_positive.reset_index(),
                x="index",
                y=0,
                labels={"index": "Фактор", "0": "SHAP значение"},
            )
            fig1.update_layout(xaxis={"categoryorder": "total descending"})
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown("**ТОП-10 факторов на неувольнение (SHAP < 0)**")
            fig2 = px.bar(
                mean_negative.reset_index(),
                x="index",
                y=0,
                labels={"index": "Фактор", "0": "SHAP значение"},
            )
            fig2.update_layout(xaxis={"categoryorder": "total descending"})
            st.plotly_chart(fig2, use_container_width=True)

        with col3:
            st.markdown("**Ключевые факторы риска (по SHAP)**")
            top_features = (
                shap_all.abs().mean().sort_values(ascending=False).head(10).index.tolist()
            )
            shap_sample = shap_all[top_features]

            fig_summary, ax = plt.subplots(figsize=(8, 6))
            shap.summary_plot(
                shap_sample.values,
                features=shap_sample,
                feature_names=top_features,
                plot_type="dot",
                show=False,
            )
            st.pyplot(fig_summary)
    else:
        st.warning("Файл SHAP-факторов не найден.")

    plot_top_departments_by_risk(df_all, latest_date, info_file)
    show_critical_department_profiles(df_all, info_file, latest_date, path_all)


if "user_info_json" not in st.session_state:
    st.session_state.user_info_json = ""

# 🔹 Загрузка конфигурации
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

# 🔹 Найдём нужный проект
project_config = None
for proj in config["projects"]:
    if PROJECT_ID in proj:
        project_config = proj[PROJECT_ID]
        break

if project_config is None:
    st.error(f"Проект с ID '{PROJECT_ID}' не найден.")
    st.stop()

allowed_users = project_config.get("allowed_users", [])
tabs_by_user = project_config.get("tabs_by_user", {})

handle_auth()

# 🔹 Доступ к дашборду (успешная авторизация)
if st.session_state.login_stage == "authenticated":
    st_autorefresh(interval=3600000, limit=None, key="auto_refresh")
    st.title("📊 Дашборд риска увольнения сотрудников")

    if info_file.exists():
        if st.button("🔄 Обновить данные"):
            st.rerun()
    else:
        st.error("❌ Файл с данными не найден")

    # 🔹 Получение доступных вкладок
    login = st.session_state.login
    available_tabs = tabs_by_user.get(login, tabs_by_user.get("default", []))

    if not available_tabs:
        st.warning("У пользователя нет доступных вкладок.")
        st.stop()

    tabs = st.tabs(available_tabs)

    for tab_name, tab in zip(available_tabs, tabs):
        with tab:
            if tab_name == "Все сотрудники":
                run_dashboard(
                    "/home/root6/python/dismissal_predict_v2/data/results/result_all.xlsx",
                    title="all",
                )
            elif tab_name == "Другие сотрудники":
                run_dashboard(
                    "/home/root6/python/dismissal_predict_v2/data/results/result_top.xlsx",
                    title="top",
                )
            elif tab_name == "По всей компании":
                run_dashboard_summary(
                    Path("/home/root6/python/dismissal_predict_v2/data/results/result_all.xlsx"),
                    Path(
                        "/home/root6/python/dismissal_predict_v2/data/results/result_all_shap.csv"
                    ),
                )
