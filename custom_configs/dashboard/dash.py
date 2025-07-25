from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import shap
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st_autorefresh(interval=3600000, limit=None, key="auto_refresh")
st.set_page_config(layout="wide", page_title="Риски увольнений")
st.title("Дашборд рисков увольнений сотрудников")

info_file = Path("~/dismissal_predict_v2/data/processed/main_all.csv")


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

        # Удаление должностей
        filtered_df = filtered_df[
            ~filtered_df["должность"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["должность 1", "должность 2"])
        ]
    else:
        st.warning(
            "Файл с должностями main_all.csv не найден. Фильтрация по должностям пропущена."
        )

    if filtered_df.empty:
        st.info("Нет сотрудников, соответствующих фильтрам.")
        return

    st.subheader(f"Сотрудники с риском увольнения на {latest_date}")
    st.info(
        """
            Выбор строки развернет профиль сотрудника ниже.
        """
    )
    top_risk_df = filtered_df.sort_values(by=latest_date, ascending=False)[
        ["фио", latest_date]
    ].rename(columns={latest_date: "риск_увольнения"})

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
                st.subheader("Информация о сотруднике:")
                columns_to_show = [
                    "дата_увольнения",
                    "id_руководителя",
                    "подчиненные",
                    "должность",
                    "дата_рождения",
                    "дата_приема_в_1с",
                    "пол",
                    "текущая_должность_на_портале",
                    "грейд",
                    "категория",
                    "бе",
                    "отдел",
                    "child_num",
                    "avg_child_age",
                    "main_child_gender",
                    "уволен",
                    "возраст",
                    "стаж",
                ]

                # Копируем значения строки
                info_display = row_info[columns_to_show].iloc[0].copy()

                # Обрабатываем id_руководителя
                manager_id = info_display["id_руководителя"]

                if pd.isna(manager_id):
                    manager_label = "nan"
                elif manager_id == -1:
                    manager_label = "-1 (nan)"
                else:
                    # Ищем фио руководителя по его id
                    manager_fio = info_df.loc[info_df["id"] == manager_id, "фио"].values
                    if len(manager_fio) > 0:
                        manager_label = f"{manager_id} ({manager_fio[0].title()})"
                    else:
                        manager_label = str(manager_id)

                # Заменяем id_руководителя на формат с ФИО
                info_display["id_руководителя"] = manager_label

                # Финальный вывод
                info_display = info_display.to_frame().reset_index()
                info_display.columns = ["Показатель", "Значение"]
                info_display["Значение"] = info_display["Значение"].astype(str)
                st.table(info_display)

            else:
                st.info("Информация о сотруднике не найдена в main_all.csv.")
        else:
            st.warning("Файл с информацией о сотрудниках не найден.")

        shap_file = (
            Path("~/dismissal_predict_v2/data/results/result_top_shap.csv")
            if title == "top"
            else Path("~/dismissal_predict_v2/data/results/result_all_shap.csv")
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


def run_dashboard_summary(path_all, path_top, shap_path_all, shap_path_top):
    # Загрузка данных
    df_all = pd.read_excel(path_all)
    df_top = pd.read_excel(path_top)

    # Подготовка столбцов и нормализация
    for df in [df_all, df_top]:
        df.columns = [str(c).strip().lower() for c in df.columns]
        df["фио"] = df["фио"].astype(str).str.strip().str.lower()

    # Определение последней даты
    date_cols = [col for col in df_all.columns if col.count(".") == 2 and col[:2].isdigit()]
    sorted_date_cols = sorted(date_cols, key=lambda d: pd.to_datetime(d, dayfirst=True))
    latest_date = sorted_date_cols[-1]

    # Объединение с усреднением
    df_combined = pd.concat([df_all[["фио", latest_date]], df_top[["фио", latest_date]]])
    df_combined = df_combined.groupby("фио", as_index=False).mean()

    # Гистограмма
    fig = px.histogram(
        df_combined, x=latest_date, nbins=50, title="Распределение рисков увольнения"
    )
    fig.update_layout(xaxis_title="Риск увольнения", yaxis_title="Количество сотрудников")
    st.plotly_chart(fig, use_container_width=True)

    # SHAP-факторы
    st.subheader("Анализ факторов риска (SHAP)")
    if shap_path_all.exists() and shap_path_top.exists():
        shap_all = pd.read_csv(shap_path_all)
        shap_top = pd.read_csv(shap_path_top)

        shap_all.drop(columns=["фио"], inplace=True, errors="ignore")
        shap_top.drop(columns=["фио"], inplace=True, errors="ignore")

        shap_combined = pd.concat([shap_all, shap_top])

        # Вычисление средних значений
        mean_positive = (
            shap_combined[shap_combined > 0].mean().dropna().sort_values(ascending=False).head(10)
        )
        mean_negative = shap_combined[shap_combined < 0].mean().dropna().sort_values().head(10)
        mean_abs = shap_combined.abs().mean().sort_values(ascending=False).head(10)

        # Формирование графиков
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**ТОП-10 факторов на увольнение (SHAP > 0)**")
            fig1 = px.bar(
                mean_positive.reset_index(),
                x="index",
                y=0,
                labels={"index": "Фактор", "0": "SHAP значение"},
                title=None,
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
                title=None,
            )
            fig2.update_layout(xaxis={"categoryorder": "total descending"})
            st.plotly_chart(fig2, use_container_width=True)

        with col3:
            st.markdown("**Ключевые факторы риска (по SHAP)**")

            # Отбираем топ-10 признаков по убыванию важности
            top_features = (
                shap_combined.abs().mean().sort_values(ascending=False).head(10).index.tolist()
            )
            shap_sample = shap_combined[top_features]

            # Строим SHAP-график (dot plot), отсортированный по убыванию важности
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
        st.warning("Файлы SHAP-факторов не найдены.")


# Вкладки
tab1, tab2, tab3 = st.tabs(["Все сотрудники", "Другие сотрудники", "По всей компании"])

with tab1:
    run_dashboard(
        "~/dismissal_predict_v2/data/results/result_all.xlsx",
        title="all",
    )

with tab2:
    run_dashboard(
        "~/dismissal_predict_v2/data/results/result_top.xlsx",
        title="top",
    )

with tab3:
    run_dashboard_summary(
        Path("~/dismissal_predict_v2/data/results/result_all.xlsx"),
        Path("~/dismissal_predict_v2/data/results/result_top.xlsx"),
        Path("~/dismissal_predict_v2/data/results/result_all_shap.csv"),
        Path("~/dismissal_predict_v2/data/results/result_top_shap.csv"),
    )
