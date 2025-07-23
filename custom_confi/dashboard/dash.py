from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import streamlit as st

st.set_page_config(layout="wide", page_title="Риски увольнений")
st.title("Дашборд рисков увольнений сотрудников")


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

        info_file = Path("~/dismissal_predict_v2/data/processed/main_all.csv")

        if info_file.exists():
            info_df = pd.read_csv(info_file)

            info_df["фио"] = info_df["фио"].str.strip().str.lower()
            fio_lower = selected_fio.strip().lower()

            row_info = info_df[info_df["фио"] == fio_lower]

            if not row_info.empty:
                st.subheader("Информация о сотруднике:")
                columns_to_show = [
                    "дата_увольнения",
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

                info_display = row_info[columns_to_show].iloc[0].to_frame().reset_index()
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


# Вкладки
tab1, tab2 = st.tabs(["Все сотрудники", "Другие сотрудники"])

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
