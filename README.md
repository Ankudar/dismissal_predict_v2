<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>dismissal_predict_v2 — Документация</title>
  <style>
    body {
      font-family: system-ui, sans-serif;
      max-width: 900px;
      margin: 40px auto;
      padding: 0 20px;
      line-height: 1.6;
    }
    h1, h2, h3 {
      color: #ffffffff;
    }
    pre {
      background: #4b4646ff;
      padding: 10px;
      border-radius: 6px;
      overflow-x: auto;
    }
    code {
      background-color: #000000ff;
      padding: 2px 4px;
      font-size: 90%;
      border-radius: 4px;
    }
  </style>
</head>
<body>

<h1>📊 dismissal_predict_v2 — Предсказание увольнений сотрудников</h1>

<p><strong>Цель проекта:</strong> разработка системы машинного обучения, которая предсказывает вероятность увольнения сотрудников на основе исторических HR-данных и выгрузок из кадровых систем (например, 1С:ZUP).</p>

<p>Проект автоматизирует процесс оценки риска увольнений, помогает HR и руководству выявлять уязвимые зоны и снижать текучесть персонала.</p>

<h2>🧩 Стек технологий</h2>
<ul>
  <li><strong>Язык:</strong> Python 3.10+</li>
  <li><strong>ML:</strong> pandas, scikit-learn, xgboost, optuna</li>
  <li><strong>Мониторинг:</strong> MLflow</li>
  <li><strong>Оркестрация:</strong> Airflow</li>
</ul>

<h2>⚙️ Этапы работы</h2>
<ol>
  <li><strong>Сбор данных:</strong> загрузка из ZUP, сохранение в <code>data/raw/zup</code></li>
  <li><strong>Предобработка:</strong> очистка, извлечение признаков, определение пола</li>
  <li><strong>Финальный датасет:</strong> кодирование, выбор признаков, формирование <code>target</code></li>
  <li><strong>Обучение:</strong> XGBoost + Optuna</li>
  <li><strong>Оценка:</strong> метрики <code>accuracy</code>, <code>ROC-AUC</code>, <code>F1</code>, <code>SHAP</code></li>
  <li><strong>Предсказание:</strong> обновление кадрового списка, генерация результатов</li>
  <li><strong>Отчёты:</strong> визуализация результатов и формирование отчётов</li>
  <li><strong>Автоматизация:</strong> DAG в Airflow для запуска пайплайнов</li>
</ol>

<h2>📁 Структура проекта</h2>
<pre>
📦 Проект: dismissal_predict_v2
├── LICENSE               <- Открытая лицензия проекта The MIT License (MIT).
├── Makefile              <- Утилиты командной строки, например: make data, make train.
├── README.md             <- Главный файл описания проекта для разработчиков и пользователей.
├── pyproject.toml        <- Конфигурация проекта и зависимостей (для black, isort, mypy и др.).
├── requirements.txt      <- Зависимости Python (можно собрать через pip freeze > requirements.txt).
├── setup.cfg             <- Конфигурация для flake8 и других инструментов линтинга.
├── __init__.py           <- Делает корень проекта исполняемым как Python-модуль.

📦airflow                 <- Оркестрация пайплайнов через Apache Airflow.
 └── 📂dags
     └── 📜dismissal_predict.py  <- DAG, реализующий пайплайн подготовки и предсказания увольнений.

📁 data                   <- Хранилище данных различных стадий обработки.
├── 📁 raw                <- Оригинальные данные, неизменяемые выгрузки.
│   └── 📁 zup            <- Подпапка с выгрузкой из ZUP (1С или других систем).
├── 📁 interim            <- Промежуточные данные (например, после очистки или первичной агрегации).
├── 📁 processed          <- Финальные подготовленные данные, готовые для моделирования.
└── 📁 results            <- Результаты прогонов, предсказания моделей, метрики.

📁 dismissal_predict      <- Основной код проекта.
├── __init__.py           <- Делает папку Python-модулем.
├── config.py             <- Глобальные настройки и конфигурации.
├── dataset.py            <- Загрузка и базовая подготовка данных.
├── prepare_dataset.py    <- Предобработка сырых данных.
├── final_prepare_df.py   <- Финальный шаг подготовки перед обучением модели.
└── 📁 modeling           <- Модуль с обучением и инференсом моделей.
    ├── __init__.py
    ├── train.py          <- Код для обучения моделей.
    └── predict.py        <- Код для генерации предсказаний.

📁 mlruns                 <- Каталог для логов MLflow (эксперименты, метрики, параметры, артефакты).
└── 📁 models             <- Сохранённые модели и артефакты через MLflow.

📁 models                 <- Дополнительные сериализованные модели, сохранённые вручную (.pkl, .joblib).

📁 notebooks              <- Jupyter-ноутбуки для исследования данных, тестов и прототипирования.

📁 reports                <- Автоматически сгенерированные отчёты, отчётные документы.
└── 📁 figures            <- Графики, визуализации и изображения, используемые в отчётах.
</pre>

<h2>🚀 Как начать</h2>

<h3>1️⃣ Подготовка окружения</h3>
<p>Рекомендуется использовать <code>virtualenv</code> или <code>conda</code> для изоляции окружения:</p>
<pre><code>
# через venv
python -m venv venv
source venv/bin/activate    # для Linux/macOS
venv\Scripts\activate.bat   # для Windows (рекомендуется использовать wsl)

# или через conda
conda create -n dismissal_predict python=3.10 -y
conda activate dismissal_predict
</code></pre>

<h3>2️⃣ Установка зависимостей</h3>
<p>Установите библиотеки из <code>requirements.txt</code>:</p>
<pre><code>pip install -r requirements.txt</code></pre>

<h3>3️⃣ Клонирование репозитория</h3>
<pre><code>git clone https://github.com/your-user/dismissal_predict_v2.git
cd dismissal_predict_v2
</code></pre>

<h3>4️⃣ Настройка MLflow</h3>
<p><strong>MLflow</strong> используется для отслеживания экспериментов, метрик и моделей.</p>
<p>📘 <a href="https://mlflow.org/docs/latest/index.html" target="_blank">Официальная документация MLflow</a></p>
<pre><code># Запуск локального интерфейса MLflow
mlflow ui --backend-store-uri ./mlruns
</code></pre>
<p>После запуска откройте в браузере: <a href="http://localhost:5000" target="_blank">http://localhost:5000</a></p>

<h3>5️⃣ Настройка Airflow</h3>
<p><strong>Apache Airflow</strong> используется для оркестрации пайплайнов обработки данных и предсказаний.</p>
<p>📘 <a href="https://airflow.apache.org/docs/apache-airflow/stable/index.html" target="_blank">Официальная документация Airflow</a></p>

<p>Перейдите в директорию проекта:</p>
<pre><code>cd /home/root6/airflow</code></pre>

<p>Создайте виртуальное окружение:</p>
<pre><code>python3 -m venv airflow_env</code></pre>

<p>Активируйте его:</p>
<pre><code>source airflow_env/bin/activate</code></pre>

<p>Обновите pip:</p>
<pre><code>pip install --upgrade pip</code></pre>

<p>Установите Airflow с необходимыми зависимостями:</p>
<pre><code>
AIRFLOW_VERSION=3.0.2
PYTHON_VERSION="$(python --version 2>&1 | cut -d ' ' -f2 | cut -d '.' -f1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip install "apache-airflow[async,postgres,google]==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
</code></pre>

<p>Инициализируйте базу данных:</p>
<pre><code>airflow db migrate</code></pre>

<p>Дополнительно установите необходимые пакеты:</p>
<pre><code>
pip install flask_appbuilder
pip install graphviz
</code></pre>

<p><strong>Запуск компонентов Airflow</strong> (в отдельных терминалах/окнах):</p>
<pre><code>airflow api-server --port 18080</code></pre>
<pre><code>airflow scheduler</code></pre>
<pre><code>airflow dag-processor</code></pre>
<pre><code>airflow triggerer</code></pre>

<p>DAG-файл проекта находится здесь:</p>
<pre><code>airflow/dags/dismissal_predict.py</code></pre>

</body>
</html>