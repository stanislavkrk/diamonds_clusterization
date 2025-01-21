import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_clean_data(file_path):
    '''
    strategy='most_frequent' -
    пропущені значення в кожному стовпці будуть замінені найчастішим значенням (модою) цього стовпця.

    :param file_path: шлях до файлу
    :return: очищені дані у форматі Pandas DF
    '''
    data = pd.read_csv(file_path)
    imputer = SimpleImputer(strategy='most_frequent')
    data.iloc[:, :] = imputer.fit_transform(data)
    return data


def preprocess_data(data, numeric_columns, categorical_columns):
    '''
    Створюємо об'єкт ColumnTransformer, який дозволяє застосовувати різні перетворення
    до різних груп стовпців у наборі даних. Він  корисний, коли в датасеті є як числові,
    так і категоріальні дані, які потребують різної обробки.
    Використовуємо StandardScaler, який нормалізує числові дані до стандартного нормального розподілу
    (середнє = 0, стандартне відхилення = 1).
    Використовуємо OneHotEncoder, який перетворює категорії в числову форму за допомогою створення
    двійкових (one-hot) стовпців.

    :param data: вхідний датасет
    :param numeric_columns: Списки стовпців, які належать до числових змінних
    :param categorical_columns: Списки стовпців, які належать до категоріальний змінних
    :return: Отримуємо перетворений масив числових значень, який об'єднує результат обробки числових
    і категоріальних даних. Це формат даних, готовий для використання в алгоритмах машинного навчання.
    Також повертаємо сам процесор для використання із заданими параметрами.
    '''
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])
    return preprocessor.fit_transform(data), preprocessor


def perform_kmeans(data, n_clusters):
    '''
    Ця функція виконує кластеризацію даних за допомогою алгоритму K-Means
    і додає новий стовпець із номером кластера для кожного об'єкта.
    Задаємо кількість кластерів, які хочемо виділити.
    Алгоритм кластеризації використовує випадкові числа для вибору початкових центрів кластерів.
    Встановлення random_state гарантує, що отримаємо однаковий результат при кожному запуску.
    fit_predict(data) - Алгоритм знаходить центри кластерів у даних (навчається на них),
    присвоює кожному об'єкту номер кластера (кластер із найближчим центром).

    :param data: набір даних для кластеризації
    :param n_clusters: кількість кластерів
    :return: Повертається оновлений набір даних, у якому з'явився новий стовпець Cluster.
    '''
    # Якщо data є розрідженою матрицею, перетворити її на щільний формат
    if hasattr(data, "toarray"):
        data = data.toarray()

    # Перетворення в DataFrame
    data = pd.DataFrame(data)

    # Виконання кластеризації
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data)
    return data


def detect_anomalies(data, contamination=0.05):
    '''
    Це алгоритм, який виявляє аномалії шляхом ізоляції точок даних. Він базується на тому,
    що аномалії ізолюються (відділяються від інших точок) швидше, ніж нормальні дані.
    Модель навчається на даних, будуючи дерева ізоляції. Кожна точка отримує мітку:
     1 — нормальні дані;
    -1 — аномалії.
    data['Anomaly'] - у новий стовпець додаються ці мітки для кожного об'єкта.
    map замінює числові мітки:
     1 → 'Normal';
    -1 → 'Anomaly'.
    Це покращує зручність аналізу, надаючи текстові позначення.

    :param data: Вхідні дані.
    :param contamination: Вказує частку даних, яку алгоритм вважає аномальними.
    :return: Повертається оновлений датасет із новим стовпцем 'Anomaly'.
    '''

    isolation_forest = IsolationForest(random_state=42, contamination=contamination)
    data.columns = data.columns.astype(str)

    data['Anomaly'] = isolation_forest.fit_predict(data)
    data['Anomaly'] = data['Anomaly'].map({1: 'Normal', -1: 'Anomaly'})
    return data


def summarize_clusters(data):
    '''
    Ця функція створює резюме кластерів, обчислюючи середні значення для всіх числових стовпців у кожному кластері.
    Групує рядки датасету на основі значень у стовпці Cluster. У кожній групі знаходяться всі об'єкти,
    які належать до одного кластера. Обчислює середнє значення для кожного числового стовпця в межах кожного кластера,
    ігнорує нечислові стовпці (категоріальні або текстові дані).
    Сенс: дозволяє зрозуміти середні властивості кожного кластера (наприклад, середня ціна, середня вага);
    легко побачити, як змінюються характеристики від одного кластеру до іншого.

    :param data: вхідний датасет
    :return: Повертається новий DataFrame, де:
    Індекси — номери кластерів;
    Колонки — середні значення числових характеристик для кожного кластеру.
    '''
    return data.groupby('Cluster').mean(numeric_only=True)


def get_anomalies(data):
    '''
    Ця функція виконує фільтрацію даних, щоб виділити всі об'єкти, які були визначені як аномалії.
    :param data: вхідний датасет
    :return: Повертається новий DataFrame, який містить тільки аномалії.
    '''
    return data[data['Anomaly'] == 'Anomaly']


def inverse_transform_clusters(processed_data, preprocessor, original_data):
    """
    Відновлює дані кластерів в оригінальному масштабі та з початковими назвами колонок.
    """
    # Відновлення числових і категоріальних даних
    numeric_data = preprocessor.named_transformers_['num'].inverse_transform(processed_data[:, :len(numeric_columns)])
    categorical_data = preprocessor.named_transformers_['cat'].inverse_transform(processed_data[:, len(numeric_columns):])

    # Створення DataFrame із оригінальними колонками
    numeric_df = pd.DataFrame(numeric_data, columns=numeric_columns)
    categorical_df = pd.DataFrame(categorical_data, columns=categorical_columns)

    # Об'єднання числових та категоріальних даних
    restored_data = pd.concat([numeric_df, categorical_df], axis=1)

    return restored_data


def plot_clusters(data, x_col, y_col, cluster_col, title, x_label, y_label):
    """
    Побудова графіка розподілу кластерів за двома характеристиками: ціна та вага.

    Parameters:
    - data: DataFrame з даними.
    - x_col: Назва стовпця для осі X.
    - y_col: Назва стовпця для осі Y.
    - cluster_col: Назва стовпця кластерів.
    - title: Заголовок графіка.
    - x_label: Підпис осі X.
    - y_label: Підпис осі Y.
    """

    palette = sns.color_palette("husl", len(data[cluster_col].unique()))
    plt.figure(figsize=(10, 6))

    # Побудова розподілу для кожного кластеру
    for i, cluster in enumerate(data[cluster_col].unique()):
        cluster_data = data[data[cluster_col] == cluster]
        plt.scatter(cluster_data[x_col], cluster_data[y_col],
                    label=f'Cluster {cluster}', alpha=0.6, color=palette[i])

    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend(title='Кластери', fontsize=10)
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":

    file_path = 'diamonds (cleaned).csv'

    # Стовпці
    numeric_columns = ["Carat Weight", "Length/Width Ratio", "Depth %", "Table %", "Length", "Width", "Height", "Price"]
    categorical_columns = ["Shape", "Cut", "Color", "Clarity", "Polish", "Symmetry", "Girdle", "Culet", "Type",
                           "Fluorescence"]

    # Завантаження даних
    data = load_and_clean_data(file_path)

    # Попередня обробка
    processed_data, preprocessor = preprocess_data(data, numeric_columns, categorical_columns)

    # Кластеризація
    clustered_data = perform_kmeans(pd.DataFrame(processed_data), n_clusters=3)

    # Виявлення аномалій
    analyzed_data = detect_anomalies(clustered_data)

    # Отримання списку аномалій
    anomalies = get_anomalies(analyzed_data)

    # Відновлення кластерів до оригінального масштабу
    restored_clusters = inverse_transform_clusters(processed_data, preprocessor, data)
    restored_clusters['Cluster'] = clustered_data['Cluster']

    # Відновлення аномалій
    restored_anomalies = restored_clusters[restored_clusters.index.isin(anomalies.index)].copy()
    restored_anomalies['Anomaly'] = 'Anomaly'

    # Сортування кластерів за стовпцем 'Cluster'
    sorted_clusters = restored_clusters.sort_values(by='Cluster')

    # Резюме кластерів
    cluster_summary = restored_clusters.groupby('Cluster').mean(numeric_only=True)
    print("Cluster Summary:")
    print(cluster_summary)

    # Збереження результатів у Excel
    sorted_clusters.to_excel("restored_clusters.xlsx", index=False)
    restored_anomalies.to_excel("restored_anomalies.xlsx", index=False)
    cluster_summary.to_excel("cluster_summary.xlsx")

    # Список аномалій
    print("\nAnomalies:")
    print(restored_anomalies)

    plot_clusters(
        data=restored_clusters,
        x_col='Carat Weight',
        y_col='Price',
        cluster_col='Cluster',
        title='Розподіл кластерів за вагою та ціною',
        x_label='Carat Weight',
        y_label='Price (USD)'
    )
