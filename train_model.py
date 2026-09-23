"""Демонстрация production-стандартов: пайплайн, SMOTE и воспроизводимость."""

from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def create_production_pipeline():
    """Создает и возвращает production-пайплайн с балансировкой SMOTE."""
    return ImbPipeline(
        [
            ("smote", SMOTE(random_state=42, sampling_strategy="auto")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=12,
                    min_samples_split=5,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main():
    """Основная функция запуска."""
    pipeline = create_production_pipeline()
    print(f"Production-пайплайн успешно создан и готов к работе:\n{pipeline}")


if __name__ == "__main__":
    main()
