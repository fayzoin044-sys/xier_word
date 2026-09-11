import argparse
from pathlib import Path

import joblib
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from data_utils import TextClassificationDataset, load_labels


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="TF-IDF 随机森林新闻分类")
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--labels", type=Path,
                        default=project_dir / "data" / "labels.txt")
    parser.add_argument(
        "--output", type=Path, default=project_dir / "outputs" / "random_forest.joblib"
    )
    parser.add_argument("--estimators", type=int, default=200)
    parser.add_argument("--max-features", type=int, default=30000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    train_data = TextClassificationDataset(
        args.data_dir / "train.txt", len(labels))
    test_data = TextClassificationDataset(
        args.data_dir / "test.txt", len(labels))
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(1, 2),
                    max_features=args.max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=args.estimators,
                    n_jobs=-1,
                    random_state=42,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )
    train_texts, train_labels = zip(*train_data.rows)
    test_texts, test_labels = zip(*test_data.rows)
    pipeline.fit(train_texts, train_labels)
    predictions = pipeline.predict(test_texts)
    print(f"accuracy={metrics.accuracy_score(test_labels, predictions):.4f}")
    print(
        metrics.classification_report(
            test_labels,
            predictions,
            target_names=labels,
            digits=4,
            zero_division=0,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.output)


if __name__ == "__main__":
    main()
