from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
RANDOM_STATE = 42

def build_model() -> VotingClassifier:
    """
    Three complementary weak-ish learners, soft-voted.

    Why an ensemble and not one big model: with 282 training calls, any single
    model's decision boundary is high-variance. Averaging three different
    inductive biases (linear, bagged trees, boosted trees) is the cheapest
    variance reduction available, and the judging criterion is robustness on
    unseen speakers and unseen TTS engines.
    """
    logreg = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000, C=0.5, class_weight="balanced",
            random_state=RANDOM_STATE)),
    ])

    rf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=1)),
    ])

    hgb = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300,
            l2_regularization=1.0, min_samples_leaf=10,
            random_state=RANDOM_STATE)),
    ])

    return VotingClassifier(
        [("logreg", logreg), ("rf", rf), ("hgb", hgb)], voting="soft"
    )
