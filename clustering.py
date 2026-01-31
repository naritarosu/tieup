# clustering.py
"""
学習フェーズ：SentenceTransformer で埋め込み → KMeans でクラスタ付与 → CSV 保存。
さらに KMeans モデルを joblib で保存し、推論時に recommend.py が再学習せずに predict できるようにする。
"""
from pathlib import Path
import argparse
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ANIME_PATH = Path("anime.csv")
BRAND_PATH = Path("brand.csv")
ANIME_OUT = Path("anime_with_clusters.csv")
BRAND_OUT = Path("brand_with_clusters.csv")
ANIME_KM_PATH = Path("anime_kmeans.joblib")
BRAND_KM_PATH = Path("brand_kmeans.joblib")


def safe_get(row, col):
    return str(row[col]) if col in row else ""


def build_anime_text(row):
    title = safe_get(row, "title")
    genre = safe_get(row, "main_genre")
    tone = safe_get(row, "tone_tags")
    synopsis = safe_get(row, "synopsis")
    text = f"{title}。ジャンル：{genre}"
    if tone:
        text += f"。雰囲気：{tone}"
    if synopsis:
        text += f"。あらすじ：{synopsis}"
    return text


def build_brand_text(row):
    name = safe_get(row, "brand_name")
    category = safe_get(row, "category")
    personality = safe_get(row, "personality_tags")
    description = safe_get(row, "description")
    text = f"{name}。カテゴリ：{category}"
    if personality:
        text += f"。イメージ：{personality}"
    if description:
        text += f"。説明：{description}"
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_anime", type=int, default=3, help="アニメクラスタ数")
    parser.add_argument("--k_brand", type=int, default=4, help="ブランドクラスタ数")
    parser.add_argument("--auto_k", action="store_true", help="silhouette_score 最大のKを自動選択 (anime 3-7, brand 3-8)")
    args = parser.parse_args()

    anime_df = pd.read_csv(ANIME_PATH).fillna("")
    brand_df = pd.read_csv(BRAND_PATH).fillna("")

    print("Anime rows:", len(anime_df))
    print("Brand rows:", len(brand_df))

    anime_texts = anime_df.apply(build_anime_text, axis=1).tolist()
    brand_texts = brand_df.apply(build_brand_text, axis=1).tolist()

    print("\nLoading model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    print("\nアニメ埋め込み生成中...")
    anime_embeddings = model.encode(anime_texts, show_progress_bar=True)

    print("\nブランド埋め込み生成中...")
    brand_embeddings = model.encode(brand_texts, show_progress_bar=True)

    # K の決定（auto_k が指定された場合は silhouette で探索）
    def choose_k(embeddings, k_default, k_min, k_max):
        if not args.auto_k:
            return k_default, None
        best_k = None
        best_score = -1
        n = len(embeddings)
        for k in range(k_min, k_max + 1):
            if k <= 1 or k >= n:
                continue
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = km.fit_predict(embeddings)
            try:
                score = silhouette_score(embeddings, labels)
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_k = k
        return (best_k if best_k else k_default), best_score

    k_anime, sil_anime = choose_k(anime_embeddings, args.k_anime, 3, 7)
    k_brand, sil_brand = choose_k(brand_embeddings, args.k_brand, 3, 8)

    if sil_anime is not None:
        print(f"auto_k anime: K={k_anime}, silhouette={sil_anime:.4f}")
    else:
        print(f"anime: K={k_anime}")
    if sil_brand is not None:
        print(f"auto_k brand: K={k_brand}, silhouette={sil_brand:.4f}")
    else:
        print(f"brand: K={k_brand}")

    print(f"\nアニメを {k_anime} クラスタに分割します...")
    anime_km = KMeans(
        n_clusters=k_anime,
        random_state=42,
        n_init="auto",
    )
    anime_df["cluster"] = anime_km.fit_predict(anime_embeddings)

    print(f"\nブランドを {k_brand} クラスタに分割します...")
    brand_km = KMeans(
        n_clusters=k_brand,
        random_state=42,
        n_init="auto",
    )
    brand_df["cluster"] = brand_km.fit_predict(brand_embeddings)

    # 保存
    anime_df.to_csv(ANIME_OUT, index=False)
    brand_df.to_csv(BRAND_OUT, index=False)
    joblib.dump(anime_km, ANIME_KM_PATH)
    joblib.dump(brand_km, BRAND_KM_PATH)

    print("\n保存完了:")
    print(f" - {ANIME_OUT}")
    print(f" - {BRAND_OUT}")
    print(f" - {ANIME_KM_PATH}")
    print(f" - {BRAND_KM_PATH}")


if __name__ == "__main__":
    main()
