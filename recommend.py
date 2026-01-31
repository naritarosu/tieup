import argparse
import warnings
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

ANIME_KM_PATH = Path("anime_kmeans.joblib")
ANIME_WITH_CLUSTERS = Path("anime_with_clusters.csv")
BRAND_WITH_CLUSTERS = Path("brand_with_clusters.csv")

TIEUP_PATH = Path("tieup.csv")
TIEUP_SOURCES_PATH = Path("tieup_sources.csv")
AFFINITY_PATH = Path("class_pair_affinity_long.csv")

SOURCE_PRIORITY = {"official": 0, "press": 1, "news": 2, "sns": 3}


def build_text(title: str, genre: str, tone: str, synopsis: str) -> str:
    return (
        f"{title}。ジャンル：{genre}"
        + (f"。雰囲気：{tone}" if tone else "")
        + (f"。あらすじ：{synopsis}" if synopsis else "")
    )


def load_affinity(anime_cluster: int) -> pd.DataFrame:
    if not AFFINITY_PATH.exists():
        warnings.warn("class_pair_affinity_long.csv not found. Run tieup_matrix.py first.")
        return pd.DataFrame()
    df = pd.read_csv(AFFINITY_PATH).fillna(0)

    for c in ["anime_cluster", "brand_cluster", "affinity_score", "weighted_count"]:
        if c not in df.columns:
            warnings.warn(f"class_pair_affinity_long.csv missing column: {c}")
            return pd.DataFrame()

    return df[df["anime_cluster"] == anime_cluster].sort_values(
        ["affinity_score", "weighted_count"], ascending=[False, False]
    )


def attach_clusters(tieup_df: pd.DataFrame, anime_df: pd.DataFrame, brand_df: pd.DataFrame) -> pd.DataFrame:
    out = tieup_df.merge(
        anime_df[["anime_id", "cluster"]],
        on="anime_id",
        how="left",
    ).rename(columns={"cluster": "anime_cluster"})
    out = out.merge(
        brand_df[["brand_id", "cluster"]],
        on="brand_id",
        how="left",
    ).rename(columns={"cluster": "brand_cluster"})
    return out


def attach_sources(tieup_df: pd.DataFrame) -> pd.DataFrame:
    if not TIEUP_SOURCES_PATH.exists():
        warnings.warn("tieup_sources.csv not found; evidence URLs will be limited to tieup.csv only.")
        return tieup_df
    sources = pd.read_csv(TIEUP_SOURCES_PATH).fillna("")
    if "tieup_id" not in tieup_df.columns or "tieup_id" not in sources.columns:
        warnings.warn("tieup_id missing in tieup.csv or tieup_sources.csv; cannot join evidence URLs.")
        return tieup_df
    return tieup_df.merge(sources, on="tieup_id", how="left")


def get_cluster_profile(anime_df: pd.DataFrame, anime_cluster: int, top_n_tones: int = 5):
    """
    指定した anime_cluster に多いジャンル・雰囲気タグ（tone_tags）を返す。
    返り値: (代表ジャンル: str, 代表トーンタグ: list[str])
    """
    if "cluster" not in anime_df.columns:
        return "", []

    subset = anime_df[anime_df["cluster"] == anime_cluster].copy()
    if subset.empty:
        return "", []

    genre_text = ""
    if "main_genre" in subset.columns:
        genres = subset["main_genre"].astype(str).fillna("").tolist()
        genres = [g for g in genres if g and g.lower() != "nan"]
        if genres:
            genre_text = Counter(genres).most_common(1)[0][0]

    tone_list = []
    if "tone_tags" in subset.columns:
        tone_counter = Counter()
        for tags in subset["tone_tags"].astype(str).fillna(""):
            for t in tags.split(","):
                t = t.strip()
                if t:
                    tone_counter[t] += 1
        tone_list = [t for t, _ in tone_counter.most_common(top_n_tones)]

    return genre_text, tone_list


def clean_tone_tags(tones, genre, max_n=3):
    """
    トーンタグを整理：
    - genre と同義・重複する語を除去
    - 最大 max_n までに制限
    """
    cleaned = []
    for t in tones:
        if not t:
            continue
        if genre and t in genre:
            continue
        cleaned.append(str(t).strip())

    uniq = []
    for t in cleaned:
        if t and t not in uniq:
            uniq.append(t)

    return uniq[:max_n]


def short_genre(g: str) -> str:
    """
    main_genre が '青春・アクション・ミリタリー' のように複合の場合、
    先頭2語までに抑える（区切りは '・' を想定）
    """
    if not g:
        return ""
    s = str(g).strip()
    parts = [p.strip() for p in s.split("・") if p.strip()]
    if len(parts) <= 2:
        return "・".join(parts)
    return "・".join(parts[:2])


def short_category(cat: str) -> str:
    """
    category をUI表示向けに短縮する:
    - かっこ書き（（...）や(...)）を削除
    - スラッシュ区切りは先頭だけ
    - 長すぎる場合は切る
    """
    if not cat:
        return ""

    s = str(cat).strip()

    for mark in ["（", "(", "[", "【"]:
        if mark in s:
            s = s.split(mark, 1)[0].strip()

    if "/" in s:
        s = s.split("/", 1)[0].strip()

    if len(s) > 14:
        s = s[:14] + "…"

    return s


def build_group_desc(cluster_genre: str, cluster_tones: list[str]) -> tuple[str, str, str]:
    """
    作品グループ判定を「読みやすい日本語」にするための要約を作る。
    返り値: (title_line, detail_line, cluster_note)
    """
    GENRE_PHRASE = {
        "SF": "近未来的な設定やテクノロジー要素",
        "サスペンス": "緊張感のある謎解き要素",
        "ミリタリー": "部隊行動や装備・戦術の要素",
        "ファンタジー": "異世界・魔法などの世界観要素",
        "ホラー": "不穏さや恐怖演出",
        "アクション": "戦闘・追跡など動きのある展開",
        "恋愛": "恋愛関係の進展や心理描写",
        "ドラマ": "人間関係や葛藤を軸にした展開",
    }
    TONE_PHRASE = {
        "ギャグ": "コミカルな表現",
        "コメディ": "笑いを重視した表現",
        "明るい": "明るいテンポ",
        "ほのぼの": "穏やかな空気感",
        "元気": "勢いのあるノリ",
        "熱血": "熱量の高い演出",
        "シリアス": "重めのドラマ性",
        "スタイリッシュ": "洗練された雰囲気",
        "ダーク": "暗めの世界観",
        "癒し": "落ち着く空気感",
    }

    genre_parts = [g.strip() for g in (cluster_genre or "").split("・") if g.strip()][:2]
    tone_parts = []
    seen = set()
    for t in (cluster_tones or []):
        t = str(t).strip()
        if t and t not in seen:
            seen.add(t)
            tone_parts.append(t)
    tone_parts = tone_parts[:3]

    phrases = []
    for g in genre_parts:
        phrases.append(GENRE_PHRASE.get(g, f"{g}要素"))
    for t in tone_parts[:2]:
        phrases.append(TONE_PHRASE.get(t, f"{t}な表現"))

    if phrases:
        title_line = "、".join(phrases) + "が目立つ作品が多いグループ"
    else:
        title_line = "作風やテーマが近い作品が多いグループ"

    if len(tone_parts) >= 3:
        t3 = tone_parts[2]
        detail_line = f"補足：このグループでは、{TONE_PHRASE.get(t3, t3)}が含まれる作品も比較的多い傾向があります。"
    else:
        detail_line = "補足：作品説明（ジャンル・雰囲気・あらすじ）の文章特徴をもとに、近い作品同士がまとめられています。"

    cluster_note = "※クラスタは、作品説明文の意味的な近さに基づいて自動的にまとめた「作品グループ」です。"

    return title_line, detail_line, cluster_note


def make_reason(row: pd.Series, w_affinity: float, w_sim: float) -> str:
    """
    読む人にわかりやすい理由文（結論→根拠→数値）
    """
    aff = float(row.get("affinity", 0.0))
    sim = float(row.get("sim", 0.0))

    aff_part = w_affinity * np.log1p(max(aff, 0.0))
    sim_part = w_sim * sim

    category = short_category(str(row.get("category", "") or "").strip())
    brand = str(row.get("brand_name", "") or "").strip()

    aff_txt = f"{aff:.1f}"
    sim_txt = f"{sim:.2f}"

    if aff_part >= sim_part:
        if category:
            return f"候補理由：この作品グループでは「{category}」のタイアップ実績が多いため（相性: {aff_txt}）。"
        return f"候補理由：この作品グループでは同系統の企業タイアップ実績が多いため（相性: {aff_txt}）。"

    if brand:
        return f"候補理由：作品内容と「{brand}」の企業イメージが近く、自然に訴求できそうなため（近さ: {sim_txt}）。"
    return f"候補理由：作品内容と企業側の説明が近く、自然に訴求できそうなため（近さ: {sim_txt}）。"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="", help="新作アニメのタイトル")
    parser.add_argument("--main_genre", default="", help="ジャンル")
    parser.add_argument("--tone_tags", default="", help="雰囲気タグ（カンマ区切り可）")
    parser.add_argument("--synopsis", default="", help="あらすじ")

    parser.add_argument("--top_k", type=int, default=5, help="最終ブランド件数")
    parser.add_argument("--top_brand_clusters", type=int, default=3, help="採用する上位ブランドクラスタ数")

    parser.add_argument("--w_affinity", type=float, default=1.0, help="score計算時の affinity 重み (log1pで使用)")
    parser.add_argument("--w_sim", type=float, default=1.0, help="score計算時の cosine 類似度重み")

    parser.add_argument("--max_sources", type=int, default=3, help="各タイアップで表示する最大URL数")
    parser.add_argument("--out_json", default="", help="推薦結果をJSONに保存（指定時）")

    args = parser.parse_args()

    if not ANIME_WITH_CLUSTERS.exists():
        raise FileNotFoundError(f"{ANIME_WITH_CLUSTERS} not found. Run clustering.py first.")
    if not BRAND_WITH_CLUSTERS.exists():
        raise FileNotFoundError(f"{BRAND_WITH_CLUSTERS} not found. Run clustering.py first.")
    if not ANIME_KM_PATH.exists():
        raise FileNotFoundError(f"{ANIME_KM_PATH} not found. Run clustering.py first.")

    anime_df = pd.read_csv(ANIME_WITH_CLUSTERS).fillna("")
    brand_df = pd.read_csv(BRAND_WITH_CLUSTERS).fillna("")
    tieup_df = pd.read_csv(TIEUP_PATH).fillna("") if TIEUP_PATH.exists() else pd.DataFrame()

    anime_km = joblib.load(ANIME_KM_PATH)

    model = SentenceTransformer(MODEL_NAME)
    new_text = build_text(args.title, args.main_genre, args.tone_tags, args.synopsis)
    new_emb = model.encode([new_text])

    anime_cluster = int(anime_km.predict(new_emb)[0])
    print(f"推定 anime_cluster: {anime_cluster}")

    cluster_genre_raw, raw_cluster_tones = get_cluster_profile(anime_df, anime_cluster, top_n_tones=5)
    cluster_genre = short_genre(cluster_genre_raw)
    cluster_tones = clean_tone_tags(raw_cluster_tones, cluster_genre, max_n=3)

    title_line, detail_line, cluster_note = build_group_desc(cluster_genre, cluster_tones)

    print("作品グループ判定")
    print(f"今回の作品は、{title_line}に近いと判定されました。")
    print(detail_line)
    print(cluster_note)
    print()

    affinity = load_affinity(anime_cluster)
    if affinity.empty:
        print("このアニメクラスタに対応する成功事例がありません。")
        return

    top_affinity = affinity.head(args.top_brand_clusters)
    print("\nブランドクラスタランキング (affinity_score, weighted_count):")
    for _, row in top_affinity.iterrows():
        print(
            f"  brand_cluster {int(row['brand_cluster'])}: "
            f"affinity={float(row['affinity_score']):.3f}, weighted_count={float(row['weighted_count']):.1f}"
        )

    if "cluster" not in brand_df.columns:
        raise RuntimeError("brand_with_clusters.csv must have 'cluster' column.")

    top_clusters = top_affinity["brand_cluster"].astype(int).tolist()
    candidates = brand_df[brand_df["cluster"].isin(top_clusters)]
    if candidates.empty:
        print("上位クラスタに属するブランドがありません。")
        return

    def build_brand_text(row: pd.Series) -> str:
        name = row.get("brand_name", "")
        category = row.get("category", "")
        personality = row.get("personality_tags", "")
        description = row.get("description", "")
        text = f"{name}。カテゴリ：{category}"
        if personality:
            text += f"。イメージ：{personality}"
        if description:
            text += f"。説明：{description}"
        return text

    brand_texts = candidates.apply(build_brand_text, axis=1).tolist()
    brand_emb = model.encode(brand_texts)
    sims = cosine_similarity(new_emb, brand_emb)[0]

    cluster_aff_map = {
        int(r["brand_cluster"]): float(r["affinity_score"])
        for _, r in affinity.iterrows()
    }

    scores = []
    for idx, (_, row) in enumerate(candidates.iterrows()):
        bc = int(row["cluster"])
        aff = cluster_aff_map.get(bc, 0.0)
        sim = float(sims[idx])
        score = args.w_affinity * np.log1p(max(aff, 0.0)) + args.w_sim * sim
        scores.append((score, aff, sim))

    candidates = candidates.copy()
    candidates["score"] = [s[0] for s in scores]
    candidates["affinity"] = [s[1] for s in scores]
    candidates["sim"] = [s[2] for s in scores]
    candidates = candidates.sort_values("score", ascending=False).head(args.top_k)

    candidates["reason"] = candidates.apply(
        lambda r: make_reason(r, args.w_affinity, args.w_sim),
        axis=1,
    )

    # JSON出力（UI用）
    if args.out_json:
        out = candidates.copy()
        out["group_title"] = title_line
        out["group_detail"] = detail_line
        out["cluster_note"] = cluster_note
        out["group_desc"] = title_line  # app互換（最低限）

        out_cols = [
            c for c in [
                "brand_id",
                "brand_name",
                "category",
                "personality_tags",
                "score",
                "affinity",
                "sim",
                "reason",
                "cluster",
                "group_desc",
                "group_title",
                "group_detail",
                "cluster_note",
            ] if c in out.columns
        ]
        out[out_cols].to_json(args.out_json, orient="records", force_ascii=False)

    print("\n具体ブランド Top{}:".format(args.top_k))
    cols_display = [
        c for c in [
            "brand_id",
            "brand_name",
            "category",
            "personality_tags",
            "score",
            "affinity",
            "sim",
            "reason",
        ] if c in candidates.columns
    ]
    print(
        candidates[cols_display].to_string(
            index=False,
            formatters={
                "score": lambda x: f"{float(x):.3f}",
                "affinity": lambda x: f"{float(x):.3f}",
                "sim": lambda x: f"{float(x):.3f}",
            },
        )
    )

    # 根拠事例の表示（任意）
    if tieup_df.empty:
        warnings.warn("tieup.csv not found; evidence cannot be shown.")
        return

    tieup_with_clusters = attach_clusters(tieup_df, anime_df, brand_df)
    tieup_with_sources = attach_sources(tieup_with_clusters)

    if "collab_type" not in tieup_with_sources.columns and "notes" not in tieup_with_sources.columns:
        warnings.warn("collab_type/notes missing; limited evidence text.")

    print("\n--- Evidence (optional) ---")
    for _, brand_row in candidates.iterrows():
        bid = brand_row.get("brand_id")
        bc = brand_row.get("cluster")
        print(f"\n[brand_id={bid}] {brand_row.get('brand_name')}")

        subset = (
            tieup_with_sources[tieup_with_sources["brand_id"] == bid]
            if "brand_id" in tieup_with_sources.columns
            else pd.DataFrame()
        )
        if subset.empty and "brand_cluster" in tieup_with_sources.columns:
            subset = tieup_with_sources[tieup_with_sources["brand_cluster"] == bc]

        subset = subset.head(3)
        if subset.empty:
            print("  (no tieup evidence)")
            continue

        if "tieup_id" not in subset.columns:
            for _, row in subset.iterrows():
                print(f"  - anime_id={row.get('anime_id','')} | collab_type={row.get('collab_type','')}")
            continue

        for tieup_id, group in subset.groupby("tieup_id"):
            first = group.iloc[0]
            print(
                f"  - tieup_id={tieup_id} | anime_id={first.get('anime_id')} | collab_type={first.get('collab_type','')}"
            )

            if "url" in group.columns:
                def url_priority(u):
                    if "source_type" not in group.columns:
                        return 99
                    st_series = group.loc[group["url"] == u, "source_type"]
                    st = st_series.iloc[0] if not st_series.empty else ""
                    return SOURCE_PRIORITY.get(str(st), 99)

                sorted_urls = sorted(group["url"].dropna().unique(), key=url_priority)
                for u in sorted_urls[: args.max_sources]:
                    if "source_type" in group.columns:
                        st_series = group.loc[group["url"] == u, "source_type"]
                        stype = st_series.iloc[0] if not st_series.empty else ""
                    else:
                        stype = ""
                    print(f"      source[{stype}]: {u}")

            evidence = first.get("notes", "")
            if "evidence_text" in first:
                evidence = first.get("evidence_text") or evidence
            if evidence:
                print(f"      evidence: {evidence}")


if __name__ == "__main__":
    main()
